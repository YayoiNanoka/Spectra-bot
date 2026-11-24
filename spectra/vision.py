"""Vision image captioner with AstrBot-aligned 3x3 animated montage and SQLite cache.

Image preprocessing, constants, and byte-budget encoding directly follow
AstrBot's ``media_utils.py``. Owns the ``image_cache`` SQLite table via
:meth:`spectra.db.Database.migrate`.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import hashlib
import io
import logging
import math
import os
import ssl
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import httpx
import openai
from openai import AsyncOpenAI
from PIL import Image, ImageOps, UnidentifiedImageError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tenacity.wait import wait_base

from spectra.config import VisionConfig
from spectra.db import Database
from spectra.providers import OpenAICompatProfile, resolve_openai_compat

logger = logging.getLogger(__name__)

_MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE image_cache (
        sha256 TEXT PRIMARY KEY,
        caption TEXT NOT NULL,
        updated_at INTEGER NOT NULL
    );
    """,
)

# Constants matched directly from AstrBot's media_utils.py.
IMAGE_COMPRESS_DEFAULT_MAX_SIZE = 1280
MODEL_IMAGE_MAX_INPUT_BYTES = 64 * 1024 * 1024
MODEL_IMAGE_MAX_BYTES = 512 * 1024
ANIMATED_MONTAGE_GRID = 3
ANIMATED_MONTAGE_FRAME_COUNT = ANIMATED_MONTAGE_GRID * ANIMATED_MONTAGE_GRID

_MONTAGE_PROMPT_NOTE = (
    "（注：这是一张动图或动态表情包按时间顺序均匀抽取的3×3帧网格图，"
    "请结合整体动作变化和文字内容进行简短描述。）"
)
_TRANSIENT_OPENAI_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


class _ImageTooLargeError(ValueError):
    """Raised when an input image exceeds ``MODEL_IMAGE_MAX_INPUT_BYTES``."""


def _read_local_file_sync(path: Path, max_bytes: int) -> bytes:
    """Read a regular local file with a strict upper bound in a worker thread."""
    if not path.is_file():
        raise FileNotFoundError(f"Local image file does not exist: {path}")
    with path.open("rb") as f:
        data = f.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise _ImageTooLargeError(
            f"Local image exceeds MODEL_IMAGE_MAX_INPUT_BYTES ({max_bytes})."
        )
    return data


def _even_frame_indices(total_frames: int, max_frames: int) -> list[int]:
    """Pick evenly spaced animation frames, including both endpoints.

    Args:
        total_frames: Number of animation frames, excluding an independent cover.
        max_frames: Maximum number of frames to select.

    Returns:
        Ascending, unique frame indices.
    """
    count = min(max_frames, total_frames)
    if count <= 1:
        return [0]
    return sorted({round(i * (total_frames - 1) / (count - 1)) for i in range(count)})


def _encode_image_frame_bytes(
    image: Image.Image,
    max_size: int | None = None,
) -> tuple[bytes, str]:
    """Encode an owned frame below ``MODEL_IMAGE_MAX_BYTES`` preserving transparency.

    Directly mirrors AstrBot's ``_encode_image_frame_bytes``.

    Args:
        image: Decoded frame owned by the caller; its pixels and metadata may change.
        max_size: Optional longest-edge limit; smaller images are not enlarged.

    Returns:
        Tuple of ``(encoded_bytes, mime_type)`` strictly below
        ``MODEL_IMAGE_MAX_BYTES``.

    Raises:
        ValueError: If the image cannot fit the byte limit even at one pixel.
    """
    ImageOps.exif_transpose(image, in_place=True)
    has_alpha = "A" in image.getbands() or "transparency" in image.info
    icc_profile = image.info.get("icc_profile")
    if image.mode not in {"RGB", "RGBA", "L", "LA", "P", "1", "I", "I;16"}:
        # A source profile is invalid after a color-space conversion such as CMYK.
        icc_profile = None
    prepared = image
    try:
        if has_alpha:
            if image.mode != "RGBA":
                prepared = image.convert("RGBA")
        elif image.mode not in {"RGB", "L", "I", "I;16"}:
            prepared = image.convert("L" if image.mode == "1" else "RGB")
        if max_size is not None:
            prepared.thumbnail(
                (max_size, max_size),
                Image.Resampling.LANCZOS,
                reducing_gap=None if prepared.mode == "I;16" else 2.0,
            )
        if prepared.mode in {"I", "I;16"}:
            # Normalize high-bit-depth pixels instead of clipping them to 255.
            low, high = prepared.getextrema()
            if (
                isinstance(low, int | float)
                and isinstance(high, int | float)
                and high > low
            ):
                with prepared.point(
                    lambda v: (v - low) * (255.0 / (high - low))
                ) as scaled:
                    prepared = scaled.convert("L")
            else:
                prepared = prepared.convert("L")
        # Do not carry EXIF, text chunks, or other unbounded metadata into previews.
        prepared.info.clear()
        save_kwargs: dict[str, Any] = (
            {"icc_profile": icc_profile} if icc_profile else {}
        )
        mime_type = "image/png" if has_alpha else "image/jpeg"
        while True:
            with io.BytesIO() as buffer:
                if has_alpha:
                    prepared.save(buffer, "PNG", **save_kwargs)
                else:
                    prepared.save(
                        buffer, "JPEG", quality=85, optimize=True, **save_kwargs
                    )
                data = buffer.getvalue()
            if len(data) < MODEL_IMAGE_MAX_BYTES:
                return data, mime_type
            if save_kwargs:
                # Oversized profiles must not defeat the byte limit.
                save_kwargs.clear()
                continue
            if prepared.size == (1, 1):
                raise ValueError("Image cannot fit the model input byte limit")
            scale = min(
                0.85,
                math.sqrt((MODEL_IMAGE_MAX_BYTES - 1) / len(data)) * 0.95,
            )
            prepared.thumbnail(
                (
                    max(1, int(prepared.width * scale)),
                    max(1, int(prepared.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
    finally:
        if prepared is not image:
            prepared.close()


def _extract_animation_montage_sync(image: Image.Image, max_size: int) -> bytes:
    """Tile evenly spaced animation frames into one white 3x3 JPEG preview.

    Directly mirrors AstrBot's ``_extract_animation_montage_sync``.

    Args:
        image: Opened animation owned by the caller.
        max_size: Longest-edge limit of the montage, without upscaling.

    Returns:
        JPEG bytes strictly below ``MODEL_IMAGE_MAX_BYTES`` with unused cells white.
    """
    total_frames = int(getattr(image, "n_frames", 1))
    # APNG's independent default image is a cover, not an animation frame.
    first_frame = 1 if image.info.get("default_image", False) else 0
    frame_indices = [
        first_frame + index
        for index in _even_frame_indices(
            total_frames - first_frame,
            ANIMATED_MONTAGE_FRAME_COUNT,
        )
    ]
    image.seek(first_frame)
    oriented_first = ImageOps.exif_transpose(image)
    try:
        display_size = oriented_first.size
    finally:
        if oriented_first is not image:
            oriented_first.close()

    # Floor the per-cell scale so the montage never exceeds max_size.
    longest_edge = max(display_size) * ANIMATED_MONTAGE_GRID
    scale = min(1.0, max(max_size, 1) / longest_edge)
    cell_size = (
        max(1, int(display_size[0] * scale)),
        max(1, int(display_size[1] * scale)),
    )
    canvas = Image.new(
        "RGB",
        (
            cell_size[0] * ANIMATED_MONTAGE_GRID,
            cell_size[1] * ANIMATED_MONTAGE_GRID,
        ),
        (255, 255, 255),
    )
    try:
        for out_index, frame_index in enumerate(frame_indices):
            image.seek(frame_index)
            oriented = ImageOps.exif_transpose(image)
            try:
                with oriented.convert("RGBA") as frame:
                    resized = frame
                    try:
                        if frame.size != cell_size:
                            resized = frame.resize(
                                cell_size,
                                Image.Resampling.LANCZOS,
                            )
                        # Alpha shows the white canvas through transparent pixels.
                        canvas.paste(
                            resized,
                            (
                                (out_index % ANIMATED_MONTAGE_GRID) * cell_size[0],
                                (out_index // ANIMATED_MONTAGE_GRID) * cell_size[1],
                            ),
                            resized,
                        )
                    finally:
                        if resized is not frame:
                            resized.close()
            finally:
                if oriented is not image:
                    oriented.close()
        encoded, _ = _encode_image_frame_bytes(canvas)
    finally:
        canvas.close()
    return encoded


def prepare_image_sync(
    source_bytes: bytes,
    max_size: int = IMAGE_COMPRESS_DEFAULT_MAX_SIZE,
) -> tuple[bytes, str, bool]:
    """Decode and prepare raw image bytes for vision model submission.

    Directly mirrors AstrBot's ``_prepare_model_image_sync``. Animated images
    become a white 3x3 JPEG montage; compliant static PNG/JPEG images below
    ``MODEL_IMAGE_MAX_BYTES`` with normal EXIF orientation pass through as-is;
    other static images are oriented, downscaled, and re-encoded strictly below
    ``MODEL_IMAGE_MAX_BYTES``.

    Args:
        source_bytes: Raw image file bytes.
        max_size: Maximum longest edge in pixels for still images and montages.

    Returns:
        Tuple of ``(encoded_bytes, mime_type, is_montage)``.

    Raises:
        PIL.UnidentifiedImageError: If bytes cannot be recognized as an image.
        PIL.Image.DecompressionBombError: If Pillow's built-in bomb limit trips.
        OSError: If decoding fails.
        ValueError: If ``max_size`` is invalid or image cannot fit byte limit.
    """
    if max_size < 1:
        raise ValueError(f"max_size must be >= 1, got {max_size}.")

    with Image.open(io.BytesIO(source_bytes)) as image:
        if getattr(image, "n_frames", 1) > 1:
            return _extract_animation_montage_sync(image, max_size), "image/jpeg", True
        if (
            image.format in {"PNG", "JPEG"}
            and len(source_bytes) < MODEL_IMAGE_MAX_BYTES
            and max(image.size) <= max_size
            and image.getexif().get(274, 1) == 1
        ):
            # Validate even byte-identical passthroughs without decoding twice.
            image.load()
            mime_type = "image/png" if image.format == "PNG" else "image/jpeg"
            return source_bytes, mime_type, False
        encoded_bytes, mime_type = _encode_image_frame_bytes(image, max_size)
        return encoded_bytes, mime_type, False


def _decode_base64_payload(payload: str, max_input_bytes: int) -> bytes:
    """Decode base64 payload with size guard and missing-padding tolerance."""
    compact = "".join(payload.split())
    if (len(compact) * 3) // 4 > max_input_bytes + 3:
        raise _ImageTooLargeError("Base64 image payload exceeds input byte limit.")
    rem = len(compact) % 4
    if rem:
        compact += "=" * (4 - rem)
    try:
        decoded = base64.b64decode(compact)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image payload.") from exc
    if len(decoded) > max_input_bytes:
        raise _ImageTooLargeError("Decoded image exceeds input byte limit.")
    return decoded


def _file_uri_to_path(file_uri: str) -> Path:
    parsed = urlparse(file_uri)
    path = url2pathname(parsed.path or "")
    if parsed.netloc and parsed.netloc.lower() != "localhost":
        path = url2pathname(f"//{parsed.netloc}{parsed.path or ''}")
    elif os.name != "nt" and path.startswith("//"):
        path = "/" + path.lstrip("/")
    return Path(path)


def _is_ssl_error(exc: BaseException) -> bool:
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ssl.SSLError):
            return True
        if "CERTIFICATE_VERIFY_FAILED" in str(cur):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _normalize_caption(raw: str) -> str:
    """Collapse whitespace/newlines and strip outer bracket wrappers."""
    text = " ".join(raw.split()).strip()
    while len(text) >= 2 and (
        (text[0] == "[" and text[-1] == "]") or (text[0] == "【" and text[-1] == "】")
    ):
        text = text[1:-1].strip()
    return text


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    return ""


class ImageCaptioner:
    """Downloads, preprocesses, caches, and captions images via an OpenAI vision model.

    Args:
        db: Connected :class:`spectra.db.Database` instance.
        config: Vision captioner configuration.
        http_client: Optional externally managed :class:`httpx.AsyncClient`.
        openai_client: Optional externally managed :class:`openai.AsyncOpenAI`.
        retry_wait: Optional tenacity wait strategy override (useful in tests).
    """

    def __init__(
        self,
        db: Database,
        config: VisionConfig,
        http_client: httpx.AsyncClient | None = None,
        openai_client: AsyncOpenAI | None = None,
        *,
        retry_wait: wait_base | None = None,
    ) -> None:
        self._db = db
        self._config = config
        self._compat: OpenAICompatProfile = resolve_openai_compat(
            base_url=config.base_url,
            model=config.model,
            custom_headers=config.custom_headers,
            extra_body=config.extra_body,
        )
        self._http_client = http_client
        self._owns_http_client = http_client is None
        self._openai_client = openai_client
        self._owns_openai_client = openai_client is None
        self._retry_wait = retry_wait or wait_exponential(
            multiplier=0.5,
            min=0.5,
            max=4.0,
        )
        self._inflight: dict[str, asyncio.Task[str | None]] = {}

    @property
    def compat(self) -> OpenAICompatProfile:
        """Return the resolved gateway compatibility profile."""
        return self._compat

    async def init(self) -> None:
        """Apply schema migrations for the ``image_cache`` module."""
        await self._db.migrate("image_cache", _MIGRATIONS)

    async def close(self) -> None:
        """Close owned HTTP and OpenAI clients."""
        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        if self._owns_openai_client and self._openai_client is not None:
            await self._openai_client.close()
            self._openai_client = None

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=self._config.timeout_sec,
                follow_redirects=True,
            )
        return self._http_client

    def _get_openai_client(self) -> AsyncOpenAI:
        if self._openai_client is None:
            self._openai_client = AsyncOpenAI(
                api_key=self._config.api_key,
                base_url=self._compat.base_url or None,
                default_headers=dict(self._compat.default_headers),
                timeout=self._config.timeout_sec,
                max_retries=0,
            )
        return self._openai_client

    async def _stream_download(
        self,
        client: httpx.AsyncClient,
        url: str,
    ) -> bytes:
        max_bytes = MODEL_IMAGE_MAX_INPUT_BYTES
        buf = bytearray()
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            if content_length is not None:
                try:
                    if int(content_length) > max_bytes:
                        raise _ImageTooLargeError(
                            f"Remote image Content-Length ({content_length}) "
                            f"exceeds {max_bytes}."
                        )
                except ValueError:
                    pass
            async for chunk in response.aiter_bytes():
                if len(buf) + len(chunk) > max_bytes:
                    raise _ImageTooLargeError(
                        f"Remote image exceeded MODEL_IMAGE_MAX_INPUT_BYTES "
                        f"({max_bytes}) during streaming."
                    )
                buf.extend(chunk)
        return bytes(buf)

    async def _load_source_bytes(self, source: str) -> bytes:
        ref = source.strip()
        if not ref:
            raise ValueError("Empty image source reference.")

        max_bytes = MODEL_IMAGE_MAX_INPUT_BYTES

        if ref.startswith("base64://"):
            return _decode_base64_payload(ref.removeprefix("base64://"), max_bytes)

        if ref.startswith("data:"):
            header, sep, payload = ref.partition(",")
            if not sep:
                raise ValueError("Malformed data URI.")
            parts = header[5:].split(";")
            if not any(part.strip().lower() == "base64" for part in parts[1:]):
                raise ValueError("Only base64-encoded data URIs are supported.")
            return _decode_base64_payload(payload, max_bytes)

        scheme = urlparse(ref).scheme.lower()
        if scheme == "file":
            file_path = _file_uri_to_path(ref)
            return await asyncio.to_thread(_read_local_file_sync, file_path, max_bytes)

        if scheme in ("http", "https"):
            client = self._get_http_client()
            try:
                return await self._stream_download(client, ref)
            except httpx.ConnectError as exc:
                if self._owns_http_client and _is_ssl_error(exc):
                    logger.warning(
                        "SSL verification failed downloading image; "
                        "retrying without TLS verify."
                    )
                    async with httpx.AsyncClient(
                        timeout=self._config.timeout_sec,
                        follow_redirects=True,
                        verify=False,
                    ) as insecure_client:
                        return await self._stream_download(insecure_client, ref)
                raise

        raise ValueError(f"Unsupported image source scheme: {scheme!r}")

    async def _get_cached_caption(self, digest: str) -> str | None:
        row = await self._db.fetchone(
            "SELECT caption FROM image_cache WHERE sha256 = ?",
            (digest,),
        )
        return str(row["caption"]) if row is not None else None

    async def _store_cached_caption(self, digest: str, caption: str) -> None:
        await self._db.execute(
            """
            INSERT INTO image_cache (sha256, caption, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(sha256) DO UPDATE SET
                caption = excluded.caption,
                updated_at = excluded.updated_at
            """,
            (digest, caption, int(time.time())),
        )

    async def _request_caption(self, data_url: str, prompt: str) -> str:
        client = self._get_openai_client()
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "timeout": self._config.timeout_sec,
        }
        req_headers = self._compat.build_request_headers(
            None,
            default_session="spectra:vision",
        )
        if req_headers:
            kwargs["extra_headers"] = req_headers
        reasoning_effort = self._config.reasoning_effort.strip()
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if self._compat.extra_body:
            kwargs["extra_body"] = copy.deepcopy(self._compat.extra_body)

        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(_TRANSIENT_OPENAI_ERRORS),
            stop=stop_after_attempt(3),
            wait=self._retry_wait,
            reraise=True,
        ):
            with attempt:
                response = await client.chat.completions.create(**kwargs)
                if not response.choices:
                    return ""
                return _extract_message_text(response.choices[0].message.content)
        return ""

    async def _caption_uncached(self, digest: str, raw_bytes: bytes) -> str | None:
        try:
            cached = await self._get_cached_caption(digest)
            if cached is not None:
                return cached

            prepared_bytes, mime_type, is_montage = await asyncio.to_thread(
                prepare_image_sync,
                raw_bytes,
                self._config.max_size,
            )
            encoded = base64.b64encode(prepared_bytes).decode("ascii")
            data_url = f"data:{mime_type};base64,{encoded}"
            prompt = (
                f"{self._config.prompt}\n{_MONTAGE_PROMPT_NOTE}"
                if is_montage
                else self._config.prompt
            )
            raw_caption = await self._request_caption(data_url, prompt)
            caption = _normalize_caption(raw_caption)
            if not caption:
                return None
            await self._store_cached_caption(digest, caption)
            return caption
        except (
            UnidentifiedImageError,
            Image.DecompressionBombError,
            openai.OpenAIError,
            OSError,
            ValueError,
        ) as exc:
            logger.warning("Failed to caption image (%s): %s", type(exc).__name__, exc)
            return None
        except Exception as exc:
            logger.warning("Unexpected error captioning image: %s", exc)
            return None

    async def caption(self, source: str) -> str | None:
        """Caption an image from a URL, data URI, ``base64://``, or ``file://`` URI.

        Returns cached captions from SQLite on SHA-256 hits and deduplicates
        concurrent in-flight requests for identical image hashes.

        Args:
            source: Image reference string.

        Returns:
            Normalized caption text, or ``None`` if vision is disabled,
            unconfigured, or captioning fails.
        """
        if not self._config.enabled or not self._config.api_key.strip():
            return None

        try:
            raw_bytes = await self._load_source_bytes(source)
        except Exception as exc:
            logger.warning(
                "Failed to load image source (%s): %s", type(exc).__name__, exc
            )
            return None

        if not raw_bytes:
            return None

        digest = hashlib.sha256(raw_bytes).hexdigest()
        cached = await self._get_cached_caption(digest)
        if cached is not None:
            return cached

        task = self._inflight.get(digest)
        if task is None:
            task = asyncio.create_task(self._caption_uncached(digest, raw_bytes))
            self._inflight[digest] = task

            def _cleanup(
                finished: asyncio.Task[str | None],
                key: str = digest,
            ) -> None:
                if self._inflight.get(key) is finished:
                    self._inflight.pop(key, None)

            task.add_done_callback(_cleanup)

        return await task
