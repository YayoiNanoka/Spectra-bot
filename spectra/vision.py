"""Vision image captioner with AstrBot-style 3x3 animated montage and SQLite cache.

Owns the ``image_cache`` SQLite table via :meth:`spectra.db.Database.migrate`.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import io
import logging
import os
import ssl
import time
import warnings
from dataclasses import dataclass
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

_MONTAGE_GRID = 3
_MONTAGE_MAX_FRAMES = _MONTAGE_GRID * _MONTAGE_GRID
_MAX_IMAGE_PIXELS = 40_000_000
_MAX_ANIMATION_FRAMES_SCANNED = 512
_MAX_ANIMATION_TOTAL_PIXELS = 250_000_000
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
    """Raised when an input image exceeds ``VisionConfig.max_input_bytes``."""


@dataclass(slots=True)
class _InflightEntry:
    task: asyncio.Task[str | None]
    waiters: int = 0


def _read_local_file_sync(path: Path, max_bytes: int) -> bytes:
    """Read a regular local file with a strict upper bound in a worker thread."""
    if not path.is_file():
        raise FileNotFoundError(f"Local image file does not exist: {path}")
    with path.open("rb") as f:
        data = f.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise _ImageTooLargeError(f"Local image exceeds max_input_bytes ({max_bytes}).")
    return data


def _even_frame_indices(total_frames: int, max_frames: int) -> list[int]:
    """Return ascending, unique evenly spaced frame indices including endpoints."""
    count = min(max_frames, total_frames)
    if count <= 1:
        return [0]
    return sorted({round(i * (total_frames - 1) / (count - 1)) for i in range(count)})


def _check_pixel_budget(image: Image.Image) -> None:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")
    if width * height > _MAX_IMAGE_PIXELS:
        raise Image.DecompressionBombError(
            f"Image pixel count ({width * height}) exceeds limit ({_MAX_IMAGE_PIXELS})."
        )


def _extract_animation_montage_sync(
    image: Image.Image,
    max_image_edge: int,
    *,
    first_frame: int,
    total_frames: int,
) -> bytes:
    """Composite up to 9 evenly spaced animation frames onto a white 3x3 JPEG."""
    frame_indices = [
        first_frame + idx
        for idx in _even_frame_indices(total_frames - first_frame, _MONTAGE_MAX_FRAMES)
    ]
    image.seek(first_frame)
    _check_pixel_budget(image)
    oriented_first = ImageOps.exif_transpose(image)
    try:
        display_size = oriented_first.size
    finally:
        if oriented_first is not image:
            oriented_first.close()

    longest_edge = max(display_size) * _MONTAGE_GRID
    scale = min(1.0, max(max_image_edge, 1) / longest_edge)
    cell_size = (
        max(1, int(display_size[0] * scale)),
        max(1, int(display_size[1] * scale)),
    )
    canvas = Image.new(
        "RGB",
        (cell_size[0] * _MONTAGE_GRID, cell_size[1] * _MONTAGE_GRID),
        (255, 255, 255),
    )
    try:
        for out_index, frame_index in enumerate(frame_indices):
            image.seek(frame_index)
            _check_pixel_budget(image)
            oriented = ImageOps.exif_transpose(image)
            try:
                frame = oriented.convert("RGBA")
                try:
                    resized = (
                        frame.resize(cell_size, Image.Resampling.LANCZOS)
                        if frame.size != cell_size
                        else frame
                    )
                    try:
                        canvas.paste(
                            resized,
                            (
                                (out_index % _MONTAGE_GRID) * cell_size[0],
                                (out_index // _MONTAGE_GRID) * cell_size[1],
                            ),
                            resized,
                        )
                    finally:
                        if resized is not frame:
                            resized.close()
                finally:
                    frame.close()
            finally:
                if oriented is not image:
                    oriented.close()

        with io.BytesIO() as buf:
            canvas.save(buf, format="JPEG", quality=90, optimize=True)
            return buf.getvalue()
    finally:
        canvas.close()


def _encode_static_image_sync(
    image: Image.Image,
    max_image_edge: int,
) -> tuple[bytes, str]:
    """Orient, downscale, and encode a static image as PNG (alpha) or JPEG."""
    _check_pixel_budget(image)
    oriented = ImageOps.exif_transpose(image)
    prepared = oriented
    try:
        has_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
        if has_alpha:
            if oriented.mode != "RGBA":
                prepared = oriented.convert("RGBA")
        elif oriented.mode in {"I", "I;16"}:
            low, high = oriented.getextrema()
            if (
                isinstance(low, int | float)
                and isinstance(high, int | float)
                and high > low
            ):
                with oriented.point(
                    lambda v: (v - low) * (255.0 / (high - low))
                ) as scaled:
                    prepared = scaled.convert("L")
            else:
                prepared = oriented.convert("L")
        elif oriented.mode not in {"RGB", "L"}:
            prepared = oriented.convert("L" if oriented.mode == "1" else "RGB")

        if max(prepared.size) > max_image_edge:
            if prepared is oriented:
                prepared = oriented.copy()
            prepared.thumbnail(
                (max_image_edge, max_image_edge),
                Image.Resampling.LANCZOS,
            )

        prepared.info.clear()
        with io.BytesIO() as buf:
            if has_alpha:
                prepared.save(buf, format="PNG")
                return buf.getvalue(), "image/png"
            prepared.save(buf, format="JPEG", quality=90, optimize=True)
            return buf.getvalue(), "image/jpeg"
    finally:
        if prepared is not oriented and prepared is not image:
            prepared.close()
        if oriented is not image:
            oriented.close()


def prepare_image_sync(
    source_bytes: bytes,
    max_image_edge: int = 1280,
) -> tuple[bytes, str, bool]:
    """Decode and prepare raw image bytes for vision model submission.

    Animated images (GIF, APNG, animated WebP) are sampled into up to 9 evenly
    spaced frames composited on a white 3x3 JPEG canvas. Static images are
    EXIF-transposed, optionally downscaled to ``max_image_edge``, and encoded
    as PNG (when alpha is present) or JPEG.

    Args:
        source_bytes: Raw image file bytes.
        max_image_edge: Maximum longest edge in pixels for the output image.

    Returns:
        Tuple of ``(encoded_bytes, mime_type, is_montage)``.

    Raises:
        PIL.UnidentifiedImageError: If bytes cannot be recognized as an image.
        PIL.Image.DecompressionBombError: If dimensions exceed pixel limits.
        OSError: If decoding fails.
        ValueError: If image dimensions are invalid.
    """
    if max_image_edge < 1:
        raise ValueError(f"max_image_edge must be >= 1, got {max_image_edge}.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(source_bytes)) as image:
            _check_pixel_budget(image)
            total_frames = int(getattr(image, "n_frames", 1))
            first_frame = 1 if image.info.get("default_image", False) else 0
            if total_frames > 1 and (total_frames - first_frame) > 1:
                if (
                    total_frames > _MAX_ANIMATION_FRAMES_SCANNED
                    or total_frames * image.width * image.height
                    > _MAX_ANIMATION_TOTAL_PIXELS
                ):
                    raise Image.DecompressionBombError(
                        f"Animation decode budget exceeded ({total_frames} frames, "
                        f"{image.width}x{image.height})."
                    )
                montage_bytes = _extract_animation_montage_sync(
                    image,
                    max_image_edge,
                    first_frame=first_frame,
                    total_frames=total_frames,
                )
                return montage_bytes, "image/jpeg", True

            if first_frame > 0 and first_frame < total_frames:
                image.seek(first_frame)
            encoded_bytes, mime_type = _encode_static_image_sync(image, max_image_edge)
            return encoded_bytes, mime_type, False


def _decode_base64_payload(payload: str, max_input_bytes: int) -> bytes:
    """Decode base64 payload with size guard and missing-padding tolerance."""
    compact = "".join(payload.split())
    # Upper bound on decoded size before allocating decoded buffer.
    if (len(compact) * 3) // 4 > max_input_bytes + 3:
        raise _ImageTooLargeError("Base64 image payload exceeds max_input_bytes.")
    rem = len(compact) % 4
    if rem:
        compact += "=" * (4 - rem)
    try:
        decoded = base64.b64decode(compact)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image payload.") from exc
    if len(decoded) > max_input_bytes:
        raise _ImageTooLargeError("Decoded image exceeds max_input_bytes.")
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
        self._http_client = http_client
        self._owns_http_client = http_client is None
        self._openai_client = openai_client
        self._owns_openai_client = openai_client is None
        self._retry_wait = retry_wait or wait_exponential(
            multiplier=0.5,
            min=0.5,
            max=4.0,
        )
        self._inflight: dict[str, _InflightEntry] = {}

    async def init(self) -> None:
        """Apply schema migrations for the ``image_cache`` module."""
        await self._db.migrate("image_cache", _MIGRATIONS)

    async def close(self) -> None:
        """Cancel in-flight caption tasks and close owned HTTP/OpenAI clients."""
        entries = list(self._inflight.values())
        self._inflight.clear()
        tasks = [entry.task for entry in entries if not entry.task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

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
                base_url=self._config.base_url,
                timeout=self._config.timeout_sec,
                max_retries=0,
            )
        return self._openai_client

    async def _stream_download(
        self,
        client: httpx.AsyncClient,
        url: str,
    ) -> bytes:
        max_bytes = self._config.max_input_bytes
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
                        f"Remote image exceeded max_input_bytes ({max_bytes}) "
                        "during streaming."
                    )
                buf.extend(chunk)
        return bytes(buf)

    async def _load_source_bytes(self, source: str) -> bytes:
        ref = source.strip()
        if not ref:
            raise ValueError("Empty image source reference.")

        max_bytes = self._config.max_input_bytes

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
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(_TRANSIENT_OPENAI_ERRORS),
            stop=stop_after_attempt(3),
            wait=self._retry_wait,
            reraise=True,
        ):
            with attempt:
                response = await client.chat.completions.create(
                    model=self._config.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        }
                    ],
                    timeout=self._config.timeout_sec,
                )
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
                self._config.max_image_edge,
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

        entry = self._inflight.get(digest)
        if entry is None:
            task = asyncio.create_task(self._caption_uncached(digest, raw_bytes))
            entry = _InflightEntry(task=task, waiters=0)
            self._inflight[digest] = entry

            def _cleanup(
                finished: asyncio.Task[str | None],
                key: str = digest,
            ) -> None:
                current = self._inflight.get(key)
                if current is not None and current.task is finished:
                    self._inflight.pop(key, None)

            task.add_done_callback(_cleanup)

        entry.waiters += 1
        try:
            return await asyncio.shield(entry.task)
        finally:
            entry.waiters -= 1
            if entry.waiters <= 0 and not entry.task.done():
                if self._inflight.get(digest) is entry:
                    self._inflight.pop(digest, None)
                entry.task.cancel()
