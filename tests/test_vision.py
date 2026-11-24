"""Unit tests for spectra.vision."""

from __future__ import annotations

import asyncio
import base64
import io
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from PIL import Image
from tenacity import wait_none

from spectra import vision
from spectra.config import VisionConfig
from spectra.db import Database
from spectra.vision import (
    MODEL_IMAGE_MAX_BYTES,
    MODEL_IMAGE_MAX_INPUT_BYTES,
    ImageCaptioner,
    prepare_image_sync,
)


def _make_static_jpeg(
    width: int = 100,
    height: int = 80,
    color: tuple[int, int, int] = (200, 30, 30),
    *,
    exif_orientation: int | None = None,
) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    if exif_orientation is not None:
        exif = img.getexif()
        exif[274] = exif_orientation
        img.save(buf, format="JPEG", exif=exif.tobytes())
    else:
        img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_rgba_png(width: int = 60, height: int = 40) -> bytes:
    img = Image.new("RGBA", (width, height), (20, 120, 220, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_animated_gif(
    frame_colors: list[tuple[int, int, int]],
    size: tuple[int, int] = (40, 30),
) -> bytes:
    frames = [Image.new("RGB", size, color) for color in frame_colors]
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    return buf.getvalue()


class _FakeCompletions:
    def __init__(self, handler: Any) -> None:
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        result = self._handler(kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return result


def _make_fake_openai(handler: Any) -> Any:
    completions = _FakeCompletions(handler)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _chat_response(text: str) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def test_astrbot_constants() -> None:
    assert MODEL_IMAGE_MAX_INPUT_BYTES == 64 * 1024 * 1024
    assert MODEL_IMAGE_MAX_BYTES == 512 * 1024


def test_prepare_image_animated_gif_3x3_montage() -> None:
    colors = [
        (255, 0, 0),
        (200, 50, 0),
        (150, 100, 0),
        (100, 150, 0),
        (50, 200, 0),
        (0, 255, 0),
        (0, 200, 50),
        (0, 150, 100),
        (0, 100, 150),
        (0, 50, 200),
        (0, 20, 230),
        (0, 0, 255),
    ]
    gif_bytes = _make_animated_gif(colors, size=(600, 400))

    prepared, mime_type, is_montage = prepare_image_sync(gif_bytes, max_size=1280)
    assert is_montage is True
    assert mime_type == "image/jpeg"
    assert len(prepared) < MODEL_IMAGE_MAX_BYTES

    with Image.open(io.BytesIO(prepared)) as out:
        assert out.format == "JPEG"
        assert max(out.size) <= 1280
        cell_w = out.width // 3
        cell_h = out.height // 3
        r0, g0, b0 = out.getpixel((cell_w // 2, cell_h // 2))
        assert r0 > 200 and g0 < 60 and b0 < 60
        r8, g8, b8 = out.getpixel((cell_w * 2 + cell_w // 2, cell_h * 2 + cell_h // 2))
        assert b8 > 200 and r8 < 60 and g8 < 60


def test_prepare_image_small_gif_no_upscale_and_white_unused_cells() -> None:
    gif_bytes = _make_animated_gif(
        [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)],
        size=(40, 30),
    )
    prepared, mime_type, is_montage = prepare_image_sync(gif_bytes, max_size=1280)
    assert is_montage is True
    assert mime_type == "image/jpeg"

    with Image.open(io.BytesIO(prepared)) as out:
        assert out.size == (120, 90)
        r, g, b = out.getpixel((100, 75))
        assert r > 245 and g > 245 and b > 245


def test_prepare_image_static_passthrough_exif_downscale_and_512kb_cap() -> None:
    # Compliant static JPEG passes through byte-identically.
    compliant = _make_static_jpeg(100, 80)
    out_bytes, mime_type, is_montage = prepare_image_sync(compliant, max_size=1280)
    assert out_bytes is compliant
    assert mime_type == "image/jpeg"
    assert is_montage is False

    # 2000x1000 rotated 90 degrees (EXIF orientation=6) -> 1000x2000 -> 320x640.
    rotated = _make_static_jpeg(2000, 1000, exif_orientation=6)
    prepared, mime_type, is_montage = prepare_image_sync(rotated, max_size=640)
    assert is_montage is False
    assert mime_type == "image/jpeg"
    assert len(prepared) < MODEL_IMAGE_MAX_BYTES
    with Image.open(io.BytesIO(prepared)) as out:
        assert out.size == (320, 640)

    # High-entropy noisy image larger than 512 KB is iteratively shrunk below 512 KB.
    noisy = Image.frombytes("RGB", (1280, 1280), os.urandom(1280 * 1280 * 3))
    noisy_buf = io.BytesIO()
    noisy.save(noisy_buf, format="PNG")
    noisy.close()
    raw_noisy = noisy_buf.getvalue()
    assert len(raw_noisy) > MODEL_IMAGE_MAX_BYTES

    shrunk, shrunk_mime, shrunk_montage = prepare_image_sync(raw_noisy, max_size=1280)
    assert shrunk_montage is False
    assert shrunk_mime == "image/jpeg"
    assert len(shrunk) < MODEL_IMAGE_MAX_BYTES


def test_prepare_image_static_alpha_preserves_png() -> None:
    raw = _make_rgba_png(80, 40)
    prepared, mime_type, is_montage = prepare_image_sync(raw, max_size=1280)
    assert is_montage is False
    assert mime_type == "image/png"
    assert prepared is raw
    with Image.open(io.BytesIO(prepared)) as out:
        assert out.format == "PNG"
        assert "A" in out.getbands()


def test_prepare_image_two_frame_and_transparent_palette_gif() -> None:
    two_frame_gif = _make_animated_gif([(255, 0, 0), (0, 0, 255)], size=(30, 30))
    prepared, mime_type, is_montage = prepare_image_sync(two_frame_gif, max_size=1280)
    assert is_montage is True
    assert mime_type == "image/jpeg"
    with Image.open(io.BytesIO(prepared)) as out:
        assert out.size == (90, 90)
        r0, g0, b0 = out.getpixel((15, 15))
        assert r0 > 200 and g0 < 60 and b0 < 60
        r1, g1, b1 = out.getpixel((45, 15))
        assert b1 > 200 and r1 < 60 and g1 < 60
        r2, g2, b2 = out.getpixel((75, 15))
        assert r2 > 245 and g2 > 245 and b2 > 245

    # Mode="P" transparent GIF (disposal=2): transparent background renders white.
    palette = [0, 0, 0, 255, 0, 0, 0, 0, 255] + [0] * (256 * 3 - 9)
    frames: list[Image.Image] = []
    for color_idx in (1, 2):
        frame = Image.new("P", (30, 30), 0)
        frame.putpalette(palette)
        for x in range(10, 20):
            for y in range(10, 20):
                frame.putpixel((x, y), color_idx)
        frames.append(frame)
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        transparency=0,
        disposal=2,
        duration=100,
        loop=0,
    )

    p_prepared, p_mime, p_montage = prepare_image_sync(buf.getvalue(), max_size=1280)
    assert p_montage is True
    assert p_mime == "image/jpeg"
    with Image.open(io.BytesIO(p_prepared)) as out:
        assert out.size == (90, 90)
        cr, cg, cb = out.getpixel((2, 2))
        assert cr > 240 and cg > 240 and cb > 240
        mr, mg, mb = out.getpixel((15, 15))
        assert mr > 200 and mg < 60 and mb < 60


def test_prepare_image_invalid_max_size_raises() -> None:
    raw = _make_static_jpeg()
    with pytest.raises(ValueError, match="max_size"):
        prepare_image_sync(raw, max_size=0)
    with pytest.raises(ValueError, match="max_size"):
        prepare_image_sync(raw, max_size=-10)


@pytest.fixture
async def db() -> Database:
    database = Database(":memory:")
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


async def test_caption_caches_in_sqlite_and_normalizes(db: Database) -> None:
    fake_openai = _make_fake_openai(
        lambda _kwargs: _chat_response("  [一只\n  橘猫在睡觉]  ")
    )
    cfg = VisionConfig(enabled=True, api_key="sk-test", model="vision-test")
    captioner = ImageCaptioner(
        db,
        cfg,
        openai_client=fake_openai,
        retry_wait=wait_none(),
    )
    await captioner.init()

    img_bytes = _make_static_jpeg()
    source = "base64://" + base64.b64encode(img_bytes).decode("ascii")

    first = await captioner.caption(source)
    second = await captioner.caption(source)
    assert first == "一只 橘猫在睡觉"
    assert second == "一只 橘猫在睡觉"
    assert len(fake_openai.chat.completions.calls) == 1

    row = await db.fetchone("SELECT caption FROM image_cache")
    assert row is not None
    assert row["caption"] == "一只 橘猫在睡觉"
    await captioner.close()


async def test_caption_deduplicates_concurrent_same_hash(db: Database) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    call_count = 0

    async def slow_handler(_kwargs: dict[str, Any]) -> Any:
        nonlocal call_count
        call_count += 1
        started.set()
        await release.wait()
        return _chat_response("同一张表情包")

    fake_openai = _make_fake_openai(slow_handler)
    cfg = VisionConfig(enabled=True, api_key="sk-test")
    captioner = ImageCaptioner(
        db,
        cfg,
        openai_client=fake_openai,
        retry_wait=wait_none(),
    )
    await captioner.init()

    source = "data:image/jpeg;base64," + base64.b64encode(_make_static_jpeg()).decode(
        "ascii"
    )
    tasks = [asyncio.create_task(captioner.caption(source)) for _ in range(5)]
    await started.wait()
    release.set()
    results = await asyncio.gather(*tasks)

    assert results == ["同一张表情包"] * 5
    assert call_count == 1
    assert captioner._inflight == {}
    await captioner.close()


async def test_caption_streaming_aborts_when_exceeding_max_input_bytes(
    db: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vision, "MODEL_IMAGE_MAX_INPUT_BYTES", 2048)
    chunks_yielded = 0

    class _HugeStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            nonlocal chunks_yielded
            for _ in range(100):
                chunks_yielded += 1
                yield b"x" * 1024

        async def aclose(self) -> None:
            return None

    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, stream=_HugeStream())
    )
    http_client = httpx.AsyncClient(transport=transport)
    fake_openai = _make_fake_openai(lambda _kwargs: _chat_response("should not run"))
    cfg = VisionConfig(enabled=True, api_key="sk-test")
    captioner = ImageCaptioner(
        db,
        cfg,
        http_client=http_client,
        openai_client=fake_openai,
        retry_wait=wait_none(),
    )
    await captioner.init()

    result = await captioner.caption("https://cdn.example.invalid/huge.jpg")
    assert result is None
    assert chunks_yielded <= 3
    assert len(fake_openai.chat.completions.calls) == 0

    await http_client.aclose()
    await captioner.close()


async def test_caption_retries_transient_error_and_falls_back_gracefully(
    db: Database,
) -> None:
    attempts = 0
    dummy_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

    def flaky_handler(_kwargs: dict[str, Any]) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise openai.APIConnectionError(request=dummy_request)
        return _chat_response("重试后成功")

    fake_flaky = _make_fake_openai(flaky_handler)
    cfg = VisionConfig(enabled=True, api_key="sk-test")
    captioner = ImageCaptioner(
        db,
        cfg,
        openai_client=fake_flaky,
        retry_wait=wait_none(),
    )
    await captioner.init()

    gif_source = "base64://" + base64.b64encode(
        _make_animated_gif([(255, 0, 0), (0, 255, 0), (0, 0, 255)])
    ).decode("ascii")
    res = await captioner.caption(gif_source)
    assert res == "重试后成功"
    assert attempts == 2
    sent_prompt = fake_flaky.chat.completions.calls[-1]["messages"][0]["content"][0][
        "text"
    ]
    assert "3×3" in sent_prompt

    def always_fail(_kwargs: dict[str, Any]) -> Any:
        raise openai.APIConnectionError(request=dummy_request)

    failing_captioner = ImageCaptioner(
        db,
        cfg,
        openai_client=_make_fake_openai(always_fail),
        retry_wait=wait_none(),
    )
    other_source = "base64://" + base64.b64encode(
        _make_static_jpeg(color=(10, 20, 30))
    ).decode("ascii")
    assert await failing_captioner.caption(other_source) is None
    await captioner.close()


async def test_caption_disabled_or_missing_api_key_and_file_uri(
    db: Database,
    tmp_path: Path,
) -> None:
    img_path = tmp_path / "sample.jpg"
    img_path.write_bytes(_make_static_jpeg())

    fake_openai = _make_fake_openai(lambda _kwargs: _chat_response("本地文件图片"))

    disabled = ImageCaptioner(
        db,
        VisionConfig(enabled=False, api_key="sk-test"),
        openai_client=fake_openai,
        retry_wait=wait_none(),
    )
    await disabled.init()
    assert await disabled.caption(img_path.as_uri()) is None
    assert len(fake_openai.chat.completions.calls) == 0

    no_key = ImageCaptioner(
        db,
        VisionConfig(enabled=True, api_key="   "),
        openai_client=fake_openai,
        retry_wait=wait_none(),
    )
    assert await no_key.caption(img_path.as_uri()) is None
    assert len(fake_openai.chat.completions.calls) == 0

    enabled = ImageCaptioner(
        db,
        VisionConfig(enabled=True, api_key="sk-test"),
        openai_client=fake_openai,
        retry_wait=wait_none(),
    )
    assert await enabled.caption(img_path.as_uri()) == "本地文件图片"
    assert len(fake_openai.chat.completions.calls) == 1
    await enabled.close()


async def test_caption_provider_compat_opencode_go_effort_and_extra_body(
    db: Database,
) -> None:
    cfg = VisionConfig(
        enabled=True,
        base_url="https://opencode.ai/zen/go/v1",
        api_key="sk-vision-oc",
        model="deepseek-v4.1-flash",
        reasoning_effort="low",
        custom_headers={"X-Custom": "vision-ok"},
        extra_body={"thinking": {"type": "enabled"}},
    )
    # Verify owned AsyncOpenAI applies URL rewrite and default headers.
    owned = ImageCaptioner(db, cfg, retry_wait=wait_none())
    try:
        internal = owned._get_openai_client()
        assert (
            str(internal.base_url).rstrip("/")
            == "https://opencode.ai/inference/go/openai/v1"
        )
        assert internal.default_headers["User-Agent"] == "spectra-bot/0.1.0"
        assert internal.default_headers["X-Custom"] == "vision-ok"
    finally:
        await owned.close()

    # Verify per-request kwargs include session header, reasoning_effort, extra_body.
    fake_openai = _make_fake_openai(lambda _kwargs: _chat_response("一只蓝猫"))
    captioner = ImageCaptioner(
        db,
        cfg,
        openai_client=fake_openai,
        retry_wait=wait_none(),
    )
    await captioner.init()
    try:
        source = "base64://" + base64.b64encode(_make_static_jpeg()).decode("ascii")
        caption = await captioner.caption(source)
        assert caption == "一只蓝猫"
        call_kwargs = fake_openai.chat.completions.calls[0]
        assert call_kwargs["extra_headers"] == {"x-opencode-session": "spectra:vision"}
        assert call_kwargs["reasoning_effort"] == "low"
        assert call_kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    finally:
        await captioner.close()
