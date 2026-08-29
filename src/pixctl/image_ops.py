from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

MAX_WIDTHS: dict[str, int | None] = {
    "original": None,
    "1280": 1280,
    "1920": 1920,
    "2560": 2560,
    "3840": 3840,
}

TARGET_SIZES: dict[str, int | None] = {
    "none": None,
    "1 MB": 1 * 1024 * 1024,
    "2 MB": 2 * 1024 * 1024,
    "5 MB": 5 * 1024 * 1024,
    "10 MB": 10 * 1024 * 1024,
}

_PIL_FORMAT: dict[str, str] = {
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "png": "PNG",
    "webp": "WEBP",
}


def _to_saveable(img: Image.Image, pil_fmt: str) -> Image.Image:
    if pil_fmt == "JPEG":
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            return bg
        if img.mode != "RGB":
            return img.convert("RGB")
    elif pil_fmt == "WEBP":
        if img.mode not in ("RGB", "RGBA"):
            return img.convert("RGB")
    return img


def _resize_to_width(img: Image.Image, width: int) -> Image.Image:
    if img.width <= width:
        return img
    ratio = width / img.width
    return img.resize((width, max(1, int(img.height * ratio))), Image.LANCZOS)


def _encode(img: Image.Image, pil_fmt: str, quality: int, extra: dict) -> bytes:
    buf = io.BytesIO()
    kw: dict = dict(extra)
    if pil_fmt in ("JPEG", "WEBP"):
        kw["quality"] = quality
        kw.setdefault("optimize", True)
    else:
        kw.setdefault("optimize", True)
    img.save(buf, format=pil_fmt, **kw)
    return buf.getvalue()


def compress_image(
    input_path: str | Path,
    output_path: str | Path,
    fmt: str,
    quality: int,
    max_width: int | None,
    target_bytes: int | None,
    strip_metadata: bool,
) -> dict:
    """
    Compress/resize an image.  Returns dict with input_size, output_size,
    output_path, and log.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    pil_fmt = _PIL_FORMAT[fmt.lower()]

    try:
        with Image.open(input_path) as orig:
            orig.load()
            input_size = input_path.stat().st_size

            extra: dict = {}
            if not strip_metadata and pil_fmt == "JPEG":
                exif = orig.info.get("exif")
                if exif:
                    extra["exif"] = exif

            img = orig
            if max_width is not None:
                img = _resize_to_width(img, max_width)
            img = _to_saveable(img, pil_fmt)
            if img is orig:
                img = img.copy()
    except UnidentifiedImageError:
        raise ValueError(f"Unsupported image format: {input_path.name}")

    log_lines: list[str] = []
    data: bytes | None = None

    if target_bytes is None:
        data = _encode(img, pil_fmt, quality, extra)
        log_lines.append(f"Saved at quality={quality}")
    else:
        if pil_fmt != "PNG":
            q = quality
            while q >= 50:
                candidate = _encode(img, pil_fmt, q, extra)
                if len(candidate) <= target_bytes:
                    data = candidate
                    log_lines.append(f"Target met at quality={q}")
                    break
                q -= 5

        if data is None:
            # Progressive resize phase
            resize_img = img.copy()
            step_q = 50
            warned = False
            while True:
                candidate = _encode(resize_img, pil_fmt, step_q, extra)
                if len(candidate) <= target_bytes:
                    data = candidate
                    log_lines.append(
                        f"Target met at width={resize_img.width}, quality={step_q}"
                    )
                    break
                new_w = max(1, int(resize_img.width * 0.9))
                if new_w == resize_img.width:
                    data = candidate
                    log_lines.append("Cannot meet target; saved at minimum achievable size")
                    break
                if new_w < 800 and not warned:
                    log_lines.append(
                        f"Warning: resizing below 800 px (width={new_w}) to satisfy target"
                    )
                    warned = True
                new_h = max(1, int(resize_img.height * new_w / resize_img.width))
                resize_img = resize_img.resize((new_w, new_h), Image.LANCZOS)

    if data is None:
        data = _encode(img, pil_fmt, 50, extra)
        log_lines.append("Fallback: saved at quality=50")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)

    return {
        "input_size": input_size,
        "output_size": len(data),
        "output_path": output_path,
        "log": "\n".join(log_lines),
    }
