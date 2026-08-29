import pytest
from pathlib import Path
from PIL import Image

from pixctl.image_ops import compress_image


def test_unsupported_format_raises_value_error(tmp_path):
    bad = tmp_path / "not_an_image.jpg"
    bad.write_bytes(b"this is not image data")
    with pytest.raises(ValueError, match="Unsupported image format"):
        compress_image(bad, tmp_path / "out.jpg", "jpg", 85, None, None, False)


def test_unsupported_format_message_contains_filename(tmp_path):
    bad = tmp_path / "garbage.png"
    bad.write_bytes(b"\x00\x01\x02\x03")
    with pytest.raises(ValueError) as exc_info:
        compress_image(bad, tmp_path / "out.png", "png", 85, None, None, False)
    assert "garbage.png" in str(exc_info.value)


def test_valid_image_unaffected(tmp_path):
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    in_path = tmp_path / "input.png"
    img.save(in_path)
    out_path = tmp_path / "output.jpg"
    result = compress_image(in_path, out_path, "jpg", 85, None, None, False)
    assert out_path.exists()
    assert result["output_size"] > 0
    assert result["input_size"] > 0
