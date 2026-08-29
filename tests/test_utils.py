import re
from pathlib import Path

import pytest

from pixctl.utils import safe_output_path, timestamped_filename


def test_timestamped_filename_includes_microseconds():
    name = timestamped_filename("img", ".png")
    # Expect format: img_YYYYMMDD_HHMMSS_ffffff.png
    assert re.match(r"img_\d{8}_\d{6}_\d{6}\.png$", name), name


def test_timestamped_filename_uniqueness():
    names = {timestamped_filename("img", ".png") for _ in range(50)}
    assert len(names) == 50, "Expected unique filenames across rapid calls"


def test_safe_output_path_returns_path(tmp_path):
    out = safe_output_path(tmp_path, "result.png")
    assert out == tmp_path / "result.png"
    assert not out.exists()


def test_safe_output_path_creates_dir(tmp_path):
    dest = tmp_path / "sub" / "dir"
    out = safe_output_path(dest, "result.png")
    assert dest.is_dir()
    assert out.parent == dest


def test_safe_output_path_collision_gets_suffix(tmp_path):
    (tmp_path / "result.png").touch()
    out = safe_output_path(tmp_path, "result.png")
    assert out != tmp_path / "result.png"
    assert out.stem.startswith("result_")
    assert out.suffix == ".png"


def test_safe_output_path_double_collision_raises(tmp_path, monkeypatch):
    (tmp_path / "result.png").touch()
    # Make token_hex always return the same value so the fallback also collides.
    monkeypatch.setattr("pixctl.utils.secrets.token_hex", lambda _: "aabbcc")
    (tmp_path / "result_aabbcc.png").touch()
    with pytest.raises(FileExistsError):
        safe_output_path(tmp_path, "result.png")
