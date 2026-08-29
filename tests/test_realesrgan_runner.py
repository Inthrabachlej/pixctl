"""Tests for run_upscale success verification in realesrgan_runner."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pixctl.realesrgan_runner import Backend, run_upscale


def _python_backend(tmp_path: Path) -> Backend:
    script = tmp_path / "inference_realesrgan.py"
    script.write_text("# stub")
    return Backend(
        kind="python",
        path=script,
        available=True,
        supports_face_enhance=True,
        python_exe=Path(sys.executable),
        cwd=tmp_path,
    )


def _fake_proc(returncode: int):
    return SimpleNamespace(returncode=returncode, stdout="", stderr="")


# ---------------------------------------------------------------------------
# actual == output_path: file already at the requested location after inference
# ---------------------------------------------------------------------------

def test_success_when_actual_equals_output_path(tmp_path):
    """No rename needed; success must be True (no false negative)."""
    in_path = tmp_path / "photo.png"
    in_path.write_bytes(b"x")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_path = out_dir / "photo.png"   # stem matches input → actual == out_path

    backend = _python_backend(tmp_path)

    def fake_run(cmd, **kw):
        out_path.write_bytes(b"upscaled")  # backend writes to actual == out_path
        return _fake_proc(0)

    with patch("subprocess.run", side_effect=fake_run):
        result = run_upscale(backend, in_path, out_path)

    assert result.success is True
    assert result.output_path == out_path
    assert out_path.exists()


# ---------------------------------------------------------------------------
# actual != output_path: rename is needed (existing behaviour preserved)
# ---------------------------------------------------------------------------

def test_success_when_actual_differs_from_output_path(tmp_path):
    """Backend writes using input stem; runner must rename to the requested path."""
    in_path = tmp_path / "photo.png"
    in_path.write_bytes(b"x")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # Requested path has a different stem → actual != out_path
    out_path = out_dir / "photo_4x.png"

    backend = _python_backend(tmp_path)

    def fake_run(cmd, **kw):
        # Backend writes to {out_dir}/{input_stem}.png, not to out_path
        (out_dir / "photo.png").write_bytes(b"upscaled")
        return _fake_proc(0)

    with patch("subprocess.run", side_effect=fake_run):
        result = run_upscale(backend, in_path, out_path)

    assert result.success is True
    assert result.output_path == out_path
    assert out_path.exists()
    assert not (out_dir / "photo.png").exists()  # moved, not copied


# ---------------------------------------------------------------------------
# Non-zero returncode: failure (success must be False regardless of files)
# ---------------------------------------------------------------------------

def test_failure_on_nonzero_returncode(tmp_path):
    in_path = tmp_path / "photo.png"
    in_path.write_bytes(b"x")
    out_path = tmp_path / "out" / "photo.png"
    out_path.parent.mkdir()

    backend = _python_backend(tmp_path)

    with patch("subprocess.run", return_value=_fake_proc(1)):
        result = run_upscale(backend, in_path, out_path)

    assert result.success is False
    assert result.output_path is None


# ---------------------------------------------------------------------------
# No backend available
# ---------------------------------------------------------------------------

def test_failure_when_no_backend():
    none_backend = Backend(
        kind="none", path=Path(), available=False, supports_face_enhance=False
    )
    result = run_upscale(none_backend, Path("in.png"), Path("out.png"))
    assert result.success is False
    assert result.returncode == -1
