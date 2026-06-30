#!/usr/bin/env python3
"""Smoke test for pixctl frontend — no browser automation required.

Prints PASS/FAIL for each check, exits 0 if all pass, 1 if any fail.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

_results: list[bool] = []


def _check(name: str, fn) -> None:
    try:
        fn()
        print(f"PASS  {name}")
        _results.append(True)
    except Exception as exc:
        print(f"FAIL  {name}: {exc}")
        _results.append(False)


# ── 1. Import check ────────────────────────────────────────────────────────────

def _test_import_src():
    from src import config, image_ops, realesrgan_runner, ui, utils  # noqa: F401

_check("Import: src/*.py modules import without error", _test_import_src)


def _test_import_app():
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", PROJECT_ROOT / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

_check("Import: app.py loads without error", _test_import_app)


# ── 2. UI build check ──────────────────────────────────────────────────────────

def _test_build_ui():
    from src.ui import build_ui
    demo = build_ui()
    assert demo is not None, "build_ui() returned None"

_check("UI build: build_ui() returns Gradio Blocks without exception", _test_build_ui)


# ── 3. Backend detection ───────────────────────────────────────────────────────

def _test_detect_none():
    from src.realesrgan_runner import detect_backend
    b = detect_backend(None)
    assert hasattr(b, "available"), "Backend missing 'available' attribute"
    assert hasattr(b, "kind"), "Backend missing 'kind' attribute"

_check("detect_backend(None): returns Backend dataclass", _test_detect_none)


def _test_detect_invalid():
    from src.realesrgan_runner import detect_backend
    b = detect_backend("/nonexistent/realesrgan-ncnn-vulkan")
    assert hasattr(b, "available"), "Backend missing 'available' attribute"

_check("detect_backend('/nonexistent/...'): returns Backend without exception", _test_detect_invalid)


# ── 4. Config load/save ────────────────────────────────────────────────────────

def _test_config_load():
    from src.config import load_local_config
    cfg = load_local_config()
    assert isinstance(cfg, dict), f"Expected dict, got {type(cfg)}"

_check("load_local_config(): returns dict", _test_config_load)


def _test_config_save():
    from src.config import load_local_config, save_local_config, _LOCAL_CONFIG_FILE
    sentinel = "_smoke_test_key_do_not_keep"
    save_local_config({sentinel: True})
    cfg = load_local_config()
    assert cfg.get(sentinel) is True, "Saved key not found after save_local_config()"
    # Clean up
    cfg.pop(sentinel, None)
    try:
        with _LOCAL_CONFIG_FILE.open("w") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass

_check("save_local_config(): round-trips and cleans up without exception", _test_config_save)


# ── 5. Upscale callback ────────────────────────────────────────────────────────

def _make_test_image() -> str:
    """Create a tiny 4×4 PNG in a temp file; return its path as a string."""
    from PIL import Image
    img = Image.new("RGB", (4, 4), color=(128, 64, 32))
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    return tmp.name


def _call_do_upscale(**overrides):
    from src.ui import _do_upscale
    defaults = dict(
        user_backend_path="",
        img_path=None,
        scale_str="2",
        model="realesrgan-x4plus",
        face_enhance=False,
        fmt="png",
        quality=90,
        auto_compress=False,
        max_width_str="original",
        target_size_str="none",
    )
    defaults.update(overrides)
    last = None
    for item in _do_upscale(**defaults):
        last = item
    return last


def _collect_do_upscale(**overrides):
    from src.ui import _do_upscale
    defaults = dict(
        user_backend_path="",
        img_path=None,
        scale_str="2",
        model="realesrgan-x4plus",
        face_enhance=False,
        fmt="png",
        quality=90,
        auto_compress=False,
        max_width_str="original",
        target_size_str="none",
    )
    defaults.update(overrides)
    return list(_do_upscale(**defaults))


def _assert_upscale_output_contract(items):
    assert items, "Expected at least one callback yield"
    for item in items:
        assert isinstance(item, tuple) and len(item) == 8, f"Expected 8-tuple, got {item!r}"


def _test_upscale_no_image():
    items = _collect_do_upscale(img_path=None, user_backend_path="")
    _assert_upscale_output_contract(items)
    result = items[-1]
    assert result[0] is None, "Expected None output image for missing input"
    assert result[1] == "", "Expected empty optimized export link for missing input"
    assert result[2] == "", "Expected empty master link for missing input"
    assert isinstance(result[3], str), "Expected str log"
    assert "No input image" in result[3], f"Unexpected log: {result[3]!r}"

_check("_do_upscale(img=None): returns 8-value UI contract — no exception", _test_upscale_no_image)


def _test_upscale_empty_backend():
    tmp = _make_test_image()
    try:
        items = _collect_do_upscale(img_path=tmp, user_backend_path="")
        _assert_upscale_output_contract(items)
        result = items[-1]
        assert isinstance(result[3], str), "Expected str log"
    finally:
        Path(tmp).unlink(missing_ok=True)

_check("_do_upscale(backend=''): returns clean tuple — no exception", _test_upscale_empty_backend)


def _test_upscale_invalid_backend():
    tmp = _make_test_image()
    try:
        items = _collect_do_upscale(
            img_path=tmp,
            user_backend_path="/nonexistent/realesrgan",
        )
        _assert_upscale_output_contract(items)
        result = items[-1]
        assert isinstance(result[3], str), "Expected str log"
    finally:
        Path(tmp).unlink(missing_ok=True)

_check("_do_upscale(backend='/nonexistent/realesrgan'): returns clean tuple — no exception", _test_upscale_invalid_backend)


def _test_upscale_success_mapping():
    from src import ui
    from src.realesrgan_runner import Backend, RunResult

    tmp = _make_test_image()
    original_detect_backend = ui.detect_backend
    original_run_upscale = ui.run_upscale

    def fake_detect_backend(_path):
        return Backend(kind="ncnn", path=Path("/tmp/fake-realesrgan"), available=True, supports_face_enhance=False)

    def fake_run_upscale(_backend, input_path, output_path, **_kwargs):
        from PIL import Image
        with Image.open(input_path) as img:
            img.save(output_path)
        return RunResult(success=True, stdout="", stderr="", returncode=0, duration=0.1, output_path=output_path)

    try:
        ui.detect_backend = fake_detect_backend
        ui.run_upscale = fake_run_upscale
        items = _collect_do_upscale(img_path=tmp, user_backend_path="")
        _assert_upscale_output_contract(items)
        result = items[-1]
        assert result[0], "Expected optimized output preview path"
        assert Path(result[0]).exists(), f"Preview path does not exist: {result[0]}"
        assert 'class="download-link"' in result[1], "Expected optimized export download link HTML"
        assert 'Download optimized export' in result[1], "Expected optimized export label"
        assert 'class="download-link"' in result[2], "Expected master download link HTML"
        assert 'Download full-quality master (PNG)' in result[2], "Expected master label"
        assert "Open full-size preview" in result[7], "Expected full-size preview link"
    finally:
        ui.detect_backend = original_detect_backend
        ui.run_upscale = original_run_upscale
        Path(tmp).unlink(missing_ok=True)

_check("_do_upscale(success): preview path plus HTML download links", _test_upscale_success_mapping)


# ── Summary ────────────────────────────────────────────────────────────────────

passed = sum(_results)
total = len(_results)
print()
print(f"Results: {passed}/{total} passed")
sys.exit(0 if passed == total else 1)
