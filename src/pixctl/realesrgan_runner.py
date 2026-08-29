import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pixctl.config import BASE_DIR

_DEBUG = os.environ.get("PIXCTL_DEBUG") == "1"

@dataclass
class Backend:
    kind: str          # "ncnn" | "python" | "none"
    path: Path
    available: bool
    supports_face_enhance: bool
    python_exe: Optional[Path] = field(default=None)
    cwd: Optional[Path] = field(default=None)


@dataclass
class RunResult:
    success: bool
    stdout: str
    stderr: str
    returncode: int
    duration: float
    output_path: Optional[Path] = field(default=None)


def _resolve_backend_from_path(p: Path, source: str) -> Optional[Backend]:
    """Classify an existing path as a Backend, or return None if unrecognized.

    p must already exist. Handles: directory with inference script or ncnn binary,
    .py script, or binary executable.
    """
    if p.is_dir():
        py_script = p / "inference_realesrgan.py"
        ncnn_bin = p / "realesrgan-ncnn-vulkan"
        if py_script.exists():
            venv_py = p / ".venv" / "bin" / "python"
            python_exe = venv_py if venv_py.exists() else Path(sys.executable)
            if _DEBUG:
                print(f"Backend source: {source} (directory)")
                print(f"Backend type:   python")
                print(f"Backend path:   {py_script}")
                print(f"Backend cwd:    {p}")
                print(f"Python exe:     {python_exe}")
            return Backend(kind="python", path=py_script, available=True,
                           supports_face_enhance=True, python_exe=python_exe, cwd=p)
        if ncnn_bin.exists():
            if _DEBUG:
                print(f"Backend source: {source} (directory)")
                print(f"Backend type:   ncnn")
                print(f"Backend path:   {ncnn_bin}")
                print(f"Backend cwd:    N/A")
                print(f"Python exe:     N/A")
            return Backend(kind="ncnn", path=ncnn_bin, available=True,
                           supports_face_enhance=False)
        if _DEBUG:
            print(f"WARNING: {source} path {str(p)!r} is a directory but contains neither "
                  "inference_realesrgan.py nor realesrgan-ncnn-vulkan — falling back.")
        return None

    if p.suffix == ".py":
        venv_py = p.parent / ".venv" / "bin" / "python"
        if not venv_py.exists() and _DEBUG:
            print(f"WARNING: venv not found at {venv_py} — using system Python interpreter.")
        python_exe = venv_py if venv_py.exists() else Path(sys.executable)
        if _DEBUG:
            print(f"Backend source: {source}")
            print(f"Backend type:   python")
            print(f"Backend path:   {p}")
            print(f"Backend cwd:    {p.parent}")
            print(f"Python exe:     {python_exe}")
        return Backend(kind="python", path=p, available=True,
                       supports_face_enhance=True, python_exe=python_exe, cwd=p.parent)

    # Binary / executable
    if _DEBUG:
        print(f"Backend source: {source}")
        print(f"Backend type:   ncnn")
        print(f"Backend path:   {p}")
        print(f"Backend cwd:    N/A")
        print(f"Python exe:     N/A")
    return Backend(kind="ncnn", path=p, available=True, supports_face_enhance=False)


def detect_backend(user_path: str | None = None) -> Backend:
    """Probe for Real-ESRGAN backends in priority order."""
    # A. User-provided path (UI input)
    if user_path:
        p = Path(user_path).resolve()
        if p.exists():
            backend = _resolve_backend_from_path(p, "UI path")
            if backend is not None:
                return backend
        else:
            if _DEBUG:
                print(f"WARNING: UI-provided path {user_path!r} does not exist — "
                      "falling back to auto-detection.")

    # B. Explicit override via REAL_ESRGAN_PATH environment variable
    env_path_str = os.environ.get("REAL_ESRGAN_PATH")
    if env_path_str:
        p = Path(env_path_str)
        if p.exists():
            backend = _resolve_backend_from_path(p, "REAL_ESRGAN_PATH env var")
            if backend is not None:
                return backend
        elif _DEBUG:
            print(f"WARNING: REAL_ESRGAN_PATH={env_path_str!r} does not exist — "
                  "falling back to auto-detection.")

    # C. realesrgan-ncnn-vulkan in PATH
    which = shutil.which("realesrgan-ncnn-vulkan")
    if which:
        if _DEBUG:
            print("Backend source: PATH")
            print(f"Backend type:   ncnn")
            print(f"Backend path:   {which}")
            print(f"Backend cwd:    N/A")
            print(f"Python exe:     N/A")
        return Backend(kind="ncnn", path=Path(which), available=True, supports_face_enhance=False)

    # D. ./realesrgan-ncnn-vulkan (local binary next to project root)
    local_ncnn = BASE_DIR / "realesrgan-ncnn-vulkan"
    if local_ncnn.exists():
        backend = _resolve_backend_from_path(local_ncnn, "BASE_DIR local binary")
        if backend is not None:
            return backend

    # E. ./Real-ESRGAN/realesrgan-ncnn-vulkan
    nested_ncnn = BASE_DIR / "Real-ESRGAN" / "realesrgan-ncnn-vulkan"
    if nested_ncnn.exists():
        backend = _resolve_backend_from_path(nested_ncnn, "BASE_DIR/Real-ESRGAN ncnn binary")
        if backend is not None:
            return backend

    # F. ./Real-ESRGAN/inference_realesrgan.py (Python script backend)
    inference_py = BASE_DIR / "Real-ESRGAN" / "inference_realesrgan.py"
    if inference_py.exists():
        backend = _resolve_backend_from_path(inference_py, "BASE_DIR/Real-ESRGAN Python script")
        if backend is not None:
            return backend

    return Backend(kind="none", path=Path(), available=False, supports_face_enhance=False)


# Map pixctl internal model IDs to inference_realesrgan.py --model_name values.
_MODEL_NAME_MAP = {
    "realesrgan-x4plus": "RealESRGAN_x4plus",
    "realesrgan-x4plus-anime": "RealESRGAN_x4plus_anime_6B",
    "realesr-animevideov3": "realesr-animevideov3",
}


def _build_command(
    backend: Backend,
    input_path: Path,
    output_path: Path,
    model: str,
    scale: int,
    face_enhance: bool,
) -> list:
    if backend.kind == "ncnn":
        return [
            str(backend.path),
            "-i", str(input_path),
            "-o", str(output_path),
            "-n", model,
            "-s", str(scale),
        ]
    if backend.kind == "python":
        # inference_realesrgan.py writes to an output directory, not a file path.
        # Pass --suffix "" so the output filename is {stem}.{ext} (no _out suffix),
        # making it easy to locate after inference.
        interpreter = str(backend.python_exe) if backend.python_exe else sys.executable
        mapped_model = _MODEL_NAME_MAP.get(model, model)
        ext = output_path.suffix.lstrip(".") or "png"
        cmd = [
            interpreter,
            str(backend.path),
            "-n", mapped_model,
            "-i", str(input_path),
            "-o", str(output_path.parent),
            "-s", str(scale),
            "--fp32",
            "--ext", ext,
            "--suffix", "",
        ]
        if face_enhance:
            cmd.append("--face_enhance")
        return cmd
    raise ValueError("No Real-ESRGAN backend available.")


_UPSCALE_TIMEOUT = 1800  # seconds; upscaling large images with Python backend can be slow


def run_upscale(
    backend: Backend,
    input_path: Path,
    output_path: Path,
    model: str = "realesrgan-x4plus",
    scale: int = 2,
    face_enhance: bool = False,
) -> RunResult:
    """Run upscaling and return a RunResult.

    output_path is the desired full output file path. The caller is responsible
    for ensuring it does not already exist (use safe_output_path).
    """
    if not backend.available:
        return RunResult(
            success=False,
            stdout="",
            stderr="No Real-ESRGAN backend found.",
            returncode=-1,
            duration=0.0,
        )

    if face_enhance and not backend.supports_face_enhance:
        if _DEBUG:
            print(
                f"WARNING: face_enhance requested but not supported by '{backend.kind}' "
                "backend — continuing without it."
            )
        face_enhance = False

    cmd = _build_command(backend, input_path, output_path, model, scale, face_enhance)
    cwd = str(backend.cwd) if backend.cwd else None
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=_UPSCALE_TIMEOUT)
    except subprocess.TimeoutExpired:
        duration = time.monotonic() - t0
        return RunResult(
            success=False,
            stdout="",
            stderr=f"Upscale process timed out after {_UPSCALE_TIMEOUT}s.",
            returncode=-1,
            duration=duration,
        )
    except FileNotFoundError as exc:
        duration = time.monotonic() - t0
        return RunResult(
            success=False,
            stdout="",
            stderr=f"Backend executable not found: {exc}",
            returncode=-1,
            duration=duration,
        )
    duration = time.monotonic() - t0

    # For the python backend the script writes output_dir/{stem}.{ext}
    # (because we pass --suffix ""). Rename to the caller-specified output_path when they differ.
    # Check path equality first (cheap) to short-circuit before the filesystem call so that the
    # "actual == output_path" case (file already in place) is never blocked by a stale exists().
    if backend.kind == "python" and proc.returncode == 0:
        ext = output_path.suffix.lstrip(".") or "png"
        actual = output_path.parent / (input_path.stem + "." + ext)
        if actual != output_path and actual.exists():
            shutil.move(str(actual), str(output_path))

    success = proc.returncode == 0 and output_path.exists()
    return RunResult(
        success=success,
        stdout=proc.stdout,
        stderr=proc.stderr,
        returncode=proc.returncode,
        duration=duration,
        output_path=output_path if success else None,
    )
