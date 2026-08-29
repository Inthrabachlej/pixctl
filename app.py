import os
import sys
import time
from pathlib import Path

# Run directly from a source checkout (no `pip install .`) -- put the src-layout
# package dir on sys.path so `from pixctl...` resolves without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# gradio-client 1.3.0 crashes when a JSON schema field is a bool (e.g. additionalProperties:true).
# Patch: return "Any" for any non-dict schema; recursive calls go through this guard too.
import gradio_client.utils as _gcu
def _gcu_safe(schema, defs, _orig=_gcu._json_schema_to_python_type):
    if not isinstance(schema, dict):
        return "Any"
    return _orig(schema, defs)
_gcu._json_schema_to_python_type = _gcu_safe
del _gcu

from pixctl.config import HOST, init_dirs, resolve_port
from pixctl.realesrgan_runner import detect_backend
from pixctl.ui import build_ui

if __name__ == "__main__":
    t0 = time.monotonic()
    init_dirs()

    port, fallback, requested = resolve_port()
    backend = detect_backend(None)

    if backend.available:
        kind = "ncnn-vulkan" if backend.kind == "ncnn" else "Python script"
        backend_line = f"OK  [{kind}]  {backend.path}"
    else:
        backend_line = "no upscale backend  (compression still works)"

    demo = build_ui()
    demo.queue()
    elapsed = time.monotonic() - t0

    print()
    print("Starting pixctl...")
    print(f"Backend:  {backend_line}")
    if fallback:
        print(f"Port {requested} busy — falling back to {port}")
    print(f"Ready in  {elapsed:.1f}s")
    print()
    print("pixctl running at:")
    print(f"  http://{HOST}:{port}")
    print()

    inbrowser = os.environ.get("PIXCTL_OPEN_BROWSER", "1") != "0"
    demo.launch(
        server_name=HOST,
        server_port=port,
        quiet=True,
        inbrowser=inbrowser,
    )
