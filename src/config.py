import json
import logging
import os
import socket
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

_LOCAL_CONFIG_FILE = BASE_DIR / ".pixctl.local.json"
_log = logging.getLogger(__name__)


def load_local_config() -> dict:
    """Load .pixctl.local.json; return {} on missing file or parse error."""
    try:
        if _LOCAL_CONFIG_FILE.exists():
            with _LOCAL_CONFIG_FILE.open() as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            _log.warning("Local config %s is not a JSON object; ignoring.", _LOCAL_CONFIG_FILE)
    except (OSError, json.JSONDecodeError) as exc:
        _log.warning("Could not read local config from %s: %s", _LOCAL_CONFIG_FILE, exc)
    return {}


def save_local_config(updates: dict) -> None:
    """Merge updates into .pixctl.local.json (creates file if absent)."""
    current = load_local_config()
    current.update(updates)
    try:
        with _LOCAL_CONFIG_FILE.open("w") as f:
            json.dump(current, f, indent=2)
    except OSError as exc:
        _log.warning("Could not save local config to %s: %s", _LOCAL_CONFIG_FILE, exc)

OUTPUTS = {
    "upscaled": BASE_DIR / "outputs" / "upscaled",
    "compressed": BASE_DIR / "outputs" / "compressed",
    "batch": BASE_DIR / "outputs" / "batch",
}
TEMP_DIR = BASE_DIR / "temp"

HOST = "127.0.0.1"
BASE_PORT = 7860
PORT = BASE_PORT  # default; use resolve_port() at startup for the actual bound port
_PORT_SCAN_RANGE = 20


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def resolve_port() -> tuple[int, bool, int]:
    """Return (chosen_port, fallback_occurred, requested_port).

    Priority: PIXCTL_PORT > GRADIO_SERVER_PORT > BASE_PORT (7860).
    Scans up to _PORT_SCAN_RANGE ports above the requested port on conflict.
    Raises OSError if no free port is found.
    """
    env_val = os.environ.get("PIXCTL_PORT") or os.environ.get("GRADIO_SERVER_PORT")
    requested = BASE_PORT
    if env_val:
        try:
            requested = int(env_val)
        except ValueError:
            pass

    for port in range(requested, requested + _PORT_SCAN_RANGE):
        if _port_is_free(HOST, port):
            return port, port != requested, requested

    raise OSError(
        f"No free port found in range {requested}–{requested + _PORT_SCAN_RANGE - 1}. "
        "Stop other services or set PIXCTL_PORT to a free port."
    )


def init_dirs() -> None:
    for path in OUTPUTS.values():
        path.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    for f in TEMP_DIR.iterdir():
        try:
            f.unlink()
        except OSError as exc:
            _log.warning("Could not remove temp file %s: %s", f, exc)
