from datetime import datetime
from pathlib import Path
import secrets


def timestamped_filename(stem: str, suffix: str) -> str:
    # Microsecond precision avoids same-second collisions in rapid bursts.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{stem}_{ts}{suffix}"


def safe_output_path(dest_dir: Path, filename: str) -> Path:
    """Return a path under dest_dir; appends a short random suffix if the path already exists."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / filename
    if not out.exists():
        return out
    # Collision fallback: insert a 6-hex token before the extension.
    stem = Path(filename).stem
    ext = Path(filename).suffix
    token = secrets.token_hex(3)
    out = dest_dir / f"{stem}_{token}{ext}"
    if out.exists():
        raise FileExistsError(f"Output file already exists after collision retry: {out}")
    return out


def placeholder_result(operation: str) -> str:
    return f"[{operation}] Not yet implemented."
