import sys
import glob
from pathlib import Path

_project_root = Path(__file__).parent.parent
_venv = _project_root / ".venv"
if _venv.is_dir():
    for _sp in glob.glob(str(_venv / "lib" / "python*" / "site-packages")):
        if _sp not in sys.path:
            sys.path.insert(0, _sp)

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_src = _project_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
