"""Locate the pinned upstream checkout without machine-specific paths."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = '28b15acb68ab3871f2ae1479964d53457ac066c6'
def configure():
    root = Path(os.environ.get('SUPERMIMIC_ROOT', ROOT / 'upstream')).resolve()
    if not (root / 'scripts/run_single_gpu.py').is_file():
        raise SystemExit('Missing upstream checkout. Run bash scripts/setup.sh or set SUPERMIMIC_ROOT.')
    for path in (root / 'vendor/mjlab/src', root / 'src', root / 'scripts'):
        sys.path.insert(0, str(path))
    return root
