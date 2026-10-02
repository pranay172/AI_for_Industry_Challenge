"""Import paths shared by the tests: the aic_model package, its tools and the scripts."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT/'scripts', ROOT/'aic_model'/'tools', ROOT/'aic_model'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
