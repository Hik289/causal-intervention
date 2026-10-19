import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("CIVIC_PROJECT_ROOT", ".")).expanduser().resolve()
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
