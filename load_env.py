import os
from pathlib import Path

def load_dotenv(dotenv_path: Path = Path(".env")):
    if not dotenv_path.exists():
        return

    with dotenv_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
