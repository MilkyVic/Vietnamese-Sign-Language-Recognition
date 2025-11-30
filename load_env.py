import os
from pathlib import Path

def load_dotenv(dotenv_path: Path = Path(".env")):
    """
    Load environment variables from a .env-style file.
    Falls back to searching parent directories so the caller does not need to
    run from the repo root. Also tries a sibling ".env.local" if it exists.
    """

    def find_file(path: Path) -> Path | None:
        candidates = []
        if path.is_absolute():
            candidates.append(path)
        else:
            cwd = Path.cwd()
            candidates.append(cwd / path)
            for parent in cwd.parents:
                candidates.append(parent / path)
        return next((p for p in candidates if p.exists()), None)

    env_file = find_file(Path(".env"))
    env_local_file = find_file(Path(".env.local"))

    for file in filter(None, (env_file, env_local_file)):
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
