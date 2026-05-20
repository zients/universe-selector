from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib

__all__ = ["__version__"]


def _version() -> str:
    try:
        return version("universe-selector")
    except PackageNotFoundError:
        pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        project = tomllib.loads(pyproject_path.read_text())["project"]
        return str(project["version"])


__version__ = _version()
