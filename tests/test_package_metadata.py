from __future__ import annotations

import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import universe_selector


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PY_TYPED = REPO_ROOT / "src/universe_selector/py.typed"


def _project_metadata() -> dict[str, object]:
    return tomllib.loads(PYPROJECT.read_text())["project"]


def test_runtime_version_matches_project_metadata() -> None:
    metadata = _project_metadata()

    assert universe_selector.__version__ == metadata["version"]


def test_source_tree_import_falls_back_to_project_metadata() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import universe_selector; print(universe_selector.__version__)",
        ],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == _project_metadata()["version"]


def test_typed_classifier_has_source_and_wheel_marker(tmp_path: Path) -> None:
    metadata = _project_metadata()
    classifiers = metadata["classifiers"]
    assert isinstance(classifiers, list)
    assert "Typing :: Typed" in classifiers
    assert PY_TYPED.is_file()

    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel_path = next(tmp_path.glob("*.whl"))

    with zipfile.ZipFile(wheel_path) as wheel:
        assert "universe_selector/py.typed" in wheel.namelist()
