"""A release has one number: the engine, the API, the web app and the extension must agree, or a bug report names the wrong build."""
import json
import tomllib
from pathlib import Path

import shield
from shield.api import app

ROOT = Path(__file__).resolve().parent.parent


def test_every_part_reports_the_same_version():
    py = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    found = {
        "pyproject.toml": py,
        "shield.__version__": shield.__version__,
        "API (OpenAPI)": app.version,
        **{f"{d}/package.json": json.loads((ROOT / d / "package.json").read_text())["version"] for d in ("web", "vscode")},
        **{f"{d}/package-lock.json": json.loads((ROOT / d / "package-lock.json").read_text())["version"] for d in ("web", "vscode")},
    }
    assert len(set(found.values())) == 1, found


def test_the_changelog_describes_this_version():
    version = shield.__version__
    assert f"## {version}" in (ROOT / "vscode/CHANGELOG.md").read_text()
    assert f"## {version}" in (ROOT / "CHANGELOG.md").read_text()
