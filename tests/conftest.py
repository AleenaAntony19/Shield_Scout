import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("SHIELD_ALLOWED_HOSTS", "testserver")  # TestClient's Host; the server otherwise accepts loopback names only


@pytest.fixture(scope="session")
def fx(tmp_path_factory):
    from demo.make_fixtures import build

    out = tmp_path_factory.mktemp("fixtures")
    return build(out)

