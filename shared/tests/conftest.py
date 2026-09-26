import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def aqi_vectors() -> Any:
    return load_fixture("aqi_vectors.json")


@pytest.fixture(scope="session")
def ist_rows() -> Any:
    return load_fixture("ist_bucket_fixture.json")["rows"]
