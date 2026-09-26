"""Contract tests: [S16] JSON Schemas vs example payloads ([S24] contract_tests, cloud [C17])."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"
BASE = "https://schemas.aqi-hybrid.dev/v1/"


def _registry() -> Registry[Any]:
    resources = []
    for p in CONTRACTS.glob("*.schema.json"):
        schema = json.loads(p.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        assert schema["$id"] == BASE + p.name
        resources.append((schema["$id"], Resource.from_contents(schema)))
    return Registry().with_resources(resources)


REGISTRY = _registry()
EXAMPLES: dict[str, Any] = json.loads(
    (ROOT / "fixtures" / "contract_examples.json").read_text("utf-8")
)


def validator(key: str) -> Draft202012Validator:
    return Draft202012Validator({"$ref": BASE + key}, registry=REGISTRY)


def errors(key: str, payload: Any) -> list[str]:
    return [e.message for e in validator(key).iter_errors(payload)]


@pytest.mark.parametrize("key", [k for k in EXAMPLES if not k.startswith("_")])
def test_examples_valid(key: str) -> None:
    for payload in EXAMPLES[key]:
        assert errors(key, payload) == []


def test_every_schema_has_examples() -> None:
    covered = {k.split("#")[0] for k in EXAMPLES}
    files = {p.name for p in CONTRACTS.glob("*.schema.json")} - {"common.schema.json"}
    assert files <= covered


def _ingest() -> Any:
    return copy.deepcopy(EXAMPLES["ingest.schema.json"][0])


def test_ingest_rejects_non_ist_bucket_ts() -> None:
    p = _ingest()
    p["records"][0]["ts"] = "2026-10-01T04:00:00Z"  # UTC hour, not IST hour start (I02)
    assert errors("ingest.schema.json", p)


def test_ingest_is_strict() -> None:
    p = _ingest()
    p["records"][0]["raw"]["co_ppm"] = 1.2  # gas sensors are never converted to ppm (I12)
    assert errors("ingest.schema.json", p)


def test_ingest_batch_limit() -> None:
    p = _ingest()
    p["records"] = p["records"][:1] * 49
    assert errors("ingest.schema.json", p)


def test_ingest_bad_schema_hash() -> None:
    p = _ingest()
    p["schema_hash"] = "sha256:..."
    assert errors("ingest.schema.json", p)


def test_forecast_is_forward_compatible() -> None:
    p = copy.deepcopy(EXAMPLES["forecast.schema.json"][0])
    p["some_future_field"] = {"x": 1}
    assert errors("forecast.schema.json", p) == []


def test_forecast_needs_six_horizons() -> None:
    p = copy.deepcopy(EXAMPLES["forecast.schema.json"][0])
    p["aqi"] = p["aqi"][:5]
    assert errors("forecast.schema.json", p)


def test_registry_enforces_sensor_pool() -> None:
    p = copy.deepcopy(EXAMPLES["registry.schema.json"][0])
    p["sensors"]["gas"].append("MiCS-6814")
    assert errors("registry.schema.json", p)


def test_event_needs_uuid4() -> None:
    p = copy.deepcopy(EXAMPLES["events.schema.json"][0])
    p["event_id"] = "not-a-uuid"
    assert errors("events.schema.json", p)
