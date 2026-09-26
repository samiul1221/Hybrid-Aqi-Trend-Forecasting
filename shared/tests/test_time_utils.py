from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from aqi_shared import time_utils as tu


def test_fixture_rows(ist_rows: Any) -> None:
    assert len(ist_rows) >= 48 * 60 // 8  # 48 h coverage
    for r in ist_rows:
        dt = tu.parse_utc(r["utc"])
        assert tu.bucket_epoch(r["epoch"]) == r["bucket_epoch"], r
        assert tu.format_utc(tu.ist_bucket_start(dt)) == r["bucket_ts"], r
        assert tu.ist_calendar(dt) == (r["ist_hour"], r["ist_dow"], r["ist_doy"]), r


def test_spec_example() -> None:
    # I02: IST 10:00 -> 04:30Z
    got = tu.ist_local_to_ts(datetime(2026, 10, 1, 10, 0))
    assert got == datetime(2026, 10, 1, 4, 30, tzinfo=UTC)


def test_is_bucket_start() -> None:
    assert tu.is_bucket_start(tu.parse_utc("2026-10-01T04:30:00Z"))
    assert not tu.is_bucket_start(tu.parse_utc("2026-10-01T04:00:00Z"))
    assert not tu.is_bucket_start(tu.parse_utc("2026-10-01T04:30:00.5+00:00"))


def test_offsets_normalised() -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    got = tu.ist_bucket_start(datetime(2026, 10, 1, 10, 59, tzinfo=ist))
    assert got == tu.parse_utc("2026-10-01T04:30:00Z")


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        tu.ist_bucket_start(datetime(2026, 1, 1))  # naive
    with pytest.raises(ValueError):
        tu.ist_local_to_ts(datetime(2026, 1, 1, 10, 30))  # not an IST hour start
