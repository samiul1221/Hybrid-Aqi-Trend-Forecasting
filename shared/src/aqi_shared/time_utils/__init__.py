"""IST clock-hour bucketing [S01] I02/I03, [S17] C01.

All instants are UTC. A bucket is the IST clock hour [HH:00, HH+1:00) IST; its ``ts`` is the UTC
instant of the IST hour start (IST 10:00 -> 04:30Z). India has no DST, so plain integer arithmetic
on epoch seconds is exact and identical in C:
    bucket = floor((epoch + 19800) / 3600) * 3600 - 19800
Calendar features use IST local time: hour 0..23, dow 0=Monday..6=Sunday, doy 1..366.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

IST_OFFSET_S = 19800  # +05:30
IST_OFFSET = timedelta(seconds=IST_OFFSET_S)
HOUR_S = 3600


def bucket_epoch(epoch_s: int) -> int:
    """UTC epoch of the start of the IST clock hour containing ``epoch_s``."""
    return (epoch_s + IST_OFFSET_S) // HOUR_S * HOUR_S - IST_OFFSET_S


def _require_aware(dt: datetime) -> None:
    if dt.tzinfo is None:
        raise ValueError("naive datetime; pass a timezone-aware UTC datetime")


def ist_bucket_start(dt: datetime) -> datetime:
    """Bucket ts (UTC, aware) for an aware datetime. Sub-second parts are dropped first."""
    _require_aware(dt)
    return datetime.fromtimestamp(bucket_epoch(int(dt.timestamp() // 1)), tz=UTC)


def is_bucket_start(dt: datetime) -> bool:
    """True if ``dt`` is exactly an IST hour start (UTC minute 30, second 0)."""
    _require_aware(dt)
    epoch = dt.timestamp()
    return epoch == int(epoch) and bucket_epoch(int(epoch)) == int(epoch)


def ist_calendar(dt: datetime) -> tuple[int, int, int]:
    """(ist_hour 0..23, ist_dow 0=Mon..6=Sun, ist_doy 1..366) of an aware datetime."""
    _require_aware(dt)
    local = dt.astimezone(UTC).replace(tzinfo=None) + IST_OFFSET
    return local.hour, local.weekday(), local.timetuple().tm_yday


def parse_utc(s: str) -> datetime:
    """Parse ISO-8601 with 'Z' or an explicit offset into an aware UTC datetime."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    _require_aware(dt)
    return dt.astimezone(UTC)


def format_utc(dt: datetime) -> str:
    """Wire format 'YYYY-MM-DDTHH:MM:SSZ' (whole seconds)."""
    _require_aware(dt)
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def ist_local_to_ts(local_ist: datetime) -> datetime:
    """CPCB 'From' (naive IST wall time of the hour start) -> bucket ts (UTC) [S09]."""
    if local_ist.tzinfo is not None:
        raise ValueError("expected naive IST wall-clock time")
    ts = (local_ist - IST_OFFSET).replace(tzinfo=UTC)
    if not is_bucket_start(ts):
        raise ValueError(f"{local_ist} is not an IST hour start")
    return ts
