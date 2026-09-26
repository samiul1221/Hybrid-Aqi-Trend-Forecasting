"""Offline tests for http cache, OpenAQ and Open-Meteo clients (httpx.MockTransport)."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest

from aqi_ml.data import open_meteo, openaq
from aqi_ml.data.http import CachedClient

FIX = Path(__file__).parent / "fixtures"


def client(tmp_path: Path, handler: Any) -> CachedClient:
    return CachedClient(tmp_path, 0.0, transport=httpx.MockTransport(handler), sleep=lambda s: None)


# ---- http ---------------------------------------------------------------------------------


def test_cache_retry_and_errors(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if req.url.params["q"] == "denied":
            return httpx.Response(401)
        if calls["n"] == 1:
            return httpx.Response(429, headers={"x-ratelimit-reset": "1"})
        return httpx.Response(200, json={"ok": req.url.params["q"]})

    c = client(tmp_path, handler)
    assert c.get_json("https://x/api", {"q": "a"}) == {"ok": "a"}  # 429 then 200
    assert c.get_json("https://x/api", {"q": "a"}) == {"ok": "a"}  # cache hit
    assert c.network_calls == 2
    with pytest.raises(httpx.HTTPStatusError):
        c.get_json("https://x/api", {"q": "denied"})
    assert len(list(tmp_path.glob("*.json"))) == 1  # errors are never cached


def test_rate_limit_waits_and_daily_stop(tmp_path: Path) -> None:
    calls = {"n": 0}

    def hourly(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 12:  # more than max_retries: waits must not consume error retries
            return httpx.Response(
                429, json={"error": True, "reason": "Hourly API request limit exceeded"}
            )
        return httpx.Response(200, json={"ok": 1})

    slept: list[float] = []
    c = CachedClient(tmp_path, 0.0, transport=httpx.MockTransport(hourly), sleep=slept.append)
    assert c.get_json("https://x/a", {"q": 1}) == {"ok": 1}
    assert slept.count(60.0) == 12

    def daily(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, json={"error": True, "reason": "Daily API request limit exceeded"}
        )

    d = CachedClient(
        tmp_path / "d", 0.0, transport=httpx.MockTransport(daily), sleep=lambda s: None
    )
    with pytest.raises(RuntimeError, match="daily"):
        d.get_json("https://x/a", {"q": 2})


# ---- OpenAQ -------------------------------------------------------------------------------


def _loc(i: int, provider: str) -> dict[str, Any]:
    return {
        "id": i, "name": f"L{i}", "isMobile": False, "provider": {"id": 1, "name": provider},
        "coordinates": {"latitude": 28.6, "longitude": 77.2},
        "sensors": [
            {"id": 10 * i, "parameter": {"name": "pm25", "units": "µg/m³"}},
            {"id": 10 * i + 1, "parameter": {"name": "no2", "units": "µg/m³"}},
            {"id": 10 * i + 2, "parameter": {"name": "um003", "units": "particles/cm³"}},
        ],
    }  # fmt: skip


def test_spec_unit_factors() -> None:
    one = np.array([1.0])
    assert openaq.convert("no2", "µg/m³", one)[0] == pytest.approx(0.532, rel=2e-3)  # [S09]
    assert openaq.convert("co", "mg/m³", one)[0] == pytest.approx(0.873, rel=1e-3)
    assert openaq.convert("nh3", "µg/m³", one)[0] == pytest.approx(1.436, rel=1e-3)
    assert openaq.convert("co", "ppb", one)[0] == pytest.approx(0.001)
    with pytest.raises(ValueError):
        openaq.convert("pm25", "ppm", one)


def test_select_stations_keeps_cpcb_only() -> None:
    locs = [_loc(1, "CPCB"), _loc(2, "AirNow"), _loc(3, "Caaqm Delhi")]
    st = openaq.select_stations(locs, "(?i)cpcb|caaqm", ["pm25", "no2"])
    assert st.station_id.tolist() == ["oaq1", "oaq3"]
    assert [s["parameter"] for s in st.sensors.iloc[0]] == ["pm25", "no2"]


def _meas(from_utc: str, to_utc: str, v: float | None) -> dict[str, Any]:
    return {
        "value": v,
        "period": {"datetimeFrom": {"utc": from_utc}, "datetimeTo": {"utc": to_utc}},
    }


def test_fetch_sensor_pages_and_parses(tmp_path: Path) -> None:
    rows = [
        _meas(f"2024-01-10T{h:02d}:30:00Z", f"2024-01-10T{h + 1:02d}:30:00Z", float(h))
        for h in range(5)
    ]
    rows.append(_meas("2024-01-10T05:30:00Z", "2024-01-10T06:30:00Z", None))

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v3/sensors/77/measurements"
        page, limit = int(req.url.params["page"]), int(req.url.params["limit"])
        return httpx.Response(200, json={"results": rows[(page - 1) * limit : page * limit]})

    df = openaq.fetch_sensor(client(tmp_path, handler), "https://api.openaq.org/v3", 77,
                             date(2024, 1, 10), date(2024, 1, 10), limit=4)  # fmt: skip
    assert df.value.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]  # null value dropped, 2 pages read


def test_station_to_ist() -> None:
    s = 1704846600 + np.arange(3) * 3600  # IST-aligned CPCB hours
    pm = pd.DataFrame({"s": s, "e": s + 3600, "value": [50.0, -1.0, 70.0]})
    no2 = pd.DataFrame({"s": s, "e": s + 3600, "value": [46.01, 46.01, 46.01]})
    ws = pd.DataFrame({"s": s, "e": s + 3600, "value": [2.0, 2.0, 2.0]})
    wd = pd.DataFrame({"s": s, "e": s + 3600, "value": [90.0, 90.0, 90.0]})
    wide, aligned = openaq.station_to_ist(
        {"pm25": ("µg/m³", pm), "no2": ("µg/m³", no2), "wind_speed": ("m/s", ws),
         "wind_direction": ("deg", wd)}, 0.75)  # fmt: skip
    w = wide.set_index("bucket")
    assert aligned["pm25"] == 1.0
    assert w.pm25.tolist()[0] == 50.0 and np.isnan(w.pm25.tolist()[1])  # negative dropped
    assert w.no2_ppb.iloc[0] == pytest.approx(24.45)
    assert w.u_stn.iloc[0] == pytest.approx(-2.0) and w.wd_stn.iloc[0] == pytest.approx(90.0)


# ---- Open-Meteo ---------------------------------------------------------------------------


def test_open_meteo_real_payload(tmp_path: Path) -> None:
    payload = json.loads((FIX / "open_meteo_era5_delhi_2024-01-09_11.json").read_text())

    def handler(req: httpx.Request) -> httpx.Response:
        p = req.url.params
        assert p["timezone"] == "GMT" and p["wind_speed_unit"] == "ms" and p["models"] == "era5"
        return httpx.Response(200, json=payload)

    hourly = list(payload["hourly_units"])[1:]
    utc = open_meteo.fetch_hourly_utc(client(tmp_path, handler), "https://archive/x", 28.6358,
                                      77.2245, date(2024, 1, 9), date(2024, 1, 11), hourly, "era5")  # fmt: skip
    assert len(utc) == 72 and (utc.epoch % 3600 == 0).all()
    ist = open_meteo.to_ist(utc, ["precipitation", "shortwave_radiation"])
    assert ((ist.bucket + 19800) % 3600 == 0).all()  # every row is an IST hour start
    assert {"t2m", "rh2m", "sp", "u10", "v10", "ws10", "wd10", "precip", "swrad", "blh"} <= set(ist)
    inner = ist.iloc[2:-2]
    assert inner.t2m.between(-5, 35).all() and inner.ws10.between(0, 20).all()
    # IST 12:00 bucket (06:30Z) is centred on 07:00Z -> 1/8 v(06Z) + 3/4 v(07Z) + 1/8 v(08Z)
    v = utc.set_index("epoch").temperature_2m
    b = 1704868200  # 2024-01-10T06:30:00Z
    exp = 0.125 * v[b - 1800] + 0.75 * v[b + 1800] + 0.125 * v[b + 5400]
    assert ist.set_index("bucket").t2m[b] == pytest.approx(exp)


def test_open_meteo_rejects_local_time(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"utc_offset_seconds": 19800, "hourly": {"time": []}})

    with pytest.raises(ValueError):
        open_meteo.fetch_hourly_utc(client(tmp_path, handler), "https://a/x", 28.6, 77.2,
                                    date(2024, 1, 1), date(2024, 1, 1), ["temperature_2m"])  # fmt: skip


# ---- OpenAQ S3 archive (keyless) ----------------------------------------------------------


def _day_file(rows: list[tuple[str, str, str, float]]) -> bytes:
    import gzip

    head = (
        '"location_id","sensors_id","location","datetime","lat","lon","parameter","units","value"\n'
    )
    body = "".join(f'1,{sid},"X-1","{dt}",28.6,77.2,"{par}","{unit}",{v}\n' for sid, dt, par, unit, v in
                   [("7", dt, par, unit, v) for dt, par, unit, v in rows])  # fmt: skip
    return gzip.compress((head + body).encode("utf-8"))


def test_archive_end_labels_and_unit_repair() -> None:
    from aqi_ml.data import openaq_archive as oa

    rows = [
        (f"2025-11-10T{h:02d}:{m:02d}:00+05:30", "co", "ppb", 1.5)
        for h in range(1, 3)
        for m in (0, 15, 30, 45)
    ]
    df = oa.infer_periods(oa.parse_day(_day_file(rows)))
    assert set(df.end - df.s) == {900}  # 15-min spacing -> 15-min periods ending at the timestamp
    first = df.sort_values("end").iloc[0]
    assert first.end == 1762716600  # 2025-11-10T01:00+05:30 = 2025-11-09T19:30Z
    reps: list[dict] = []
    out = oa.to_canonical(df, ["co"], reps)
    assert out["co"][1].value.median() == pytest.approx(
        1.5 * 24.45 / 28.01, rel=1e-3
    )  # read as mg/m3
    assert reps and reps[0]["label"] == "ppb" and reps[0]["used"] == "mg/m³"
    nox = df.assign(parameter="nox", value=0.058)
    got = oa.to_canonical(nox, ["nox"], [])["nox"][1].value.median()
    assert got == pytest.approx(58.0, rel=1e-3)  # ppm-labelled-as-ppb slip, keeps NOx >= NO2
