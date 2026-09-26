"""Polite HTTP GET for downloaders: disk cache (resumable), rate limit, retry with backoff.

Cache key = sha256(url + sorted params). Headers (API keys) are never part of the key or the
file. Only 200 responses are cached, so a rerun resumes exactly where a failed run stopped.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx

RETRY_STATUS = {429, 500, 502, 503, 504}


def _reason(r: httpx.Response) -> str:
    try:
        body = r.json()
    except ValueError:
        return r.text[:200]
    return str(body.get("reason", "")) if isinstance(body, dict) else ""


class CachedClient:
    def __init__(
        self,
        cache_dir: Path,
        min_interval_s: float,
        headers: dict[str, str] | None = None,
        max_retries: int = 5,
        max_limit_waits: int = 60,
        transport: httpx.BaseTransport | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self.cache_dir = cache_dir
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.max_limit_waits = max_limit_waits  # 429 waits (1 min each) before giving up
        self._sleep = sleep
        self._last = 0.0
        self.network_calls = 0
        self._client = httpx.Client(headers=headers or {}, timeout=60.0, transport=transport)

    def _path(self, url: str, params: dict[str, Any]) -> Path:
        key = url + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
        return self.cache_dir / (hashlib.sha256(key.encode()).hexdigest() + ".json")

    def get_json(self, url: str, params: dict[str, Any], cache: bool = True) -> Any:
        path = self._path(url, params)
        if cache and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        limit_waits = 0
        attempt = -1
        while attempt < self.max_retries:
            attempt += 1
            wait = self.min_interval_s - (time.monotonic() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = time.monotonic()
            self.network_calls += 1
            try:
                r = self._client.get(url, params=params)
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
                self._sleep(2**attempt)
                continue
            if r.status_code == 200:
                data = r.json()
                if cache:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(data), encoding="utf-8")
                return data
            if r.status_code == 429:
                reason = _reason(r)
                if (
                    "daily" in reason.lower()
                ):  # Open-Meteo daily cap: stop; a rerun resumes from cache
                    raise RuntimeError(f"daily API limit reached ({reason}); rerun later to resume")
                if limit_waits < self.max_limit_waits:
                    limit_waits += 1
                    reset = r.headers.get("x-ratelimit-reset") or r.headers.get("retry-after")
                    self._sleep(float(reset) if reset and reset.isdigit() else 60.0)
                    attempt -= 1  # rate-limit waits do not use up error retries
                    continue
            if r.status_code in RETRY_STATUS and attempt < self.max_retries:
                self._sleep(2**attempt)
                continue
            r.raise_for_status()
        raise RuntimeError("unreachable")
