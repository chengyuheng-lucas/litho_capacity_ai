from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


Source = Literal["fred", "stooq", "file"]
Kind = Literal["price", "level"]


@dataclass(frozen=True)
class SeriesDef:
    name: str
    source: Source
    key: str
    kind: Kind

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "source": self.source, "key": self.key, "kind": self.kind}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SeriesDef":
        return cls(name=str(d["name"]), source=str(d["source"]), key=str(d["key"]), kind=str(d["kind"]))


DEFAULT_SERIES: list[SeriesDef] = [
    SeriesDef(name="SP500", source="fred", key="SP500", kind="price"),
    SeriesDef(name="VIX", source="fred", key="VIXCLS", kind="level"),
    SeriesDef(name="WTI", source="fred", key="DCOILWTICO", kind="price"),
    SeriesDef(name="GOLD", source="fred", key="GOLDAMGBD228NLBM", kind="price"),
    SeriesDef(name="COPPER", source="fred", key="PCOPPUSDM", kind="price"),
    SeriesDef(name="USD_EUR", source="fred", key="DEXUSEU", kind="price"),
    SeriesDef(name="USD_CNY", source="fred", key="DEXCHUS", kind="price"),
    SeriesDef(name="US10Y", source="fred", key="DGS10", kind="level"),
    SeriesDef(name="US2Y", source="fred", key="DGS2", kind="level"),
    SeriesDef(name="10Y2Y", source="fred", key="T10Y2Y", kind="level"),
    SeriesDef(name="FEDFUNDS", source="fred", key="FEDFUNDS", kind="level"),
    SeriesDef(name="PMI", source="fred", key="NAPM", kind="level"),
    SeriesDef(name="INDPRO", source="fred", key="INDPRO", kind="level"),
    SeriesDef(name="UNRATE", source="fred", key="UNRATE", kind="level"),
    SeriesDef(name="SOXX", source="stooq", key="soxx.us", kind="price"),
    SeriesDef(name="SMH", source="stooq", key="smh.us", kind="price"),
    SeriesDef(name="ASML", source="stooq", key="asml.us", kind="price"),
    SeriesDef(name="AMAT", source="stooq", key="amat.us", kind="price"),
    SeriesDef(name="LRCX", source="stooq", key="lrcx.us", kind="price"),
]


def _http_get(url: str, timeout_s: float = 15.0, retries: int = 2) -> bytes:
    last: Exception | None = None
    for i in range(max(1, retries)):
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                return resp.read()
        except Exception as e:
            last = e
            time.sleep(0.8 * (2**i))
    raise last if last is not None else RuntimeError("http_get_failed")


def _cached_fetch(url: str, cache_dir: str | Path, ttl_hours: float, timeout_s: float, retries: int) -> bytes:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    path = cache_dir / f"{h}.bin"
    now = time.time()
    if path.exists():
        age_h = (now - path.stat().st_mtime) / 3600.0
        if age_h <= ttl_hours:
            return path.read_bytes()
    data = _http_get(url, timeout_s=timeout_s, retries=retries)
    path.write_bytes(data)
    return data


def _fetch_fred_series(
    series_id: str,
    start: str | None,
    end: str | None,
    cache_dir: str | Path,
    ttl_hours: float,
    timeout_s: float,
    retries: int,
) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    if start is not None:
        url += f"&cosd={start}"
    if end is not None:
        url += f"&coed={end}"
    payload = _cached_fetch(url, cache_dir=cache_dir, ttl_hours=ttl_hours, timeout_s=timeout_s, retries=retries)
    df = pd.read_csv(pd.io.common.BytesIO(payload))
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        dt_col = "date"
    elif "observation_date" in df.columns:
        dt_col = "observation_date"
    else:
        raise ValueError("missing date col")
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True).dt.tz_localize(None)
    df = df.set_index(dt_col).sort_index()

    val_col = series_id.lower()
    if val_col not in df.columns:
        other = [c for c in df.columns if c not in {"date", "observation_date"}]
        if len(other) == 1:
            val_col = other[0]
        else:
            raise ValueError("missing value col")

    s = pd.to_numeric(df[val_col], errors="coerce")
    s = s.replace(".", np.nan).astype(float)
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    s = s.dropna()
    s.name = series_id
    return s


def _fetch_stooq_close(
    symbol: str,
    start: str | None,
    end: str | None,
    cache_dir: str | Path,
    ttl_hours: float,
    timeout_s: float,
    retries: int,
) -> pd.Series:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    payload = _cached_fetch(url, cache_dir=cache_dir, ttl_hours=ttl_hours, timeout_s=timeout_s, retries=retries)
    df = pd.read_csv(pd.io.common.BytesIO(payload))
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError(f"bad stooq payload: {symbol}")
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    df = df.set_index("Date").sort_index()
    s = pd.to_numeric(df["Close"], errors="coerce").astype(float).dropna()
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    s.name = symbol
    return s


def _fetch_file_series(path: str | Path, start: str | None, end: str | None) -> pd.Series:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "date" in cols and "value" in cols:
        dt_col = cols["date"]
        val_col = cols["value"]
    elif "date" in df.columns and "close" in cols:
        dt_col = "date"
        val_col = cols["close"]
    elif "date" in cols and "close" in cols:
        dt_col = cols["date"]
        val_col = cols["close"]
    else:
        raise ValueError(f"bad csv: {path}")
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True).dt.tz_localize(None)
    s = pd.to_numeric(df[val_col], errors="coerce").astype(float)
    s.index = df[dt_col]
    s = s.sort_index().dropna()
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    return s


def fetch_weekly_frame(
    series: list[SeriesDef],
    start: str | None,
    end: str | None,
    cache_dir: str | Path,
    ttl_hours: float,
    timeout_s: float,
    retries: int,
) -> pd.DataFrame:
    out: dict[str, pd.Series] = {}
    errors: dict[str, str] = {}

    for s in series:
        try:
            if s.source == "fred":
                raw = _fetch_fred_series(
                    s.key,
                    start=start,
                    end=end,
                    cache_dir=cache_dir,
                    ttl_hours=ttl_hours,
                    timeout_s=timeout_s,
                    retries=retries,
                )
            elif s.source == "stooq":
                raw = _fetch_stooq_close(
                    s.key,
                    start=start,
                    end=end,
                    cache_dir=cache_dir,
                    ttl_hours=ttl_hours,
                    timeout_s=timeout_s,
                    retries=retries,
                )
            elif s.source == "file":
                raw = _fetch_file_series(s.key, start=start, end=end)
            else:
                raise ValueError(s.source)
            wk = raw.resample("W-FRI").last().ffill()
            wk.name = s.name
            out[s.name] = wk
        except Exception as e:
            errors[s.name] = str(e)

    if not out:
        raise RuntimeError(json.dumps(errors, ensure_ascii=False))

    df = pd.concat(out.values(), axis=1).sort_index()
    df = df.ffill().dropna()
    return df


def build_features(frame: pd.DataFrame, series: list[SeriesDef]) -> tuple[pd.DataFrame, list[str]]:
    cols = []
    names = []

    kind_map = {s.name: s.kind for s in series}

    for col in frame.columns:
        kind = kind_map.get(col, "level")
        s = frame[col].astype(float)

        if kind == "price":
            s = s.clip(lower=1e-8)
            lvl = np.log(s)
            ret1 = lvl.diff().fillna(0.0)
            mom4 = lvl.diff(4).fillna(0.0)
            vol8 = ret1.rolling(8).std().fillna(0.0)
            cols.extend([lvl, ret1, mom4, vol8])
            names.extend([f"{col}_log", f"{col}_ret1", f"{col}_mom4", f"{col}_vol8"])
        else:
            d1 = s.diff().fillna(0.0)
            d4 = s.diff(4).fillna(0.0)
            cols.extend([s, d1, d4])
            names.extend([f"{col}_lvl", f"{col}_d1", f"{col}_d4"])

    feat = pd.concat(cols, axis=1)
    feat.columns = names
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feat, names


def _default_target(frame: pd.DataFrame) -> str:
    for c in ["SOXX", "SMH", "ASML", "SP500"]:
        if c in frame.columns:
            return c
    return frame.columns[0]


def build_weak_labels(
    frame: pd.DataFrame,
    target_col: str,
    horizon_weeks: int,
    max_ratio: float,
) -> tuple[pd.Series, pd.Series]:
    s = frame[target_col].astype(float).clip(lower=1e-8)
    lvl = np.log(s)
    ret1 = lvl.diff().fillna(0.0)

    fut_ret = (lvl.shift(-horizon_weeks) - lvl).fillna(0.0)
    fut_vol = ret1.rolling(horizon_weeks).std().shift(-horizon_weeks + 1).fillna(0.0)

    ratio = (0.25 * fut_ret - 0.10 * fut_vol).clip(-max_ratio, max_ratio)
    inv = (8.0 + 10.0 * (-fut_ret).clip(lower=0.0) + 8.0 * fut_vol).clip(2.0, 26.0)

    ratio.name = "y_adj_ratio"
    inv.name = "y_inv_weeks"
    return inv.astype(float), ratio.astype(float)


def make_public_samples(
    series: list[SeriesDef],
    lookback_weeks: int,
    horizon_weeks: int,
    max_ratio: float,
    start: str | None,
    end: str | None,
    cache_dir: str | Path,
    ttl_hours: float,
    timeout_s: float,
    retries: int,
    target_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], str, pd.DatetimeIndex]:
    frame = fetch_weekly_frame(
        series=series,
        start=start,
        end=end,
        cache_dir=cache_dir,
        ttl_hours=ttl_hours,
        timeout_s=timeout_s,
        retries=retries,
    )
    target = target_col or _default_target(frame)
    inv, ratio = build_weak_labels(frame=frame, target_col=target, horizon_weeks=horizon_weeks, max_ratio=max_ratio)
    feat, feat_names = build_features(frame=frame, series=series)

    idx = feat.index
    valid_end = len(idx) - horizon_weeks
    if valid_end <= lookback_weeks:
        raise ValueError("not enough history")

    xs = []
    ys_inv = []
    ys_ratio = []
    xs_index = []

    feat_np = feat.to_numpy(dtype=np.float32)
    inv_np = inv.to_numpy(dtype=np.float32)
    ratio_np = ratio.to_numpy(dtype=np.float32)

    for t in range(lookback_weeks - 1, valid_end):
        x = feat_np[t - lookback_weeks + 1 : t + 1]
        y1 = inv_np[t]
        y2 = ratio_np[t]
        xs.append(x)
        ys_inv.append(y1)
        ys_ratio.append(y2)
        xs_index.append(idx[t])

    x_arr = np.stack(xs, axis=0).astype(np.float32)
    y_inv_arr = np.array(ys_inv, dtype=np.float32)
    y_ratio_arr = np.array(ys_ratio, dtype=np.float32)
    return x_arr, y_inv_arr, y_ratio_arr, feat_names, target, pd.DatetimeIndex(xs_index)


def latest_public_window(
    series: list[SeriesDef],
    lookback_weeks: int,
    start: str | None,
    end: str | None,
    cache_dir: str | Path,
    ttl_hours: float,
    timeout_s: float,
    retries: int,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    frame = fetch_weekly_frame(
        series=series,
        start=start,
        end=end,
        cache_dir=cache_dir,
        ttl_hours=ttl_hours,
        timeout_s=timeout_s,
        retries=retries,
    )
    feat, _ = build_features(frame=frame, series=series)
    if len(feat) < lookback_weeks:
        raise ValueError("not enough history")
    window = feat.iloc[-lookback_weeks:].copy()
    ts = feat.index[-1]
    return window, ts


def default_public_end() -> str:
    return date.today().isoformat()
