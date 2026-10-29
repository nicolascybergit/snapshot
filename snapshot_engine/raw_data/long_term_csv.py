"""
Raw Data — long-term CSV history (Nasdaq ~25y, US stocks 50y+).
Always-live access. The base every other level rests on.
Format: Nasdaq export (Close/Last, MM/DD/YYYY).

Implementation notes (Nasdaq export quirks handled here):
  - columns: Date, Close/Last, Volume, Open, High, Low
  - the Volume column feeds applied_data/volume.py (close and volume come from
    the SAME parsed frame, so both series share one index)
  - price cells may be prefixed with "$" and contain thousands commas -> strip both
  - dates are MM/DD/YYYY
  - exports are newest-first -> we sort ascending by date
  - histories differ in length (a ticker may start mid-series) -> never assume
    alignment; each symbol is returned as its own independent series
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pandas as pd

_PRICE_COLUMNS = ["Close/Last", "Open", "High", "Low"]


def _clean_price(series: pd.Series) -> pd.Series:
    """Strip '$' and thousands commas, coerce to float."""
    cleaned = (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_dates(column: pd.Series) -> pd.Series:
    """Nasdaq exports use MM/DD/YYYY; fall back to free parsing for other exports."""
    try:
        return pd.to_datetime(column, format="%m/%d/%Y")
    except (ValueError, TypeError):
        return pd.to_datetime(column, errors="coerce")


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Shared cleanup: strip headers, parse dates, coerce prices, sort ascending."""
    df.columns = [c.strip() for c in df.columns]
    if "Close/Last" not in df.columns and "Close" in df.columns:
        df = df.rename(columns={"Close": "Close/Last"})
    if "Date" not in df.columns or "Close/Last" not in df.columns:
        raise ValueError("CSV needs a 'Date' and a 'Close/Last' (or 'Close') column")
    df["Date"] = _parse_dates(df["Date"])
    for col in _PRICE_COLUMNS:
        if col in df.columns:
            df[col] = _clean_price(df[col])
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df.dropna(subset=["Date", "Close/Last"])
    return df.sort_values("Date").set_index("Date")


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Load one Nasdaq CSV file into a clean, date-sorted (ascending) DataFrame
    indexed by ``Date`` with float OHLC/Close columns.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    return _normalise(pd.read_csv(p))


def load_csv_text(text: str) -> pd.DataFrame:
    """
    Same as :func:`load_csv` but for CSV content held in memory — used by the
    dashboard when a file is opened in the browser instead of read from disk.
    """
    return _normalise(pd.read_csv(io.StringIO(text)))


def close_series_from_text(text: str) -> pd.Series:
    """Ascending daily close series (float) from CSV content."""
    return load_csv_text(text)["Close/Last"].astype(float)


def close_series(path: str | Path) -> pd.Series:
    """Ascending daily close series (float) from one CSV file."""
    return load_csv(path)["Close/Last"].astype(float)


def volume_series(path: str | Path) -> pd.Series:
    """Ascending daily traded-volume series (float) from one CSV file."""
    df = load_csv(path)
    if "Volume" not in df.columns:
        return pd.Series(dtype=float, index=df.index)
    return df["Volume"].astype(float)


def volume_series_from_text(text: str) -> pd.Series:
    """Same as :func:`volume_series` but for CSV content held in memory."""
    df = load_csv_text(text)
    if "Volume" not in df.columns:
        return pd.Series(dtype=float, index=df.index)
    return df["Volume"].astype(float)


def resolve_path(symbol: str, csv_dir: str | Path, csv_map: dict[str, str]) -> Optional[Path]:
    """Map a symbol to its CSV path via ``csv_map``; None if unmapped/missing."""
    filename = csv_map.get(symbol)
    if not filename:
        return None
    path = Path(csv_dir) / filename
    return path if path.exists() else None


def load(symbol: str, csv_dir: str | Path, csv_map: dict[str, str]) -> Optional[pd.Series]:
    """
    Load a symbol's normalised, date-sorted close series. Returns None when the
    symbol is unmapped or its file is absent (histories are handled gracefully).
    """
    path = resolve_path(symbol, csv_dir, csv_map)
    if path is None:
        return None
    return close_series(path)


def load_volume(symbol: str, csv_dir: str | Path,
                csv_map: dict[str, str]) -> Optional[pd.Series]:
    """Same as :func:`load` but for the traded-volume column."""
    path = resolve_path(symbol, csv_dir, csv_map)
    if path is None:
        return None
    return volume_series(path)
