#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for phase 2/3 physician distribution analyses."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
RESULTS_DIR = DATA_DIR / "results"
OD_YEARS = [2016, 2018, 2020, 2022, 2024]
POP_YEARS = [2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
# The static rounds of the Survey of Medical Institutions, which is triennial. The list
# was [2014, 2017, 2020, 2023, 2024] while the infrastructure files were prefecture totals
# allocated by physician share and one could be written for any year; the 2014 entry was
# never reached by nearest_year() from a flow wave, and the 2024 entry was a fourth
# allocation of the 2023 round's national totals rather than a survey. Under
# cc_facilities_pipeline.py only the three real rounds exist, and nearest_year() over them
# is the wave assignment: 2016 and 2018 -> 2017, 2020 -> 2020, 2022 and 2024 -> 2023.
INFRA_YEARS = [2017, 2020, 2023]


def ensure_results_dir() -> Path:
    """Create and return the results directory."""

    RESULTS_DIR.mkdir(exist_ok=True)
    return RESULTS_DIR


def smsa_code(value: object) -> str:
    """Normalize an SMSA code to a four-character string."""

    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(4)


def pref_code_from_smsa(code: object) -> str:
    """Return a two-character prefecture code from an SMSA code."""

    return smsa_code(code)[:2]


def load_smsa_master() -> pd.DataFrame:
    """Load the canonical SMSA code/name table from the physician panel."""

    panel = pd.read_csv(DATA_DIR / "data/superseded/physician_panel_smsa__PRE_REPAIR.csv")
    panel["smsa_code"] = panel["smsa_code"].map(smsa_code)
    if "pref_code" in panel.columns:
        panel["pref_code"] = panel["pref_code"].map(lambda x: str(int(x)).zfill(2) if pd.notna(x) else "")
    else:
        panel["pref_code"] = panel["smsa_code"].str[:2]
    return panel[["smsa_code", "smsa_name", "pref_code"]].copy()


def smsa_codes() -> list[str]:
    """Return canonical SMSA codes."""

    return load_smsa_master()["smsa_code"].tolist()


def load_centroids(codes: Iterable[str] | None = None) -> pd.DataFrame:
    """Load SMSA centroids with normalized codes."""

    df = pd.read_csv(DATA_DIR / "smsa_centroids_final.csv")
    df["smsa_code"] = df["smsa_code"].map(smsa_code)
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    if codes is not None:
        df = df.set_index("smsa_code").reindex(list(codes)).reset_index()
    return df


def load_matrix(path: Path, codes: list[str] | None = None) -> pd.DataFrame:
    """Load a square matrix CSV with normalized row and column SMSA codes."""

    df = pd.read_csv(path)
    first = df.columns[0]
    df[first] = df[first].map(smsa_code)
    df = df.set_index(first)
    df.columns = [smsa_code(col) for col in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if codes is not None:
        df = df.reindex(index=codes, columns=codes).fillna(0.0)
    return df


def load_od_matrix(year: int, codes: list[str] | None = None) -> pd.DataFrame:
    """Load an OD matrix for a year."""

    return load_matrix(DATA_DIR / f"od_matrix_{year}.csv", codes=codes)


def load_adjacency(codes: list[str] | None = None) -> pd.DataFrame:
    """Load the geographic adjacency matrix."""

    return load_matrix(DATA_DIR / "adjacency_final.csv", codes=codes)


def load_physician_panel() -> pd.DataFrame:
    """Load the SMSA physician-stock panel with normalized codes."""

    df = pd.read_csv(DATA_DIR / "data/superseded/physician_panel_smsa__PRE_REPAIR.csv")
    df["smsa_code"] = df["smsa_code"].map(smsa_code)
    for year in POP_YEARS:
        if str(year) in df.columns:
            df[str(year)] = pd.to_numeric(df[str(year)], errors="coerce")
    return df


def stock_series(year: int, codes: list[str] | None = None) -> pd.Series:
    """Return physician stock indexed by SMSA code for one year."""

    panel = load_physician_panel().set_index("smsa_code")
    series = pd.to_numeric(panel[str(year)], errors="coerce").fillna(0.0)
    if codes is not None:
        series = series.reindex(codes).fillna(0.0)
    return series


def nearest_year(target: int, available: Iterable[int]) -> int:
    """Return the available year nearest to a target year."""

    return min(available, key=lambda y: (abs(y - target), y))


CENSUS_DIR = DATA_DIR / "census_based"


class MissingDenominator(ValueError):
    """A density or population is absent where a model is about to consume it.

    Filling it with zero is not a neutral default. A circle with no density enters a
    gravity model as log1p(0) = 0 -- the density of an empty place -- and the estimate
    absorbs it silently. On 2026-08-29 that was happening to eight circles in the 2016
    wave, Yokohama, Nagoya and Kumamoto among them, because census_based/ was stale by one
    upstream repair. Nothing reported it: .fillna(0.0) at five call sites and twice more
    inside this module turned a missing input into a plausible number.

    So this raises instead. A missing denominator is a broken input, and the caller has to
    know. Physician COUNTS keep .fillna(0.0): a circle with no physicians recorded in a
    wave genuinely has none, and zero is the value, not a stand-in for one.
    """


def _no_missing(series: pd.Series, what: str, year: int, source) -> pd.Series:
    bad = series[series.isna()]
    if len(bad):
        raise MissingDenominator(
            f"{len(bad)} of {len(series)} circles have no {what} for {year} in {source}: "
            + ", ".join(str(c) for c in list(bad.index)[:12])
            + (" ..." if len(bad) > 12 else "")
            + ". Filling these with zero would enter them into a model as the density of an "
              "empty place; fix the input instead.")
    return series


def load_population(year: int, codes: list[str] | None = None) -> pd.Series:
    """Load SMSA population. Prefer census-based circle population (true 国勢調査
    aggregation); fall back to the legacy physician-share-allocated file.

    Raises MissingDenominator rather than filling a gap with zero."""

    census_path = CENSUS_DIR / f"smsa_population_census_{year}.csv"
    if census_path.exists():
        df = pd.read_csv(census_path)
        df["smsa_code"] = df["smsa_code"].map(smsa_code)
        series = pd.to_numeric(df.set_index("smsa_code")["population"], errors="coerce")
        series = _no_missing(series, "population", year, census_path)
        if codes is not None:
            series = _no_missing(series.reindex(codes), "population", year,
                                 f"{census_path} (reindexed to {len(codes)} codes)")
        return series

    chosen = year if (DATA_DIR / f"smsa_population_{year}.csv").exists() else nearest_year(year, POP_YEARS)
    df = pd.read_csv(DATA_DIR / f"smsa_population_{chosen}.csv")
    df["smsa_code"] = df["smsa_code"].map(smsa_code)
    series = pd.to_numeric(df.set_index("smsa_code")["population"], errors="coerce")
    series = _no_missing(series, "population", chosen, f"smsa_population_{chosen}.csv")
    if codes is not None:
        series = _no_missing(series.reindex(codes), "population", chosen,
                             f"smsa_population_{chosen}.csv (reindexed)")
    return series


def load_density(year: int, codes: list[str] | None = None) -> pd.Series:
    """Load SMSA physician density. Prefer census-based circle density (true
    circle-level); fall back to the legacy file (which is degenerate at the
    prefecture level)."""

    census_path = CENSUS_DIR / f"smsa_density_census_{year}.csv"
    if census_path.exists():
        df = pd.read_csv(census_path)
        df["smsa_code"] = df["smsa_code"].map(smsa_code)
        series = pd.to_numeric(df.set_index("smsa_code")["physicians_per_100k"], errors="coerce")
        series = _no_missing(series, "density", year, census_path)
        if codes is not None:
            series = _no_missing(series.reindex(codes), "density", year,
                                 f"{census_path} (reindexed to {len(codes)} codes)")
        return series

    chosen = year if (DATA_DIR / f"smsa_density_{year}.csv").exists() else nearest_year(year, POP_YEARS)
    df = pd.read_csv(DATA_DIR / f"smsa_density_{chosen}.csv")
    df["smsa_code"] = df["smsa_code"].map(smsa_code)
    series = pd.to_numeric(df.set_index("smsa_code")["physicians_per_100k"], errors="coerce")
    series = _no_missing(series, "density", chosen, f"smsa_density_{chosen}.csv")
    if codes is not None:
        series = _no_missing(series.reindex(codes), "density", chosen,
                             f"smsa_density_{chosen}.csv (reindexed)")
    return series


def load_infrastructure(year: int, codes: list[str] | None = None) -> pd.DataFrame:
    """Load SMSA infrastructure using the nearest available year."""

    chosen = year if (DATA_DIR / f"smsa_infrastructure_{year}.csv").exists() else nearest_year(year, INFRA_YEARS)
    df = pd.read_csv(DATA_DIR / f"smsa_infrastructure_{chosen}.csv")
    df["smsa_code"] = df["smsa_code"].map(smsa_code)
    for col in ["hospitals", "clinics", "total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["source_year"] = chosen
    if codes is not None:
        df = df.set_index("smsa_code").reindex(codes).reset_index()
        for col in ["hospitals", "clinics", "total"]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
    return df


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Compute Haversine distance in kilometers."""

    radius = 6371.0088
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def distance_matrix(codes: list[str]) -> pd.DataFrame:
    """Return an SMSA-by-SMSA Haversine distance matrix."""

    centroids = load_centroids(codes).set_index("smsa_code")
    lat = centroids["latitude"].to_numpy()
    lon = centroids["longitude"].to_numpy()
    dist = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    return pd.DataFrame(dist, index=codes, columns=codes)


def gini(values: np.ndarray | pd.Series) -> float:
    """Compute the unweighted Gini coefficient for non-negative values."""

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    arr = np.clip(arr, 0.0, None)
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * arr) / (n * np.sum(arr))) - ((n + 1.0) / n))


def matrix_flows(matrix: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Return inflow, outflow, net inflow, and off-diagonal matrix."""

    values = matrix.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(values, 0.0)
    offdiag = pd.DataFrame(values, index=matrix.index, columns=matrix.columns)
    outflow = offdiag.sum(axis=1)
    inflow = offdiag.sum(axis=0)
    net = inflow - outflow
    return inflow, outflow, net, offdiag


def save_json(path: Path, payload: dict) -> None:
    """Write a JSON file with stable formatting."""

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            val = float(obj)
            return None if math.isnan(val) or math.isinf(val) else val
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        return str(obj)

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=convert), encoding="utf-8-sig")


def save_figure(path: Path) -> None:
    """Save the current Matplotlib figure and close it."""

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
