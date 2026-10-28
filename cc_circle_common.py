"""
Shared helpers for utilization-validation (Phase 3). Fixed 330-circle basis.
Encodes the Akita 335->330 aggregation rule (see data/raw/external/README_ALIGNMENT.md):
never naive code-join a 335-basis source; old Akita 0501-0508 -> 県北/県央/県南.
"""
import json, re
import numpy as np, pandas as pd

PANEL = pd.read_csv("data/superseded/physician_panel_smsa__PRE_REPAIR.csv")
CODES = PANEL["smsa_code"].apply(lambda c: str(c).zfill(4)).tolist()

# 335-basis Akita (old) -> 330-basis (new) circle. counts SUM, rates POP-WEIGHT.
AKITA_OLD2NEW = {"0501": "0501", "0502": "0501", "0503": "0501",
                 "0504": "0502", "0505": "0502",
                 "0506": "0503", "0507": "0503", "0508": "0503"}


def _to330(code):
    return AKITA_OLD2NEW.get(code, code)


def census_pop(year):
    df = pd.read_csv(f"census_based/smsa_population_census_{year}.csv")
    df["code4"] = df["smsa_code"].apply(lambda c: str(c).zfill(4))
    return df.set_index("code4")["population"].reindex(CODES).astype(float)


def eff_panel(year):
    """effective-supply panel row set for a year, indexed by 4-digit code (330)."""
    d = pd.read_csv("results/effective_supply_panel.csv")
    d = d[d["year"] == year].copy()
    d["code4"] = d["smsa_code"].apply(lambda c: str(int(c)).zfill(4))
    return d.set_index("code4").reindex(CODES)


def _ps_meta_map(metafile):
    """internal cat01 code -> 4-digit SMA code (335-basis), from the meta labels."""
    m = json.load(open(metafile, encoding="utf-8"))
    out = {}
    for k, lab in m["cat01"]["codes"].items():
        mm = re.match(r"^(\d{4})\s", str(lab).strip())
        if mm:
            out[k] = mm.group(1)
    return out


def ps_raw335(metafile, valuesfile, cat02):
    """Raw 335-basis dict {4-digit code -> value} for one cat02 selection."""
    cmap = _ps_meta_map(metafile)
    vals = json.load(open(valuesfile, encoding="utf-8"))["values"]
    raw = {}
    for v in vals:
        if v.get("@cat02") != cat02:
            continue
        c335 = cmap.get(v.get("@cat01"))
        if not c335:
            continue
        raw[c335] = float(str(v["$"]).replace(",", "")) if v["$"] not in ("-", "", "…") else np.nan
    return raw


def ps_series(metafile, valuesfile, cat02, agg="sum", weight335=None):
    """Patient-survey SMA series -> Series on the 330 basis (Akita aggregated).
    agg='sum' for counts (推計患者数); agg='wmean' for %/rates, weighted by weight335
    (a {335-code -> weight} dict on the SAME basis; Akita districts pop/patient-weighted)."""
    raw = ps_raw335(metafile, valuesfile, cat02)
    if agg == "sum":
        acc = {c: 0.0 for c in CODES}
        for c335, x in raw.items():
            acc[_to330(c335)] += (0 if np.isnan(x) else x)
        return pd.Series(acc).reindex(CODES)
    num = {c: 0.0 for c in CODES}; den = {c: 0.0 for c in CODES}
    for c335, x in raw.items():
        if np.isnan(x):
            continue
        w = (weight335 or {}).get(c335, 0.0); c330 = _to330(c335)
        num[c330] += x * w; den[c330] += w
    return pd.Series({c: (num[c] / den[c] if den[c] > 0 else np.nan) for c in CODES}).reindex(CODES)


def underserved():
    return pd.read_csv("outputs/validation/underserved_by_circle.csv", dtype={"smsa_code": str}).set_index("smsa_code").reindex(CODES)
