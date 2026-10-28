"""
Emit the extrapolated age-share matrix s_ia and its simplex diagnostics.

WHY NOT MODIFY cc_stage2_needweight.py AND RUN IT, as the brief expected. That script
reads `data/superseded/physician_panel_smsa__PRE_REPAIR.csv` -- the PRE-repair panel -- and writes
`exploratory/S2_gini_by_denominator.csv` and `exploratory/S2_decomposition.csv`, which
round 13 moved to `outputs/archive_prerepair/` precisely because they are pre-repair. It
has no switch to run only the age part, so running it would recreate the artefacts just
retired. The age functions themselves are panel-independent (census only), so they are
reproduced here VERBATIM with a drift guard: the source text of both functions is
compared against the original file, and this script refuses to run if either has changed.

What the implementation actually does, read off cc_stage2_needweight.py:

    def shares_for_year(ay):
        if ay <= 2015:  v = s2010 + (s2015 - s2010) * ((ay - 2010) / 5.0)
        else:           v = s2015 + (s2020 - s2015) * ((ay - 2015) / 5.0)
        v = np.clip(v, 0, None); renormalise

So 2006 and 2008 are obtained by extrapolating BACKWARD along the 2010-2015 slope
(ay - 2010 is negative). A4 documents only forward extrapolation for 2022 and 2024.

Writes outputs/audit/i06_age_shares.csv        per analysis year: simplex diagnostics
       outputs/audit/i06_age_min_share.csv     where the minimum share occurs
       outputs/audit/i06_elderly_share.csv     national 65+ share per analysis year
"""
import sys, io, os, re, inspect
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

SRC = open("cc_stage2_needweight.py", encoding="utf-8").read()
GUARD = """def shares_for_year(ay):
    s2010, s2015, s2020 = CENSUS_SHARES[2010], CENSUS_SHARES[2015], CENSUS_SHARES[2020]
    if ay <= 2015:
        v = s2010 + (s2015 - s2010) * ((ay - 2010) / 5.0)
    else:
        v = s2015 + (s2020 - s2015) * ((ay - 2015) / 5.0)
    v = np.clip(v, 0, None)
    rs = v.sum(axis=1, keepdims=True); rs[rs == 0] = np.nan
    return v / rs"""
assert GUARD in SRC, ("cc_stage2_needweight.py::shares_for_year has changed since this emit "
                      "was written -- re-read it before trusting these numbers")
print("drift guard: shares_for_year matches cc_stage2_needweight.py verbatim")

YEARS = [2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
CENSUS_AGE_YEARS = [2010, 2015, 2020]
GROUPS = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44",
          "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85-89", "90+"]
ELDERLY = [i for i, g in enumerate(GROUPS) if g in
           ("65-69", "70-74", "75-79", "80-84", "85-89", "90+")]

vm = pd.read_csv("data/crosswalk/vintage_sma_municipalities.csv", dtype=str, encoding="utf-8-sig")
m24 = vm[vm.year == "2024"][["muni", "sma"]].drop_duplicates()
MAP = dict(zip(m24.muni, m24.sma))
CODES = sorted(set(m24.sma))
cidx = {c: i for i, c in enumerate(CODES)}


def sma_age_shares(cy):
    # Round 38 (2026-10-06): the area age table, built once by cc_build_census_age.py through
    # the code-change tables. Summing census_age/age_muni_ over the 2024 map's codes, as this
    # did until then, lost every municipality whose code had changed since the census --
    # 熊本市 in 2010, the seven Hamamatsu wards in 2010-2020 -- and gave those areas the age
    # mix of what was left.
    t = pd.read_csv(f"census_age/age_circle_{cy}.csv", dtype={"code4": str}, encoding="utf-8")
    mat = t.set_index("code4").reindex(CODES)[GROUPS].to_numpy(float)
    tot = mat.sum(axis=1, keepdims=True)
    return np.divide(mat, tot, out=np.full_like(mat, np.nan), where=tot > 0)


CENSUS_SHARES = {cy: sma_age_shares(cy) for cy in CENSUS_AGE_YEARS}


def shares_raw(ay):
    s2010, s2015, s2020 = CENSUS_SHARES[2010], CENSUS_SHARES[2015], CENSUS_SHARES[2020]
    if ay <= 2015:
        return s2010 + (s2015 - s2010) * ((ay - 2010) / 5.0)
    return s2015 + (s2020 - s2015) * ((ay - 2015) / 5.0)


def shares_used(ay):
    v = np.clip(shares_raw(ay), 0, None)
    rs = v.sum(axis=1, keepdims=True); rs[rs == 0] = np.nan
    return v / rs


rows, mins = [], []
for ay in YEARS:
    raw, used = shares_raw(ay), shares_used(ay)
    extrap = (ay < 2010) or (ay > 2020)
    direction = "backward" if ay < 2010 else ("forward" if ay > 2020 else "interpolated")
    n_neg = int((raw < 0).sum())
    dev = float(np.nanmax(np.abs(raw.sum(axis=1) - 1.0)))
    k = int(np.nanargmin(raw))
    ci, gi = divmod(k, raw.shape[1])
    rows.append({"year": ay, "extrapolated": extrap, "direction": direction,
                 "raw_min_share": float(np.nanmin(raw)), "raw_n_negative": n_neg,
                 "raw_max_abs_sum_dev": dev,
                 "clip_triggered": bool(n_neg > 0),
                 "used_min_share": float(np.nanmin(used))})
    mins.append({"year": ay, "extrapolated": extrap, "circle": CODES[ci],
                 "age_group": GROUPS[gi], "min_share": float(np.nanmin(raw))})

A = pd.DataFrame(rows); A.to_csv("outputs/audit/i06_age_shares.csv", index=False,
                                 encoding="utf-8-sig")
M = pd.DataFrame(mins); M.to_csv("outputs/audit/i06_age_min_share.csv", index=False,
                                 encoding="utf-8-sig")

# ---- national 65+ share, from the same shares, weighted by circle population
def circle_pop(ay):
    d = pd.read_csv(f"census_based/smsa_population_census_{ay}.csv")
    d["code4"] = d["smsa_code"].apply(lambda c: str(c).zfill(4))
    return d.set_index("code4")["population"].reindex(CODES).astype(float).to_numpy()


er = []
for ay in YEARS:
    sh = shares_used(ay)
    pop = circle_pop(ay)
    e65 = np.nansum(pop * sh[:, ELDERLY].sum(axis=1))
    er.append({"year": ay, "national_65plus_share": float(e65 / np.nansum(pop)),
               "basis": "backward extrapolation 2010-2015 slope" if ay < 2010 else
                        ("forward extrapolation 2015-2020 slope" if ay > 2020 else
                         "census / interpolated")})
E = pd.DataFrame(er); E.to_csv("outputs/audit/i06_elderly_share.csv", index=False,
                               encoding="utf-8-sig")

pd.set_option("display.width", 200)
print()
print(A.to_string(index=False))
print()
print("minimum share, extrapolated waves only:")
print(M[M.extrapolated].to_string(index=False))
print()
print(E.to_string(index=False))
print()
ext = A[A.extrapolated]
print(f"A4 'smallest share observed' over the extrapolated waves : {ext.raw_min_share.min():.6f}"
      f"   printed 0.011   {'ok' if abs(ext.raw_min_share.min()-0.011) <= 0.0005001 else 'MISMATCH'}")
print(f"A4 max |row sum - 1|                                     : "
      f"{A.raw_max_abs_sum_dev.max():.3e}   printed 4e-16")
print(f"A4 clip triggered anywhere                               : {bool(A.clip_triggered.any())}"
      f"   printed 'never triggered'")
for y, t in [(2006, 0.20), (2024, 0.30)]:
    v = float(E[E.year == y].national_65plus_share.iloc[0])
    print(f"A4 national 65+ share {y}                              : {v:.4f}   printed {t:.2f}"
          f"   {'ok' if abs(v-t) <= 0.005001 else 'MISMATCH'}")
v20 = float(E[E.year == 2020].national_65plus_share.iloc[0])
v24 = float(E[E.year == 2024].national_65plus_share.iloc[0])
print(f"2020 census-measured {v20:.4f} vs 2024 extrapolated {v24:.4f}  "
      f"(difference attributable to the age extrapolation: {v24-v20:+.4f})")
