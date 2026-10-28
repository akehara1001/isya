"""
Emit how each census year's municipality units reach the 330 areas, as cc_census_pipeline.py
builds them.

This is NOT a reimplementation. It imports `area_population` from the pipeline and calls it,
so the figures written out here are literally those the denominator is built from.

ROUND 38 (2026-10-06). Until this round this file emitted the coverage of the COUNT-BASED
PRO-RATA BACK-FILL that supplementary A1.3 prints (95.6% in 2005, 99.0% in 2010, ...): the
share of the published national population that the back-filled circle totals reproduced.
The back-fill is withdrawn. Every census unit is now carried to the 2024 area map by the
explicit code-change tables, and the pipeline asserts that the 330 areas sum to the published
national total and to every published prefecture total with zero difference -- so
coverage_pct is 100.0 in every census year BY CONSTRUCTION, and A1.3 has to be rewritten in a
manuscript round rather than re-bound here. The column names are kept so that h11's binding
reports the mismatch instead of losing the column.

Writes outputs/audit/i05_census_coverage.csv  (per circle, per census year)
       outputs/audit/i05_census_coverage_national.csv  (one row per census year)
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import cc_census_pipeline as ccp                      # the pipeline itself
import census_crosswalk as X                          # noqa: E402
from census_inputs import municipality_csv            # noqa: E402

PRINTED = {2005: 95.6, 2010: 99.0, 2015: 99.6, 2020: 99.5, 2025: 100.0}

m24, _ = X.fixed_map()
rows, nat = [], []
for cy in ccp.CENSUS_YEARS:
    s, T = ccp.area_population(cy)
    natl = float(X.reference(cy)["00"])
    reached = {t for v in T.codes_2024 for t in v.split()}
    per_circle = {}
    for code, circ in m24.items():
        per_circle.setdefault(circ, [0, 0])
        per_circle[circ][0] += 1
        per_circle[circ][1] += int(code in reached)
    cov = pd.Series({c: v[1] / v[0] for c, v in per_circle.items()}).reindex(ccp.CODES4)
    nat.append({"census_year": cy,
                "mean_circle_coverage_unweighted_pct": float(100.0 * cov.mean()),
                "backfilled_circle_pop": float(s.sum()),
                "national_population": natl,
                "coverage_pct": float(100.0 * s.sum() / natl), "printed_in_A1_3": PRINTED.get(cy),
                "circles_below_100pct": int((cov < 0.999).sum()),
                "min_circle_coverage": float(cov.min()),
                "census_units": int(T.census_code.nunique()),
                "units_carried_by_code_change_tables": int(T.moved.sum()),
                "rows_from_declared_splits": int((T.share < 1).sum())})
    moved = T[T.moved].groupby("area").census_code.nunique()
    for c in ccp.CODES4:
        rows.append({"census_year": cy, "circle": c, "muni_coverage_count_based": float(cov[c]),
                     "backfilled_population": float(s[c]),
                     "units_carried_by_code_change_tables": int(moved.get(c, 0))})

# ---------------------------------------------------------------------------------------
# The gap between the total-population table and the sum of the age table.
#
# This block used to record that gap under one name -- `unstated_age_persons` -- for every
# year alike, and that name is a CLAIM about where the difference comes from. For a table
# that imputes unknown age it is false: such a table is exhaustive by construction and has
# no unstated age at all, so a positive gap there is something the name does not describe.
#
# It was. On 2026-08-29 the 2020 row read 80,469, and 126,226,568 - 126,146,099 = 80,469:
# the preliminary tabulation minus the confirmed one. Total population came from the
# 速報集計 and age composition from the 不詳補完結果 keyed to the confirmed 人口等基本集計.
# A revision between two editions of the census had been sitting in a column called
# "unstated age" for as long as anyone had looked, absorbed in silence because nothing
# printed the difference or asked what it was.
#
# So: which edition each age table is comes from split/census_age_tables.csv, quoting the
# e-Stat STATISTICS_NAME for the id the fetch script passes -- input-side metadata, not an
# inference from the data. Every year prints its gap and what the gap is attributed to,
# and a year whose table imputes unknown age must show ZERO or this script fails.
# ---------------------------------------------------------------------------------------
import os as _os
AGE_TABLES = "split/census_age_tables.csv"
_AT = pd.read_csv(AGE_TABLES, encoding="utf-8-sig", comment="#",
                  dtype={"statsdataid": str})   # 0000033787 is not 33787
_AT["unknown_age_imputed"] = (_AT["unknown_age_imputed"].astype(str).str.strip().str.lower()
                              == "true")
_AT = _AT.set_index("census_year")
assert len(_AT), "the age-table declaration is empty; this check would examine nothing"

_gap_failures = []
for _cy in (2005, 2010, 2015, 2020):
    _f = f"census_age/age_muni_{_cy}.csv"
    if not _os.path.exists(_f):
        continue
    assert _cy in _AT.index, (
        f"census_age/age_muni_{_cy}.csv exists and {AGE_TABLES} does not declare which "
        f"table it is. An undeclared age table cannot be checked.")
    _a = pd.read_csv(_f, dtype={"area_code": str})
    _a["area_code"] = _a.area_code.str.zfill(5)
    _aged = _a[_a.area_code.str.endswith("000") & (_a.area_code != "00000")].iloc[:, 1:].sum().sum()
    _t = pd.read_csv(municipality_csv(_cy), dtype={"area_code": str})
    _t["area_code"] = _t.area_code.str.zfill(5)
    _t["population"] = pd.to_numeric(_t.population, errors="coerce")
    _tot = _t[_t.area_code.str.endswith("000") & (_t.area_code != "00000")].population.sum()
    _imputed = bool(_AT.loc[_cy, "unknown_age_imputed"])
    _gap = float(_tot - _aged)
    _attr = ("unstated age (the table reports 年齢不詳 separately and the fetch drops it)"
             if not _imputed else
             ("nothing: the table imputes unknown age and the sums agree" if _gap == 0 else
              "UNEXPLAINED -- the table imputes unknown age, so this is NOT unstated age"))
    nat_row = [r for r in nat if r["census_year"] == _cy]
    if nat_row:
        nat_row[0]["age_statsdataid"] = str(_AT.loc[_cy, "statsdataid"])
        nat_row[0]["unknown_age_imputed"] = _imputed
        nat_row[0]["age_table_sum"] = float(_aged)
        nat_row[0]["total_population"] = float(_tot)
        nat_row[0]["total_minus_age_sum"] = _gap
        nat_row[0]["gap_attributed_to"] = _attr
        # The historical column keeps its historical meaning and is populated only where
        # that meaning is true. NaN for an imputed year says the column does not apply --
        # it does not delete the number, which is in total_minus_age_sum for every year.
        # Writing 0 here would have been worse than the old behaviour: it would assert
        # there is nothing to explain.
        nat_row[0]["unstated_age_persons"] = _gap if not _imputed else np.nan
        nat_row[0]["unstated_age_pct"] = (float(100.0 * _gap / _tot) if not _imputed
                                          else np.nan)
    if _imputed and _gap != 0:
        _gap_failures.append((_cy, _gap, str(_AT.loc[_cy, "statsdataid"])))

C = pd.DataFrame(rows)
C.to_csv("outputs/audit/i05_census_coverage.csv", index=False, encoding="utf-8-sig")
N = pd.DataFrame(nat)
N.to_csv("outputs/audit/i05_census_coverage_national.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 200)
print(N.to_string(index=False))
print()
print("A1.3 printed the back-fill's coverage; the back-fill is withdrawn (round 38).")
print("coverage_pct is now 100.0 in every census year by construction, and A1.3 is to be")
print("rewritten in a manuscript round. The printed values, for the record:")
for _, r in N.iterrows():
    print(f"   census {int(r.census_year)}: now {r.coverage_pct:6.2f}%   printed {r.printed_in_A1_3:5.1f}%   "
          f"units {int(r.census_units):,}, carried by the code-change tables "
          f"{int(r.units_carried_by_code_change_tables):,}")
assert (N.coverage_pct.round(9) == 100.0).all(), "an area total does not reproduce the census"

print("\ntotal population against the sum of the age table, per census year:")
print(f"  age-table declarations read from {AGE_TABLES} : {len(_AT)}")
for _, r in N.iterrows():
    if "total_minus_age_sum" not in r or pd.isna(r.get("total_minus_age_sum")):
        continue
    print(f"   census {int(r.census_year)}  id {r.age_statsdataid}  "
          f"imputed={'yes' if r.unknown_age_imputed else 'no ':<3}  "
          f"total {r.total_population:>13,.0f}  age sum {r.age_table_sum:>13,.0f}  "
          f"gap {r.total_minus_age_sum:>+10,.0f}")
    print(f"        -> {r.gap_attributed_to}")
if _gap_failures:
    print("\n  FAIL: an age table that imputes unknown age does not sum to the total.")
    for _cy, _gap, _id in _gap_failures:
        print(f"    {_cy}: gap {_gap:+,.0f} against statsDataId {_id}, a 不詳補完結果.")
    print("  The age classes of an imputed-unknown table are exhaustive, so this difference")
    print("  is not unstated age. The two sides are different editions of the same census.")
assert not _gap_failures, (
    "imputed-unknown age table(s) do not sum to the total population: "
    + "; ".join(f"{c} off by {g:+,.0f}" for c, g, _ in _gap_failures)
    + ". Recording this as unstated age is what this assertion exists to stop.")
