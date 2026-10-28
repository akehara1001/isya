"""Area age tables: the census age table of each year carried to the 330 areas by the code-change
tables, the same way as the area populations.

Round 38, stage 4 (internal work record). No network.

WHY. Every producer of the need-weighted and 65-plus denominators built its own area age
composition by summing census_age/age_muni_{cy}.csv over the municipality codes of the 2024
area map. A municipality whose code changed between the census and 2024 matched nothing and
was left out, so in 2010 the age composition of 熊本・上益城 was that of 上益城郡 alone (熊本市
43201 missing) and in 2010, 2015 and 2020 that of 西部 was 湖西市 alone (the seven Hamamatsu
wards missing). That is the population denominator's defect again, one level down. These
tables are built ONCE, through census_crosswalk, and the producers read them.

Reads   census_age/age_muni_{cy}.csv            (the fetched municipality age tables, unchanged)
        census_raw via census_inputs, the code-change tables and the declarations, through
        census_crosswalk.unit_table
Writes  {OUT}/age_circle_{cy}.csv for cy in 2005, 2010, 2015, 2020
        columns: code4, the 19 age groups (age-stated persons), age_stated, population,
        age_unknown (= population - age_stated)

THE TWO SIDES OF EACH ROW. The age groups are the AGE-STATED persons, as published.
`population` is the area's published total, built by the same unit table as
census_based. Under the standing rule (needbase.py, B-100) the need weight applies the
age-stated SHARES to the published total, which apportions the unstated-age residents pro
rata; this table keeps both sides so the apportionment is visible rather than implied.

ASSERTED, per census year, before anything is written:
  * every census unit with a population has a row in the age table; a unit missing from it
    must have population 0 (the evacuated towns of 2015 and 2020);
  * the areas' age-stated sums, grouped by prefecture, equal the age table's own prefecture
    rows -- nothing lost, nothing counted twice;
  * 2015 and 2020 (不詳補完結果, exhaustive): age_stated == population in every area;
  * 2005 and 2010 (年齢不詳 published separately): 0 <= age_unknown in every area. No floor is
    put on the age-stated fraction: an early draft required 0.95 and 2010's 区西部 (1304) is
    0.9244, which is the census and not a fault -- 2010 had its highest 年齢不詳 in central
    Tokyo. A threshold with no source is not a check.

OUT defaults to census_age; round 38 first writes census_age_v2 (CENSUS_AGE_OUT) and swaps it in
only after q28 passes.

    python cc_build_census_age.py
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import census_crosswalk as X  # noqa: E402

OUT = os.environ.get("CENSUS_AGE_OUT", "census_age")
GROUPS = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44",
          "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85-89", "90+"]
AGE_YEARS = (2005, 2010, 2015, 2020)
EXHAUSTIVE = (2015, 2020)
TOL = 1e-6


def build(cy, codes):
    age = pd.read_csv(f"census_age/age_muni_{cy}.csv", dtype={"area_code": str}, encoding="utf-8")
    age["area_code"] = age["area_code"].str.zfill(5)
    age = age.set_index("area_code")[GROUPS].astype(float)
    T, _ = X.unit_table(cy, values=age)
    absent = T[T[GROUPS].isna().any(axis=1)]
    bad = absent[absent.population != 0]
    assert bad.empty, (f"{cy}: census units with population and no age row: "
                       f"{bad.census_code.tolist()}")
    T[GROUPS] = T[GROUPS].fillna(0.0)
    A = T.groupby("area")[GROUPS + ["population"]].sum().reindex(codes)
    assert A.notna().all().all(), f"{cy}: an area has no units"
    A["age_stated"] = A[GROUPS].sum(axis=1)
    A["age_unknown"] = A["population"] - A["age_stated"]

    pref_rows = age[age.index.str.endswith("000") & (age.index != "00000")]
    by_pref = A["age_stated"].groupby(A.index.str[:2]).sum()
    diff = {p[:2]: float(by_pref.get(p[:2], 0.0) - pref_rows.loc[p].sum()) for p in pref_rows.index}
    off = {p: d for p, d in diff.items() if abs(d) > TOL}
    assert len(pref_rows) == 47 and not off, (
        f"{cy}: area age-stated sums do not equal the age table's prefecture rows: {off}")

    if cy in EXHAUSTIVE:
        dev = (A["age_stated"] - A["population"]).abs()
        assert (dev <= TOL).all(), (
            f"{cy}: the table is 不詳補完結果, so every area's age-stated sum must equal its "
            f"population; worst {dev.idxmax()} off by {dev.max():,.6f}")
    else:
        assert (A["age_unknown"] >= -TOL).all(), (
            f"{cy}: an area has more age-stated persons than residents: "
            f"{A[A.age_unknown < -TOL].index.tolist()}")
    return A, T


# The code-change tables, read here by name so that the dependency graph (n08) orders this
# script after cc_build_code_changes.py, which writes them. census_crosswalk reads the same
# files; main() asserts that the two reads agree.
CHANGE_TABLES = {name: pd.read_csv(f"data/crosswalk/municipality_code_changes_{name}.csv",
                                   dtype=str, encoding="utf-8")
                 for name in ("2005_2010", "2010_2024")}


def main():
    mine = pd.concat(CHANGE_TABLES.values(), ignore_index=True)
    theirs, _ = X.changes()
    assert mine.sort_values(list(mine.columns)).reset_index(drop=True).equals(
        theirs.sort_values(list(theirs.columns)).reset_index(drop=True)), (
        "census_crosswalk read different code-change tables from the ones named here")
    panel = pd.read_csv("data/crosswalk/physician_panel_smsa_repaired.csv",
                        dtype={"code4": str}, encoding="utf-8-sig")
    codes = panel["code4"].tolist()
    os.makedirs(OUT, exist_ok=True)
    print(f"area age tables -> {OUT}/")
    print(f"{'year':>5} {'units':>6} {'split rows':>10} {'age-stated':>14} {'population':>14} "
          f"{'unstated age':>12} {'min frac':>9}")
    for cy in AGE_YEARS:
        A, T = build(cy, codes)
        out = A.reset_index().rename(columns={"area": "code4", "index": "code4"})
        out.insert(0, "code4", out.pop("code4"))
        out = out[["code4"] + GROUPS + ["age_stated", "population", "age_unknown"]]
        out.to_csv(f"{OUT}/age_circle_{cy}.csv", index=False, encoding="utf-8",
                   lineterminator="\n")
        print(f"{cy:>5} {T.census_code.nunique():>6} {int((T.share < 1).sum()):>10} "
              f"{A.age_stated.sum():>14,.0f} {A.population.sum():>14,.0f} "
              f"{A.age_unknown.sum():>12,.0f} {(A.age_stated / A.population).min():>9.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
