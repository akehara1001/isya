"""
STEP 3-5: Build census-based circle population, density, Gini; bidirectional paradox.
Writes ONLY to census_based/ (or CENSUS_BASED_OUT). Reads census_raw/ and existing raw files
(read-only). Does not modify or delete any existing file.

ROUND 38 (2026-10-06, internal work record). THE COUNT-BASED BACK-FILL
IS GONE. Until this round each census year's municipality codes were matched to the 2024 codes
of the area map, and an area whose codes failed to match was scaled up by the inverse of the
COUNT of municipalities that did. That fails worst exactly where it matters: 熊本市 (43201 until
its wards were created in 2012) left 熊本・上益城 at 174,804 in 2010 against 821,876, the seven
old Hamamatsu wards left 西部 at about a quarter of its size in 2010, 2015 and 2020, and an
absorbed town whose absorber kept its code (鳩ケ谷市 into 川口市) vanished with the coverage
still reading 100%.

Each census unit is now carried to the 2024 map by the explicit code-change tables
(data/crosswalk/municipality_code_changes_*.csv, every row sourced to the e-Stat LOD), in
census_crosswalk.unit_table. A unit that cannot be carried, a map code no unit reaches, or a
unit split between two areas without a published apportionment STOPS the run, and the codes
are printed. Before anything is written, every census year's 330 areas must sum to the
published national total and to each of the 47 published prefecture totals
(data/reference/census_official_totals.csv) with zero difference.

Writes
  census_year_population_{cy}.csv   the five census anchors (2005 ... 2025), as built
  smsa_population_census_{ay}.csv   the ten analysis waves, linear between anchors (unchanged rule)
  smsa_density_census_{ay}.csv, gini_census_2006_2024.csv   as before
"""
import os, sys, numpy as np, pandas as pd
from phase2_common import DATA_DIR
import census_crosswalk as X

# Round 38 writes census_based_v2/ first and swaps it in only after q28 passes.
OUT = os.environ.get("CENSUS_BASED_OUT", "census_based")

ANALYSIS_YEARS = [2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
CENSUS_YEARS = [2005, 2010, 2015, 2020, 2025]
# FIXED modern circle definition for ALL years (consistent 330-circle panel across time)
FIXED_MAP_VINTAGE = X.MAP_VINTAGE

# The circle codes, names and physician stock come from the REPAIRED panel (round 38, stage 0).
# This read the superseded pre-repair panel until 2026-10-05. The codes and their order are the
# same in both files; the stock differs in 2006-2014 only, which moves the density files of those
# waves and nothing that is read for 2016-2024. smsa_code is kept as the integer the old file
# carried, because every reader rebuilds code4 from it with zfill(4).
panel = pd.read_csv("data/crosswalk/physician_panel_smsa_repaired.csv",
                    dtype={"code4": str}, encoding="utf-8-sig")
panel["smsa_code"] = panel["code4"].astype(int)
CODES4 = panel["code4"].tolist()

# The code-change tables, read here by name so that the dependency graph (n08) sees this script
# depend on cc_build_code_changes.py, which writes them. census_crosswalk reads the same files;
# the assertion in area_population() makes the two one read.
CHANGE_TABLES = {name: pd.read_csv(f"data/crosswalk/municipality_code_changes_{name}.csv",
                                   dtype=str, encoding="utf-8")
                 for name in ("2005_2010", "2010_2024")}


def area_population(cy):
    """The 330 area populations of census year cy, carried by the code-change tables.
    Raises (census_crosswalk.CrosswalkError) rather than fill a gap."""
    mine = pd.concat(CHANGE_TABLES.values(), ignore_index=True)
    theirs, _ = X.changes()
    assert mine.sort_values(list(mine.columns)).reset_index(drop=True).equals(
        theirs.sort_values(list(theirs.columns)).reset_index(drop=True)), (
        "census_crosswalk read different code-change tables from the ones named here")
    T, _ = X.unit_table(cy)
    s = T.groupby("area")["population"].sum().reindex(CODES4)
    assert s.notna().all() and len(s) == 330, f"census {cy}: an area has no census unit"
    nat, pref = X.reconcile(cy, s)
    assert nat == 0 and not pref, (
        f"census {cy}: the 330 areas do not reconcile to the published totals: national "
        f"{nat:+,}, prefectures {pref}")
    return s, T


def main():
    os.makedirs(OUT, exist_ok=True)
    # ---- aggregate each census year to circles ----
    circle_census = {}
    for cy in CENSUS_YEARS:
        s, T = area_population(cy)
        circle_census[cy] = s
        ref = X.reference(cy)
        print(f"census {cy}: units {T.census_code.nunique():,}, carried by the code-change tables "
              f"{int(T.moved.sum()):,}, split by a declared apportionment "
              f"{int((T.share < 1).sum()):,}; 330-area sum {s.sum():,.0f} = published "
              f"{int(ref['00']):,} (difference 0; 47 prefectures, difference 0)")
        pd.DataFrame({"smsa_code": panel["smsa_code"], "code4": CODES4,
                      "population": s.values}).to_csv(
            f"{OUT}/census_year_population_{cy}.csv", index=False, lineterminator="\n")

    # ---- interpolate each analysis year ----
    pop_by_year = {}
    for ay in ANALYSIS_YEARS:
        if ay in CENSUS_YEARS:
            pop_by_year[ay] = circle_census[ay].copy()
        else:
            lo = max([c for c in CENSUS_YEARS if c < ay])
            hi = min([c for c in CENSUS_YEARS if c > ay])
            w = (ay - lo) / (hi - lo)
            pop_by_year[ay] = circle_census[lo] * (1 - w) + circle_census[hi] * w
        # save population
        dfp = pd.DataFrame({"smsa_code": panel["smsa_code"], "code4": CODES4,
                            "population": pop_by_year[ay].values})
        dfp.to_csv(f"{OUT}/smsa_population_census_{ay}.csv", index=False, lineterminator="\n")

    # ---- STEP 4: density, within-pref std, distinct count, Gini ----
    def gini(x):
        x = np.asarray(x, float); x = x[~np.isnan(x)]; n = x.size
        if n == 0 or x.sum() == 0:
            return np.nan
        xs = np.sort(x); idx = np.arange(1, n + 1)
        return (2 * np.sum(idx * xs)) / (n * xs.sum()) - (n + 1) / n

    pref = pd.Series([c[:2] for c in CODES4], index=CODES4)
    gini_rows = []
    print("\n=== STEP 4: census-based density ===")
    print(f"{'year':>5} {'distinct_density':>16} {'mean_withinpref_std':>20} {'stock_Gini':>11} {'density_Gini':>13}")
    for ay in ANALYSIS_YEARS:
        stock = panel.set_index("code4")[str(ay)].reindex(CODES4).astype(float)
        pop = pop_by_year[ay]
        dens = stock / pop * 1e5
        # save density
        pd.DataFrame({"smsa_code": panel["smsa_code"], "code4": CODES4,
                      "physicians_per_100k": dens.values}).to_csv(
            f"{OUT}/smsa_density_census_{ay}.csv", index=False, lineterminator="\n")
        # within-pref std
        wstd = dens.groupby(pref).std().mean()
        ndist = dens.round(4).nunique()
        sg = gini(stock.values); dg = gini(dens.values)
        gini_rows.append({"year": ay, "stock_Gini": sg, "density_Gini": dg,
                          "distinct_density": int(ndist), "mean_withinpref_density_std": float(wstd)})
        print(f"{ay:>5} {ndist:>16} {wstd:>20.4f} {sg:>11.4f} {dg:>13.4f}")
    gdf = pd.DataFrame(gini_rows)
    gdf.to_csv(f"{OUT}/gini_census_2006_2024.csv", index=False, lineterminator="\n")

    # ---- old vs new density Gini ----
    print("\n=== OLD vs NEW density Gini (per year) ===")
    print(f"{'year':>5} {'old_density_Gini':>16} {'new_density_Gini':>16}")
    def gini_old(stock, pop):
        d = stock / pop * 1e5
        return gini(d.values)
    for ay in ANALYSIS_YEARS:
        oldpop = pd.read_csv(os.path.join(DATA_DIR, f"smsa_population_{ay}.csv"))
        oldpop["code4"] = oldpop["smsa_code"].apply(lambda c: str(c).zfill(4))
        op = oldpop.set_index("code4")["population"].reindex(CODES4)
        stock = panel.set_index("code4")[str(ay)].reindex(CODES4).astype(float)
        og = gini_old(stock, op.replace(0, np.nan))
        ng = gdf.set_index("year").loc[ay, "density_Gini"]
        print(f"{ay:>5} {og:>16.4f} {ng:>16.4f}")

    # ---- STEP 5: bidirectional paradox with NEW density ----
    def od_flows(year):
        od = pd.read_csv(os.path.join(DATA_DIR, f"od_matrix_{year}.csv"))
        F = od.drop(columns=["smsa_code"]).to_numpy(float)
        np.fill_diagonal(F, 0)
        return F.sum(axis=0), F.sum(axis=1)  # inflow_i, outflow_i

    print("\n=== STEP 5: bidirectional paradox (theta=0.5), census-based density ===")
    for year in [2016, 2024]:
        stock = panel.set_index("code4")[str(year)].reindex(CODES4).astype(float).values
        pop = pop_by_year[year].values
        inflow, outflow = od_flows(year)
        eff = np.clip(stock + 0.5 * (inflow - outflow), 0, None)
        dprim = stock / pop * 1e5
        deff = eff / pop * 1e5
        print(f"  {year}: stock_Gini   primary={gini(stock):.4f}  effective={gini(eff):.4f}  "
              f"({'shrink' if gini(eff)<gini(stock) else 'widen'})")
        print(f"  {year}: density_Gini primary={gini(dprim):.4f}  effective={gini(deff):.4f}  "
              f"({'WIDEN' if gini(deff)>gini(dprim) else 'shrink'})")


if __name__ == "__main__":
    main()
