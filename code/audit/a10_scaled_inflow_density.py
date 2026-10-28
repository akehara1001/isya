"""
co-author review round 2, item B-4 -- is corr(n_i, d_i) mechanical?

THE OBJECTION, verbatim. n_i is a net flow whose outflow leg is sum_j F_ij, and d_i has
x_i in its numerator; an area with many physicians necessarily moves many of them, so the
two variables share x_i and part of the negative correlation follows from that sharing
rather than from any equalising behaviour.

WHAT IS COMPUTED. The same Pearson correlation with the net inflow put on a scale that no
longer grows with the area's own size, two ways:

  population-scaled   10^5 * n_i / P_i   net inflow per 100 000 residents; the same
                                         denominator d_i already carries, so the two
                                         variables are then in the same units
  stock-scaled        n_i / x_i          net inflow as a fraction of the area's own stock;
                                         divides out the shared quantity directly

NO NEW METHOD AND NO NEW DATA. The data path is the one Figure S5 draws from -- repaired
panel, od_matrix_{year}.csv, census_based population -- and the run asserts that the
UNSCALED correlation reproduces the published -0.6665 (2016) and -0.7061 (2024) before
reporting anything else, so a divergence upstream shows up here as a failure rather than
as four new numbers. Intervals are percentile bootstrap over paired area resamples,
NB = 5,000, the procedure and draw count cc_validation_tests.py uses for every correlation
it reports; the generator is seeded per (year, scaling) so each interval is independent of
the order the rows are computed in.

Writes outputs/audit/a10_scaled_inflow_density.csv
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)

SEED = 20260906
NB = 5000
YEARS = [2016, 2024]

# The published unscaled correlations, to four decimals. Not a target to be reproduced by
# adjustment: the assert below fails the run if the data path stops giving them.
PUBLISHED = {2016: -0.7049, 2024: -0.7061}
# Round 40 §3 (2026-10-06): 2016 moved from -0.6665 to -0.7049 when the census denominators
# (round 38), the flow matrices (round 39) and the confirmed 2025 census were applied to the
# manuscript; 2024 is unchanged. The guard stays: it now asserts what the paper prints.

panel = pd.read_csv("data/crosswalk/physician_panel_smsa_repaired.csv",
                    encoding="utf-8-sig", dtype={"code4": str}).set_index("code4")

rows = []
for year in YEARS:
    od = pd.read_csv(f"od_matrix_{year}.csv", index_col=0, encoding="utf-8-sig")
    od.index = [f"{int(i):04d}" for i in od.index]
    od.columns = [f"{int(c):04d}" for c in od.columns]
    assert list(od.index) == list(od.columns), f"{year}: OD matrix is not square-labelled"
    M = od.values.astype(float).copy()
    np.fill_diagonal(M, 0.0)                      # main text Eq. 3 convention
    n = pd.Series(M.sum(0) - M.sum(1), index=od.index)

    pop = pd.read_csv(f"census_based/smsa_population_census_{year}.csv",
                      encoding="utf-8-sig",
                      dtype={"code4": str}).set_index("code4")["population"]
    x = panel[str(year)].reindex(od.index).astype(float)
    P = pop.reindex(od.index).astype(float)
    d = 1e5 * x / P
    assert n.notna().all() and d.notna().all() and P.notna().all(), f"{year}: incomplete area set"
    assert (x > 0).all(), f"{year}: an area has no primary stock, so n_i/x_i is undefined"
    assert len(n) == 330, f"{year}: {len(n)} areas, not 330"

    r_raw = float(np.corrcoef(n.to_numpy(), d.to_numpy())[0, 1])
    assert abs(r_raw - PUBLISHED[year]) < 5e-5, (
        f"{year}: unscaled r = {r_raw:.6f}, published {PUBLISHED[year]}")

    for scaling, v in [("unscaled", n),
                       ("per 100 000 residents", 1e5 * n / P),
                       ("share of own stock", n / x)]:
        a, b = v.to_numpy(float), d.to_numpy(float)
        r = float(stats.pearsonr(a, b)[0])
        g = np.random.default_rng([SEED, year, len(scaling)])
        draws = np.empty(NB)
        for k in range(NB):
            idx = g.integers(0, len(a), len(a))
            draws[k] = stats.pearsonr(a[idx], b[idx])[0]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append({"year": year, "scaling": scaling, "n_areas": len(a),
                     # SIX decimals: the page prints two, so a value on the rounding
                     # boundary can still be adjudicated against this file.
                     "r": round(r, 6),
                     "ci_lo": round(float(lo), 6), "ci_hi": round(float(hi), 6),
                     "boot_median": round(float(np.median(draws)), 6),
                     "n_boot": NB, "excludes_0": bool(lo > 0 or hi < 0)})

out = pd.DataFrame(rows)
out.to_csv("outputs/audit/a10_scaled_inflow_density.csv", index=False,
           encoding="utf-8-sig")
pd.set_option("display.width", 200)
print(out.to_string(index=False))

sc = out[out.scaling != "unscaled"]
print(f"\nscaled rows: {len(sc)}   all negative: {bool((sc.r < 0).all())}   "
      f"weakest |r|: {float(sc.r.abs().min()):.4f}")
print("wrote outputs/audit/a10_scaled_inflow_density.csv")
