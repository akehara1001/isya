"""
STAGE G (part) -- generating scripts for the two unbound tables that have simple sources.

Table S1  tab:degeneracy  within-prefecture SD of density under the two denominators
Table 2   tab:quint       net secondary-practice flow by destination-density quintile, 2024

Both are recomputed on the repaired panel and the consistent 330-circle set. The three
failure modes this project has actually produced are checked for explicitly:
  (i)   an inconsistent circle set  -> assert_units(330)
  (ii)  a pre-repair derived file   -> reads the repaired panel and census_based populations
  (iii) silent missing removal      -> no_missing on every vector before it is aggregated

Writes outputs/audit/j02_table_s1.csv and j02_table_2.csv
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from guards import assert_units, no_missing

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
N = 330

rep = pd.read_csv("data/crosswalk/physician_panel_smsa_repaired.csv",
                  dtype={"code4": str}, encoding="utf-8-sig").set_index("code4")
CODES = list(rep.index)
assert_units(CODES, N, "repaired panel circles")
PREF = np.array([c[:2] for c in CODES])


def pop(y):
    d = pd.read_csv(f"census_based/smsa_population_census_{y}.csv")
    d["code4"] = d.smsa_code.apply(lambda c: str(c).zfill(4))
    v = d.set_index("code4")["population"].reindex(CODES).astype(float).to_numpy()
    assert_units(v.size, N, f"{y} circle population")
    return no_missing(v, f"{y} circle population")


def stock(y):
    v = rep[str(y)].astype(float).to_numpy()
    assert_units(v.size, N, f"{y} physician stock")
    return no_missing(v, f"{y} physician stock")


# ================================================================ Table S1
rows = []
for y in (2006, 2016, 2024):
    x, P = stock(y), pop(y)
    d_census = 1e5 * x / P
    # physician-share allocation: P_i = P_pref * x_i / sum_pref x  ->  density is a
    # prefecture constant, so the within-prefecture SD is floating-point zero
    d_share = np.empty(N)
    for p in np.unique(PREF):
        m = PREF == p
        d_share[m] = 1e5 * x[m].sum() / P[m].sum()
    # ddof=1, unweighted mean over all 47 prefectures. Established by enumerating all
    # eight combinations of (ddof 0/1) x (drop single-circle prefectures) x (unweighted /
    # population-weighted): only ddof=1 unweighted reproduces the printed 2024 value
    # (83.07 against 83.1). No combination reproduces 2006 or 2016 on any panel or circle
    # set, so those two printed values were never reproducible and are replaced.
    sd_c = np.mean([d_census[PREF == p].std(ddof=1) if (PREF == p).sum() > 1 else 0.0
                    for p in np.unique(PREF)])
    sd_s = np.mean([d_share[PREF == p].std(ddof=1) if (PREF == p).sum() > 1 else 0.0
                    for p in np.unique(PREF)])
    rows.append({"year": y, "within_pref_sd_physician_share": sd_s,
                 "within_pref_sd_census": sd_c,
                 "distinct_density_values_census": int(len(np.unique(np.round(d_census, 6)))),
                 "distinct_density_values_share": int(len(np.unique(np.round(d_share, 6))))})
S1 = pd.DataFrame(rows)
S1.to_csv("outputs/audit/j02_table_s1.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200)
print("Table S1 (degeneracy):")
print(S1.to_string(index=False))
print("  printed in S1: physician-share ~1e-14 in all three years; census 113.6 / 82.8 / 83.1")

# ================================================================ Table 2
y = 2024
x, P = stock(y), pop(y)
dens = 1e5 * x / P
od = pd.read_csv(f"od_matrix_{y}.csv")
F = od.drop(columns=["smsa_code"]).to_numpy(float)
assert_units(F.shape[0], N, "OD matrix rows")
assert_units(F.shape[1], N, "OD matrix columns")
np.fill_diagonal(F, 0)
net = no_missing(F.sum(0) - F.sum(1), "net secondary-practice flow")

q = pd.qcut(pd.Series(dens, index=CODES), 5, labels=[1, 2, 3, 4, 5])
rows = []
for k in (1, 2, 3, 4, 5):
    m = (q == k).to_numpy()
    rows.append({"quintile": k, "n_circles": int(m.sum()),
                 "mean_density": float(dens[m].mean()),
                 "total_net_flow": float(net[m].sum()),
                 "mean_net_per_circle": float(net[m].mean())})
T2 = pd.DataFrame(rows)
assert_units(int(T2.n_circles.sum()), N, "Table 2 quintiles")
T2.to_csv("outputs/audit/j02_table_2.csv", index=False, encoding="utf-8-sig")
print()
print("Table 2 (net flow by destination-density quintile, 2024):")
print(T2.to_string(index=False))
print("  printed in Table 2: 153.6/195.0/225.3/268.1/392.0 mean density; "
      "+2633/+3075/+2870/+1115/-9693 total; +39.9/+46.6/+43.5/+16.9/-146.9 per circle")
print(f"  net flow sums to {net.sum():+.1f} (must be 0 by construction)")
