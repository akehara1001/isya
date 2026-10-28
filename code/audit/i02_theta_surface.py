"""
ROUND 13, WORK 3.2 -- regenerate the theta surface and the denominator reversal.

These support Supplementary B1 and Table S4, and were the only remaining quantities
carried by files with no generating script (`B_theta_year_surface.csv`,
`B_crossover_theta.csv`, `A_paradox_by_density_quintile.csv`,
`C_paradox_direction_old_vs_new.csv`, `C_robustness_old_vs_new.csv`). Everything here is
recomputed from the repaired panel, the census denominators and the OD matrices, so the
release set no longer depends on artefacts nobody can rebuild.

No new method: this is the theta sensitivity already described in Methods 2.4 and B1.

  effective supply   S_i = max(0, x_i + theta * n_i),  n_i = inflow_i - outflow_i
  density            10^5 * S_i / P_i
  two denominators   census aggregation, and the physician-share allocation of A1.1

Writes outputs/audit/i02_theta_surface.csv and .txt
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
FLOW_YEARS = [2016, 2018, 2020, 2022, 2024]
THETAS = np.round(np.arange(0, 1.0001, 0.1), 2)
log = []


def say(s=""):
    print(s)
    log.append(s)


panel = pd.read_csv("data/crosswalk/physician_panel_smsa_repaired.csv",
                    dtype={"code4": str}, encoding="utf-8-sig").set_index("code4")
CODES = list(panel.index)


def stock(y):
    return panel[str(y)].reindex(CODES).astype(float).to_numpy()


def census_pop(y):
    d = pd.read_csv(f"census_based/smsa_population_census_{y}.csv")
    d["code4"] = d["smsa_code"].apply(lambda c: str(c).zfill(4))
    return d.set_index("code4")["population"].reindex(CODES).astype(float).to_numpy()


def flows(y):
    od = pd.read_csv(f"od_matrix_{y}.csv")
    F = od.drop(columns=["smsa_code"]).to_numpy(float)
    np.fill_diagonal(F, 0)
    return F.sum(0), F.sum(1)          # inflow, outflow


def gini(v):
    xs = np.sort(np.asarray(v, float)); n = xs.size; ii = np.arange(1, n + 1)
    return (2 * (xs * ii).sum()) / (n * xs.sum()) - (n + 1) / n


def share_pop(y):
    """Physician-share allocation: prefecture population split by physician share."""
    x = stock(y); P = census_pop(y)
    pref = np.array([c[:2] for c in CODES])
    out = np.zeros_like(P)
    for p in np.unique(pref):
        m = pref == p
        out[m] = P[m].sum() * x[m] / x[m].sum()
    return out


rows = []
say(f"theta grid {THETAS[0]}..{THETAS[-1]} step 0.1;  flow years {FLOW_YEARS}")
say("")
say(f"{'year':>5} {'theta':>6} {'primary (census)':>17} {'effective (census)':>19} "
    f"{'primary (share)':>16} {'effective (share)':>18}")
for y in FLOW_YEARS:
    x, Pc, Ps = stock(y), census_pop(y), share_pop(y)
    inflow, outflow = flows(y)
    n = inflow - outflow
    for th in THETAS:
        S = np.maximum(0.0, x + th * n)
        gc_p, gc_e = gini(x / Pc * 1e5), gini(S / Pc * 1e5)
        gs_p, gs_e = gini(x / Ps * 1e5), gini(S / Ps * 1e5)
        rows.append({"year": y, "theta": float(th),
                     "gini_primary_census": gc_p, "gini_effective_census": gc_e,
                     "dir_census": "shrink" if gc_e < gc_p else "widen",
                     "gini_primary_share": gs_p, "gini_effective_share": gs_e,
                     "dir_share": "shrink" if gs_e < gs_p else "widen",
                     "min_effective_supply": float(np.min(x + th * n)),
                     "truncation_binds": bool(np.min(x + th * n) < 0)})
        if th in (0.5,):
            say(f"{y:>5} {th:>6.1f} {gc_p:>17.4f} {gc_e:>19.4f} {gs_p:>16.4f} {gs_e:>18.4f}")

D = pd.DataFrame(rows)
D.to_csv("outputs/audit/i02_theta_surface.csv", index=False, encoding="utf-8-sig")

say("")
cross = D[(D.theta > 0) & (D.dir_census != "shrink")]
say(f"census denominator: theta-year cells where effective Gini is NOT below primary: "
    f"{len(cross)} of {len(D[D.theta > 0])}  -> no crossover" if not len(cross) else
    f"CROSSOVER at: {cross[['year','theta']].to_string(index=False)}")
w = D[(D.theta > 0) & (D.dir_share != "widen")]
say(f"physician-share denominator: cells where the direction is not 'widen': {len(w)}")
say(f"truncation ever binding: {bool(D.truncation_binds.any())}; "
    f"minimum of (x + theta n) over all cells: {D.min_effective_supply.min():.1f}")

say("")
say("Table S4 check (theta = 0.5):")
say(f"{'year':>5} {'share primary':>14} {'share effective':>16} {'census primary':>15} "
    f"{'census effective':>17}")
for y in (2016, 2024):
    r = D[(D.year == y) & (D.theta == 0.5)].iloc[0]
    say(f"{y:>5} {r.gini_primary_share:>14.3f} {r.gini_effective_share:>16.3f} "
        f"{r.gini_primary_census:>15.3f} {r.gini_effective_census:>17.3f}")

Path("outputs/audit/i02_theta_surface.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
print("\nwrote outputs/audit/i02_theta_surface.{csv,txt}")
