"""
ROUND 11, Q14 -- the fixed-cohort decile profile, as a saveable table.

c02_decile_reconciliation.py already computes both share series; it only *prints* the
population side and saves the physician side. This regenerates both into one CSV so
Table S12 can be built from a file rather than from a console log. Same construction:
circles ranked by 2010 density, split into ten groups of 33, then followed as a FIXED
cohort to 2024 (no re-ranking in 2024) -- the same definition as Figure 3(a).

Writes outputs/audit/h06_decile_profile.csv
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)

rep = pd.read_csv("data/crosswalk/physician_panel_smsa_repaired.csv",
                  dtype={"code4": str}, encoding="utf-8-sig").set_index("code4")
CODES = list(rep.index)
X = {y: rep[str(y)].astype(float).to_numpy() for y in (2010, 2024)}


def pop(y):
    d = pd.read_csv(f"census_based/smsa_population_census_{y}.csv")
    d["code4"] = d["smsa_code"].apply(lambda c: str(c).zfill(4))
    return d.set_index("code4")["population"].reindex(CODES).astype(float).to_numpy()


P = {y: pop(y) for y in (2010, 2024)}
N = len(CODES)

d2010 = X[2010] / P[2010] * 1e5
order = np.argsort(d2010, kind="stable")
grp = np.zeros(N, int)
edges = np.linspace(0, N, 11).round().astype(int)
for g in range(10):
    grp[order[edges[g]:edges[g + 1]]] = g + 1

rows = []
for g in range(1, 11):
    m = grp == g
    ps0 = 100 * X[2010][m].sum() / X[2010].sum()
    ps1 = 100 * X[2024][m].sum() / X[2024].sum()
    qs0 = 100 * P[2010][m].sum() / P[2010].sum()
    qs1 = 100 * P[2024][m].sum() / P[2024].sum()
    d0 = X[2010][m].sum() / P[2010][m].sum() * 1e5
    d1 = X[2024][m].sum() / P[2024][m].sum() * 1e5
    rows.append({"decile": g, "n_circles": int(m.sum()),
                 "density_2010": d0, "density_2024": d1,
                 "density_growth_pct": 100 * (d1 / d0 - 1),
                 "phys_2010": X[2010][m].sum(), "phys_2024": X[2024][m].sum(),
                 "phys_growth_pct": 100 * (X[2024][m].sum() / X[2010][m].sum() - 1),
                 "pop_2010": P[2010][m].sum(), "pop_2024": P[2024][m].sum(),
                 "pop_growth_pct": 100 * (P[2024][m].sum() / P[2010][m].sum() - 1),
                 "phys_share_2010": ps0, "phys_share_2024": ps1, "phys_share_change_pp": ps1 - ps0,
                 "pop_share_2010": qs0, "pop_share_2024": qs1, "pop_share_change_pp": qs1 - qs0})
# national row -- decile 0 -- so that "against a national +17.9%" has a named field too
rows.append({"decile": 0, "n_circles": N,
             "density_2010": X[2010].sum() / P[2010].sum() * 1e5,
             "density_2024": X[2024].sum() / P[2024].sum() * 1e5,
             "density_growth_pct": 100 * ((X[2024].sum() / P[2024].sum())
                                          / (X[2010].sum() / P[2010].sum()) - 1),
             "phys_2010": X[2010].sum(), "phys_2024": X[2024].sum(),
             "phys_growth_pct": 100 * (X[2024].sum() / X[2010].sum() - 1),
             "pop_2010": P[2010].sum(), "pop_2024": P[2024].sum(),
             "pop_growth_pct": 100 * (P[2024].sum() / P[2010].sum() - 1),
             "phys_share_2010": 100.0, "phys_share_2024": 100.0, "phys_share_change_pp": 0.0,
             "pop_share_2010": 100.0, "pop_share_2024": 100.0, "pop_share_change_pp": 0.0})

D = pd.DataFrame(rows)
D.to_csv("outputs/audit/h06_decile_profile.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 200)
print(D.round(2).to_string(index=False))
print()
print("checks against the two claims the main text makes from this table:")
mx = D.phys_share_change_pp.abs().max()
print(f"  largest |physician-share change|  : {mx:.4f} pp   (text: 'no more than 0.63')  "
      f"-> {'OK' if abs(mx - 0.63) < 0.005 else 'MISMATCH'}")
top = D.loc[D.density_2010.idxmax(), "pop_share_change_pp"]
bot = D.loc[D.density_2010.idxmin(), "pop_share_change_pp"]
print(f"  densest decile population share  : {top:+.4f} pp   (text: '+1.56')  "
      f"-> {'OK' if abs(top - 1.56) < 0.005 else 'MISMATCH'}")
print(f"  sparsest decile population share : {bot:+.4f} pp   (text: '-0.39')  "
      f"-> {'OK' if abs(bot + 0.39) < 0.005 else 'MISMATCH'}")
print(f"  shares sum 2010 / 2024           : phys {D.phys_share_2010.sum():.2f} / "
      f"{D.phys_share_2024.sum():.2f}; pop {D.pop_share_2010.sum():.2f} / {D.pop_share_2024.sum():.2f}")
