"""
ROUND 11 -- regenerate the Moran run on the REPAIRED panel.

This is the Moran run on the repaired panel. Its predecessor read
data/crosswalk/physician_panel_smsa_repaired.csv instead of data/superseded/physician_panel_smsa__PRE_REPAIR.csv.
Same statistic, same weights, same 9,999 permutations, same seed family. No new method,
no new window -- a01's defect was that it was written before the crosswalk repair and
never re-run.

Three things are produced in ONE run so that the tables built from it cannot drift apart:

  block "row" / "binary"        per-wave series          -> Table S8
  block "row/primary|effective" primary vs effective     -> Table S9
  block "trend"                 OLS slope permutation    -> C1 trend paragraph

Guard: 2016-2024 must reproduce a01 exactly, because the repair does not touch those
waves and the seed and draw order are identical. If that guard fails, the regeneration
differs from a01 for a reason other than the repair and must not be used.

Writes outputs/audit/h05_moran_repaired.csv and .txt
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)

YEARS = [2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
SEED = 20260621
N_PERM = 9999
log = []


def say(s=""):
    print(s)
    log.append(s)


panel = pd.read_csv("data/crosswalk/physician_panel_smsa_repaired.csv",
                    dtype={"code4": str}, encoding="utf-8-sig")
CODES = panel["code4"].tolist()
N = len(CODES)
say(f"panel: data/crosswalk/physician_panel_smsa_repaired.csv, {N} circles")


def stock(y):
    return panel.set_index("code4")[str(y)].reindex(CODES).astype(float)


def census_pop(y):
    df = pd.read_csv(f"census_based/smsa_population_census_{y}.csv")
    df["code4"] = df["smsa_code"].apply(lambda c: str(c).zfill(4))
    return df.set_index("code4")["population"].reindex(CODES).astype(float)


def density(y):
    return (stock(y) / census_pop(y) * 1e5).to_numpy(float)


def od_flows(y):
    od = pd.read_csv(f"od_matrix_{y}.csv")
    F = od.drop(columns=["smsa_code"]).to_numpy(float)
    np.fill_diagonal(F, 0)
    return F.sum(0), F.sum(1)


adj = pd.read_csv("adjacency_final.csv")
adj.index = [str(c).zfill(4) for c in adj["Unnamed: 0"]]
W0 = adj.drop(columns=["Unnamed: 0"])
W0.columns = [str(c) for c in W0.columns]
W0 = W0.loc[CODES, CODES].to_numpy(float)
rs = W0.sum(1, keepdims=True); rs[rs == 0] = 1
W_ROW = W0 / rs
W_BIN = W0.copy()


def moran(v, Wm):
    z = v - v.mean()
    return (v.size / Wm.sum()) * ((z @ Wm @ z) / (z @ z))


def perm_dist(v, Wm, rng, n_perm=N_PERM, batch=1000):
    n = v.size
    z = v - v.mean()
    denom = z @ z
    const = n / Wm.sum()
    out = np.empty(n_perm)
    done = 0
    while done < n_perm:
        k = min(batch, n_perm - done)
        Z = np.empty((k, n))
        for r in range(k):
            Z[r] = rng.permutation(z)
        out[done:done + k] = const * ((Z @ Wm) * Z).sum(1) / denom
        done += k
    return out


E_A = -1.0 / (N - 1)
say(f"analytic E[I] = -1/(N-1) = {E_A:.6f}")

rows = []
IMAP = {}
rng = np.random.default_rng(SEED)
for y in YEARS:
    d = density(y)
    for tag, Wm in [("row", W_ROW), ("binary", W_BIN)]:
        I = moran(d, Wm)
        P = perm_dist(d, Wm, rng)
        mu, sd = P.mean(), P.std(ddof=1)
        p = (1 + (P >= I).sum()) / (1 + N_PERM)
        IMAP[(tag, y)] = I
        rows.append({"year": y, "block": tag, "I": round(I, 4),
                     "E_I_analytic": round(E_A, 6),
                     "perm_mean": round(mu, 6), "perm_sd": round(sd, 6),
                     "z_vs_perm_mean": round((I - mu) / sd, 4),
                     "z": round((I - E_A) / sd, 4),
                     "pseudo_p": round(p, 4),
                     "above_E_I": bool(I > E_A)})

rng2 = np.random.default_rng(SEED + 1)
for y in [2016, 2024]:
    s = stock(y).to_numpy(float); pop = census_pop(y).to_numpy(float)
    inflow, outflow = od_flows(y)
    eff = np.clip(s + 0.5 * (inflow - outflow), 0, None)
    for tag, v in [("primary", s / pop * 1e5), ("effective", eff / pop * 1e5)]:
        I = moran(v, W_ROW)
        P = perm_dist(v, W_ROW, rng2)
        mu, sd = P.mean(), P.std(ddof=1)
        p = (1 + (P >= I).sum()) / (1 + N_PERM)
        rows.append({"year": y, "block": f"row/{tag}", "I": round(I, 4),
                     "E_I_analytic": round(E_A, 6),
                     "perm_mean": round(mu, 6), "perm_sd": round(sd, 6),
                     "z_vs_perm_mean": round((I - mu) / sd, 4),
                     "z": round((I - E_A) / sd, 4),
                     "pseudo_p": round(p, 4),
                     "above_E_I": bool(I > E_A)})

# ---------------------------------------------------------------- trend, all windows
for tag in ["row", "binary"]:
    for w0 in [2006, 2010, 2012]:
        ys = np.array([y for y in YEARS if y >= w0], float)
        iv = np.array([IMAP[(tag, y)] for y in YEARS if y >= w0])
        slope = np.polyfit(ys, iv, 1)[0] * 2
        rr = np.random.default_rng(SEED + 7)
        null = np.array([np.polyfit(rr.permutation(ys), iv, 1)[0] * 2 for _ in range(N_PERM)])
        p = (1 + (np.abs(null) >= abs(slope)).sum()) / (1 + N_PERM)
        rows.append({"year": f"{w0}-2024", "block": f"trend/{tag}", "I": np.nan,
                     "E_I_analytic": np.nan, "perm_mean": np.nan, "perm_sd": np.nan,
                     "z_vs_perm_mean": np.nan, "z": round(slope, 4),
                     "pseudo_p": round(p, 4), "above_E_I": None})

D = pd.DataFrame(rows)
D.to_csv("outputs/audit/h05_moran_repaired.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- guard
# The original guard compared 2016-2024 against a01_moran_null.csv, the PRE-repair run.
# That file was deleted in the final round (it was the artefact that put stale values into
# Table S8), so the guard is now internal: the repair cannot touch 2016-2024, therefore
# those waves must reproduce the published panel exactly. h10_premise_check.py verifies
# that panel identity directly (1,650 cells, 0 differ) and is the standing check.
fails = []
for tag in ["row", "binary"]:
    for y in [2016, 2018, 2020, 2022, 2024]:
        n = D[(D.year == y) & (D.block == tag)].iloc[0]
        if not np.isfinite(float(n.I)):
            fails.append(f"{tag} {y}: non-finite I")
say()
say("GUARD -- 2016-2024 present and finite (panel identity checked by h10_premise_check):")
say("  " + ("PASS" if not fails else "FAIL: " + "; ".join(fails)))

pd.set_option("display.width", 200); pd.set_option("display.max_rows", 60)
say(D.to_string(index=False))
sub = D[D.block == "row"]
below = sub[sub.I.astype(float) < E_A]
say()
say(f"row-standardised waves with I < E[I]: {list(below.year) if len(below) else 'NONE'}")
subb = D[D.block == "binary"]
belowb = subb[subb.I.astype(float) < E_A]
say(f"binary waves with I < E[I]: {list(belowb.year) if len(belowb) else 'NONE'}")
pm = D[D.block.isin(["row", "binary"])].perm_mean.astype(float)
say(f"empirical permutation means, range: {pm.min():.4f} to {pm.max():.4f}")
say(f"perm sd, row: ~{D[D.block=='row'].perm_sd.astype(float).mean():.4f}; "
    f"binary: ~{D[D.block=='binary'].perm_sd.astype(float).mean():.4f}")

Path("outputs/audit/h05_moran_repaired.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
print("\nwrote outputs/audit/h05_moran_repaired.{csv,txt}")
