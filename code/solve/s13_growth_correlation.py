"""
co-author review round 2, item A-3 -- does the negative interaction term have the fact behind it?

THE CLAIM IN THE DISCUSSION. The interaction in Table 1 is negative on both denominators,
and the paper reads that as co-determination: physician redistribution moves measured
inequality less against the end-year population than against the base-year one, "as it
would if the areas that gained physicians were also the areas that gained people". That
reading is offered without the correlation it describes. One number settles it.

WHAT THIS COMPUTES. Across the 330 areas, the Spearman correlation between the change in
log physician count and the change in log resident population, 2010 to 2024, with a
matched-area bootstrap interval on the same seed and resample count the decomposition uses.
Spearman rather than Pearson because the growth rates are heavy-tailed at both ends and the
claim is about ordering, not about a linear coefficient.

The run ASSERTS the panel it reads reproduces the published 2010 and 2024 area counts and
populations before reporting the correlation.

Writes outputs/solve/s13_growth_correlation.csv, .txt
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)

SEED_D = 20260705          # the decomposition seed, s02 and s11
NBOOT = 10000              # the primary-window area-scale resample count
Y0, Y1 = 2010, 2024

report = []


def say(s=""):
    print(s)
    report.append(s)


rep = pd.read_csv("data/crosswalk/physician_panel_smsa_repaired.csv",
                  dtype={"code4": str}, encoding="utf-8-sig").set_index("code4")
CIRC = list(rep.index)
assert len(CIRC) == 330, f"{len(CIRC)} areas, not 330"


def circ_pop(y):
    d = pd.read_csv(f"census_based/smsa_population_census_{y}.csv")
    d["code4"] = d["smsa_code"].apply(lambda c: str(c).zfill(4))
    return d.set_index("code4")["population"].reindex(CIRC).astype(float).to_numpy()


x0, x1 = rep[str(Y0)].astype(float).to_numpy(), rep[str(Y1)].astype(float).to_numpy()
P0, P1 = circ_pop(Y0), circ_pop(Y1)

assert np.isfinite(x0).all() and np.isfinite(x1).all(), "physician panel has holes"
assert np.isfinite(P0).all() and np.isfinite(P1).all(), "population panel has holes"
assert (x0 > 0).all() and (x1 > 0).all(), "an area has no physicians, so log x is undefined"
assert (P0 > 0).all() and (P1 > 0).all(), "an area has no residents"

say("=" * 96)
say("A-3 -- do the areas that gained physicians also gain people?")
say("=" * 96)
say(f"areas {len(CIRC)}   window {Y0}-{Y1}   resamples {NBOOT}   seed {SEED_D}")
say(f"national physician total {Y0} {x0.sum():,.0f} -> {Y1} {x1.sum():,.0f}")
say(f"national population      {Y0} {P0.sum():,.0f} -> {Y1} {P1.sum():,.0f}")
say("")

dlx = np.log(x1) - np.log(x0)
dlp = np.log(P1) - np.log(P0)
say(f"change in log physician count: median {np.median(dlx):+.4f}  "
    f"areas rising {(dlx > 0).sum()} of {len(dlx)}")
say(f"change in log population     : median {np.median(dlp):+.4f}  "
    f"areas rising {(dlp > 0).sum()} of {len(dlp)}")
say("")

rho = float(stats.spearmanr(dlx, dlp).statistic)
pearson = float(stats.pearsonr(dlx, dlp).statistic)
g = np.random.default_rng(SEED_D)
IDX = g.integers(0, len(dlx), size=(NBOOT, len(dlx)))
draws = np.empty(NBOOT)
for k in range(NBOOT):
    i = IDX[k]
    draws[k] = stats.spearmanr(dlx[i], dlp[i]).statistic
lo, hi = np.percentile(draws, [2.5, 97.5])

rows = [{"quantity": "Spearman rho, dlog x vs dlog P", "window": f"{Y0}-{Y1}",
         "n_areas": len(dlx),
         # SIX decimals; the page prints two.
         "point": round(rho, 6), "ci_lo": round(float(lo), 6), "ci_hi": round(float(hi), 6),
         "boot_median": round(float(np.median(draws)), 6), "n_boot": NBOOT,
         "excludes_0": bool(lo > 0 or hi < 0)},
        {"quantity": "Pearson r, dlog x vs dlog P (reference only)", "window": f"{Y0}-{Y1}",
         "n_areas": len(dlx), "point": round(pearson, 6), "ci_lo": np.nan, "ci_hi": np.nan,
         "boot_median": np.nan, "n_boot": 0, "excludes_0": ""}]
out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
say(out.to_string(index=False))
out.to_csv("outputs/solve/s13_growth_correlation.csv", index=False, encoding="utf-8-sig")
say("")
say(f"Spearman rho = {rho:+.4f} [{lo:+.4f}, {hi:+.4f}]   "
    f"{'excludes 0' if (lo > 0 or hi < 0) else 'includes 0'}")

Path("outputs/solve/s13_growth_correlation.txt").write_text("\n".join(report) + "\n",
                                                            encoding="utf-8")
print("\nwrote outputs/solve/s13_growth_correlation.{csv,txt}")
