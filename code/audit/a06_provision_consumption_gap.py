"""
AUDIT item 8 -- is the provision/consumption "separation" itself a tested difference?

The main text contrasts r(supply, provision) = 0.50 with r(supply, consumption) = 0.00
and calls it a separation, but the DIFFERENCE between those two correlations is never
tested.  Table S11 already carries a paired-circle bootstrap of Delta r along one axis
(effective minus primary density); this rotates the same machinery to the other axis
(facility-located minus residence-based utilisation), on the identical resamples.

No new method: same paired circle resample, same Pearson/Spearman estimators, same
NB = 5,000 draws and the same data path as cc_validation_tests.py test A.

Writes outputs/audit/a06_provision_consumption_gap.csv
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import cc_circle_common as cc

PS = "data/raw/external/patient_survey"
SEED = 20260805
NB = 5000

RES_META = {2023: "0004026138", 2020: "0004002360"}
RES_VAL = {2023: "ps2023_res_tot_0004026138", 2020: "ps2020_res_tot_0004002360"}
FAC_META = {2023: "0004026139", 2020: "0004002361"}
FAC_VAL = {2023: "ps2023_fac_tot_0004026139", 2020: "ps2020_fac_tot_0004002361"}


def frame(ps_year, eff_year, cens_year):
    ep = cc.eff_panel(eff_year)
    pop = cc.census_pop(cens_year)
    eff = ep["effective_supply_theta05"].astype(float) / pop * 1e5
    prim = ep["primary_stock"].astype(float) / pop * 1e5
    res_inp = cc.ps_series(f"{PS}/meta/{RES_META[ps_year]}.json",
                           f"{PS}/{RES_VAL[ps_year]}.json", cat02="1", agg="sum") * 1000.0
    fac_inp = cc.ps_series(f"{PS}/meta/{FAC_META[ps_year]}.json",
                           f"{PS}/{FAC_VAL[ps_year]}.json", cat02="1", agg="sum") * 1000.0
    return pd.DataFrame({"eff": eff, "prim": prim,
                         "util_res": res_inp / pop * 1e5,
                         "util_fac": fac_inp / pop * 1e5}
                        ).replace([np.inf, -np.inf], np.nan).dropna()


rows = []
for ps_year, eff_year, cens_year, tag in [(2023, 2024, 2024, "PS2023x2024"),
                                          (2020, 2020, 2020, "PS2020x2020")]:
    df = frame(ps_year, eff_year, cens_year)
    n = len(df)
    for xname in ["eff", "prim"]:
        x = df[xname].to_numpy()
        yf = df["util_fac"].to_numpy(); yr = df["util_res"].to_numpy()
        for stat, fn in [("pearson", lambda a, b: stats.pearsonr(a, b)[0]),
                         ("spearman", lambda a, b: stats.spearmanr(a, b)[0])]:
            r_fac, r_res = fn(x, yf), fn(x, yr)
            g = np.random.default_rng(SEED)
            diffs = np.empty(NB)
            for k in range(NB):
                idx = g.integers(0, n, n)
                diffs[k] = fn(x[idx], yf[idx]) - fn(x[idx], yr[idx])
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            rows.append({"wave": tag, "supply_measure": xname, "statistic": stat,
                         "n": n,
                         # SIX decimals. Table S11 prints these to three, so storing three
                         # left no digit between the file and the page: a value on the .0005
                         # boundary could not be adjudicated, which is exactly how Table A5's
                         # dyad width came to be rounded twice. Rounding belongs to printing.
                         "r_provision": round(float(r_fac), 6),
                         "r_consumption": round(float(r_res), 6),
                         "diff_point": round(float(r_fac - r_res), 6),
                         "diff_boot_median": round(float(np.median(diffs)), 6),
                         "ci_lo": round(float(lo), 6), "ci_hi": round(float(hi), 6),
                         "excludes_0": bool(lo > 0 or hi < 0)})

out = pd.DataFrame(rows)
out.to_csv("outputs/audit/a06_provision_consumption_gap.csv", index=False)
pd.set_option("display.width", 200)
print(out.to_string(index=False))

print("\nHeadline (effective density, Pearson, PS2023 x 2024 -- the pair quoted in the main text):")
h = out[(out["wave"] == "PS2023x2024") & (out["supply_measure"] == "eff") &
        (out["statistic"] == "pearson")].iloc[0]
print(f"  r(provision) - r(consumption) = {h.diff_point:+.3f} "
      f"[{h.ci_lo:+.3f}, {h.ci_hi:+.3f}]  excludes zero: {h.excludes_0}")
