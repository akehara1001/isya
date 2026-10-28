"""
AUDIT items 2, 3, 4, 15 -- four checks that need no new method.

ITEM 2  Gini scale invariance of the need-weighted denominator.
        Recompute the need-weighted density Gini with and WITHOUT the normalising
        constant r-bar.  Gini is scale invariant, so the two must agree to
        floating-point.  This falsifies the current supplement justification
        ("leaving the national total unchanged so that the resulting Gini is
        directly comparable in scale").

ITEM 3  Is the max(0, .) truncation in Eq. (3) ever binding?
        Count, over every flow wave and every theta on {0, 0.1, ..., 1.0},
        the circles with x_i + theta*n_i < 0.  Also verify sum_i n_i == 0 and,
        where truncation never binds, that sum_i S_i == sum_i x_i exactly.

ITEM 4  How many flow waves does the gravity model actually pool?
        Enumerate the OD matrices that exist, count positive off-diagonal flows
        per wave and pooled, and compute the implied zero share against
        330 * 329 * (number of waves).

ITEM 15 Do the extrapolated 2022/2024 age shares stay on the simplex?
        Check non-negativity and sum-to-one for every circle x every analysis year,
        and report what the implementation does (clip + renormalise).

Writes outputs/audit/a02_*.csv / .txt
"""
import sys, io, os, glob, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from phase2_common import read_csv_rows, parse_code_name

YEARS = [2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
GROUPS = ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44",
          "45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84","85-89","90+"]

panel = pd.read_csv("data/crosswalk/physician_panel_smsa_repaired.csv",
                    dtype={"code4": str}, encoding="utf-8-sig")
if "smsa_code" not in panel.columns:
    panel["smsa_code"] = panel["code4"]
panel["code4"] = panel["smsa_code"].apply(lambda c: str(c).zfill(4))
CODES = panel["code4"].tolist()
report = []


def say(s=""):
    print(s)
    report.append(s)


def gini(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = x.size
    xs = np.sort(x); idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * xs)) / (n * xs.sum()) - (n + 1) / n


def stock(y):
    return panel.set_index("code4")[str(y)].reindex(CODES).astype(float).to_numpy()


def total_pop(y):
    df = pd.read_csv(f"census_based/smsa_population_census_{y}.csv")
    df["code4"] = df["smsa_code"].apply(lambda c: str(c).zfill(4))
    return df.set_index("code4")["population"].reindex(CODES).astype(float).to_numpy()


def build_map():
    m = {}; cur = None
    for row in read_csv_rows(Path("smsa_2024.csv")):
        if not row:
            continue
        code, name, kind = parse_code_name(row[0])
        if kind == "smsa" and code:
            cur = str(code).zfill(4)
        elif kind == "municipality" and code and cur:
            m[str(code).zfill(5)] = cur
    return m


MAP = build_map()


def sma_age_counts(cy):
    # Round 38 (2026-10-06): the area age table, built once by cc_build_census_age.py through
    # the code-change tables. Summing census_age/age_muni_ over the 2024 map's codes, as this
    # did until then, lost every municipality whose code had changed since the census --
    # 熊本市 in 2010, the seven Hamamatsu wards in 2010-2020 -- and gave those areas the age
    # mix of what was left.
    t = pd.read_csv(f"census_age/age_circle_{cy}.csv", dtype={"code4": str}, encoding="utf-8")
    return t.set_index("code4").reindex(CODES)[GROUPS].to_numpy(float)


CNT = {cy: sma_age_counts(cy) for cy in [2010, 2015, 2020]}
SH = {cy: CNT[cy] / CNT[cy].sum(1, keepdims=True) for cy in CNT}
rate = pd.read_csv("census_age/juryoritsu_2020.csv").set_index("age_group")["juryoritsu_per100k"]
W = rate.reindex(GROUPS).to_numpy(float)


def shares_raw(ay):
    """Interpolated / extrapolated shares BEFORE the clip-and-renormalise step
    that cc_stage2_needweight.py applies."""
    if ay <= 2015:
        return SH[2010] + (SH[2015] - SH[2010]) * ((ay - 2010) / 5.0)
    return SH[2015] + (SH[2020] - SH[2015]) * ((ay - 2015) / 5.0)


def shares_for_year(ay):
    """Exactly as implemented in cc_stage2_needweight.py / cc_decomp_ci.py."""
    v = np.clip(shares_raw(ay), 0, None)
    rs = v.sum(1, keepdims=True); rs[rs == 0] = np.nan
    return v / rs


from needbase import fraction_from_sums, assert_unity


def _area_sums(cy):
    """Age-stated persons and published population per area, from the area age table.
    The census-edition guard is taken over the same areas as the shares (round 38)."""
    t = pd.read_csv(f"census_age/age_circle_{cy}.csv", dtype={"code4": str},
                    encoding="utf-8").set_index("code4").reindex(CODES)
    return t[GROUPS].to_numpy(float).sum(1), t["population"].to_numpy(float)



# The age-stated fraction, computed as a GUARD on the census edition and NOT applied to the
# base -- see needbase.py. This file's item 2 asks what the normalising constant r-bar does
# and item 3 what truncating the top age band does; both are questions ABOUT the
# need-weighted denominator, so both must ask them of the same denominator the rest of the
# repository uses, which is the published total reweighted by the age shares.
GFRAC = {cy: fraction_from_sums(*_area_sums(cy)) for cy in [2010, 2015, 2020]}
assert_unity(GFRAC, "circle")


# =====================================================================
say("=" * 78)
say("ITEM 2 -- need-weighted Gini with and without the normalising constant r-bar")
say("=" * 78)
say(f"{'year':>5} {'Gini(need, /rbar)':>18} {'Gini(need, no rbar)':>20} {'abs diff':>12} {'rbar':>10}")
rows2 = []
for ay in YEARS:
    raw = total_pop(ay); s = stock(ay)          # PRO RATA -- see needbase.py
    nm = (shares_for_year(ay) * W).sum(1)                # per-circle mean utilisation rate
    rbar = np.nansum(raw * nm) / np.nansum(raw)
    P_norm = raw * nm / rbar                             # as published
    P_unnorm = raw * nm                                  # r-bar omitted
    g_norm = gini(s / P_norm * 1e5)
    g_unnorm = gini(s / P_unnorm * 1e5)
    rows2.append({"year": ay, "gini_need_normalised": g_norm,
                  "gini_need_unnormalised": g_unnorm,
                  "abs_diff": abs(g_norm - g_unnorm), "rbar": rbar})
    say(f"{ay:>5} {g_norm:>18.12f} {g_unnorm:>20.12f} {abs(g_norm-g_unnorm):>12.2e} {rbar:>10.1f}")
d2 = pd.DataFrame(rows2)
d2.to_csv("outputs/audit/a02_item2_scale_invariance.csv", index=False)
say(f"max |difference| over all waves = {d2.abs_diff.max():.3e}  "
    f"(machine epsilon scale => Gini is invariant to r-bar, as it must be)")
say("")

# =====================================================================
say("=" * 78)
say("ITEM 3 -- is max(0, x_i + theta*n_i) ever binding?")
say("=" * 78)
OD_FILES = sorted(glob.glob("od_matrix_*.csv"))
OD_YEARS = [int(re.search(r"(\d{4})", f).group(1)) for f in OD_FILES]
say(f"OD matrices present on disk: {OD_YEARS}")
THETAS = np.round(np.arange(0, 1.01, 0.1), 2)
rows3 = []
for y in OD_YEARS:
    od = pd.read_csv(f"od_matrix_{y}.csv")
    F = od.drop(columns=["smsa_code"]).to_numpy(float)
    np.fill_diagonal(F, 0)
    n_i = F.sum(0) - F.sum(1)          # inflow - outflow
    x = stock(y)
    say(f"  {y}: sum_i n_i = {n_i.sum():+.6g}  (must be 0 by construction); "
        f"min_i n_i = {n_i.min():.0f}, total off-diagonal flow = {F.sum():.0f}")
    for th in THETAS:
        v = x + th * n_i
        nneg = int((v < 0).sum())
        rows3.append({"year": y, "theta": th, "n_negative": nneg,
                      "min_x_plus_theta_n": round(float(v.min()), 2),
                      "argmin_circle": CODES[int(np.argmin(v))],
                      "sum_S": float(np.maximum(v, 0).sum()),
                      "sum_x": float(x.sum()),
                      "conservation_gap": float(np.maximum(v, 0).sum() - x.sum())})
d3 = pd.DataFrame(rows3)
d3.to_csv("outputs/audit/a02_item3_truncation.csv", index=False)
say("")
say(f"{'year':>5} {'theta':>6} {'#circles with x+theta*n<0':>26} {'min(x+theta*n)':>16} {'argmin':>8}")
for _, r in d3.iterrows():
    say(f"{int(r.year):>5} {r.theta:>6.1f} {int(r.n_negative):>26d} {r.min_x_plus_theta_n:>16.1f} {r.argmin_circle:>8}")
tot_bind = int(d3.n_negative.sum())
say("")
say(f"TOTAL binding cases over {len(OD_YEARS)} waves x {len(THETAS)} theta values = {tot_bind}")
say(f"global min over the whole grid of min_i (x_i + theta n_i) = {d3.min_x_plus_theta_n.min():.1f} "
    f"(year {int(d3.loc[d3.min_x_plus_theta_n.idxmin(),'year'])}, "
    f"theta {d3.loc[d3.min_x_plus_theta_n.idxmin(),'theta']:.1f}, "
    f"circle {d3.loc[d3.min_x_plus_theta_n.idxmin(),'argmin_circle']})")
say(f"max |sum_i S_i - sum_i x_i| over the grid = {d3.conservation_gap.abs().max():.6g}")
say("")

# =====================================================================
say("=" * 78)
say("ITEM 4 -- how many flow waves does the pooled gravity model use?")
say("=" * 78)
rows4 = []
for y in OD_YEARS:
    od = pd.read_csv(f"od_matrix_{y}.csv")
    F = od.drop(columns=["smsa_code"]).to_numpy(float)
    np.fill_diagonal(F, 0)
    npos = int((F > 0).sum())
    rows4.append({"year": y, "n_positive_offdiag": npos,
                  "possible_pairs": 330 * 329,
                  "zero_share_pct": round(100 * (1 - npos / (330 * 329)), 2),
                  "total_flow": float(F.sum())})
d4 = pd.DataFrame(rows4)
d4.to_csv("outputs/audit/a02_item4_flow_waves.csv", index=False)
say(d4.to_string(index=False))
npos_tot = int(d4.n_positive_offdiag.sum())
poss_tot = 330 * 329 * len(OD_YEARS)
say("")
say(f"pooled positive directed flows over {len(OD_YEARS)} waves = {npos_tot:,}")
say(f"possible directed pairs = 330 x 329 x {len(OD_YEARS)} = {poss_tot:,}")
say(f"share of possible pairs that are ZERO = {100*(1-npos_tot/poss_tot):.2f}%")
say(f"manuscript n for the gravity model = 26,011  ->  "
    f"{'MATCHES the 5-wave pooling' if npos_tot == 26011 else 'DOES NOT MATCH'}")
say("")

# =====================================================================
say("=" * 78)
say("ITEM 15 -- do the interpolated/extrapolated age shares stay on the simplex?")
say("=" * 78)
rows15 = []
for ay in YEARS:
    if ay < 2010:
        continue                     # need weight is defined from 2010 (census age anchors)
    raw = shares_raw(ay)
    used = shares_for_year(ay)
    rows15.append({"year": ay,
                   "extrapolated": ay > 2020,
                   "raw_min_share": float(np.nanmin(raw)),
                   "raw_n_negative": int((raw < 0).sum()),
                   "raw_max_abs_sum_dev": float(np.nanmax(np.abs(raw.sum(1) - 1))),
                   "used_min_share": float(np.nanmin(used)),
                   "used_n_negative": int((used < 0).sum()),
                   "used_max_abs_sum_dev": float(np.nanmax(np.abs(used.sum(1) - 1)))})
d15 = pd.DataFrame(rows15)
d15.to_csv("outputs/audit/a02_item15_simplex.csv", index=False)
say(d15.to_string(index=False))
say("")
say(f"raw (pre-clip) negative share cells over all circles x years: {int(d15.raw_n_negative.sum())}")
say(f"raw (pre-clip) max |sum of shares - 1|: {d15.raw_max_abs_sum_dev.max():.3e}")
say(f"as-used max |sum of shares - 1|: {d15.used_max_abs_sum_dev.max():.3e}; "
    f"negative cells as used: {int(d15.used_n_negative.sum())}")
say("NOTE: the implementation applies np.clip(v, 0, None) then divides by the row sum, "
    "so the shares used are on the simplex by construction; the question answered here is "
    "whether that clip/renormalise step ever had anything to do.")

Path("outputs/audit/a02_items_2_3_4_15.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
print("\nwrote outputs/audit/a02_items_2_3_4_15.txt")
