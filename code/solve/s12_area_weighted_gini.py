"""
co-author review round 2, item A-1 -- the population-weighted Gini at the paper's own scale.

THE ASYMMETRY. s11_municipality_noise.py runs a population-weighted Gini as treatment (b)
at MUNICIPALITY scale, and Supplementary B5 reports it: weighting pulls the municipality
estimate of A from +0.0439 down to +0.0071. The 330-area scale -- the scale every headline
number is computed at -- has no weighted counterpart, so a reader who accepts that
weighting matters at one scale cannot see what it does at the other. Nothing but the order
the work happened in explains that, which is what makes it worth closing.

WHAT THIS COMPUTES. The same treatment at 330 areas, primary window 2010-2024, both
denominators: the density-Gini change, the physician component A and the denominator
component B under the Shapley allocation, each unweighted and population-weighted, with
matched-area bootstrap intervals.

NOTHING HERE IS NEW MACHINERY. The weighted Gini and its row form are s11's, character for
character; the area panel, the need denominator and the two seeds are s02's; the resample
count is c01's 10,000 (s11's 2,000 for treatment (b) until round 41-2), so the unweighted
column reproduces the intervals of Tables S5 and S7, which is asserted. The run ASSERTS that the UNWEIGHTED area-scale
figures reproduce the published +0.0157 (raw) and +0.0243 (need; +0.0152 and +0.0238 until round 40) before reporting any
weighted number, so a divergence upstream fails here rather than arriving as a new result.

Writes outputs/solve/s12_area_weighted_gini.csv, .txt
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

SEED_G, SEED_D = 20260805, 20260705      # s02 and s11 use the same pair
# Round 41-2 ruling 0-1 (2026-10-06): 10,000, the count of c01's paired and joint bootstraps
# (Tables S5 and S7). Was 2,000, s11's count for its weighted treatment, which printed the same
# quantity -- the unweighted 2010-2024 density-Gini change -- with a second interval beside
# Table S5's (upper ends -0.0001 against +0.0001 on the exact panel). With the same seeds,
# the same integers() draw and the same percentile, the unweighted column now reproduces
# c01's intervals; the run asserts it below.
NBOOT = 10000
GROUPS = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44",
          "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84",
          "85-89", "90+"]
Y0, Y1 = 2010, 2024

# The published unweighted area-scale physician component, Shapley, from
# outputs/audit/c01_round3_numbers.csv (block "1c/2 decomposition", repaired column).
# Not a target to be reproduced by adjustment: the asserts below fail the run if the
# data path stops giving them.
PUBLISHED_A = {"raw": 0.0155320949, "need": 0.0241719582}
# Round 41 §1 (2026-10-06): moved from 0.0156583038 / 0.0243016826 when the physician panel
# stopped apportioning the physicians of municipalities whose code changed (2006-2010 move);
# c01's "raw/need 2010-2024: A Shapley" on the exact panel. The guard stays. The output the
# guard would have stopped is in an internal work record.
# Round 40 §3 (2026-10-06): moved from 0.0152054163 / 0.0237796 when the census denominators
# (round 38) and the confirmed 2025 census were applied to the manuscript; the values are c01's
# "raw/need 2010-2024: A Shapley" on the current outputs. The guard stays.

report = []


def say(s=""):
    print(s)
    report.append(s)


# ---------------------------------------------------------------- estimators (s02, s11)
def gini(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    xs = np.sort(v); n = xs.size; ii = np.arange(1, n + 1)
    return (2 * (xs * ii).sum()) / (n * xs.sum()) - (n + 1) / n


def gini_rows(M):
    S = np.sort(M, axis=1); n = M.shape[1]; ii = np.arange(1, n + 1)
    return (2 * (S * ii).sum(1)) / (n * S.sum(1)) - (n + 1) / n


def wgini(x, w):
    o = np.argsort(x); x = np.asarray(x)[o]; w = np.asarray(w)[o]
    p = w / w.sum(); S = np.cumsum(p * x); S = np.concatenate([[0.0], S])
    return float(1 - (p * (S[:-1] + S[1:])).sum() / S[-1])


def wgini_rows(X, W_):
    o = np.argsort(X, axis=1)
    Xs = np.take_along_axis(X, o, 1); Ws = np.take_along_axis(W_, o, 1)
    P_ = Ws / Ws.sum(1, keepdims=True); S = np.cumsum(P_ * Xs, axis=1)
    S0 = np.concatenate([np.zeros((len(X), 1)), S], axis=1)
    return 1 - (P_ * (S0[:, :-1] + S0[:, 1:])).sum(1) / S0[:, -1]


# ---------------------------------------------------------------- the area panel (s02)
MU = pd.read_csv("data/crosswalk/vintage_sma_municipalities.csv",
                 dtype={"pref": str, "sma": str, "muni": str}, encoding="utf-8-sig")
MU["muni"] = MU["muni"].str.zfill(5); MU["sma"] = MU["sma"].str.zfill(4)

rep = pd.read_csv("data/crosswalk/physician_panel_smsa_repaired.csv",
                  dtype={"code4": str}, encoding="utf-8-sig").set_index("code4")
CIRC = list(rep.index)
assert len(CIRC) == 330, f"{len(CIRC)} areas, not 330"


def circ_pop(y):
    d = pd.read_csv(f"census_based/smsa_population_census_{y}.csv")
    d["code4"] = d["smsa_code"].apply(lambda c: str(c).zfill(4))
    return d.set_index("code4")["population"].reindex(CIRC).astype(float).to_numpy()


CIRC_AGE, CIRC_SUMS = {}, {}
_m24 = MU[MU.year == 2024].set_index("muni")["sma"]
_ci = {c: i for i, c in enumerate(CIRC)}
for cy in (2010, 2015, 2020):
    # Round 38 (2026-10-06): the area age table, built once by cc_build_census_age.py through
    # the code-change tables. Summing census_age/age_muni_ over the 2024 map's codes, as this
    # did until then, lost every municipality whose code had changed since the census --
    # 熊本市 in 2010, the seven Hamamatsu wards in 2010-2020 -- and gave those areas the age
    # mix of what was left.
    _t = pd.read_csv(f"census_age/age_circle_{cy}.csv", dtype={"code4": str}, encoding="utf-8")
    _t = _t.set_index("code4").reindex(CIRC)
    mat = _t[GROUPS].to_numpy(float)
    CIRC_AGE[cy] = mat / np.clip(mat.sum(1, keepdims=True), 1e-9, None)
    CIRC_SUMS[cy] = (mat.sum(1), _t["population"].to_numpy(float))

rate = pd.read_csv("census_age/juryoritsu_2020.csv").set_index("age_group")["juryoritsu_per100k"]
W = rate.reindex(GROUPS).to_numpy(float)

# THE AGE-STATED FRACTION IS BUILT WHERE EVERY OTHER PRODUCER BUILDS IT. needbase.py is the
# single construction (c11 enforces it), and this file needs the fraction at area scale with
# numerator and denominator summed separately before the ratio, exactly as s02 does at three
# scales. The fraction is a GUARD on the census edition rather than a factor in the
# denominator: assert_unity fails the run if an age table turns out not to be the
# 不詳補完結果 edition, in which case the shares below would not sum to the published total
# and the need denominator computed from them would be wrong without saying so.
from needbase import fraction_from_sums, assert_unity  # noqa: E402

# Round 38: numerator and denominator are the area table's own sums, so the guard covers the
# same areas -- and the same municipalities -- as the shares.
_GFRAC = {}
for cy in (2010, 2015, 2020):
    _GFRAC[cy] = fraction_from_sums(*CIRC_SUMS[cy])
assert_unity(_GFRAC, "area")


def circ_shares(ay):
    v = (CIRC_AGE[2010] + (CIRC_AGE[2015] - CIRC_AGE[2010]) * ((ay - 2010) / 5.0)) \
        if ay <= 2015 else \
        (CIRC_AGE[2015] + (CIRC_AGE[2020] - CIRC_AGE[2015]) * ((ay - 2015) / 5.0))
    v = np.clip(v, 0, None); rs = v.sum(1, keepdims=True); rs[rs == 0] = np.nan
    return v / rs


X = {y: rep[str(y)].astype(float).to_numpy() for y in (Y0, Y1)}
POP = {y: circ_pop(y) for y in (Y0, Y1)}


def denom(ay, kind):
    raw = POP[ay]
    if kind == "raw":
        return raw
    sh = circ_shares(ay)
    nm = np.nan_to_num((sh * W).sum(1), nan=np.nanmean((sh * W).sum(1)))
    return raw * nm / (np.nansum(raw * nm) / np.nansum(raw))


# ---------------------------------------------------------------- the two treatments
n = len(CIRC)
IDX = np.random.default_rng(SEED_G).integers(0, n, size=(NBOOT, n))
ID = np.random.default_rng(SEED_D).integers(0, n, size=(NBOOT, n))
WEIGHTS = POP[Y0]                       # s11 treatment (b): base-year population

say("=" * 100)
say("A-1 -- population-weighted Gini at 330 areas, 2010->2024")
say("=" * 100)
say(f"units {n}   resamples {NBOOT}   seeds {SEED_G} (Gini) / {SEED_D} (decomposition)")
say("weights: base-year (2010) area population, as in s11 treatment (b)")
say("")

recs = []
for kind in ("raw", "need"):
    P0, P1 = denom(Y0, kind), denom(Y1, kind)
    x0, x1 = X[Y0], X[Y1]
    cells = {k: np.nan_to_num(v) * 1e5 for k, v in
             {"00": x0 / P0, "10": x1 / P0, "01": x0 / P1, "11": x1 / P1}.items()}
    for weighted in (False, True):
        if weighted:
            w = WEIGHTS
            g = {k: wgini(v, w) for k, v in cells.items()}
            G = {k: wgini_rows(v[ID], w[ID]) for k, v in cells.items()}
            dgb = wgini_rows(cells["11"][IDX], w[IDX]) - wgini_rows(cells["00"][IDX], w[IDX])
        else:
            g = {k: gini(v) for k, v in cells.items()}
            G = {k: gini_rows(v[ID]) for k, v in cells.items()}
            dgb = gini_rows(cells["11"][IDX]) - gini_rows(cells["00"][IDX])
        Ax, Ap = g["10"] - g["00"], g["11"] - g["01"]
        Bp, Bx = g["01"] - g["00"], g["11"] - g["10"]
        Axb, Apb = G["10"] - G["00"], G["11"] - G["01"]
        Bpb, Bxb = G["01"] - G["00"], G["11"] - G["10"]
        As, Asb = (Ax + Ap) / 2, (Axb + Apb) / 2
        Bs, Bsb = (Bx + Bp) / 2, (Bxb + Bpb) / 2
        dG = g["11"] - g["00"]
        for name, pt, draws in (("total change", dG, dgb),
                                ("A (Shapley)", As, Asb),
                                ("B (Shapley)", Bs, Bsb)):
            lo, hi = np.percentile(draws, [2.5, 97.5])
            recs.append({"denominator": kind,
                         "weighting": "population-weighted" if weighted else "unweighted",
                         "quantity": name,
                         # SIX decimals: the appendix prints four, so a value on the
                         # rounding boundary can still be adjudicated against this file.
                         "point": round(float(pt), 6),
                         "ci_lo": round(float(lo), 6), "ci_hi": round(float(hi), 6),
                         "excludes_0": bool(lo > 0 or hi < 0)})

out = pd.DataFrame(recs)

# ---------------------------------------------------------------- the guard
for kind, want in PUBLISHED_A.items():
    r = out[(out.denominator == kind) & (out.weighting == "unweighted")
            & (out.quantity == "A (Shapley)")]
    assert len(r) == 1, f"{kind}: unweighted A is not unique"
    got = float(r.point.iloc[0])
    assert abs(got - want) < 5e-5, (
        f"{kind}: unweighted area-scale A (Shapley) = {got:.6f}, published {want}. "
        "The data path no longer reproduces the manuscript, so the weighted figures "
        "beside it cannot be trusted either.")
say(f"guard: unweighted A (Shapley) reproduces the published "
    f"{PUBLISHED_A['raw']:+.4f} (raw) and {PUBLISHED_A['need']:+.4f} (need)")

# Round 41-2 ruling 0-1: one quantity, one interval. The unweighted total change here is the
# quantity Table S5 prints from c01's paired bootstrap; with the same seed, resample count and
# percentile it must be the same interval, so a divergence in either producer fails here.
import re as _re  # noqa: E402
_C01 = pd.read_csv("outputs/audit/c01_round3_numbers.csv", encoding="utf-8-sig")
for kind in ("raw", "need"):
    row = _C01[(_C01.block == "1b/2 paired bootstrap")
               & (_C01.quantity == f"{kind} Gini change 2010-2024")]
    assert len(row) == 1, f"c01 has no {kind} 2010-2024 change"
    want = [float(v) for v in _re.findall(r"[-+]?\d+\.\d+", str(row.repaired.iloc[0]))]
    r = out[(out.denominator == kind) & (out.weighting == "unweighted")
            & (out.quantity == "total change")].iloc[0]
    got = [r.point, r.ci_lo, r.ci_hi]
    assert all(abs(a - b) < 5e-7 for a, b in zip(got, want)), (
        f"{kind}: unweighted total change {got} differs from Table S5's {want}")
say("guard: the unweighted total change and its interval equal c01's paired bootstrap "
    "(Table S5) for both denominators")
say("")

pd.set_option("display.width", 200)
say(out.to_string(index=False))
out.to_csv("outputs/solve/s12_area_weighted_gini.csv", index=False, encoding="utf-8-sig")

wA = out[(out.weighting == "population-weighted") & (out.quantity == "A (Shapley)")]
say("")
say("weighted physician component, both denominators:")
for r in wA.itertuples():
    say(f"    {r.denominator:>5}: {r.point:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]"
        f"   {'excludes 0' if r.excludes_0 else 'includes 0'}")
say("")
say(f"weighted A negative anywhere: {bool((wA.point < 0).any())}")

Path("outputs/solve/s12_area_weighted_gini.txt").write_text("\n".join(report) + "\n",
                                                            encoding="utf-8")
print("\nwrote outputs/solve/s12_area_weighted_gini.{csv,txt}")
