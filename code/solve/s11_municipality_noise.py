"""
WORK 2 -- is the municipality-scale result real, or small-unit noise?

The scale gradient in component A (prefecture -0.014, circle +0.015, municipality +0.044)
is the strongest available support for the paper's measurement thesis, but municipality
density can be dominated by tiny units: a few physicians over a few thousand residents
produces an extreme ratio, and the Gini then measures heterogeneity of unit size rather
than of physician distribution.

Four treatments of the same municipality panel, all with the identical estimators and
seeds used at circle scale:

  (a) all municipalities on the fixed basis
  (b) population-weighted Gini (each unit weighted by its population)
  (c) population floors: drop units under 1,000 / 5,000 / 10,000 residents
  (d) composite units: municipalities under 10,000 residents are merged, within their
      own circle, into one composite unit per circle. Documented rule; no municipality
      adjacency matrix is required or invented.

DECISION CRITERION, fixed in advance: does component A stay positive at municipality
scale, and does the prefecture < circle < municipality ordering survive, under every
treatment? If yes, promotion to the main result is on the table. If it depends on small
units, it stays in the supplement.

Bootstrap: 10,000 resamples for (a); 2,000 for (b)-(d), which is stated in the output --
these are robustness variants and the reduction is visible in the interval widths.

Writes outputs/solve/s11_municipality_noise.{csv,txt}
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from census_inputs import municipality_csv  # noqa: E402
YEARS = [2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
CENSUS = [2005, 2010, 2015, 2020, 2025]
SEED_G, SEED_D = 20260805, 20260705
GROUPS = ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44",
          "45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84","85-89","90+"]
report, recs = [], []


def say(s=""):
    print(s); report.append(s)


def gini(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    xs = np.sort(v); n = xs.size; ii = np.arange(1, n + 1)
    return (2 * (xs * ii).sum()) / (n * xs.sum()) - (n + 1) / n


def gini_rows(M):
    S = np.sort(M, axis=1); n = M.shape[1]; ii = np.arange(1, n + 1)
    return (2 * (S * ii).sum(1)) / (n * S.sum(1)) - (n + 1) / n


def wgini(x, w):
    """population-weighted Gini of x with weights w."""
    o = np.argsort(x); x = np.asarray(x)[o]; w = np.asarray(w)[o]
    p = w / w.sum(); S = np.cumsum(p * x); S = np.concatenate([[0.0], S])
    return float(1 - (p * (S[:-1] + S[1:])).sum() / S[-1])


def wgini_rows(X, W):
    o = np.argsort(X, axis=1)
    Xs = np.take_along_axis(X, o, 1); Ws = np.take_along_axis(W, o, 1)
    P = Ws / Ws.sum(1, keepdims=True); S = np.cumsum(P * Xs, axis=1)
    S0 = np.concatenate([np.zeros((len(X), 1)), S], axis=1)
    return 1 - (P * (S0[:, :-1] + S0[:, 1:])).sum(1) / S0[:, -1]


# ---------------------------------------------------------------- panel
MU = pd.read_csv("data/crosswalk/vintage_sma_municipalities.csv",
                 dtype={"pref": str, "sma": str, "muni": str}, encoding="utf-8-sig")
MU["muni"] = MU["muni"].str.zfill(5); MU["sma"] = MU["sma"].str.zfill(4)
cen = {}
for c in CENSUS:
    d = pd.read_csv(municipality_csv(c), dtype={"area_code": str})
    d["area_code"] = d["area_code"].str.zfill(5)
    d = d[~d["area_code"].str.endswith("000")]
    v = pd.to_numeric(d["population"], errors="coerce")
    cen[c] = pd.Series(v.to_numpy(float), index=d["area_code"].to_numpy()).dropna()
m24 = MU[MU.year == 2024]
BASIS = sorted(set(m24["muni"]) & set(cen[2020].index) & set(cen[2010].index))
CIRCLE_OF = dict(zip(m24["muni"], m24["sma"]))


def mpop(y):
    lo = max([c for c in CENSUS if c <= y]); hi = min([c for c in CENSUS if c >= y])
    if lo == hi:
        return cen[lo].reindex(BASIS).to_numpy(float)
    w = (y - lo) / (hi - lo)
    return ((1 - w) * cen[lo].reindex(BASIS) + w * cen[hi].reindex(BASIS)).to_numpy(float)


def mphys(y):
    return (MU[MU.year == y].groupby("muni")["physicians"].sum()
            .reindex(BASIS).fillna(0.0).to_numpy(float))


X = {y: mphys(y) for y in YEARS}
P = {y: mpop(y) for y in YEARS}
AGE = {}
for cy in [2010, 2015, 2020]:
    a = pd.read_csv(f"census_age/age_muni_{cy}.csv", dtype={"area_code": str})
    a["area_code"] = a["area_code"].str.zfill(5)
    a = a.set_index("area_code").reindex(BASIS)[GROUPS].astype(float).to_numpy()
    AGE[cy] = a / np.clip(a.sum(1, keepdims=True), 1e-9, None)
W_ = pd.read_csv("census_age/juryoritsu_2020.csv").set_index("age_group")[
    "juryoritsu_per100k"].reindex(GROUPS).to_numpy(float)


def shares(ay):
    v = (AGE[2010] + (AGE[2015] - AGE[2010]) * ((ay - 2010) / 5.0)) if ay <= 2015 else \
        (AGE[2015] + (AGE[2020] - AGE[2015]) * ((ay - 2015) / 5.0))
    v = np.clip(np.nan_to_num(v), 0, None)
    rs = v.sum(1, keepdims=True); rs[rs == 0] = np.nan
    return v / rs


from needbase import muni_age_table, fraction_from_sums, assert_unity

# The age-stated fraction, computed as a GUARD on the census edition and NOT applied to the
# base -- see needbase.py. This file's unit IS the municipality, so the two sums need no
# aggregation and it calls fraction_from_sums directly; the published side comes from
# cen[], which this file already built. assert_unity is what earns the computation.
GF = {}
for _cy in (2010, 2015, 2020):
    _asum = muni_age_table(_cy).reindex(BASIS)[GROUPS].astype(float).to_numpy().sum(1)
    GF[_cy] = fraction_from_sums(_asum, cen[_cy].reindex(BASIS).to_numpy(float))
assert_unity(GF, "municipality")


def denom(y, kind, sel=None, agg=None):
    raw = P[y]
    if kind == "raw":
        d = raw
    else:
        # PRO RATA: the age shares multiply the PUBLISHED total, distributing the
        # unstated-age residents in the age-stated proportions. The manuscript's Eq.
        # (needweight); see needbase.py for why the age-stated base was reverted.
        nm = (shares(y) * W_).sum(1)
        nm = np.nan_to_num(nm, nan=np.nanmean(nm))
        d = raw * nm / (np.nansum(raw * nm) / np.nansum(raw))
    return d


# ---------------------------------------------------------------- descriptives
say("=" * 104)
say("WORK 2 -- municipality-scale noise check")
say("=" * 104)
say(f"basis: {len(BASIS):,} municipalities (2024 MHLW vintage n 2010 and 2020 censuses)")
say("")
say(f"{'year':>5} {'zero-physician units':>21} {'pop min':>9} {'pop p25':>9} "
    f"{'pop median':>11} {'pop p75':>9} {'pop max':>11} {'units <10k':>11}")
for y in YEARS:
    p_ = P[y]
    say(f"{y:>5} {int((X[y]==0).sum()):>21,} {np.nanmin(p_):>9,.0f} "
        f"{np.nanpercentile(p_,25):>9,.0f} {np.nanmedian(p_):>11,.0f} "
        f"{np.nanpercentile(p_,75):>9,.0f} {np.nanmax(p_):>11,.0f} "
        f"{int((p_<10000).sum()):>11,}")
    recs.append({"block": "descriptives", "year": y,
                 "zero_physician_units": int((X[y] == 0).sum()),
                 "pop_min": float(np.nanmin(p_)), "pop_median": float(np.nanmedian(p_)),
                 "units_under_10k": int((p_ < 10000).sum())})

d24 = X[2024] / P[2024] * 1e5
top = np.argsort(d24)[::-1][:10]
say("")
say("ten highest municipality densities, 2024 (the units a noise story would rest on):")
say(f"{'municipality':>12} {'circle':>7} {'physicians':>11} {'population':>11} {'per 100k':>10}")
nm24 = dict(zip(m24["muni"], m24["muni_name"]))
for i in top:
    mu = BASIS[i]
    say(f"{mu:>12} {CIRCLE_OF.get(mu,''):>7} {X[2024][i]:>11,.0f} {P[2024][i]:>11,.0f} "
        f"{d24[i]:>10,.0f}   {nm24.get(mu,'')}")

# municipal mergers inside the analysis window
say("")
n_by_year = {y: MU[MU.year == y]["muni"].nunique() for y in YEARS}
say("distinct municipality codes in the source, by wave: "
    + ", ".join(f"{y}:{n_by_year[y]:,}" for y in YEARS))
say(f"codes present in 2010 but absent in 2024: "
    f"{len(set(MU[MU.year==2010]['muni']) - set(m24['muni'])):,}; "
    f"absent in 2012 but present in 2010: "
    f"{len(set(MU[MU.year==2010]['muni']) - set(MU[MU.year==2012]['muni'])):,}")
say("The fixed 2024 basis absorbs any merger inside the window by construction: a unit that")
say("disappears simply stops contributing, and coverage is reported per wave (98.0% from 2010).")

# ---------------------------------------------------------------- treatments
def analyse(tag, sel, weights, nboot):
    """sel: boolean mask over BASIS (or None). weights: None or per-unit weights."""
    idx = np.arange(len(BASIS)) if sel is None else np.where(sel)[0]
    n = len(idx)
    out = {"treatment": tag, "units": n}
    rng = np.random.default_rng(SEED_G)
    IDX = rng.integers(0, n, size=(nboot, n))
    for kind in ["raw", "need"]:
        a = np.nan_to_num(X[2010][idx] / denom(2010, kind)[idx] * 1e5)
        b = np.nan_to_num(X[2024][idx] / denom(2024, kind)[idx] * 1e5)
        if weights is None:
            g0, g1 = gini(a), gini(b)
            d = gini_rows(b[IDX]) - gini_rows(a[IDX])
        else:
            w = weights[idx]
            g0, g1 = wgini(a, w), wgini(b, w)
            d = wgini_rows(b[IDX], w[IDX]) - wgini_rows(a[IDX], w[IDX])
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[f"dG_{kind}"] = g1 - g0
        out[f"dG_{kind}_ci"] = (lo, hi)
    rngd = np.random.default_rng(SEED_D)
    ID = rngd.integers(0, n, size=(nboot, n))
    for kind in ["raw", "need"]:
        x0, x1 = X[2010][idx], X[2024][idx]
        P0, P1 = denom(2010, kind)[idx], denom(2024, kind)[idx]
        dd = {k: np.nan_to_num(v) for k, v in
              {"00": x0/P0, "10": x1/P0, "01": x0/P1, "11": x1/P1}.items()}
        if weights is None:
            g = {k: gini(v*1e5) for k, v in dd.items()}
            G = {k: gini_rows(v[ID]*1e5) for k, v in dd.items()}
        else:
            w = weights[idx]
            g = {k: wgini(v*1e5, w) for k, v in dd.items()}
            G = {k: wgini_rows(v[ID]*1e5, w[ID]) for k, v in dd.items()}
        Ax, Ap = g["10"]-g["00"], g["11"]-g["01"]
        Axb, Apb = G["10"]-G["00"], G["11"]-G["01"]
        As, Asb = (Ax+Ap)/2, (Axb+Apb)/2
        out[f"A_x_{kind}"] = Ax; out[f"A_S_{kind}"] = As
        out[f"A_S_{kind}_ci"] = tuple(np.percentile(Asb, [2.5, 97.5]))
        out[f"A_x_{kind}_ci"] = tuple(np.percentile(Axb, [2.5, 97.5]))
        out[f"A_P_{kind}"] = Ap
        out[f"A_P_{kind}_ci"] = tuple(np.percentile(Apb, [2.5, 97.5]))
    return out


pop24 = P[2024]
TREAT = [("(a) all municipalities", None, None, 10000),
         ("(b) population-weighted", None, P[2010], 2000),
         ("(c) pop >= 1,000", P[2010] >= 1000, None, 2000),
         ("(c) pop >= 5,000", P[2010] >= 5000, None, 2000),
         ("(c) pop >= 10,000", P[2010] >= 10000, None, 2000)]

say("")
say("=" * 104)
say("TREATMENTS -- component A and the density-Gini change, 2010->2024")
say("=" * 104)
rows = []
for tag, sel, w, nb in TREAT:
    r = analyse(tag, sel, w, nb)
    rows.append(r)
    say("")
    say(f"--- {tag}  ({r['units']:,} units, {nb:,} resamples) ---")
    for kind in ["raw", "need"]:
        lo, hi = r[f"dG_{kind}_ci"]
        say(f"    {kind:>4}  dG = {r[f'dG_{kind}']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        for sch in ["x", "P", "S"]:
            lo2, hi2 = r[f"A_{sch}_{kind}_ci"]
            say(f"          A ({sch}) = {r[f'A_{sch}_{kind}']:+.4f} [{lo2:+.4f}, {hi2:+.4f}]"
                f"  {'SIG' if (lo2>0 or hi2<0) else ''}")
    recs.append({"block": "treatment", **{k: (v if not isinstance(v, tuple) else str(v))
                                          for k, v in r.items()}})

# ---------------------------------------------------------------- (d) composite units
say("")
say("--- (d) composite units: municipalities under 10,000 residents merged within their circle ---")
small = P[2010] < 10000
circ = np.array([CIRCLE_OF.get(mu, "----") for mu in BASIS])
keys = np.where(small, np.char.add("COMP_", circ), np.array(BASIS))
uk, inv = np.unique(keys, return_inverse=True)
def agg(v):
    return np.bincount(inv, weights=v, minlength=len(uk))
say(f"    {int(small.sum()):,} municipalities merged into {len(set(keys[small])):,} composites; "
    f"{len(uk):,} units in total")
n = len(uk)
rng = np.random.default_rng(SEED_G); IDX = rng.integers(0, n, size=(2000, n))
rngd = np.random.default_rng(SEED_D); ID = rngd.integers(0, n, size=(2000, n))
rd = {"treatment": "(d) composites <10k merged in-circle", "units": n}
for kind in ["raw", "need"]:
    x0, x1 = agg(X[2010]), agg(X[2024])
    P0, P1 = agg(denom(2010, kind)), agg(denom(2024, kind))
    a, b = np.nan_to_num(x0/P0*1e5), np.nan_to_num(x1/P1*1e5)
    d = gini_rows(b[IDX]) - gini_rows(a[IDX]); lo, hi = np.percentile(d, [2.5, 97.5])
    say(f"    {kind:>4}  dG = {gini(b)-gini(a):+.4f} [{lo:+.4f}, {hi:+.4f}]")
    rd[f"dG_{kind}"] = gini(b)-gini(a); rd[f"dG_{kind}_ci"] = (lo, hi)
    dd = {k: np.nan_to_num(v) for k, v in
          {"00": x0/P0, "10": x1/P0, "01": x0/P1, "11": x1/P1}.items()}
    g = {k: gini(v*1e5) for k, v in dd.items()}
    G = {k: gini_rows(v[ID]*1e5) for k, v in dd.items()}
    Ax, Ap = g["10"]-g["00"], g["11"]-g["01"]
    Axb, Apb = G["10"]-G["00"], G["11"]-G["01"]
    for sch, val, bt in [("x", Ax, Axb), ("P", Ap, Apb), ("S", (Ax+Ap)/2, (Axb+Apb)/2)]:
        lo2, hi2 = np.percentile(bt, [2.5, 97.5])
        say(f"          A ({sch}) = {val:+.4f} [{lo2:+.4f}, {hi2:+.4f}]"
            f"  {'SIG' if (lo2>0 or hi2<0) else ''}")
        rd[f"A_{sch}_{kind}"] = val; rd[f"A_{sch}_{kind}_ci"] = (lo2, hi2)
rows.append(rd)
recs.append({"block": "treatment", **{k: (v if not isinstance(v, tuple) else str(v))
                                      for k, v in rd.items()}})

# ---------------------------------------------------------------- verdict
say("")
say("=" * 104)
say("DECISION CRITERION")
say("=" * 104)
say(f"{'treatment':>40} {'units':>8} {'A Shapley raw':>16} {'A>0':>5} {'sig':>5} "
    f"{'A Shapley need':>16} {'A>0':>5} {'sig':>5}")
allpos = True
for r in rows:
    lo, hi = r["A_S_raw_ci"]; lo2, hi2 = r["A_S_need_ci"]
    pos = r["A_S_raw"] > 0 and r["A_S_need"] > 0
    allpos &= pos
    say(f"{r['treatment']:>40} {r['units']:>8,} {r['A_S_raw']:>+16.4f} "
        f"{str(r['A_S_raw']>0):>5} {'yes' if lo>0 else 'no':>5} "
        f"{r['A_S_need']:>+16.4f} {str(r['A_S_need']>0):>5} {'yes' if lo2>0 else 'no':>5}")
say("")
say(f"component A positive under EVERY treatment, both denominators: {allpos}")
say("")
say("scale ordering, A Shapley raw:  prefecture -0.0142  <  circle +0.0152  <  municipality:")
for r in rows:
    say(f"    {r['treatment']:>40} {r['A_S_raw']:>+9.4f}  "
        f"{'ordering holds' if r['A_S_raw'] > 0.0152 else 'ordering BREAKS'}")

pd.DataFrame(recs).to_csv("outputs/solve/s11_municipality_noise.csv", index=False,
                          encoding="utf-8-sig")
Path("outputs/solve/s11_municipality_noise.txt").write_text("\n".join(report) + "\n",
                                                            encoding="utf-8")
print("\nwrote outputs/solve/s11_municipality_noise.{csv,txt}")
