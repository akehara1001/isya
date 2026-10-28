"""
SOLVE 2 -- answer the modifiable areal unit problem instead of conceding it.

The crosswalk repair established that the MHLW source tabulates physicians by
MUNICIPALITY within each circle. Municipality population is already held for every
census year. So the same analysis can be run at three nested scales and the MAUP
becomes a testable question rather than a caveat.

  prefecture    47 units   (aggregate of the repaired circle panel)
  circle       330 units   (the paper's unit)
  municipality ~1,850 units (new)

Method is identical at every scale -- nothing is invented. Gini, matched-circle paired
bootstrap of the change (seed 20260805, 10,000 resamples), and the same counterfactual
decomposition (seed 20260705) under all three attribution schemes and both denominators.

Fixed basis. Municipalities are held to the 2024 MHLW vintage intersected with the
census, exactly as circles are held to the 2024 boundary. Physicians in municipality
codes extinguished by merger are excluded rather than reallocated; the resulting
coverage is reported per wave, and the primary 2010-2024 window is where it is highest.

DECISION CRITERION: does component A keep its positive sign -- physician redistribution
widening inequality -- at every scale? If yes the paper's central result is
scale-invariant and the MAUP is answered. If not, the scale dependence is itself the
result and is reported as such.

Writes outputs/solve/s02_scale_invariance.csv and .txt
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
N_BOOT = 10000
SEED_GINI, SEED_DECOMP = 20260805, 20260705
GROUPS = ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44",
          "45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84","85-89","90+"]
report, recs = [], []


def say(s=""):
    print(s)
    report.append(s)


def gini(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    xs = np.sort(v); n = xs.size; ii = np.arange(1, n + 1)
    return (2 * (xs * ii).sum()) / (n * xs.sum()) - (n + 1) / n


def gini_rows(M):
    S = np.sort(M, axis=1); n = M.shape[1]; ii = np.arange(1, n + 1)
    return (2 * (S * ii).sum(1)) / (n * S.sum(1)) - (n + 1) / n


# ================================================== municipality physicians
MU = pd.read_csv("data/crosswalk/vintage_sma_municipalities.csv",
                 dtype={"pref": str, "sma": str, "muni": str}, encoding="utf-8-sig")
MU["muni"] = MU["muni"].str.zfill(5)
MUNI24 = sorted(MU[MU.year == 2024]["muni"].unique())

# ================================================== municipality population
cen = {}
for c in CENSUS:
    d = pd.read_csv(municipality_csv(c), dtype={"area_code": str})
    d["area_code"] = d["area_code"].str.zfill(5)
    d = d[~d["area_code"].str.endswith("000")]          # drop prefecture/national rows
    v = pd.to_numeric(d["population"], errors="coerce")
    cen[c] = pd.Series(v.to_numpy(float), index=d["area_code"].to_numpy()).dropna()

BASIS = sorted(set(MUNI24) & set(cen[2020].index) & set(cen[2010].index))
say("=" * 100)
say("SOLVE 2 -- the same analysis at three nested scales")
say("=" * 100)
say(f"municipality basis: {len(MUNI24)} in the 2024 MHLW vintage, "
    f"{len(BASIS)} also present in the 2010 and 2020 censuses -> analysis basis")


def muni_pop(year):
    lo = max([c for c in CENSUS if c <= year], default=CENSUS[0])
    hi = min([c for c in CENSUS if c >= year], default=CENSUS[-1])
    if lo == hi:
        return cen[lo].reindex(BASIS).astype(float)
    w = (year - lo) / (hi - lo)
    return ((1 - w) * cen[lo].reindex(BASIS) + w * cen[hi].reindex(BASIS)).astype(float)


def muni_phys(year):
    d = MU[MU.year == year].groupby("muni")["physicians"].sum()
    return d.reindex(BASIS).fillna(0.0).astype(float)


say("")
say("municipality-scale coverage (physicians on the fixed basis, of the national total):")
say(f"{'year':>5} {'phys on basis':>15} {'national':>10} {'coverage':>10} "
    f"{'zero-physician municipalities':>30}")
cov = {}
for y in YEARS:
    p = muni_phys(y); nat = MU[MU.year == y]["physicians"].sum()
    cov[y] = p.sum() / nat
    say(f"{y:>5} {p.sum():>15,.0f} {nat:>10,.0f} {100*p.sum()/nat:>9.2f}% "
        f"{int((p == 0).sum()):>30,}")

# ================================================== age shares at municipality level
AGE = {}
for cy in [2010, 2015, 2020]:
    a = pd.read_csv(f"census_age/age_muni_{cy}.csv", dtype={"area_code": str})
    a["area_code"] = a["area_code"].str.zfill(5)
    a = a.set_index("area_code").reindex(BASIS)[GROUPS].astype(float)
    AGE[cy] = (a.to_numpy() / np.clip(a.to_numpy().sum(1, keepdims=True), 1e-9, None))
rate = pd.read_csv("census_age/juryoritsu_2020.csv").set_index("age_group")["juryoritsu_per100k"]
W = rate.reindex(GROUPS).to_numpy(float)


def muni_shares(ay):
    v = (AGE[2010] + (AGE[2015] - AGE[2010]) * ((ay - 2010) / 5.0)) if ay <= 2015 else \
        (AGE[2015] + (AGE[2020] - AGE[2015]) * ((ay - 2015) / 5.0))
    v = np.clip(np.nan_to_num(v), 0, None)
    rs = v.sum(1, keepdims=True); rs[rs == 0] = np.nan
    return v / rs


# ================================================== assemble the three scales
rep = pd.read_csv("data/crosswalk/physician_panel_smsa_repaired.csv",
                  dtype={"code4": str}, encoding="utf-8-sig").set_index("code4")
CIRC = list(rep.index)


def circ_pop(y):
    d = pd.read_csv(f"census_based/smsa_population_census_{y}.csv")
    d["code4"] = d["smsa_code"].apply(lambda c: str(c).zfill(4))
    return d.set_index("code4")["population"].reindex(CIRC).astype(float).to_numpy()


CIRC_AGE = {}
for cy in [2010, 2015, 2020]:
    # Round 38 (2026-10-06): the area age table, built once by cc_build_census_age.py through
    # the code-change tables. Summing census_age/age_muni_ over the 2024 map's codes, as this
    # did until then, lost every municipality whose code had changed since the census --
    # 熊本市 in 2010, the seven Hamamatsu wards in 2010-2020 -- and gave those areas the age
    # mix of what was left.
    # The MUNICIPALITY scale above is unchanged: its basis is the municipalities present
    # under the same code in 2010, 2020 and the 2024 vintage, a declared restriction.
    a = pd.read_csv(f"census_age/age_circle_{cy}.csv", dtype={"code4": str}, encoding="utf-8")
    mat = a.set_index("code4").reindex(CIRC)[GROUPS].to_numpy(float)
    CIRC_AGE[cy] = mat / np.clip(mat.sum(1, keepdims=True), 1e-9, None)


def circ_shares(ay):
    v = (CIRC_AGE[2010] + (CIRC_AGE[2015] - CIRC_AGE[2010]) * ((ay - 2010) / 5.0)) \
        if ay <= 2015 else \
        (CIRC_AGE[2015] + (CIRC_AGE[2020] - CIRC_AGE[2015]) * ((ay - 2015) / 5.0))
    v = np.clip(v, 0, None); rs = v.sum(1, keepdims=True); rs[rs == 0] = np.nan
    return v / rs


SCALES = {}
# municipality
SCALES["municipality"] = {
    "x": {y: muni_phys(y).to_numpy() for y in YEARS},
    "P": {y: muni_pop(y).to_numpy() for y in YEARS},
    "sh": muni_shares}
# circle
SCALES["circle"] = {
    "x": {y: rep[str(y)].astype(float).to_numpy() for y in YEARS},
    "P": {y: circ_pop(y) for y in YEARS},
    "sh": circ_shares}
# prefecture
pref_of_c = np.array([c[:2] for c in CIRC])
PREFS = sorted(set(pref_of_c))
M = np.array([[1.0 if pref_of_c[j] == p else 0.0 for j in range(len(CIRC))] for p in PREFS])
SCALES["prefecture"] = {
    "x": {y: M @ SCALES["circle"]["x"][y] for y in YEARS},
    "P": {y: M @ SCALES["circle"]["P"][y] for y in YEARS},
    "sh": lambda ay: (M @ (circ_shares(ay) * SCALES["circle"]["P"][2020][:, None])) /
                     np.clip((M @ SCALES["circle"]["P"][2020])[:, None], 1e-9, None)}


# ============================== the age-stated base, matching c01 (B-67, B-85)
# The shares above are normalised by the AGE-BAND SUM, so they are shares of the
# age-stated population. Multiplying them by the PUBLISHED total -- which this file did
# until 2026-08-31 -- distributes the unstated-age population pro rata, and it does so
# only at the anchors whose tables report 年齢不詳 separately. Of s02's three anchors
# that is 2010 alone: 2015 (0004019348) and 2020 (0004019309) are 不詳補完結果 tables
# whose classes are exhaustive, so their fraction is exactly 1.
#
# c01 was unified in item 2 and this file was not, which left ONE QUANTITY COMPUTED TWO
# WAYS INSIDE ONE PAPER -- Table 1 from c01 on the age-stated base, section 3.2 and
# Supplementary B5 from here on the published total. This removes that divergence.
#
# Numerator and denominator are summed separately before the ratio is taken, so a circle
# or prefecture fraction is population-weighted rather than an average of ratios.
from needbase import (muni_age_table, muni_published, fraction_from_sums,
                      assert_unity, interpolate)


# THE CONSTRUCTION LIVES IN needbase.py. This file needs the fraction at THREE scales,
# with numerator and denominator summed separately before the ratio at each, so that a
# circle or prefecture figure is population-weighted rather than an average of ratios.
# It therefore builds the two sums itself and calls fraction_from_sums per scale.
def _muni_agesum(cy):
    return muni_age_table(cy).reindex(BASIS)[GROUPS].astype(float).to_numpy().sum(1)


def _muni_published(cy):
    return muni_published(cy).reindex(BASIS).to_numpy(float)


_m24 = MU[MU.year == 2024].set_index("muni")["sma"]
_c_of_m = np.array([_m24.get(m, "") for m in BASIS])
_CIDX = {c: i for i, c in enumerate(CIRC)}
GFRAC = {}
for _cy in (2010, 2015, 2020):
    _a, _p = _muni_agesum(_cy), _muni_published(_cy)
    _ok = np.isfinite(_a) & np.isfinite(_p) & (_p > 0)
    _ca = np.zeros(len(CIRC)); _cp = np.zeros(len(CIRC))
    for _i, _c in enumerate(_c_of_m):
        _j = _CIDX.get(_c)
        if _j is not None and _ok[_i]:
            _ca[_j] += _a[_i]; _cp[_j] += _p[_i]
    _pa, _pp = M @ _ca, M @ _cp
    GFRAC[_cy] = {
        "municipality": fraction_from_sums(_a, _p),
        "circle": fraction_from_sums(_ca, _cp),
        "prefecture": fraction_from_sums(_pa, _pp)}
for _sc in ("municipality", "circle", "prefecture"):
    assert_unity({_cy: GFRAC[_cy][_sc] for _cy in GFRAC}, _sc)


def denom(scale, ay, kind):
    S = SCALES[scale]
    raw = S["P"][ay]
    if kind == "raw":
        return raw
    # PRO RATA at every scale: the age shares multiply the PUBLISHED total, distributing
    # the unstated-age residents in the age-stated proportions. The manuscript's Eq.
    # (needweight); see needbase.py for why the age-stated base was reverted. GFRAC above
    # is kept as the census-edition guard (assert_unity), not as a factor.
    sh = S["sh"](ay)
    nm = np.nan_to_num((sh * W).sum(1), nan=np.nanmean((sh * W).sum(1)))
    return raw * nm / (np.nansum(raw * nm) / np.nansum(raw))


# ================================================== Gini series and paired bootstrap
say("")
say("=" * 100)
say("GINI AND ITS CHANGE, 2010->2024, AT THREE SCALES")
say("=" * 100)
say(f"{'scale':>13} {'units':>7} {'measure':>8} {'G 2010':>8} {'G 2024':>8} {'dG':>9} "
    f"{'95% CI':>22}")
for scale in ["prefecture", "circle", "municipality"]:
    S = SCALES[scale]
    n = len(S["x"][2024])
    rng = np.random.default_rng(SEED_GINI)
    IDX = rng.integers(0, n, size=(N_BOOT, n))
    for measure in ["count", "raw", "need"]:
        if measure == "count":
            a, b = S["x"][2010], S["x"][2024]
        else:
            a = S["x"][2010] / denom(scale, 2010, measure) * 1e5
            b = S["x"][2024] / denom(scale, 2024, measure) * 1e5
        a = np.nan_to_num(a); b = np.nan_to_num(b)
        d = gini_rows(b[IDX]) - gini_rows(a[IDX])
        lo, hi = np.percentile(d, [2.5, 97.5])
        say(f"{scale:>13} {n:>7} {measure:>8} {gini(a):>8.4f} {gini(b):>8.4f} "
            f"{gini(b)-gini(a):>+9.4f} [{lo:>+8.4f},{hi:>+8.4f}]")
        recs.append({"block": "Gini change 2010-2024", "scale": scale, "units": n,
                     "measure": measure, "G_2010": round(gini(a), 4),
                     "G_2024": round(gini(b), 4), "dG": round(gini(b) - gini(a), 4),
                     "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                     "excludes_zero": bool(lo > 0 or hi < 0)})
    say("")

# ================================================== decomposition at each scale
say("=" * 100)
say("DECOMPOSITION 2010->2024 AT THREE SCALES (Gini units, all three schemes)")
say("=" * 100)
for scale in ["prefecture", "circle", "municipality"]:
    S = SCALES[scale]
    n = len(S["x"][2024])
    rngd = np.random.default_rng(SEED_DECOMP)
    IDXD = rngd.integers(0, n, size=(N_BOOT, n))
    for kind in ["raw", "need"]:
        x0, x1 = np.nan_to_num(S["x"][2010]), np.nan_to_num(S["x"][2024])
        P0, P1 = denom(scale, 2010, kind), denom(scale, 2024, kind)
        d = {"00": x0 / P0, "10": x1 / P0, "01": x0 / P1, "11": x1 / P1}
        d = {k: np.nan_to_num(v) for k, v in d.items()}
        g = {k: gini(v * 1e5) for k, v in d.items()}
        Gb = {k: gini_rows(v[IDXD] * 1e5) for k, v in d.items()}
        tot = g["11"] - g["00"]
        Ax, Ap = g["10"] - g["00"], g["11"] - g["01"]
        Bp, Bx = g["01"] - g["00"], g["11"] - g["10"]
        Axb, Apb = Gb["10"] - Gb["00"], Gb["11"] - Gb["01"]
        Bpb, Bxb = Gb["01"] - Gb["00"], Gb["11"] - Gb["10"]
        ci = lambda v: np.percentile(v, [2.5, 97.5])
        say("")
        say(f"--- {scale} ({n} units), {kind} denominator ---   total dG = {tot:+.4f}")
        for lab, pt, bt in [("A x-first", Ax, Axb), ("A P-first", Ap, Apb),
                            ("A Shapley", (Ax + Ap) / 2, (Axb + Apb) / 2),
                            ("B x-first", Bx, Bxb), ("B P-first", Bp, Bpb),
                            ("B Shapley", (Bx + Bp) / 2, (Bxb + Bpb) / 2)]:
            lo, hi = ci(bt)
            say(f"{lab:>14} {pt:>+9.4f} [{lo:>+9.4f},{hi:>+9.4f}] "
                f"{'SIG' if (lo > 0 or hi < 0) else '   '}")
            recs.append({"block": "decomposition 2010-2024", "scale": scale, "units": n,
                         "measure": kind, "component": lab, "point": round(pt, 4),
                         "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                         "excludes_zero": bool(lo > 0 or hi < 0)})
        lo, hi = ci(Bpb - Axb)
        say(f"{'B - A':>14} {Bp-Ax:>+9.4f} [{lo:>+9.4f},{hi:>+9.4f}] "
            f"{'SIG' if (lo > 0 or hi < 0) else '   '}")
        recs.append({"block": "decomposition 2010-2024", "scale": scale, "units": n,
                     "measure": kind, "component": "B - A", "point": round(Bp - Ax, 4),
                     "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                     "excludes_zero": bool(lo > 0 or hi < 0)})

R = pd.DataFrame(recs)
R.to_csv("outputs/solve/s02_scale_invariance.csv", index=False, encoding="utf-8-sig")

say("")
say("=" * 100)
say("VERDICT: is component A positive at every scale?")
say("=" * 100)
dec = R[R.block == "decomposition 2010-2024"]
for scale in ["prefecture", "circle", "municipality"]:
    for kind in ["raw", "need"]:
        sub = dec[(dec.scale == scale) & (dec.measure == kind) &
                  (dec.component.str.startswith("A "))]
        pos = bool((sub.point > 0).all())
        sig = int(sub.excludes_zero.sum())
        say(f"{scale:>13} {kind:>5}: A positive under all 3 schemes = {str(pos):>5}; "
            f"interval excludes zero in {sig}/3")

Path("outputs/solve/s02_scale_invariance.txt").write_text("\n".join(report) + "\n",
                                                          encoding="utf-8")
print("\nwrote outputs/solve/s02_scale_invariance.{csv,txt}")
