"""
ROUND 3 -- the numbers the repair round did not report, plus the 2012-2024 window.

Nothing new is estimated. The same three procedures from rounds 1-2 are run over more
windows, more denominators and both weightings, with IDENTICAL seeds:

    paired circle bootstrap of a Gini change      seed 20260805, 10,000 resamples
    joint circle bootstrap of the decomposition   seed 20260705, 10,000 resamples
    Moran permutation and trend permutation       seed 20260621, 9,999 draws

Both panels (published, repaired) are computed on one code path so every difference is
attributable to the repair.

Covers:
  ITEM 1  count-Gini series and dG (2006-2024, 2010-2024); need-weighted A/B/I/total and
          B-A; need-weighted density-Gini series; Moran trend row+binary; e65 decomposition
  ITEM 2  every one of the above repeated on the 2012-2024 window
  ITEM 6  Moran trend restricted to 2012-2024, to test how far the drift depends on the
          three most heavily reconstructed waves

Writes outputs/audit/c01_round3_numbers.csv and c01_round3_numbers.txt
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from phase2_common import read_csv_rows, parse_code_name

YEARS = [2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
N_BOOT, N_PERM = 10000, 9999
SEED_GINI, SEED_DECOMP, SEED_MORAN = 20260805, 20260705, 20260621
GROUPS = ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44",
          "45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84","85-89","90+"]
report, recs = [], []


def say(s=""):
    print(s)
    report.append(s)


def rec(**kw):
    recs.append(kw)


# ---------------------------------------------------------------- panels
pub = pd.read_csv("data/superseded/physician_panel_smsa__PRE_REPAIR.csv")
pub["code4"] = pub["smsa_code"].apply(lambda c: str(c).zfill(4))
CODES = sorted(pub["code4"])
PUB = pub.set_index("code4")[[str(y) for y in YEARS]].astype(float).reindex(CODES)
PUB.columns = YEARS
rp = pd.read_csv("data/crosswalk/physician_panel_smsa_repaired.csv",
                 dtype={"code4": str}, encoding="utf-8-sig").set_index("code4")
REP = rp[[str(y) for y in YEARS]].astype(float).reindex(CODES)
REP.columns = YEARS
PANELS = {"published": PUB, "repaired": REP}


def total_pop(y):
    d = pd.read_csv(f"census_based/smsa_population_census_{y}.csv")
    d["code4"] = d["smsa_code"].apply(lambda c: str(c).zfill(4))
    return d.set_index("code4")["population"].reindex(CODES).astype(float).to_numpy()


POP = {y: total_pop(y) for y in YEARS}


def gini(v):
    xs = np.sort(np.asarray(v, float)); n = xs.size; ii = np.arange(1, n + 1)
    return (2 * (xs * ii).sum()) / (n * xs.sum()) - (n + 1) / n


def gini_rows(M):
    S = np.sort(M, axis=1); n = M.shape[1]; ii = np.arange(1, n + 1)
    return (2 * (S * ii).sum(1)) / (n * S.sum(1)) - (n + 1) / n


# ---------------------------------------------------------------- denominators
def build_map():
    m, cur = {}, None
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


def sma_age_shares(cy):
    """Circle age composition: the age-stated persons of every census municipality the
    code-change tables carry to the circle, normalised to shares."""
    # Round 38 (2026-10-06): the area age table, built once by cc_build_census_age.py through
    # the code-change tables. Summing census_age/age_muni_ over the 2024 map's codes, as this
    # did until then, lost every municipality whose code had changed since the census --
    # 熊本市 in 2010, the seven Hamamatsu wards in 2010-2020 -- and gave those areas the age
    # mix of what was left.
    # The prefecture substitution this function carried for an area with no matched
    # municipality is gone with it: the area table reaches every area or the build stops.
    t = pd.read_csv(f"census_age/age_circle_{cy}.csv", dtype={"code4": str}, encoding="utf-8")
    mat = t.set_index("code4").reindex(CODES)[GROUPS].to_numpy(float)
    tot = mat.sum(1, keepdims=True)
    if (tot <= 0).any():
        raise AssertionError(f"{cy}: circle with zero age total after substitution")
    out = mat / tot
    if out.shape[0] != len(CODES) or not np.isfinite(out).all():
        raise AssertionError(f"{cy}: circle count changed or non-finite age share")
    return out


SH = {cy: sma_age_shares(cy) for cy in [2005, 2010, 2015, 2020]}


from needbase import fraction_from_sums, assert_unity


def _area_sums(cy):
    """Age-stated persons and published population per area, from the area age table.
    The census-edition guard is taken over the same areas as the shares (round 38)."""
    t = pd.read_csv(f"census_age/age_circle_{cy}.csv", dtype={"code4": str},
                    encoding="utf-8").set_index("code4").reindex(CODES)
    return t[GROUPS].to_numpy(float).sum(1), t["population"].to_numpy(float)



# The age-stated fraction, computed as a GUARD and not applied to the base. See needbase.py
# for why: the manuscript defines P^need = P * sum_a s_ia r_a / rbar on the resident
# population, which is pro-rata apportionment of the unstated-age residents, and that is
# what denom() does below. assert_unity is what earns the computation: at the 不詳補完
# anchors the fraction must be exactly 1, so a re-fetch that returned a different census
# edition would change what s_ia means and fails here instead of moving a Gini quietly.
GFRAC = {cy: fraction_from_sums(*_area_sums(cy)) for cy in [2005, 2010, 2015, 2020]}
assert_unity(GFRAC, "circle")


def shares_for_year(ay):
    """Census anchors at 2005/2010/2015/2020, interpolated between; only 2022 and 2024 are
    extrapolated, forward along the 2015-2020 slope. Anchor years are returned untouched so
    a missing value in a far anchor cannot propagate into a measured year."""
    if ay in SH:
        v = SH[ay].copy()
    elif ay < 2010:
        v = SH[2005] + (SH[2010] - SH[2005]) * ((ay - 2005) / 5.0)
    elif ay < 2015:
        v = SH[2010] + (SH[2015] - SH[2010]) * ((ay - 2010) / 5.0)
    elif ay < 2020:
        v = SH[2015] + (SH[2020] - SH[2015]) * ((ay - 2015) / 5.0)
    else:
        v = SH[2020] + (SH[2020] - SH[2015]) * ((ay - 2020) / 5.0)
    v = np.clip(v, 0, None); rs = v.sum(1, keepdims=True); rs[rs == 0] = np.nan
    return v / rs


rate = pd.read_csv("census_age/juryoritsu_2020.csv").set_index("age_group")["juryoritsu_per100k"]
W = rate.reindex(GROUPS).to_numpy(float)


def denom(ay, kind):
    raw = POP[ay]
    if kind == "raw":
        return raw
    sh = shares_for_year(ay)
    # PRO RATA, in every year and for both weighted forms: the age SHARES multiply the
    # PUBLISHED total, which distributes the unstated-age residents in the age-stated
    # proportions. This is the manuscript's Eq. (needweight) literally -- see needbase.py
    # for the three checks that settled it against the age-stated base.
    if kind == "e65":
        return raw * sh[:, 13:].sum(1)
    nm = (sh * W).sum(1)
    return raw * nm / (np.nansum(raw * nm) / np.nansum(raw))


# ================================================================= 1. Gini series
say("=" * 104)
say("1a. COUNT AND DENSITY GINI SERIES (raw and need-weighted), published vs repaired")
say("=" * 104)
say(f"{'year':>5} | {'count pub':>9} {'count rep':>9} | {'dens pub':>9} {'dens rep':>9} | "
    f"{'need pub':>9} {'need rep':>9} | {'e65 rep':>8}")
SER = {}
for y in YEARS:
    row = {}
    for k, P in PANELS.items():
        x = P[y].to_numpy()
        row[(k, "count")] = gini(x)
        for kind in ["raw", "need", "e65"]:
            row[(k, kind)] = gini(x / denom(y, kind) * 1e5)
    SER[y] = row
    say(f"{y:>5} | {row[('published','count')]:>9.4f} {row[('repaired','count')]:>9.4f} | "
        f"{row[('published','raw')]:>9.4f} {row[('repaired','raw')]:>9.4f} | "
        f"{row[('published','need')]:>9.4f} {row[('repaired','need')]:>9.4f} | "
        f"{row[('repaired','e65')]:>8.4f}")
    for kind in ["count", "raw", "need", "e65"]:
        # FULL PRECISION AT STORAGE, rounded once at print (B-11). Storing round(x, 4)
        # against a page that prints 3dp left ONE digit between the file and the page,
        # and a value whose 4th decimal became a 5 was then rounded outward a second
        # time -- three interval bounds in tab:decomp were printed wrong that way and a
        # checker comparing the page to this file could not see it (B-66).
        rec(block="1a Gini series", quantity=f"{kind} Gini {y}",
            published=row[("published", kind)],
            repaired=row[("repaired", kind)])

# ---- decomposition of the repaired raw decline into sub-windows (for item 5) ----
g10, g16, g24 = SER[2010][("repaired", "raw")], SER[2016][("repaired", "raw")], \
    SER[2024][("repaired", "raw")]
say("")
say(f"Where the repaired 2010-2024 raw decline occurs: total {g24-g10:+.4f}; "
    f"2010-2016 {g16-g10:+.4f} ({100*(g16-g10)/(g24-g10):.1f}%), "
    f"2016-2024 {g24-g16:+.4f} ({100*(g24-g16)/(g24-g10):.1f}%).")
say("2016 onward is bit-identical between panels, so that share of the decline is "
    "untouched by the repair.")
rec(block="1a Gini series", quantity="share of 2010-2024 raw decline occurring 2016-2024",
    published=None, repaired=f"{100*(g24-g16)/(g24-g10):.1f}%")

# ================================================================= paired bootstrap
say("")
say("=" * 104)
say("1b/2. PAIRED BOOTSTRAP OF THE GINI CHANGE -- all measures, all windows")
say("=" * 104)
rng = np.random.default_rng(SEED_GINI)
IDX = rng.integers(0, len(CODES), size=(N_BOOT, len(CODES)))
WINDOWS = [(2006, 2024), (2010, 2024), (2012, 2024), (2016, 2024)]
say(f"{'measure':>8} {'window':>11} {'panel':>10} {'G base':>8} {'G end':>8} {'dG':>9} "
    f"{'95% CI':>22} {'Pr(dG>0)':>9} {'Pr(dG<0)':>9}")
for measure in ["count", "raw", "need"]:
    for base, end in WINDOWS:
        vals = {}
        for k, P in PANELS.items():
            if measure == "count":
                a, b = P[base].to_numpy(), P[end].to_numpy()
            else:
                a = P[base].to_numpy() / denom(base, measure) * 1e5
                b = P[end].to_numpy() / denom(end, measure) * 1e5
            d = gini_rows(b[IDX]) - gini_rows(a[IDX])
            lo, hi = np.percentile(d, [2.5, 97.5])
            # STORED at full precision; the say() below prints 4dp. B-11.
            vals[k] = f"{gini(b)-gini(a):+.10f} [{lo:+.10f}, {hi:+.10f}]"
            say(f"{measure:>8} {base}-{end} {k:>10} {gini(a):>8.4f} {gini(b):>8.4f} "
                f"{gini(b)-gini(a):>+9.4f} [{lo:>+8.4f},{hi:>+8.4f}] "
                f"{(d>0).mean():>9.4f} {(d<0).mean():>9.4f}")
        rec(block="1b/2 paired bootstrap", quantity=f"{measure} Gini change {base}-{end}",
            published=vals["published"], repaired=vals["repaired"])
    say("")

# ================================================================= Fig. 2: every wave against 2010
# Round 43-r52 (2026-10-12). Fig. 2 now plots, for every wave t, the change in the Gini from 2010,
# dG(t) = G(t) - G(2010), with its matched-area interval. Nothing new is estimated: this is block
# 1b/2 applied to each wave, on the repaired panel, with the SAME draw IDX applied to t and to 2010.
# Its 2010-2024 rows therefore equal "count/raw Gini change 2010-2024" above to the last digit,
# which d01 asserts before drawing. 2010 itself is 0 with no interval and is not recorded.
say("=" * 104)
say("1e. FIG. 2 -- matched-area change from 2010 at every wave (the draws of 1b/2)")
say("=" * 104)
for measure in ["count", "raw"]:
    P = PANELS["repaired"]
    if measure == "count":
        a = P[2010].to_numpy()
    else:
        a = P[2010].to_numpy() / denom(2010, measure) * 1e5
    ga = gini_rows(a[IDX])
    for t in YEARS:
        if t == 2010:
            continue
        b = P[t].to_numpy() if measure == "count" else P[t].to_numpy() / denom(t, measure) * 1e5
        d = gini_rows(b[IDX]) - ga
        lo, hi = np.percentile(d, [2.5, 97.5])
        say(f"{measure:>8} 2010->{t} {gini(b)-gini(a):>+9.4f} [{lo:>+8.4f},{hi:>+8.4f}]")
        rec(block="1e Fig2 change from 2010", quantity=f"{measure} Gini change from 2010 to {t}",
            published=None, repaired=f"{gini(b)-gini(a):+.10f} [{lo:+.10f}, {hi:+.10f}]")
say("")

# ================================================================= decomposition
say("=" * 104)
say("1c/2. DECOMPOSITION -- raw / need / e65, base 2010 and base 2012, all three schemes")
say("=" * 104)
rngd = np.random.default_rng(SEED_DECOMP)
IDXD = rngd.integers(0, len(CODES), size=(N_BOOT, len(CODES)))
ci = lambda v: np.percentile(v, [2.5, 97.5])


def decompose(x0, x1, P0, P1):
    """The counterfactual decomposition of a density-Gini change, for ANY numerator.

    PARAMETERISED 2026-09-03 for C-1 (co-author review). The body below is the loop that used to sit
    here inline, moved behind four arrays and not otherwise touched: the numerator came
    from the fixed PANELS dictionary and the window from a literal `for base in [2010,
    2012]` with 2024 hard-coded, so the decomposition could not be run on the effective-
    supply series, nor on the 2016-2024 flow window, without editing this file
    (split/ozaki_round1_findings.md section 6-3 measured exactly that). The existing calls
    pass the same arrays as before and every stored value is byte-identical; the guard is
    code/audit/c12_decomposition_parameterised.py.

    x0, x1  numerator at the base and end of the window (one value per circle)
    P0, P1  denominator at the same two points, in the same circle order
    """
    d = {"00": x0 / P0, "10": x1 / P0, "01": x0 / P1, "11": x1 / P1}
    g = {kk: gini(v * 1e5) for kk, v in d.items()}
    Gb = {kk: gini_rows(v[IDXD] * 1e5) for kk, v in d.items()}
    tot = g["11"] - g["00"]
    Ax, Bp = g["10"] - g["00"], g["01"] - g["00"]
    Ap, Bx = g["11"] - g["01"], g["11"] - g["10"]
    Axb, Bpb = Gb["10"] - Gb["00"], Gb["01"] - Gb["00"]
    Apb, Bxb = Gb["11"] - Gb["01"], Gb["11"] - Gb["10"]
    return dict(
        g0=g["00"], g1=g["11"], tot=tot, I=tot - Ax - Bp,
        A_x=Ax, A_P=Ap, A_S=(Ax + Ap) / 2,
        B_x=Bx, B_P=Bp, B_S=(Bx + Bp) / 2,
        A_x_ci=ci(Axb), A_P_ci=ci(Apb), A_S_ci=ci((Axb + Apb) / 2),
        B_x_ci=ci(Bxb), B_P_ci=ci(Bpb), B_S_ci=ci((Bxb + Bpb) / 2),
        I_ci=ci((Gb["11"] - Gb["00"]) - Axb - Bpb),
        tot_ci=ci(Gb["11"] - Gb["00"]),
        BmA_x=Bp - Ax, BmA_x_ci=ci(Bpb - Axb),
        BmA_S=(Bx + Bp) / 2 - (Ax + Ap) / 2,
        BmA_S_ci=ci((Bxb + Bpb) / 2 - (Axb + Apb) / 2),
        P_dom=float((np.abs(Bpb) > np.abs(Axb)).mean()),
        A_all_pos=bool(Ax > 0 and Ap > 0),
        A_x_sig=bool(ci(Axb)[0] > 0), A_P_sig=bool(ci(Apb)[0] > 0),
        A_S_sig=bool(ci((Axb + Apb) / 2)[0] > 0),
        BmA_sig=bool(ci(Bpb - Axb)[1] < 0))


DEC = {}
for k, P in PANELS.items():
    for kind in ["raw", "need", "e65"]:
        for base in [2010, 2012]:
            DEC[(k, kind, base)] = decompose(
                P[base].to_numpy(), P[2024].to_numpy(),
                denom(base, kind), denom(2024, kind))

for kind in ["raw", "need", "e65"]:
    for base in [2010, 2012]:
        p_, r_ = DEC[("published", kind, base)], DEC[("repaired", kind, base)]
        say("")
        say(f"--- {kind} denominator, {base} -> 2024 ---")
        say(f"{'':>24} {'published':>30} {'repaired':>30}")
        for lab, key in [("G(base)", "g0"), ("G(2024)", "g1"), ("total dG", "tot")]:
            say(f"{lab:>24} {p_[key]:>30.4f} {r_[key]:>30.4f}")
            rec(block="1c/2 decomposition", quantity=f"{kind} {base}-2024: {lab}",
                published=p_[key], repaired=r_[key])
        for lab, kp, kc in [("A x-first", "A_x", "A_x_ci"), ("A P-first", "A_P", "A_P_ci"),
                            ("A Shapley", "A_S", "A_S_ci"), ("B x-first", "B_x", "B_x_ci"),
                            ("B P-first", "B_P", "B_P_ci"), ("B Shapley", "B_S", "B_S_ci"),
                            ("interaction", "I", "I_ci"),
                            ("B-A (main conv)", "BmA_x", "BmA_x_ci"),
                            ("B-A (Shapley)", "BmA_S", "BmA_S_ci")]:
            # PRINTED at 4dp, STORED at full precision. These are the cells whose double
            # rounding put three wrong interval bounds in tab:decomp (B-66).
            ps = f"{p_[kp]:+.4f} [{p_[kc][0]:+.4f},{p_[kc][1]:+.4f}]"
            rs = f"{r_[kp]:+.4f} [{r_[kc][0]:+.4f},{r_[kc][1]:+.4f}]"
            ps_full = f"{p_[kp]:+.10f} [{p_[kc][0]:+.10f},{p_[kc][1]:+.10f}]"
            rs_full = f"{r_[kp]:+.10f} [{r_[kc][0]:+.10f},{r_[kc][1]:+.10f}]"
            say(f"{lab:>24} {ps:>30} {rs:>30}")
            rec(block="1c/2 decomposition", quantity=f"{kind} {base}-2024: {lab}",
                published=ps_full, repaired=rs_full)
        say(f"{'Pr(|B|>|A|)':>24} {p_['P_dom']:>30.2f} {r_['P_dom']:>30.2f}")
        say(f"{'A>0 under all schemes':>24} {str(p_['A_all_pos']):>30} {str(r_['A_all_pos']):>30}")
        say(f"{'A sig>0 (x/P/Shapley)':>24} "
            f"{str((p_['A_x_sig'],p_['A_P_sig'],p_['A_S_sig'])):>30} "
            f"{str((r_['A_x_sig'],r_['A_P_sig'],r_['A_S_sig'])):>30}")
        say(f"{'B-A CI excludes 0':>24} {str(p_['BmA_sig']):>30} {str(r_['BmA_sig']):>30}")
        for lab, key in [("Pr(|B|>|A|)", "P_dom"), ("A>0 all schemes", "A_all_pos"),
                         ("B-A CI excludes 0", "BmA_sig")]:
            rec(block="1c/2 decomposition", quantity=f"{kind} {base}-2024: {lab}",
                published=p_[key], repaired=r_[key])

# ================================================================= Moran
say("")
say("=" * 104)
say("1d/6. MORAN'S I -- values, and the trend over 2006-2024 versus 2012-2024")
say("=" * 104)
adj = pd.read_csv("adjacency_final.csv")
adj.index = [str(c).zfill(4) for c in adj["Unnamed: 0"]]
W0 = adj.drop(columns=["Unnamed: 0"]); W0.columns = [str(c) for c in W0.columns]
W0 = W0.loc[CODES, CODES].to_numpy(float)
rs_ = W0.sum(1, keepdims=True); rs_[rs_ == 0] = 1
WM = {"row": W0 / rs_, "binary": W0.copy()}
E_I = -1.0 / (len(CODES) - 1)


def moran(v, Wm):
    z = v - v.mean()
    return (len(v) / Wm.sum()) * ((z @ Wm @ z) / (z @ z))


def perm_dist(v, Wm, rng_, n_perm=N_PERM, batch=1000):
    z = v - v.mean(); dn = z @ z; const = len(v) / Wm.sum()
    out = np.empty(n_perm); done = 0
    while done < n_perm:
        kk = min(batch, n_perm - done)
        Z = np.empty((kk, len(v)))
        for r in range(kk):
            Z[r] = rng_.permutation(z)
        out[done:done + kk] = const * ((Z @ Wm) * Z).sum(1) / dn
        done += kk
    return out


MOR = {}
for k, P in PANELS.items():
    for tag, Wm in WM.items():
        rngm = np.random.default_rng(SEED_MORAN)
        for y in YEARS:
            v = P[y].to_numpy() / POP[y] * 1e5
            I = moran(v, Wm); D_ = perm_dist(v, Wm, rngm)
            MOR[(k, tag, y)] = (I, (I - D_.mean()) / D_.std(ddof=1),
                                (1 + (D_ >= I).sum()) / (1 + N_PERM))
say(f"E[I] = -1/(N-1) = {E_I:.6f}")
for tag in WM:
    say("")
    say(f"--- {tag} weights ---")
    say(f"{'year':>5} | {'I pub':>8} {'I rep':>8} {'z rep':>7} {'p rep':>7} {'I>E[I]':>7}")
    for y in YEARS:
        ip = MOR[("published", tag, y)][0]
        ir, zr, pr = MOR[("repaired", tag, y)]
        say(f"{y:>5} | {ip:>8.4f} {ir:>8.4f} {zr:>7.2f} {pr:>7.4f} {str(ir>E_I):>7}")
        rec(block="1d Moran", quantity=f"{tag} I {y}", published=ip, repaired=ir)

say("")
say("--- trend permutation test (OLS slope per biennial wave, year labels permuted) ---")
say(f"{'weights':>8} {'window':>11} {'panel':>10} {'slope/wave':>11} {'two-sided p':>12} "
    f"{'n waves':>8}")
for tag in WM:
    for w0 in [2006, 2010, 2012]:
        yrs = [y for y in YEARS if y >= w0]
        vals = {}
        for k in PANELS:
            ys = np.array(yrs, float)
            iv = np.array([MOR[(k, tag, y)][0] for y in yrs])
            slope = np.polyfit(ys, iv, 1)[0] * 2
            rr = np.random.default_rng(SEED_MORAN + 7)
            null = np.array([np.polyfit(rr.permutation(ys), iv, 1)[0] * 2
                             for _ in range(N_PERM)])
            p = (1 + (np.abs(null) >= abs(slope)).sum()) / (1 + N_PERM)
            vals[k] = f"{slope:+.10f} (p={p:.10f})"
            say(f"{tag:>8} {w0}-2024 {k:>10} {slope:>+11.4f} {p:>12.4f} {len(yrs):>8}")
        rec(block="1d/6 Moran trend", quantity=f"{tag} slope per wave, {w0}-2024",
            published=vals["published"], repaired=vals["repaired"])
    say("")

# WRITES ONLY WHEN RUN, never when imported. c13_effective_supply_decomp.py imports this
# module for denom() and decompose(), and an import that rewrote these two files would make
# c13 a second writer of c01's artefacts -- the shape c09 exists to catch for figures. The
# computation above still runs on import, which is the cost of reusing it; nothing else in
# the repository imports this file.
if __name__ == "__main__":
    pd.DataFrame(recs).to_csv("outputs/audit/c01_round3_numbers.csv", index=False,
                              encoding="utf-8-sig")
    Path("outputs/audit/c01_round3_numbers.txt").write_text("\n".join(report) + "\n",
                                                            encoding="utf-8")
    print("\nwrote outputs/audit/c01_round3_numbers.csv and .txt")
