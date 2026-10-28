"""
STAGE C -- recompute every affected quantity on the repaired panel.

Seeds are IDENTICAL to the previous round so that any change is attributable to the
data repair and not to Monte Carlo noise:
    Gini series / paired bootstrap        seed 20260805   (a04)
    decomposition + B-A difference        seed 20260705   (a05 / cc_decomp_ci.py)
    Moran permutation                     seed 20260621   (a01 / cc_moran_supp.py)
    bottom decile                         deterministic

Everything is computed twice -- once on the published panel, once on the repaired one --
so the three-column comparison is produced by one program on one code path.

Writes outputs/audit/b06_recompute_after_crosswalk.csv (long form, machine readable)
       outputs/audit/b06_recompute_all.txt
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
N_BOOT = 10000
SEED_GINI = 20260805
SEED_DECOMP = 20260705
SEED_MORAN = 20260621
N_PERM = 9999
GROUPS = ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44",
          "45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84","85-89","90+"]
report, rows = [], []


def say(s=""):
    print(s)
    report.append(s)


def rec(item, quantity, published, repaired, note=""):
    rows.append({"item": item, "quantity": quantity, "published_panel": published,
                 "repaired_panel": repaired, "note": note})


# ---------------------------------------------------------------- panels
pub = pd.read_csv("data/superseded/physician_panel_smsa__PRE_REPAIR.csv")
pub["code4"] = pub["smsa_code"].apply(lambda c: str(c).zfill(4))
CODES = sorted(pub["code4"])
PUB = pub.set_index("code4")[[str(y) for y in YEARS]].astype(float).reindex(CODES)
PUB.columns = YEARS
rep = pd.read_csv("data/crosswalk/physician_panel_smsa_repaired.csv",
                  dtype={"code4": str}, encoding="utf-8-sig").set_index("code4")
NAME = rep["smsa_name"].to_dict()
REP = rep[[str(y) for y in YEARS]].astype(float).reindex(CODES)
REP.columns = YEARS
PANELS = {"published": PUB, "repaired": REP}


def total_pop(y):
    df = pd.read_csv(f"census_based/smsa_population_census_{y}.csv")
    df["code4"] = df["smsa_code"].apply(lambda c: str(c).zfill(4))
    return df.set_index("code4")["population"].reindex(CODES).astype(float).to_numpy()


POP = {y: total_pop(y) for y in YEARS}


def gini(v):
    xs = np.sort(np.asarray(v, float)); n = xs.size; ii = np.arange(1, n + 1)
    return (2 * (xs * ii).sum()) / (n * xs.sum()) - (n + 1) / n


def gini_rows(M):
    S = np.sort(M, axis=1); n = M.shape[1]; ii = np.arange(1, n + 1)
    return (2 * (S * ii).sum(1)) / (n * S.sum(1)) - (n + 1) / n


# ================================================================= 1. Gini series
say("=" * 100)
say("1. GINI SERIES (count and density), published vs repaired")
say("=" * 100)
say(f"{'year':>5} | {'count pub':>10} {'count rep':>10} {'diff':>8} | "
    f"{'dens pub':>9} {'dens rep':>9} {'diff':>8}")
G = {k: {"count": {}, "density": {}} for k in PANELS}
for y in YEARS:
    for k, P in PANELS.items():
        G[k]["count"][y] = gini(P[y].to_numpy())
        G[k]["density"][y] = gini(P[y].to_numpy() / POP[y] * 1e5)
    say(f"{y:>5} | {G['published']['count'][y]:>10.4f} {G['repaired']['count'][y]:>10.4f} "
        f"{G['repaired']['count'][y]-G['published']['count'][y]:>+8.4f} | "
        f"{G['published']['density'][y]:>9.4f} {G['repaired']['density'][y]:>9.4f} "
        f"{G['repaired']['density'][y]-G['published']['density'][y]:>+8.4f}")
    rec("Gini series", f"count Gini {y}", round(G["published"]["count"][y], 4),
        round(G["repaired"]["count"][y], 4))
    rec("Gini series", f"density Gini {y}", round(G["published"]["density"][y], 4),
        round(G["repaired"]["density"][y], 4))

# ================================================================= 2. paired bootstrap
say("")
say("=" * 100)
say("2. PAIRED (MATCHED-CIRCLE) BOOTSTRAP OF THE CHANGE  [item 6]")
say("=" * 100)
rng = np.random.default_rng(SEED_GINI)
IDX = rng.integers(0, len(CODES), size=(N_BOOT, len(CODES)))
say(f"{'panel':>10} {'measure':>8} {'window':>11} {'G base':>8} {'G end':>8} {'dG':>9} "
    f"{'95% CI':>22} {'Pr(dG<0)':>9}")
for k, P in PANELS.items():
    for measure in ["count", "density"]:
        for base, end in [(2006, 2024), (2010, 2024), (2016, 2024)]:
            a = P[base].to_numpy() if measure == "count" else P[base].to_numpy() / POP[base] * 1e5
            b = P[end].to_numpy() if measure == "count" else P[end].to_numpy() / POP[end] * 1e5
            d = gini_rows(b[IDX]) - gini_rows(a[IDX])
            lo, hi = np.percentile(d, [2.5, 97.5])
            say(f"{k:>10} {measure:>8} {base}-{end} {gini(a):>8.4f} {gini(b):>8.4f} "
                f"{gini(b)-gini(a):>+9.4f} [{lo:>+8.4f},{hi:>+8.4f}] "
                f"{(d<0).mean():>9.4f}")
            rec("Item 6 paired bootstrap", f"{measure} dG {base}-{end}",
                None, f"{gini(b)-gini(a):+.4f} [{lo:+.4f}, {hi:+.4f}]")
            if k == "repaired":
                rows[-1]["published_panel"] = None
        say("")

# recorded properly: rebuild the item-6 rows with both panels side by side
rows[:] = [r for r in rows if r["item"] != "Item 6 paired bootstrap"]
for measure in ["count", "density"]:
    for base, end in [(2006, 2024), (2010, 2024), (2016, 2024)]:
        vals = {}
        for k, P in PANELS.items():
            a = P[base].to_numpy() if measure == "count" else P[base].to_numpy() / POP[base] * 1e5
            b = P[end].to_numpy() if measure == "count" else P[end].to_numpy() / POP[end] * 1e5
            d = gini_rows(b[IDX]) - gini_rows(a[IDX])
            lo, hi = np.percentile(d, [2.5, 97.5])
            vals[k] = f"{gini(b)-gini(a):+.4f} [{lo:+.4f}, {hi:+.4f}]"
        rec("Item 6 paired bootstrap", f"{measure} Gini change {base}-{end}",
            vals["published"], vals["repaired"])

# ================================================================= 3. decomposition
say("=" * 100)
say("3. DECOMPOSITION AND B-A DIFFERENCE  [items 7, 9]")
say("=" * 100)


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
    # Round 38 (2026-10-06): the area age table, built once by cc_build_census_age.py through
    # the code-change tables. Summing census_age/age_muni_ over the 2024 map's codes, as this
    # did until then, lost every municipality whose code had changed since the census --
    # 熊本市 in 2010, the seven Hamamatsu wards in 2010-2020 -- and gave those areas the age
    # mix of what was left.
    t = pd.read_csv(f"census_age/age_circle_{cy}.csv", dtype={"code4": str}, encoding="utf-8")
    mat = t.set_index("code4").reindex(CODES)[GROUPS].to_numpy(float)
    tot = mat.sum(1, keepdims=True)
    return np.divide(mat, tot, out=np.full_like(mat, np.nan), where=tot > 0)


SH = {cy: sma_age_shares(cy) for cy in [2010, 2015, 2020]}


from needbase import fraction_from_sums, assert_unity


def _area_sums(cy):
    """Age-stated persons and published population per area, from the area age table.
    The census-edition guard is taken over the same areas as the shares (round 38)."""
    t = pd.read_csv(f"census_age/age_circle_{cy}.csv", dtype={"code4": str},
                    encoding="utf-8").set_index("code4").reindex(CODES)
    return t[GROUPS].to_numpy(float).sum(1), t["population"].to_numpy(float)



# The census-edition GUARD. THE CONSTRUCTION LIVES IN needbase.py -- see that file for
# what the fraction is, why 2015/2020 must come back exactly 1, and why it is NOT applied
# to the base. The resident denominator stays the published total, as it always did.
GFRAC = {cy: fraction_from_sums(*_area_sums(cy)) for cy in [2010, 2015, 2020]}
assert_unity(GFRAC, "circle")


def shares_for_year(ay):
    v = (SH[2010] + (SH[2015] - SH[2010]) * ((ay - 2010) / 5.0)) if ay <= 2015 else \
        (SH[2015] + (SH[2020] - SH[2015]) * ((ay - 2015) / 5.0))
    v = np.clip(v, 0, None); rs = v.sum(1, keepdims=True); rs[rs == 0] = np.nan
    return v / rs


rate = pd.read_csv("census_age/juryoritsu_2020.csv").set_index("age_group")["juryoritsu_per100k"]
W = rate.reindex(GROUPS).to_numpy(float)


def denom(ay, kind):
    raw = POP[ay]
    if kind == "raw":
        return raw
    # PRO RATA, for both weighted forms: the age shares multiply the PUBLISHED total,
    # distributing the unstated-age residents in the age-stated proportions. The
    # manuscript's Eq. (needweight); see needbase.py for why the age-stated base went.
    nm = (shares_for_year(ay) * W).sum(1)
    if kind == "e65":
        return raw * shares_for_year(ay)[:, 13:].sum(1)
    return raw * nm / (np.nansum(raw * nm) / np.nansum(raw))


rngd = np.random.default_rng(SEED_DECOMP)
IDXD = rngd.integers(0, len(CODES), size=(N_BOOT, len(CODES)))
DEC = {}
for k, P in PANELS.items():
    for kind in ["raw", "need"]:
        x0, x1 = P[2010].to_numpy(), P[2024].to_numpy()
        P0, P1 = denom(2010, kind), denom(2024, kind)
        d = {"00": x0 / P0, "10": x1 / P0, "01": x0 / P1, "11": x1 / P1}
        g = {kk: gini(v * 1e5) for kk, v in d.items()}
        Gb = {kk: gini_rows(v[IDXD] * 1e5) for kk, v in d.items()}
        tot = g["11"] - g["00"]
        A, B, I = g["10"] - g["00"], g["01"] - g["00"], tot - (g["10"] - g["00"]) - (g["01"] - g["00"])
        Ash = ((g["10"] - g["00"]) + (g["11"] - g["01"])) / 2
        Bsh = ((g["11"] - g["10"]) + (g["01"] - g["00"])) / 2
        Ab, Bb = Gb["10"] - Gb["00"], Gb["01"] - Gb["00"]
        Asb = ((Gb["10"] - Gb["00"]) + (Gb["11"] - Gb["01"])) / 2
        Bsb = ((Gb["11"] - Gb["10"]) + (Gb["01"] - Gb["00"])) / 2
        ci = lambda v: np.percentile(v, [2.5, 97.5])
        DEC[(k, kind)] = dict(
            g0=g["00"], g1=g["11"], tot=tot, A=A, B=B, I=I, Ash=Ash, Bsh=Bsh,
            A_ci=ci(Ab), B_ci=ci(Bb), Ash_ci=ci(Asb), Bsh_ci=ci(Bsb),
            A_xfirst_pct=100 * A / tot, A_Pfirst_pct=100 * (tot - B) / tot,
            A_shap_pct=100 * Ash / tot,
            BmA=B - A, BmA_ci=ci(Bb - Ab),
            absBmA=abs(B) - abs(A), absBmA_ci=ci(np.abs(Bb) - np.abs(Ab)),
            P_dom=float((np.abs(Bb) > np.abs(Ab)).mean()))

for kind in ["raw", "need"]:
    say("")
    say(f"--- {kind} denominator, 2010->2024 ---")
    say(f"{'':>22} {'published':>28} {'repaired':>28}")
    p_, r_ = DEC[("published", kind)], DEC[("repaired", kind)]
    for lab, key in [("G(2010)", "g0"), ("G(2024)", "g1"), ("total dG", "tot"),
                     ("A (physician)", "A"), ("B (denominator)", "B"), ("interaction", "I"),
                     ("A Shapley", "Ash"), ("B Shapley", "Bsh")]:
        say(f"{lab:>22} {p_[key]:>28.4f} {r_[key]:>28.4f}")
        rec("Items 7/9 decomposition", f"{kind}: {lab}", round(p_[key], 4), round(r_[key], 4))
    for lab, key in [("A 95% CI", "A_ci"), ("B 95% CI", "B_ci"),
                     ("A Shapley 95% CI", "Ash_ci"), ("B Shapley 95% CI", "Bsh_ci"),
                     ("B-A 95% CI", "BmA_ci"), ("|B|-|A| 95% CI", "absBmA_ci")]:
        ps = f"[{p_[key][0]:+.4f}, {p_[key][1]:+.4f}]"
        rs = f"[{r_[key][0]:+.4f}, {r_[key][1]:+.4f}]"
        say(f"{lab:>22} {ps:>28} {rs:>28}")
        rec("Items 7/9 decomposition", f"{kind}: {lab}", ps, rs)
    say(f"{'A share x-first':>22} {p_['A_xfirst_pct']:>27.1f}% {r_['A_xfirst_pct']:>27.1f}%")
    say(f"{'A share P-first':>22} {p_['A_Pfirst_pct']:>27.1f}% {r_['A_Pfirst_pct']:>27.1f}%")
    say(f"{'A share Shapley':>22} {p_['A_shap_pct']:>27.1f}% {r_['A_shap_pct']:>27.1f}%")
    say(f"{'B-A':>22} {p_['BmA']:>28.4f} {r_['BmA']:>28.4f}")
    say(f"{'Pr(|B|>|A|)':>22} {p_['P_dom']:>28.2f} {r_['P_dom']:>28.2f}")
    for lab, key in [("A share x-first %", "A_xfirst_pct"), ("A share P-first %", "A_Pfirst_pct"),
                     ("A share Shapley %", "A_shap_pct"), ("Pr(|B|>|A|)", "P_dom")]:
        rec("Items 7/9 decomposition", f"{kind}: {lab}", round(p_[key], 2), round(r_[key], 2))

# ================================================================= 4. Moran
say("")
say("=" * 100)
say("4. MORAN'S I  [items 1, 10]")
say("=" * 100)
adj = pd.read_csv("adjacency_final.csv")
adj.index = [str(c).zfill(4) for c in adj["Unnamed: 0"]]
W0 = adj.drop(columns=["Unnamed: 0"]); W0.columns = [str(c) for c in W0.columns]
W0 = W0.loc[CODES, CODES].to_numpy(float)
rs_ = W0.sum(1, keepdims=True); rs_[rs_ == 0] = 1
W_ROW, W_BIN = W0 / rs_, W0.copy()
E_I = -1.0 / (len(CODES) - 1)


def moran(v, Wm):
    z = v - v.mean()
    return (len(v) / Wm.sum()) * ((z @ Wm @ z) / (z @ z))


def perm_dist(v, Wm, rng, n_perm=N_PERM, batch=1000):
    z = v - v.mean(); denom_ = z @ z; const = len(v) / Wm.sum()
    out = np.empty(n_perm); done = 0
    while done < n_perm:
        kk = min(batch, n_perm - done)
        Z = np.empty((kk, len(v)))
        for r in range(kk):
            Z[r] = rng.permutation(z)
        out[done:done + kk] = const * ((Z @ Wm) * Z).sum(1) / denom_
        done += kk
    return out


MOR = {}
for k, P in PANELS.items():
    rngm = np.random.default_rng(SEED_MORAN)
    for tag, Wm in [("row", W_ROW), ("binary", W_BIN)]:
        for y in YEARS:
            d = P[y].to_numpy() / POP[y] * 1e5
            I = moran(d, Wm); D_ = perm_dist(d, Wm, rngm)
            MOR[(k, tag, y)] = (I, (I - D_.mean()) / D_.std(ddof=1),
                                (1 + (D_ >= I).sum()) / (1 + N_PERM))
say(f"E[I] = -1/(N-1) = {E_I:.6f}")
for tag in ["row", "binary"]:
    say("")
    say(f"--- {tag}-standardised weights ---" if tag == "row" else "--- binary weights ---")
    say(f"{'year':>5} | {'I pub':>8} {'z pub':>7} {'p pub':>7} | {'I rep':>8} {'z rep':>7} "
        f"{'p rep':>7} | {'I > E[I] rep':>13}")
    for y in YEARS:
        ip, zp, pp = MOR[("published", tag, y)]
        ir, zr, pr = MOR[("repaired", tag, y)]
        say(f"{y:>5} | {ip:>8.4f} {zp:>7.2f} {pp:>7.4f} | {ir:>8.4f} {zr:>7.2f} {pr:>7.4f} | "
            f"{str(ir > E_I):>13}")
        rec("Items 1/10 Moran", f"{tag} I {y}", round(ip, 4), round(ir, 4))
        rec("Items 1/10 Moran", f"{tag} z {y}", round(zp, 2), round(zr, 2))
        rec("Items 1/10 Moran", f"{tag} pseudo-p {y}", round(pp, 4), round(pr, 4))

# trend permutation test
say("")
say("--- trend permutation test (OLS slope, year labels permuted, 9,999 draws) ---")
for tag in ["row", "binary"]:
    for k in PANELS:
        ys = np.array(YEARS, float)
        iv = np.array([MOR[(k, tag, y)][0] for y in YEARS])
        slope = np.polyfit(ys, iv, 1)[0] * 2       # per biennial wave
        rr = np.random.default_rng(SEED_MORAN + 7)
        null = np.array([np.polyfit(rr.permutation(ys), iv, 1)[0] * 2 for _ in range(N_PERM)])
        p = (1 + (np.abs(null) >= abs(slope)).sum()) / (1 + N_PERM)
        say(f"  {tag:>7} {k:>10}: slope {slope:+.4f} per wave, two-sided p = {p:.4f}")
        rec("Item 10 Moran trend", f"{tag} slope per wave", None, round(slope, 4))
        rows[-1]["published_panel"] = round(slope, 4) if k == "published" else rows[-1]["published_panel"]
rows[:] = [r for r in rows if r["item"] != "Item 10 Moran trend"]
for tag in ["row", "binary"]:
    vals = {}
    for k in PANELS:
        ys = np.array(YEARS, float)
        iv = np.array([MOR[(k, tag, y)][0] for y in YEARS])
        slope = np.polyfit(ys, iv, 1)[0] * 2
        rr = np.random.default_rng(SEED_MORAN + 7)
        null = np.array([np.polyfit(rr.permutation(ys), iv, 1)[0] * 2 for _ in range(N_PERM)])
        vals[k] = f"{slope:+.4f} (p={(1 + (np.abs(null) >= abs(slope)).sum())/(1+N_PERM):.4f})"
    rec("Item 10 Moran trend", f"{tag} slope per wave", vals["published"], vals["repaired"])

# ================================================================= 5. bottom decile
say("")
say("=" * 100)
say("5. BOTTOM DECILE / QUINTILE  [item 16]")
say("=" * 100)
DENS = {k: {y: P[y].to_numpy() / POP[y] * 1e5 for y in YEARS} for k, P in PANELS.items()}
bd_rows = []
for base in [2006, 2016]:
    for cut, frac in [("decile", 0.10), ("quintile", 0.20)]:
        kk = int(round(len(CODES) * frac))
        for k, P in PANELS.items():
            fixed = np.argsort(DENS[k][base], kind="stable")[:kk]
            x0, p0 = P[base].to_numpy()[fixed].sum(), POP[base][fixed].sum()
            x1, p1 = P[2024].to_numpy()[fixed].sum(), POP[2024][fixed].sum()
            nx0, nx1 = P[base].to_numpy().sum(), P[2024].to_numpy().sum()
            np0, np1 = POP[base].sum(), POP[2024].sum()
            bd_rows.append({"panel": k, "base": base, "cut": cut, "k": kk,
                            "members": set(np.array(CODES)[fixed]),
                            "phys_pct": 100 * (x1 / x0 - 1), "pop_pct": 100 * (p1 / p0 - 1),
                            "dens_pct": 100 * ((x1 / p1) / (x0 / p0) - 1),
                            "share_stock_0": 100 * x0 / nx0, "share_stock_1": 100 * x1 / nx1,
                            "share_pop_0": 100 * p0 / np0, "share_pop_1": 100 * p1 / np1,
                            "nat_phys_pct": 100 * (nx1 / nx0 - 1)})
BD = pd.DataFrame(bd_rows)
for base in [2006, 2016]:
    for cut in ["decile", "quintile"]:
        s = BD[(BD.base == base) & (BD.cut == cut)]
        p_, r_ = s[s.panel == "published"].iloc[0], s[s.panel == "repaired"].iloc[0]
        say("")
        say(f"--- lowest {cut}, fixed cohort defined on {base}, followed to 2024 (k={p_.k}) ---")
        say(f"{'':>26} {'published':>14} {'repaired':>14}")
        for lab, key in [("physicians %", "phys_pct"), ("population %", "pop_pct"),
                         ("density %", "dens_pct"),
                         (f"share of stock {base} %", "share_stock_0"),
                         ("share of stock 2024 %", "share_stock_1"),
                         (f"share of pop {base} %", "share_pop_0"),
                         ("share of pop 2024 %", "share_pop_1"),
                         ("national physicians %", "nat_phys_pct")]:
            say(f"{lab:>26} {p_[key]:>+14.2f} {r_[key]:>+14.2f}")
            rec("Item 16 bottom group", f"{base} base, {cut}: {lab}",
                round(float(p_[key]), 2), round(float(r_[key]), 2))
        swapped_out = sorted(p_.members - r_.members)
        swapped_in = sorted(r_.members - p_.members)
        say(f"  cohort membership changed: {len(swapped_out)} circles leave, "
            f"{len(swapped_in)} enter ({len(p_.members & r_.members)} common)")
        if swapped_out:
            say(f"    leave : " + ", ".join(f"{c}({NAME.get(c,'')})" for c in swapped_out))
            say(f"    enter : " + ", ".join(f"{c}({NAME.get(c,'')})" for c in swapped_in))
        rec("Item 16 bottom group", f"{base} base, {cut}: cohort members changed",
            None, f"{len(swapped_out)} out / {len(swapped_in)} in")

pd.DataFrame(rows).to_csv("outputs/audit/b06_recompute_after_crosswalk.csv",
                          index=False, encoding="utf-8-sig")
Path("outputs/audit/b06_recompute_all.txt").write_text("\n".join(report) + "\n",
                                                       encoding="utf-8")
print("\nwrote outputs/audit/b06_recompute_after_crosswalk.csv and b06_recompute_all.txt")
