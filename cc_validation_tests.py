"""
PHASE 3: three area-level (ecological) validation tests for the borrowed-access
hypothesis. All associations ecological, no causal claim. Each test: coefficient(s)
+ bootstrap 95% CI + one figure. Writes to outputs/validation/.

test A: effective supply density vs actual INPATIENT utilisation (patient survey,
        SMA level = inpatient only; outpatient is prefecture-only, excluded).
test B: net secondary-practice inflow vs under-served (無医地区) status.
        (i) binary logistic [PRIMARY, extensive margin] + (ii) Spearman + (iii) NB
        with log-pop offset; 3 rounds for time robustness; independent net-outflow-
        ratio proxy (demand-side) reported separately, NOT combined with (i)-(iii).
test C: prefecture-level residence-vs-worksite borrowing (47 units, COARSE proxy).

NO interpretation / Results wording here — numbers only.
"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib as mpl, matplotlib.pyplot as plt
import cc_circle_common as cc

OUT = "outputs/validation"; os.makedirs(OUT, exist_ok=True)
PS = "data/raw/external/patient_survey"
SEED = 20260703
rng = np.random.default_rng(SEED)
NB = 5000

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.right": False,
    "axes.spines.top": False, "axes.linewidth": 0.7, "legend.frameon": False, "figure.dpi": 150})
C_EFF = "#e08214"; C_PRIM = "#5c677d"; C_DENS = "#c1440e"; C_STOCK = "#1b4965"; GRID = "#d9d9d9"


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    print("  saved fig", name)


def boot_ci(fn, *arrs, n=NB, gen=None):
    """percentile 95% CI of statistic fn over paired circle resamples.
    gen: optional dedicated Generator; tests B/proxy/C use per-test seeded
    generators so their CIs are decoupled from upstream RNG consumption
    (test A keeps the global stream, whose values are frozen)."""
    g = gen if gen is not None else rng
    N = len(arrs[0]); out = []
    for _ in range(n):
        idx = g.integers(0, N, N)
        try:
            out.append(fn(*[a[idx] for a in arrs]))
        except Exception:
            pass
    out = np.array([x for x in out if np.isfinite(x)])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def pearson(x, y):
    return stats.pearsonr(x, y)[0]


def spearman(x, y):
    return stats.spearmanr(x, y)[0]


results = {}

# ======================================================================
# TEST A — effective supply density vs inpatient utilisation
# ======================================================================
def test_A(ps_year, eff_year, cens_year, tag):
    ep = cc.eff_panel(eff_year)
    pop = cc.census_pop(cens_year)
    eff_density = ep["effective_supply_theta05"].astype(float) / pop * 1e5
    prim_density = ep["primary_stock"].astype(float) / pop * 1e5
    metf = f"{PS}/meta/{RES_META[ps_year]}.json"; valf = f"{PS}/{RES_VAL[ps_year]}.json"
    facm = f"{PS}/meta/{FAC_META[ps_year]}.json"; facv = f"{PS}/{FAC_VAL[ps_year]}.json"
    res_inp = cc.ps_series(metf, valf, cat02="1", agg="sum") * 1000.0     # 千人 -> persons
    fac_inp = cc.ps_series(facm, facv, cat02="1", agg="sum") * 1000.0
    util_res = res_inp / pop * 1e5    # residence inpatient census per 100k residents
    util_fac = fac_inp / pop * 1e5    # facility inpatient census per 100k residents
    df = pd.DataFrame({"eff": eff_density, "prim": prim_density,
                       "util_res": util_res, "util_fac": util_fac}).replace([np.inf, -np.inf], np.nan).dropna()
    res = {}
    # 2x2 contrast: (effective vs primary density) x (facility-provided vs resident-consumed)
    for xname in ["eff", "prim"]:
        x = df[xname].to_numpy()
        for yname in ["util_fac", "util_res"]:
            y = df[yname].to_numpy()
            r = pearson(x, y); rho = spearman(x, y)
            rlo, rhi = boot_ci(pearson, x, y); slo, shi = boot_ci(spearman, x, y)
            res[f"{xname}~{yname}"] = dict(n=len(df), pearson=round(r, 3), pearson_CI=[round(rlo, 3), round(rhi, 3)],
                                           spearman=round(rho, 3), spearman_CI=[round(slo, 3), round(shi, 3)],
                                           # unrounded, for checking a 2-decimal page against
                                           # the source: 0.165 at 3 decimals is 0.16497
                                           full=dict(pearson=float(r), pearson_CI=[rlo, rhi],
                                                     spearman=float(rho), spearman_CI=[slo, shi]))
    # paired difference: does effective track PROVISION better than primary does?
    # bootstrap CI of  r(eff,fac) - r(prim,fac)  and  r(eff,res) - r(prim,res)
    for yname in ["util_fac", "util_res"]:
        y = df[yname].to_numpy(); xe = df["eff"].to_numpy(); xp = df["prim"].to_numpy()
        def _dr(a, b, yy):
            return pearson(a, yy) - pearson(b, yy)
        dlo, dhi = boot_ci(_dr, xe, xp, y)
        res[f"delta_r_eff_minus_prim~{yname}"] = dict(
            delta=round(pearson(xe, y) - pearson(xp, y), 3), CI=[round(dlo, 3), round(dhi, 3)],
            full=dict(delta=float(pearson(xe, y) - pearson(xp, y)), CI=[dlo, dhi]))
    # figure: eff density vs facility inpatient utilisation
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.scatter(df["eff"], df["util_fac"], s=9, c=C_EFF, alpha=0.7, edgecolors="none")
    b, a = np.polyfit(df["eff"], df["util_fac"], 1)
    xs = np.linspace(df["eff"].min(), df["eff"].max(), 50)
    ax.plot(xs, a + b * xs, color=C_DENS, lw=1.2)
    ax.set_xlabel("Effective physician density (per 100k, %d)" % eff_year)
    ax.set_ylabel("Inpatient census, facility (per 100k, %d)" % ps_year)
    ax.grid(True, color=GRID, lw=0.4)
    ax.set_title("Test A  (r=%.2f, $\\rho$=%.2f, n=%d)" % (res["eff~util_fac"]["pearson"], res["eff~util_fac"]["spearman"], len(df)))
    save(fig, f"figA_effdensity_utilisation_{tag}"); plt.close(fig)
    return res


RES_META = {2023: "0004026138", 2020: "0004002360"}
RES_VAL = {2023: "ps2023_res_tot_0004026138", 2020: "ps2020_res_tot_0004002360"}
FAC_META = {2023: "0004026139", 2020: "0004002361"}
FAC_VAL = {2023: "ps2023_fac_tot_0004026139", 2020: "ps2020_fac_tot_0004002361"}

print("=" * 80, "\nTEST A: effective supply density vs inpatient utilisation (INPATIENT ONLY)")
results["A_2023x2024"] = test_A(2023, 2024, 2024, "2023x2024")
results["A_2020x2020"] = test_A(2020, 2020, 2020, "2020x2020")
for k in ["A_2023x2024", "A_2020x2020"]:
    print(f"  {k}:")
    for yn, d in results[k].items():
        if yn.startswith("delta"):
            print(f"    {yn}: delta_r={d['delta']} CI{d['CI']}")
        else:
            print(f"    {yn}: Pearson {d['pearson']} CI{d['pearson_CI']}  Spearman {d['spearman']} CI{d['spearman_CI']}  (n={d['n']})")

# ======================================================================
# TEST B — net secondary-inflow vs under-served (無医地区)
# ======================================================================
def test_B(und_col, net_year, cens_year, tag, make_fig=False):
    gen_b = np.random.default_rng([20260704, int(und_col[-4:])])   # decoupled per-round rng
    ep = cc.eff_panel(net_year); und = cc.underserved()
    pop = cc.census_pop(cens_year)
    net = ep["net_inflow"].astype(float)
    mui = und[und_col].astype(float).fillna(0)
    df = pd.DataFrame({"net": net, "mui": mui, "pop": pop}).dropna()
    df = df[df["pop"] > 0]
    y_bin = (df["mui"] > 0).astype(int).to_numpy()
    netv = df["net"].to_numpy(); z = (netv - netv.mean()) / netv.std()
    muiv = df["mui"].to_numpy(); logpop = np.log(df["pop"].to_numpy())
    out = {"n": len(df), "n_underserved": int(y_bin.sum()), "n_zero": int((y_bin == 0).sum())}
    # (i) logistic  P(mui>0) ~ z(net)
    m = sm.Logit(y_bin, sm.add_constant(z)).fit(disp=0)
    def _lfit(zz, yy):
        return sm.Logit(yy, sm.add_constant(zz)).fit(disp=0).params[1]
    llo, lhi = boot_ci(_lfit, z, y_bin, n=2000, gen=gen_b)
    out["logit_beta_perSD"] = round(float(m.params[1]), 3)
    out["logit_OR_perSD"] = round(float(np.exp(m.params[1])), 3)
    out["logit_OR_CI"] = [round(float(np.exp(llo)), 3), round(float(np.exp(lhi)), 3)]
    out["logit_p"] = round(float(m.pvalues[1]), 4)
    out["pointbiserial_r"] = round(pearson(y_bin.astype(float), netv), 3)
    # (ii) spearman(net, mui)
    rho = spearman(netv, muiv); slo, shi = boot_ci(spearman, netv, muiv, gen=gen_b)
    out["spearman"] = round(float(rho), 3); out["spearman_CI"] = [round(slo, 3), round(shi, 3)]
    # (iii) NB  mui ~ z(net) + offset(log pop). Stable GLM-NB (fixed dispersion
    #       alpha=1); plain NB is a robustness leg only (with ~40% structural zeros
    #       a hurdle/ZINB is the honest spec — flagged, not fit here).
    NBfam = sm.families.NegativeBinomial(alpha=1.0)
    nb = sm.GLM(muiv, sm.add_constant(z), family=NBfam, offset=logpop).fit()
    def _nbfit(zz, yy, off):
        return sm.GLM(yy, sm.add_constant(zz), family=NBfam, offset=off).fit().params[1]
    nlo, nhi = boot_ci(_nbfit, z, muiv, logpop, n=1500, gen=gen_b)
    out["nb_beta_perSD"] = round(float(nb.params[1]), 3)
    out["nb_IRR_perSD"] = round(float(np.exp(nb.params[1])), 3)
    out["nb_IRR_CI"] = [round(float(np.exp(nlo)), 3), round(float(np.exp(nhi)), 3)]
    out["nb_p"] = round(float(nb.pvalues[1]), 4)
    if make_fig:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.2, 3.0))
        g0 = netv[y_bin == 0]; g1 = netv[y_bin == 1]
        a1.boxplot([g0, g1], labels=["no\nunderserved", "has\nunderserved"], widths=0.5,
                   showfliers=False, medianprops=dict(color=C_DENS))
        a1.scatter(np.random.default_rng(1).normal(1, 0.06, len(g0)), g0, s=6, c=C_PRIM, alpha=0.4)
        a1.scatter(np.random.default_rng(2).normal(2, 0.06, len(g1)), g1, s=6, c=C_EFF, alpha=0.5)
        a1.axhline(0, color="0.5", lw=0.6, ls="--"); a1.set_ylabel("Net secondary-practice inflow (physicians)")
        a1.grid(True, axis="y", color=GRID, lw=0.4)
        a1.set_title("Extensive margin (OR/SD=%.2f)" % out["logit_OR_perSD"])
        order = np.argsort(netv)
        a2.scatter(netv, muiv, s=10, c=C_EFF, alpha=0.6, edgecolors="none")
        a2.set_xlabel("Net secondary-practice inflow"); a2.set_ylabel("Number of underserved areas (mui-chiku)")
        a2.grid(True, color=GRID, lw=0.4); a2.set_title("Intensity ($\\rho$=%.2f)" % out["spearman"])
        fig.suptitle("Test B  net inflow vs under-served  (%s, n=%d)" % (tag, len(df)), fontsize=9)
        save(fig, f"figB_netinflow_underserved_{tag}"); plt.close(fig)
    return out


print("=" * 80, "\nTEST B: net secondary-inflow vs under-served (無医地区)")
results["B_main_2022"] = test_B("mui_2022", 2022, 2022, "2022", make_fig=True)
results["B_rob_2019"] = test_B("mui_2019", 2020, 2020, "2019")   # underserved 2019 x nearest net 2020
results["B_rob_2014"] = test_B("mui_2014", 2016, 2016, "2014")   # underserved 2014 x nearest net 2016
for k in ["B_main_2022", "B_rob_2019", "B_rob_2014"]:
    d = results[k]
    print(f"  {k}: n={d['n']} underserved={d['n_underserved']} zero={d['n_zero']}")
    print(f"     (i) logistic OR/SD={d['logit_OR_perSD']} CI{d['logit_OR_CI']} p={d['logit_p']}  point-biserial r={d['pointbiserial_r']}")
    print(f"     (ii) Spearman={d['spearman']} CI{d['spearman_CI']}")
    print(f"     (iii) NB IRR/SD={d.get('nb_IRR_perSD')} CI{d.get('nb_IRR_CI')} p={d.get('nb_p')}" + (f"  ERR:{d.get('nb_error')}" if d.get('nb_error') else ""))

# independent demand-side proxy: net inpatient OUTFLOW ratio (outflow% - inflow%) vs net inflow
def proxy(flow_year, net_year, tag):
    fm = f"{PS}/meta/{FLOW_META[flow_year]}.json"; fv = f"{PS}/{FLOW_VAL[flow_year]}.json"
    rm = f"{PS}/meta/{RES_META[flow_year]}.json"; rv = f"{PS}/{RES_VAL[flow_year]}.json"
    w335 = cc.ps_raw335(rm, rv, cat02="1")   # residence inpatient count, 335 basis, as Akita weight
    inflow = cc.ps_series(fm, fv, cat02="1", agg="wmean", weight335=w335)
    outflow = cc.ps_series(fm, fv, cat02="2", agg="wmean", weight335=w335)
    ep = cc.eff_panel(net_year); net = ep["net_inflow"].astype(float)
    df = pd.DataFrame({"net": net, "netout": (outflow - inflow)}).dropna()
    gen_p = np.random.default_rng([20260704, flow_year])
    rho = spearman(df["net"].to_numpy(), df["netout"].to_numpy())
    lo, hi = boot_ci(spearman, df["net"].to_numpy(), df["netout"].to_numpy(), gen=gen_p)
    return dict(n=len(df), spearman=round(float(rho), 3), spearman_CI=[round(lo, 3), round(hi, 3)])


FLOW_META = {2023: "0004026140", 2020: "0004002362"}
FLOW_VAL = {2023: "ps2023_flow_0004026140", 2020: "ps2020_flow_0004002362"}
results["B_proxy_2020"] = proxy(2020, 2020, "2020")
results["B_proxy_2023"] = proxy(2023, 2024, "2023")
print("  demand-side proxy (net outflow ratio vs net secondary-inflow) — INDEPENDENT signal:")
for k in ["B_proxy_2020", "B_proxy_2023"]:
    print(f"     {k}: Spearman={results[k]['spearman']} CI{results[k]['spearman_CI']} (n={results[k]['n']})")

# ======================================================================
# TEST C — prefecture residence vs worksite borrowing (47, COARSE)
# ======================================================================
def test_C(year):
    import csv
    raw = open(f"data/raw/external/physician_residence/phys_juusho_juugyou_{year}.csv", encoding="cp932").read()
    z2h = str.maketrans("０１２３４５６７８９", "0123456789")
    rows = list(csv.reader(raw.splitlines()))
    import re
    recs = {}
    for r in rows:
        if not r:
            continue
        c0 = r[0].strip().translate(z2h)
        m = re.match(r"^(\d{2})\D", c0)
        if m and 1 <= int(m.group(1)) <= 47 and len(r) > 3:
            try:
                res = float(r[2]); work = float(r[3])
                recs[m.group(1).zfill(2)] = (res, work)
            except ValueError:
                pass
    # Prefecture population = the paper's own fixed-map area populations summed by prefecture
    # (round 39, internal work record). This read pref_population.csv, a file
    # of unknown provenance that differs from the confirmed census in every prefecture; it is
    # now in the archive beside the project root. The area populations reconcile to the
    # published prefecture totals of the census (q28).
    _a = pd.read_csv(f"census_based/smsa_population_census_{year}.csv")
    popm = _a.groupby(_a["smsa_code"].astype(str).str.zfill(4).str[:2])["population"].sum()
    rr = []
    for pc, (res, work) in recs.items():
        pop = float(popm.get(pc, np.nan))
        rr.append((pc, res, work, (work - res) / res, work / pop * 1e5))
    d = pd.DataFrame(rr, columns=["pc", "res", "work", "borrow_idx", "density"]).dropna()
    gen_c = np.random.default_rng([20260704, 47, year])
    rho = spearman(d["borrow_idx"].to_numpy(), d["density"].to_numpy())
    lo, hi = boot_ci(spearman, d["borrow_idx"].to_numpy(), d["density"].to_numpy(), gen=gen_c)
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.scatter(d["density"], d["borrow_idx"] * 100, s=14, c=C_STOCK, alpha=0.75, edgecolors="none")
    ax.axhline(0, color="0.5", lw=0.6, ls="--")
    ax.set_xlabel("Physician density, worksite (per 100k, %d)" % year)
    ax.set_ylabel("Net worksite import (%, (work−res)/res)")
    ax.grid(True, color=GRID, lw=0.4)
    ax.set_title("Test C  prefectures (n=%d, $\\rho$=%.2f)  COARSE" % (len(d), rho))
    save(fig, f"figC_borrowing_pref_{year}"); plt.close(fig)
    return dict(n=len(d), spearman=round(float(rho), 3), spearman_CI=[round(lo, 3), round(hi, 3)],
                mean_borrow_pct=round(float(d["borrow_idx"].mean() * 100), 2),
                n_import=int((d["borrow_idx"] > 0).sum()), n_export=int((d["borrow_idx"] < 0).sum()))


print("=" * 80, "\nTEST C: prefecture residence-vs-worksite borrowing (COARSE, 47 units)")
results["C_2024"] = test_C(2024)
d = results["C_2024"]
print(f"  2024: n={d['n']}  Spearman(borrow_idx, density)={d['spearman']} CI{d['spearman_CI']}")
print(f"        mean net worksite import={d['mean_borrow_pct']}%  import>0:{d['n_import']}  export<0:{d['n_export']}")

json.dump(results, open(f"{OUT}/validation_results.json", "w"), ensure_ascii=False, indent=2)
print("\nsaved", f"{OUT}/validation_results.json")
