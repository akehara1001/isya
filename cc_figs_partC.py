"""
PART C previews: figA (2-panel provision/consumption), figB (symlog boxplot fix),
figC (labelled extremes), Fig3 (font enlargement + overlap fix).
Writes ONLY to fig_preview/ (pending user confirmation) + dumps plotting data CSVs.
Font floors: axis labels >=11, ticks >=9, titles >=12, annotations >=10.
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import statsmodels.api as sm
import cc_circle_common as cc

# Default to where the manuscript actually reads its figures. Defaulting to a
# preview directory meant a clean re-run left paper/figures/ empty and the shipped
# figures depended on someone remembering to copy them across.
OUT = os.environ.get("CC_FIGDIR", "paper/figures"); os.makedirs(OUT, exist_ok=True)
ORANGE, BLUE, RED = "#E8A03D", "#4A6274", "#C0384B"
GRID = "#d9d9d9"
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "none",
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 10,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8,
    "legend.frameon": False, "figure.dpi": 150})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    print("saved", name)


SIDE = []


def side(fig_name, item, value):
    """Record a number that is actually drawn, taken from the same variable the
    drawing call receives -- never recomputed afterwards."""
    SIDE.append({'figure': fig_name, 'item': item, 'value': float(value)})


# ================= data assembly + dump =================
PS = "data/raw/external/patient_survey"
pop24 = cc.census_pop(2024)
ep24 = cc.eff_panel(2024)
ep22 = cc.eff_panel(2022)
und = cc.underserved()
u_fac = cc.ps_series(f"{PS}/meta/0004026139.json", f"{PS}/ps2023_fac_tot_0004026139.json", "1") * 1000 / pop24 * 1e5
u_res = cc.ps_series(f"{PS}/meta/0004026138.json", f"{PS}/ps2023_res_tot_0004026138.json", "1") * 1000 / pop24 * 1e5
D = pd.DataFrame({
    "smsa_code": cc.CODES,
    "pref": und["pref"].values, "sma": und["sma"].values,
    "d_eff_2024": (ep24["effective_supply_theta05"].astype(float) / pop24 * 1e5).values,
    "d_prim_2024": (ep24["primary_stock"].astype(float) / pop24 * 1e5).values,
    "u_fac_2023": u_fac.values, "u_res_2023": u_res.values,
    "net_inflow_2022": ep22["net_inflow"].astype(float).values,
    "net_inflow_2024": ep24["net_inflow"].astype(float).values,
    "mui_2022": und["mui_2022"].values,
    "pop_2024": pop24.values,
})
D.to_csv("outputs/validation/fig_validation_data.csv", index=False, encoding="utf-8-sig")
print("dumped fig_validation_data.csv", D.shape)


def ols_band(x, y, ax, color=RED):
    X = sm.add_constant(x)
    fit = sm.OLS(y, X).fit()
    xs = np.linspace(x.min(), x.max(), 120)
    pred = fit.get_prediction(sm.add_constant(xs))
    mean = pred.predicted_mean; lo, hi = pred.conf_int().T
    ax.fill_between(xs, lo, hi, color=color, alpha=0.15, lw=0, zorder=2)
    ax.plot(xs, mean, color=color, lw=1.6, zorder=3)


# ================= figA : two-panel provision vs consumption =================
def figA(dens_col="d_eff_2024", name="FigS1_validation_testA",
         xlab="Effective physician density (per 100 000, 2024)"):
    """Test A, drawn against whichever density the citing paper actually reports.

    Two variants are emitted from the SAME assembled frame, because the two documents
    report different densities and a figure must show the quantity its own text names:

      d_eff_2024  -> FigS1_validation_testA          the source manuscript, whose caption
                                                     and Table S7 reading both say
                                                     "effective density"
      d_prim_2024 -> FigS1_validation_testA_primary  paper A, which after the split reports
                                                     no effective-supply series at all --
                                                     "effective" appears 0 times in its body

    This is a change of drawn variable, not a new analysis: both columns are already in D,
    both are already dumped to outputs/validation/fig_validation_data.csv, and the
    coefficients of both are already printed in Table S7 (d: 0.47 / 0.58 and -0.04 / 0.03).
    """
    d = D.dropna(subset=[dens_col, "u_fac_2023", "u_res_2023"])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), sharex=True, sharey=True)
    # r, rho and n were hardcoded into these titles. They are now computed from the
    # same arrays the scatter receives, so the caption cannot disagree with the points.
    from scipy.stats import pearsonr, spearmanr
    panels = [("u_fac_2023", "Provision",
               "Inpatient census at area facilities\n(per 100 000 residents, 2023)"),
              ("u_res_2023", "Consumption",
               "Inpatient census of area residents\n(per 100 000 residents, 2023)")]
    for ax, (col, lead, ylab) in zip(axes, panels):
        x = d[dens_col].to_numpy(); y = d[col].to_numpy()
        _r = float(pearsonr(x, y)[0])
        _rho = float(spearmanr(x, y)[0])
        side(name, col + "_pearson", _r)
        side(name, col + "_spearman", _rho)
        title = lead + " ($r=" + format(_r, '.2f') + "$, $\\rho=" \
                + format(_rho, '.2f') + "$)"
        ax.scatter(x, y, s=16, c=ORANGE, alpha=0.5, edgecolors="none", zorder=4)
        ols_band(x, y, ax)
        ax.set_title(title, fontsize=12, loc="left")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(True, color=GRID, lw=0.4, zorder=0)
        ax.tick_params(labelleft=True)   # sharey hides right panel's ticks; both are needed
    side(name, "n", len(d))
    fig.suptitle("Test A: supply density predicts provision, not consumption "
                 "($n=" + str(len(d)) + "$)", fontsize=13, y=1.005)
    fig.tight_layout()
    save(fig, name)
    plt.close(fig)


# ================= figB : symlog boxplot + intensity =================
def figB():
    d = D.dropna(subset=["net_inflow_2022", "mui_2022"])
    y0 = d.loc[d.mui_2022 == 0, "net_inflow_2022"].to_numpy()
    y1 = d.loc[d.mui_2022 > 0, "net_inflow_2022"].to_numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.6))

    side("FigS2_validation_testB", "n_no_underserved", len(y0))
    side("FigS2_validation_testB", "n_has_underserved", len(y1))
    bp = a1.boxplot([y0, y1],
                    labels=["no underserved\narea ($n=" + str(len(y0)) + "$)",
                            "has underserved\narea ($n=" + str(len(y1)) + "$)"],
                    widths=0.45, showfliers=False, zorder=4,
                    medianprops=dict(color=RED, lw=1.8),
                    boxprops=dict(color=BLUE, lw=1.2), whiskerprops=dict(color=BLUE, lw=1.1),
                    capprops=dict(color=BLUE, lw=1.1))
    rng = np.random.default_rng(7)
    a1.scatter(rng.normal(1, 0.07, len(y0)), y0, s=11, c=BLUE, alpha=0.4, edgecolors="none", zorder=3)
    a1.scatter(rng.normal(2, 0.07, len(y1)), y1, s=11, c=ORANGE, alpha=0.5, edgecolors="none", zorder=3)
    a1.set_yscale("symlog", linthresh=150)
    a1.set_yticks([-1500, -500, -150, -50, 0, 50, 150, 500])
    a1.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    a1.axhline(0, color="0.55", lw=0.7, ls="--", zorder=2)
    # annotate the extreme exporters (both in the no-underserved group)
    ext = d.nsmallest(2, "net_inflow_2022")
    labels = {"区中央部": "Tokyo core wards", "区西部": "Tokyo western wards"}
    for k, (_, r) in enumerate(ext.iterrows()):
        side('FigS2_validation_testB', 'annot_' + str(r['sma']), r['net_inflow_2022'])
        a1.annotate(f"{labels.get(r['sma'], r['sma'])}  ({r['net_inflow_2022']:+,.0f})",
                    xy=(1.045, r["net_inflow_2022"]), xytext=(1.16, r["net_inflow_2022"] * (1.0 + 0.35 * k)),
                    fontsize=10, color=BLUE, va="center",
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9))
    a1.set_ylabel("Net secondary-practice inflow (physicians, 2022; symlog)")
    _B = json.load(open("outputs/validation/validation_results.json",
                        encoding="utf-8"))["B_main_2022"]
    side("FigS2_validation_testB", "logit_OR_perSD", _B["logit_OR_perSD"])
    side("FigS2_validation_testB", "logit_p", _B["logit_p"])
    a1.set_title("Extensive margin (OR/SD $=" + format(_B["logit_OR_perSD"], ".2f")
                 + "$, $p=" + format(_B["logit_p"], ".2f") + "$)",
                 fontsize=12, loc="left")
    a1.grid(True, axis="y", color=GRID, lw=0.4, zorder=0)

    a2.scatter(d["net_inflow_2022"], d["mui_2022"], s=22, c=ORANGE, alpha=0.5,
               edgecolors="none", zorder=4)
    a2.set_xscale("symlog", linthresh=150)
    a2.set_xticks([-1500, -500, -150, 0, 150, 500])
    a2.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    a2.axvline(0, color="0.55", lw=0.7, ls="--", zorder=2)
    a2.set_xlabel("Net secondary-practice inflow (physicians, 2022; symlog)")
    a2.set_ylabel("Designated physicianless districts (count, 2022)")
    side("FigS2_validation_testB", "intensity_spearman", _B["spearman"])
    a2.set_title("Intensity ($\\rho=" + format(_B["spearman"], ".2f") + "$)",
                 fontsize=12, loc="left")
    a2.grid(True, color=GRID, lw=0.4, zorder=0)

    side("FigS2_validation_testB", "n", len(d))
    fig.suptitle("Test B: net inflow versus under-served status ($n="
                 + str(len(d)) + "$)", fontsize=13, y=1.005)
    fig.tight_layout()
    save(fig, "FigS2_validation_testB")
    plt.close(fig)


# ================= figC : labelled extremes =================
ROMAJI = {"01": "Hokkaido", "05": "Akita", "11": "Saitama", "12": "Chiba", "13": "Tokyo",
          "14": "Kanagawa", "26": "Kyoto", "27": "Osaka", "28": "Hyogo", "31": "Tottori",
          "32": "Shimane", "39": "Kochi", "40": "Fukuoka", "47": "Okinawa", "08": "Ibaraki",
          "10": "Gunma", "19": "Yamanashi", "29": "Nara", "33": "Okayama", "36": "Tokushima"}


def figC():
    import csv, re
    raw = open("data/raw/external/physician_residence/phys_juusho_juugyou_2024.csv", encoding="cp932").read()
    z2h = str.maketrans("０１２３４５６７８９", "0123456789")
    recs = {}
    for r in csv.reader(raw.splitlines()):
        if not r:
            continue
        m = re.match(r"^(\d{2})\D", r[0].strip().translate(z2h))
        if m and 1 <= int(m.group(1)) <= 47 and len(r) > 3:
            try:
                recs[m.group(1).zfill(2)] = (float(r[2]), float(r[3]))
            except ValueError:
                pass
    # Prefecture population = the paper's own fixed-map area populations summed by prefecture
    # (round 39, internal work record). This read pref_population.csv, a file
    # of unknown provenance that differs from the confirmed census in every prefecture; it is
    # now in the archive beside the project root. The area populations reconcile to the
    # published prefecture totals of the census (q28).
    _a = pd.read_csv("census_based/smsa_population_census_2024.csv")
    popm = _a.groupby(_a["smsa_code"].astype(str).str.zfill(4).str[:2])["population"].sum()
    rows = [(pc, res, work, (work - res) / res * 100, work / float(popm[pc]) * 1e5)
            for pc, (res, work) in recs.items()]
    d = pd.DataFrame(rows, columns=["pc", "res", "work", "borrow", "dens"])
    d.to_csv("outputs/validation/fig_validation_data_pref.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.6, 4.9))
    ax.scatter(d["dens"], d["borrow"], s=34, c=BLUE, alpha=0.8, edgecolors="white", lw=0.5, zorder=4)
    ax.axhline(0, color="0.55", lw=0.7, ls="--", zorder=2)
    ax.set_xlabel("Physician density, worksite basis (per 100 000, 2024)")
    ax.set_ylabel("Net worksite import (%, (worksite $-$ residence)/residence)")
    from scipy.stats import spearmanr as _sp
    _rhoC = float(_sp(d["dens"].to_numpy(), d["borrow"].to_numpy())[0])
    side("FigS3_validation_testC", "n", len(d))
    side("FigS3_validation_testC", "spearman", _rhoC)
    ax.set_title("Test C: prefecture-level borrowing, coarse ($n=" + str(len(d))
                 + "$, $\\rho=" + format(_rhoC, ".2f") + "$)",
                 fontsize=12, loc="left")
    ax.grid(True, color=GRID, lw=0.4, zorder=0)
    # deterministic manual offsets for the four extreme prefectures (adjustText
    # scattered labels outside the axes here); all chosen clear of neighbours
    lab = d.reindex(d["borrow"].abs().sort_values(ascending=False).index).head(4)
    MANUAL = {"11": (10, 2, "left", "center"),    # Saitama, top-left corner -> right of point
              "12": (10, 2, "left", "center"),    # Chiba -> right of point
              "13": (-10, 2, "right", "center"),  # Tokyo (dense right edge) -> left of point
              "14": (10, 2, "left", "center")}    # Kanagawa -> right of point
    for _, r in lab.iterrows():
        dx, dy, ha, va = MANUAL.get(r["pc"], (10, 2, "left", "center"))
        side('FigS3_validation_testC', 'annot_' + str(r['pc']), r['borrow'])
        ax.annotate(f"{ROMAJI.get(r['pc'], r['pc'])} ({r['borrow']:+.0f}%)",
                    (r["dens"], r["borrow"]), fontsize=10.5, color=RED,
                    xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, zorder=5)
    save(fig, "FigS3_validation_testC")
    plt.close(fig)


# ================= Fig3 v2 : enlarged fonts, no overlap =================
def fig3v2():
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    Q = {"Q1": dict(dens=153.6, net=+2633), "Q2": dict(dens=195.0, net=+3075),
         "Q3": dict(dens=225.3, net=+2870), "Q4": dict(dens=268.1, net=+1115),
         "Q5": dict(dens=392.0, net=-9693)}
    norm = Normalize(vmin=140, vmax=400); cmap = cm.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    pos = {"Q1": (0.05, 153.6), "Q2": (1.10, 195.0), "Q3": (2.20, 225.3),
           "Q4": (3.30, 268.1), "Q5": (1.68, 392.0)}
    maxgain = 3075; x5, y5 = pos["Q5"]
    # arrow-label offsets: push each label outward, away from the arrow fan
    lbl_off = {"Q1": (-30, 4), "Q2": (-26, -2), "Q3": (26, -2), "Q4": (30, 4)}
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        x1, y1 = pos[q]; frac = Q[q]["net"] / maxgain
        ar = FancyArrowPatch((x5, y5 - 16), (x1, y1 + 16), arrowstyle="-|>",
                             mutation_scale=12 + 13 * frac, lw=1.2 + 6 * frac,
                             color=RED, alpha=0.85,
                             connectionstyle="arc3,rad=0.18", zorder=3, capstyle="round")
        ax.add_patch(ar)
        xm, ym = (x5 + x1) / 2, (y5 + y1) / 2
        ax.annotate(f"+{Q[q]['net']:,}", (xm, ym), xytext=lbl_off[q],
                    textcoords="offset points", fontsize=11.5, fontweight="bold",
                    color=RED, ha="center", va="center", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    for q, (x, y) in pos.items():
        ax.scatter([x], [y], s=1050, c=[cmap(norm(Q[q]["dens"]))], edgecolor="k",
                   linewidth=0.8, zorder=5)
        ax.text(x, y, q, fontsize=11, fontweight="bold", ha="center", va="center",
                color="white" if Q[q]["dens"] < 300 else "black", zorder=6)
        lbl = f"{Q[q]['dens']:.0f}/100k\nnet {Q[q]['net']:+,}"
        if q == "Q5":
            ax.annotate(lbl, (x, y), xytext=(30, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=11)
        else:
            ax.annotate(lbl, (x, y), xytext=(0, -34), textcoords="offset points",
                        ha="center", va="top", fontsize=11)
    ax.set_xlim(-0.95, 4.55); ax.set_ylim(112, 430)
    ax.set_ylabel("Mean physician density (per 100 000)", fontsize=11)
    ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9.5)
    ax.set_title("Net secondary-practice physician flow by density quintile, 2024",
                 fontsize=12.5, loc="left")
    fig.tight_layout()
    save(fig, "Fig3_flow_mechanism")
    plt.close(fig)


if __name__ == "__main__":
    figA()                                          # effective density, for the source
    figA(dens_col="d_prim_2024",                    # primary density, for paper A
         name="FigS1_validation_testA_primary",
         xlab="Physician density (per 100 000, 2024)")
    figB(); figC(); fig3v2()
    print("figures ->", OUT)

pd.DataFrame(SIDE).to_csv('outputs/audit/cc_figs_partC_sidecar.csv', index=False,
                         encoding='utf-8-sig')
print('sidecar: ' + str(len(SIDE)) + ' drawn values -> outputs/audit/cc_figs_partC_sidecar.csv')
