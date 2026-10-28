"""
Two supplementary figures requested in review, plus their sidecar.

  FigS6_theta_curve      -- the theta surface B1 names but never draws. Effective-supply
                            Gini against theta, one curve per flow wave, census and
                            physician-share denominators side by side. Marks theta = 0.2,
                            the weight the official maldistribution index adopted in 2022,
                            and theta = 0.5, this paper's base case.
  FigS7_flow_density     -- net secondary-practice inflow against own density, 2016 and
                            2024, so a reader can see whether r = -0.67 / -0.71 rests on a
                            handful of large exporters.

No new statistic. FigS6 reads outputs/audit/i02_theta_surface.csv as it stands. FigS7
recomputes nothing that is not already the definition of n_i and d_i in the main text:
the OD matrix with its diagonal zeroed, the repaired panel, and the census denominator --
the same three inputs, and the same arithmetic, as i02_theta_surface.py and
h05_moran_repaired.py.

Deliberately NOT read: census_based/smsa_density_census_*.csv (pre-repair, unregistered)
and od_network_metrics_*.csv (pre-repair). Both would have been convenient and both are
stale.

Writes paper/figures/FigS6_*, paper/figures/FigS7_* and
outputs/audit/q01_supp_figures_sidecar.csv.
"""
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)

OUT = os.environ.get("CC_FIGDIR", "paper/figures")
os.makedirs(OUT, exist_ok=True)
ORANGE, BLUE, RED = "#E8A03D", "#4A6274", "#C0384B"
GRID = "#d9d9d9"
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "none",
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 10,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8,
    "legend.frameon": False, "figure.dpi": 150})

SIDE = []


def side(fig_name, item, value):
    """Record a number that is actually drawn, taken from the same variable the
    drawing call receives -- never recomputed afterwards."""
    SIDE.append({"figure": fig_name, "item": item, "value": float(value)})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    print("saved", name)


WAVES = [2016, 2018, 2020, 2022, 2024]

# ===================== FigS6: the theta surface =====================
TH = pd.read_csv("outputs/audit/i02_theta_surface.csv", encoding="utf-8-sig")
assert set(TH["year"]) == set(WAVES), "theta surface does not carry the five flow waves"

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.4))
cmap = plt.get_cmap("viridis")
cols = {y: cmap(i / (len(WAVES) - 1) * 0.86) for i, y in enumerate(WAVES)}

for ax, col, lab in [(a1, "gini_effective_census", "Census denominator"),
                     (a2, "gini_effective_share", "Physician-share denominator")]:
    for y in WAVES:
        g = TH[TH["year"] == y].sort_values("theta")
        ax.plot(g["theta"], g[col], color=cols[y], lw=1.6, zorder=3, label=str(y))
        # the theta = 0 endpoint IS that wave's primary-density Gini
        y0 = float(g[g["theta"] == 0][col].iloc[0])
        ax.plot([0], [y0], "o", ms=5.5, mfc="white", mec=cols[y], mew=1.4, zorder=4)
        side("FigS6_theta_curve", f"{lab.split()[0].lower()}_{y}_theta0", y0)
        side("FigS6_theta_curve", f"{lab.split()[0].lower()}_{y}_theta02",
             float(g[np.isclose(g["theta"], 0.2)][col].iloc[0]))
        side("FigS6_theta_curve", f"{lab.split()[0].lower()}_{y}_theta05",
             float(g[np.isclose(g["theta"], 0.5)][col].iloc[0]))
        side("FigS6_theta_curve", f"{lab.split()[0].lower()}_{y}_theta1",
             float(g[np.isclose(g["theta"], 1.0)][col].iloc[0]))
    for xv, txt, cl in [(0.2, "MHLW index (2022)\n0.8 / 0.2", RED),
                        (0.5, "base case", "0.45")]:
        ax.axvline(xv, color=cl, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel(r"$\theta$  (weight on the secondary workplace)")
    ax.set_title(lab, pad=8)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)

a1.set_ylabel("Gini of effective-supply density")
a1.annotate("MHLW index (2022)", xy=(0.2, a1.get_ylim()[1]), xytext=(0.235, a1.get_ylim()[1]),
            fontsize=8.5, color=RED, va="top", ha="left")
a1.annotate("base case", xy=(0.5, a1.get_ylim()[0]), xytext=(0.53, a1.get_ylim()[0]),
            fontsize=8.5, color="0.35", va="bottom", ha="left")
a2.set_ylabel("Gini of effective-supply density")
a1.legend(title="Flow wave", loc="upper right", ncol=1, handlelength=1.6)
side("FigS6_theta_curve", "n_waves", len(WAVES))
side("FigS6_theta_curve", "n_theta_points", TH["theta"].nunique())
save(fig, "FigS6_theta_curve")
plt.close(fig)

# ===================== FigS7: n_i against d_i =====================
panel = pd.read_csv("data/crosswalk/physician_panel_smsa_repaired.csv",
                    encoding="utf-8-sig", dtype={"code4": str}).set_index("code4")
NAME = {"1301": "Tokyo core wards", "1304": "Tokyo western wards",
        "2304": "Owari-tobu", "2701": "Toyono", "0905": "Tochigi-kennan",
        "2604": "Kyoto-Otokuni", "1404": "Kawasaki-hokubu"}

fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.5), sharey=True)
for ax, year in zip(axes, [2016, 2024]):
    od = pd.read_csv(f"od_matrix_{year}.csv", index_col=0, encoding="utf-8-sig")
    od.index = [f"{int(i):04d}" for i in od.index]
    od.columns = [f"{int(c):04d}" for c in od.columns]
    assert list(od.index) == list(od.columns)
    M = od.values.astype(float).copy()
    np.fill_diagonal(M, 0.0)                       # main text Eq. 3 convention
    n = pd.Series(M.sum(0) - M.sum(1), index=od.index)
    pop = pd.read_csv(f"census_based/smsa_population_census_{year}.csv",
                      encoding="utf-8-sig", dtype={"code4": str}).set_index("code4")["population"]
    x = panel[str(year)].reindex(od.index).astype(float)
    d = 1e5 * x / pop.reindex(od.index)
    assert n.notna().all() and d.notna().all(), f"{year}: incomplete circle set"

    r = float(np.corrcoef(n, d)[0, 1])
    ax.axvline(0, color="0.55", lw=0.7, ls="--", zorder=2)
    ax.scatter(n, d, s=17, facecolor=BLUE, edgecolor="none", alpha=0.55, zorder=3)
    ax.set_xscale("symlog", linthresh=150)         # same linthresh as Fig S4
    ax.set_xticks([-1500, -500, -150, -50, 0, 50, 150, 500])
    ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Net secondary-practice inflow $n_i$ (physicians; symlog)")
    ax.set_title(f"{year}    $r={r:+.2f}$".replace("-", "−"), pad=8)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    side("FigS7_flow_density", f"r_{year}", r)
    side("FigS7_flow_density", f"n_circles_{year}", len(n))

    for k, (code, rw) in enumerate(pd.DataFrame({"n": n, "d": d}).nsmallest(3, "n").iterrows()):
        side("FigS7_flow_density", f"annot_{year}_{code}", rw["n"])
        ax.annotate(f"{NAME.get(code, code)} ({rw['n']:+,.0f})",
                    xy=(rw["n"], rw["d"]), xytext=(30, 12 - 30 * k),
                    textcoords="offset points", fontsize=9, color=BLUE,
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))

axes[0].set_ylabel(r"Primary physician density $d_i$ (per 100 000)")

# The largest exporter is annotated on this figure at 2016 and 2024 and on Figure S4 at
# 2022, and the three values do not line up monotonically. Emit the whole series so the
# caption can state it rather than leave a reader to notice the discrepancy unaided.
for year in WAVES:
    od = pd.read_csv(f"od_matrix_{year}.csv", index_col=0, encoding="utf-8-sig")
    od.index = [f"{int(i):04d}" for i in od.index]
    od.columns = [f"{int(c):04d}" for c in od.columns]
    M = od.values.astype(float).copy()
    np.fill_diagonal(M, 0.0)
    n = pd.Series(M.sum(0) - M.sum(1), index=od.index)
    side("FigS7_flow_density", f"n_1301_{year}", n["1301"])

save(fig, "FigS7_flow_density")
plt.close(fig)

# ===================== sidecar =====================
pd.DataFrame(SIDE).to_csv("outputs/audit/q01_supp_figures_sidecar.csv",
                          index=False, encoding="utf-8-sig")
print("sidecar: " + str(len(SIDE)) + " drawn values -> outputs/audit/q01_supp_figures_sidecar.csv")
