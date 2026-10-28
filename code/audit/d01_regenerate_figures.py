"""
FINAL APPLY -- regenerate every figure whose content depends on the repaired panel.

Figure numbering after inserting the lowest-decile figure at position 3:
    Fig1_density_map_2024      unchanged (2024 only; identical in both panels)
    Fig2_gini_divergence       REGENERATED on the repaired panel
    Fig3_bottom_decile         NEW  (2010 base primary, 2016 base robustness)
    Fig4_flow_mechanism        renamed from Fig3_flow_mechanism, content unchanged

Round 45: the fifth figure, printed in neither paper, was retired from the closure together
with the lines that drew it; both are kept in that round's archive folder. This script draws
Fig2 and Fig3.

All values are read from outputs/audit/c01_round3_numbers.csv (repaired column) and the
repaired panel, never hard-coded, so the figures cannot drift from the text.

Deterministic. Writes to paper/figures/.

Round 43-r52 (2026-10-12): Fig2 is the change from 2010 with matched-area intervals (c01 block
1e), and Fig3 carries round 51's text sizes and an opaque legend with the grid stopped at its box.
"""
import sys, io, os, re, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.path  # noqa: F401  (mpl.path in fig3)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
FIG = "paper/figures"
YEARS = [2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 10,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.7,
    "legend.frameon": False, "figure.dpi": 150})
C_STOCK, C_DENS = "#1b4965", "#c1440e"
GRID = "#d9d9d9"
BAND = "#cccccc"


SIDE = []


def side(fig_name, panel, item, value):
    """Record a number that is actually drawn. Called with the same variable the
    plotting call receives, never a recomputation."""
    SIDE.append({"figure": fig_name, "panel": panel, "item": item, "value": float(value)})


def save(fig, name):
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(f"{FIG}/{name}.{ext}", dpi=600 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- data
num = pd.read_csv("outputs/audit/c01_round3_numbers.csv", encoding="utf-8-sig")
ser = num[num.block == "1a Gini series"]


def series(prefix):
    d = ser[ser.quantity.str.startswith(prefix + " ")]
    return [float(d[d.quantity == f"{prefix} {y}"].repaired.iloc[0]) for y in YEARS]


G_COUNT = series("count Gini")
G_DENS = series("raw Gini")

rep =pd.read_csv("data/crosswalk/physician_panel_smsa_repaired.csv",
                  dtype={"code4": str}, encoding="utf-8-sig").set_index("code4")
CODES = list(rep.index)
X = {y: rep[str(y)].astype(float).to_numpy() for y in YEARS}


def pop(y):
    d = pd.read_csv(f"census_based/smsa_population_census_{y}.csv")
    d["code4"] = d["smsa_code"].apply(lambda c: str(c).zfill(4))
    return d.set_index("code4")["population"].reindex(CODES).astype(float).to_numpy()


P = {y: pop(y) for y in YEARS}
D = {y: X[y] / P[y] * 1e5 for y in YEARS}


# ================================================================= FIG 2
# Round 43-r52 (2026-10-12). Fig. 2 is the change in each Gini from 2010 with its matched-area
# interval, the figure delivered in round 43 (internal work record). It replaces the dual-axis
# level plot. The values are c01's block "1e Fig2 change from 2010" -- block 1b/2 applied to every
# wave with the same draws -- never recomputed here, and the 2010-2024 pair is asserted equal to
# 1b/2's own 2010-2024 row before anything is drawn.
FIG2_SERIES = (("stock", "count", "o", "-", "black", "Physician counts (stock)"),
               ("density", "raw", "s", "--", "white", "Physician density"))
NUM3 = re.compile(r"[+-]\d+\.\d+")


def fig2_values():
    blk = num[num.block == "1e Fig2 change from 2010"]
    out = {}
    for series, measure, *_ in FIG2_SERIES:
        d = {2010: (0.0, None, None)}
        for y in YEARS:
            if y == 2010:
                continue
            cell = blk[blk.quantity == f"{measure} Gini change from 2010 to {y}"].repaired
            assert len(cell) == 1, f"Fig 2: c01 has {len(cell)} rows for {measure} {y}"
            v = [float(x) for x in NUM3.findall(cell.iloc[0])]
            assert len(v) == 3 and v[1] <= v[0] <= v[2], f"Fig 2: {measure} {y} outside its interval"
            d[y] = tuple(v)
        ref = num[(num.block == "1b/2 paired bootstrap")
                  & (num.quantity == f"{measure} Gini change 2010-2024")].repaired.iloc[0]
        assert [float(x) for x in NUM3.findall(ref)] == list(d[2024]), \
            f"Fig 2: {measure} 2010->2024 differs from Table S5's 2010-2024 row"
        out[series] = d
    return out


def fig2():
    V = fig2_values()
    # Sizes are the printed sizes (the figures docx places this 6.27 in wide).
    with mpl.rc_context({"axes.linewidth": 0.8}):
        fig, ax = plt.subplots(figsize=(6.27, 3.6), constrained_layout=True)
        ax.axvspan(2005.0, 2011.0, color=BAND, alpha=0.22, zorder=0, lw=0)
        ax.axhline(0, color="black", lw=0.7, zorder=1)
        handles, drawn = [], []
        for (series, _m, marker, ls, face, label), dx in zip(FIG2_SERIES, (-0.25, +0.25)):
            d = V[series]
            x = np.array(YEARS, float) + dx
            y = np.array([d[t][0] for t in YEARS])
            has = np.array([d[t][1] is not None for t in YEARS])
            lo = np.array([d[t][1] if d[t][1] is not None else d[t][0] for t in YEARS])
            hi = np.array([d[t][2] if d[t][2] is not None else d[t][0] for t in YEARS])
            err = np.vstack([y - lo, hi - y])
            ax.plot(x, y, ls=ls, color="black", lw=1.2, zorder=2)
            ax.errorbar(x[has], y[has], yerr=err[:, has], fmt="none", ecolor="black",
                        elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=3)
            ax.plot(x, y, ls="none", marker=marker, ms=5.5, mfc=face, mec="black", mew=1.0,
                    zorder=4)
            handles.append(Line2D([], [], color="black", ls=ls, lw=1.2, marker=marker, ms=5.5,
                                  mfc=face, mec="black", mew=1.0, label=label))
            drawn.append((d, dx))
            for t in YEARS:
                side("Fig2_gini_divergence", "a", f"{series}_delta_{t}", d[t][0])
                if d[t][1] is not None:
                    side("Fig2_gini_divergence", "a", f"{series}_lo_{t}", d[t][1])
                    side("Fig2_gini_divergence", "a", f"{series}_hi_{t}", d[t][2])
        ax.set_xlim(2004.5, 2025.5)
        ax.set_xticks(YEARS)
        ax.set_xticklabels([str(y) for y in YEARS])
        ax.set_ylim(-0.020, 0.025)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.01))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
            lambda t, _: "0.00" if abs(t) < 1e-12 else f"{t:.2f}".replace("-", "−")))
        ax.tick_params(axis="both", labelsize=10)
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel("Change in Gini from 2010 (ΔG)", fontsize=11)
        leg = ax.legend(handles=handles, loc="upper left", fontsize=10, handlelength=2.6,
                        borderaxespad=0.6)
        # The legend sits inside the axes; fail the run if it covers any point or interval.
        fig.canvas.draw()
        lb = leg.get_window_extent(fig.canvas.get_renderer())
        for d, dx in drawn:
            for t in YEARS:
                lo = d[t][1] if d[t][1] is not None else d[t][0]
                hi = d[t][2] if d[t][2] is not None else d[t][0]
                (x0, y0), (x1, y1) = ax.transData.transform([[t + dx - 0.2, lo], [t + dx + 0.2, hi]])
                assert not lb.overlaps(mpl.transforms.Bbox([[x0, y0 - 4], [x1, y1 + 4]])), \
                    f"Fig 2: legend covers the {t} point"
    save(fig, "Fig2_gini_divergence")


# ================================================================= FIG 3
# Round 43-r52 (2026-10-12), carrying round 51's redraw into the closure: every text 1.25 times
# its earlier size, so that at the 190 mm the journal prints the smallest is 8.8 pt (it was 7.2);
# the y ticks pinned at 5, which is what the auto locator chose before the text grew; and the
# legend on an opaque white box with a thin grey edge, above the grid. The grid is drawn by hand
# and stops at the legend box: an opaque box hides a gridline from the eye but not from a geometry
# check, which reads the stroke under the text all the same (nature-figure's collision audit).
FIG3_FS = {"title": 11 * 1.25, "label": 11 * 1.25, "tick": 9.5 * 1.25, "legend": 9 * 1.25}
GRID3 = "#e6e6e6"


def grid_around(ax, box):
    """Major gridlines as segments that stop at the legend's box (both in display space)."""
    inv = ax.transData.inverted()
    (lx0, ly0), (lx1, ly1) = inv.transform([[box.x0, box.y0], [box.x1, box.y1]])
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    segs = []
    for x in ax.get_xticks():
        if not x0 <= x <= x1:
            continue
        if lx0 <= x <= lx1:
            segs += [((x, y0), (x, max(y0, ly0)))] + ([((x, ly1), (x, y1))] if ly1 < y1 else [])
        else:
            segs.append(((x, y0), (x, y1)))
    for y in ax.get_yticks():
        if not y0 <= y <= y1:
            continue
        if ly0 <= y <= ly1:
            segs += [((x0, y), (max(x0, lx0), y))] + ([((lx1, y), (x1, y))] if lx1 < x1 else [])
        else:
            segs.append(((x0, y), (x1, y)))
    for (a, b) in segs:
        if a != b:
            ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=GRID3, lw=0.4, zorder=1.5))
    return len(segs)


def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
    legs = []
    for ax, base, tag in zip(axes, [2010, 2016], ["a", "b"]):
        yrs = [y for y in YEARS if y >= base]
        k = 33
        fixed = np.argsort(D[base], kind="stable")[:k]
        bx = X[base][fixed].sum(); bp = P[base][fixed].sum(); bd = bx / bp * 1e5
        dens = [100 * (X[y][fixed].sum() / P[y][fixed].sum() * 1e5) / bd for y in yrs]
        phys = [100 * X[y][fixed].sum() / bx for y in yrs]
        popn = [100 * P[y][fixed].sum() / bp for y in yrs]
        ax.axhline(100, color="#bbbbbb", lw=0.7, zorder=0)
        ax.plot(yrs, dens, "-o", ms=4, lw=1.7, color=C_DENS, label="Density (per 100 000)")
        ax.plot(yrs, phys, "-s", ms=4, lw=1.7, color=C_STOCK, label="Physicians")
        ax.plot(yrs, popn, "-^", ms=4, lw=1.7, color="#7a8b99", label="Population")
        ax.set_ylabel(f"Index ({base} = 100)", fontsize=FIG3_FS["label"])
        ax.set_xlabel("Year", fontsize=FIG3_FS["label"])
        ax.set_xticks(yrs[::2] if base == 2010 else yrs)
        ax.set_xticklabels([str(v) for v in (yrs[::2] if base == 2010 else yrs)])
        ax.tick_params(labelsize=FIG3_FS["tick"])
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
        ax.set_title(f"({tag}) Lowest density decile in {base} "
                     f"({'primary window' if base == 2010 else 'robustness window'})",
                     fontsize=FIG3_FS["title"], loc="left")
        leg = ax.legend(loc="upper left", fontsize=FIG3_FS["legend"], frameon=True,
                        facecolor="white", edgecolor="#999999", framealpha=1.0, fancybox=False,
                        labelspacing=0.2, handlelength=1.5)  # 0.5 -> 0.2 and 2.0 -> 1.5, font unchanged: the box would otherwise cover (b)'s 2020 point
        leg.get_frame().set_linewidth(0.5)
        leg.set_zorder(5)
        legs.append(leg)
        for _y, _de, _ph, _po in zip(yrs, dens, phys, popn):
            side("Fig3_bottom_decile", tag, f"density_index_{_y}", _de)
            side("Fig3_bottom_decile", tag, f"physicians_index_{_y}", _ph)
            side("Fig3_bottom_decile", tag, f"population_index_{_y}", _po)
        side("Fig3_bottom_decile", tag, "base_year", base)
        print(f"  Fig3 panel {tag}: base {base}, physicians {phys[-1]-100:+.1f}%, "
              f"population {popn[-1]-100:+.1f}%, density {dens[-1]-100:+.1f}%")
    # Fix the view the auto-scaling chose, then lay the grid around each legend's box.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax, leg in zip(axes, legs):
        ax.set_xlim(ax.get_xlim()); ax.set_ylim(ax.get_ylim())
        grid_around(ax, leg.get_frame().get_window_extent(r))
    fig.canvas.draw()
    # Fail the run if a legend box comes within 1 pt of any line, or within a marker's radius
    # plus 1 pt of any marker (the reference line included).
    pt = fig.dpi / 72
    for ax, leg in zip(axes, legs):
        box = leg.get_frame().get_window_extent(r)
        for ln in ax.get_lines():
            if ln.get_color() == GRID3:
                continue
            xy = ln.get_transform().transform(np.column_stack(ln.get_data()))
            assert not mpl.path.Path(xy).intersects_bbox(box.padded(1 * pt), filled=False), \
                f"Fig 3: the legend box is within 1 pt of a line ({ln.get_label()!r})"
            if ln.get_marker() not in (None, "None", "none", ""):
                rad = (ln.get_markersize() / 2 + 1) * pt
                for px_, py_ in xy:
                    assert not box.padded(rad).contains(px_, py_), \
                        f"Fig 3: the legend box is within 1 pt of a marker ({ln.get_label()!r})"
    save(fig, "Fig3_bottom_decile")


# ---------------------------------------------------------------- renames
# RETIRED 2026-09-03 (listing action row C-2). The Fig3 -> Fig4 rename was a one-off from
# the round that renumbered the figures. Its source has not existed for months, so the
# move never fired -- but it stayed in the closure as a SECOND WRITER of
# paper/figures/Fig4_flow_mechanism.{pdf,png,svg}, running after cc_figures.py:270 and
# winning if it ever did fire. c09 counted it as one of three multi-writer artefacts.
# A completed rename is not a pipeline step; it is a fact about the past.
for stale in ["Fig4_tension", "Fig_bottom_decile"]:
    for ext in ["pdf", "svg", "png"]:
        f = f"{FIG}/{stale}.{ext}"
        if os.path.exists(f):
            os.remove(f)
            print(f"removed stale {stale}.{ext}")

fig2(); fig3()
pd.DataFrame(SIDE).to_csv("outputs/audit/d01_figure_sidecar.csv", index=False,
                          encoding="utf-8-sig")
print(f"sidecar: {len(SIDE)} drawn values -> outputs/audit/d01_figure_sidecar.csv")
print("\nfigures in paper/figures/:")
for f in sorted(os.listdir(FIG)):
    if f.endswith(".pdf"):
        print("   ", f)
