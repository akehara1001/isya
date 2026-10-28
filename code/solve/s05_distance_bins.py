"""
WORK 1 -- non-parametric check on the functional form of distance decay.

Distance is now the only load-bearing gravity covariate whose coefficient more than
doubled when the model became zero-inclusive (-0.669 -> -1.455 descriptive, -2.030 with
fixed effects). Whether log(1+dist) is the right shape therefore matters more than the
adjacency sign reversal, which is treated here as a possible symptom of it.

Bins, not splines: no knot placement to defend, no smoothness assumption, and the decay
curve is read straight off the coefficients.

  - centroid distances of adjacent pairs are described first, so the bin design is
    driven by the data rather than chosen
  - distance is cut at deciles of the positive-flow distance distribution, with the
    nearest bin subdivided if it is too coarse to resolve the adjacency range
  - PPML on all 542,850 directed pairs, two-way clustered on origin and destination,
    in both the descriptive and the origin/destination-fixed-effect specification
  - each fitted with and without the adjacency dummy, so we can see whether adjacency
    still carries a signal once distance is unconstrained

Deterministic. Writes outputs/solve/s05_distance_bins.csv, .txt
       and paper/figures/FigS5_distance_decay.{pdf,png,svg}
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd
import scipy.sparse as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)

# A clean re-run starts with outputs/ absent; create the directory rather than
# assuming a previous run left it behind.
os.makedirs("outputs/solve", exist_ok=True)
sys.path.insert(0, ROOT)
from phase3_common import (OD_YEARS, distance_matrix, load_adjacency, load_density,
                           load_infrastructure, load_od_matrix, smsa_codes, stock_series)

report, recs = [], []


def say(s=""):
    print(s)
    report.append(s)


# ---------------------------------------------------------------- data
CODES = smsa_codes()
N = len(CODES)
dist = distance_matrix(CODES).to_numpy()
adjm = load_adjacency(codes=CODES).to_numpy()
ca = np.array(CODES)
ri, ci = np.where(~np.eye(N, dtype=bool))
frames = []
for year in OD_YEARS:
    od = load_od_matrix(year, codes=CODES).to_numpy(float, copy=True)
    np.fill_diagonal(od, 0.0)
    stock = stock_series(year, codes=CODES).reindex(CODES).to_numpy(float)
    dens = load_density(year, codes=CODES).to_numpy(float)   # raises if any is missing
    inf_ = load_infrastructure(year, codes=CODES).set_index("smsa_code")
    inf = (inf_["hospitals"].reindex(CODES).fillna(0.0).to_numpy(float)
           + inf_["clinics"].reindex(CODES).fillna(0.0).to_numpy(float))
    frames.append(pd.DataFrame({
        "year": year, "origin": ca[ri], "dest": ca[ci], "F": od[ri, ci],
        "dist_km": dist[ri, ci], "log_dist": np.log1p(dist[ri, ci]),
        "adj": adjm[ri, ci].astype(float),
        "same_pref": (ca[ri].astype("<U2") == ca[ci].astype("<U2")).astype(float),
        "log_x_i": np.log1p(stock[ri]), "log_density_j": np.log1p(dens[ci]),
        "log_infra_total_j": np.log1p(inf[ci])}))
D = pd.concat(frames, ignore_index=True)

say("=" * 100)
say("WORK 1 -- distance decay without a functional form")
say("=" * 100)
adjd = D.loc[D.adj == 1, "dist_km"]
posd = D.loc[D.F > 0, "dist_km"]
say(f"centroid distance of ADJACENT pairs (n={len(adjd):,}): "
    f"min {adjd.min():.1f}, p25 {adjd.quantile(.25):.1f}, median {adjd.median():.1f}, "
    f"p75 {adjd.quantile(.75):.1f}, max {adjd.max():.1f} km")
say(f"centroid distance of pairs WITH FLOW (n={len(posd):,}): "
    f"min {posd.min():.1f}, p25 {posd.quantile(.25):.1f}, median {posd.median():.1f}, "
    f"p75 {posd.quantile(.75):.1f}, max {posd.max():.1f} km")
say(f"all pairs: median {D.dist_km.median():.1f}, max {D.dist_km.max():.1f} km")
say("")
# "flow-weighted" was wrong, and this line is where the phrase entered the repository.
# posd is D.loc[D.F > 0, "dist_km"]: one entry per pair that carries any flow, each counted
# once whatever its volume. Weighting by flow puts the quantiles somewhere else entirely --
# code/solve/s10_distance_range_test.py measured it: the median among positive-flow pairs is
# 68.6 km, the flow-weighted median falls in band 2 at 29.9 km. Same family as the distance
# three-value mislabel corrected in B-S13 (paper_B/PHASE_B_LEDGER.md), where the same word
# was wrong in a different sentence: the number was right and the name given to it was not.
# The phrase reached paper_B/main.tex line 235 and paper_B/supplementary.tex line 401 from
# here. Those are round 4's to fix; registered in split/ozaki_round1b_instruments.md 1-4.
# The docstring above has always said "positive-flow", so the two disagreed inside one file.
say("Adjacent pairs sit almost entirely below the first decile of the POSITIVE-FLOW "
    "distance distribution -- pairs with F > 0, counted once each and NOT weighted by "
    "flow -- so the nearest decile is subdivided to resolve them.")

# ---------------------------------------------------------------- bin design
q = np.quantile(posd, np.arange(0, 1.01, 0.1))
inner = np.quantile(adjd, [0.25, 0.5, 0.75])
EDGES = np.unique(np.concatenate([[0], inner, q[1:-1], [D.dist_km.max() + 1]]))
D["bin"] = pd.cut(D.dist_km, EDGES, right=False, labels=False)
NB_ = int(D["bin"].max()) + 1
say("")
say(f"{NB_} distance bins:")
say(f"{'bin':>4} {'range (km)':>18} {'pairs':>9} {'positive':>9} {'flow':>9} "
    f"{'adjacent':>9} {'median km':>10}")
BINMID = []
for b in range(NB_):
    m = D["bin"] == b
    BINMID.append(float(D.loc[m, "dist_km"].median()))
    say(f"{b:>4} {f'{EDGES[b]:.0f}-{EDGES[b+1]:.0f}':>18} {int(m.sum()):>9,} "
        f"{int((D.loc[m,'F']>0).sum()):>9,} {D.loc[m,'F'].sum():>9,.0f} "
        f"{int(D.loc[m,'adj'].sum()):>9,} {BINMID[-1]:>10.1f}")

# ---------------------------------------------------------------- PPML machinery
y = D["F"].to_numpy(float)
o_idx = pd.factorize(D["origin"])[0]
d_idx = pd.factorize(D["dest"])[0]
yr_idx = pd.factorize(D["year"])[0]
bin_idx = D["bin"].to_numpy(int)


def dumm(idx, drop_first=True):
    k = idx.max() + 1
    m = sp.csr_matrix((np.ones(len(idx)), (np.arange(len(idx)), idx)), shape=(len(idx), k))
    return m[:, np.arange(1, k)] if drop_first else m


def drop_sep(fes, yv):
    keep = np.ones(len(yv), bool)
    while True:
        drop = np.zeros(len(yv), bool)
        for idx in fes:
            tot = np.bincount(idx[keep], weights=yv[keep], minlength=idx.max() + 1)
            bad = np.where(tot == 0)[0]
            if len(bad):
                drop |= keep & np.isin(idx, bad)
        if not drop.any():
            return keep
        keep &= ~drop


def ppml(X, yv, oc, dc, maxit=200, tol=1e-10):
    n, k = X.shape
    b = np.zeros(k); b[0] = np.log(max(yv.mean(), 1e-8))
    ll = lambda bb: float((yv * np.clip(X @ bb, -30, 30)
                           - np.exp(np.clip(X @ bb, -30, 30))).sum())
    cur = ll(b)
    for _ in range(maxit):
        mu = np.exp(np.clip(X @ b, -30, 30))
        H = np.asarray((X.T @ X.multiply(mu[:, None]).tocsr()).todense())
        H[np.diag_indices_from(H)] += 1e-7
        step = np.linalg.lstsq(H, X.T @ (yv - mu), rcond=None)[0]
        t = 1.0
        for _ in range(30):
            if np.isfinite(ll(b + t * step)) and ll(b + t * step) >= cur - 1e-9:
                break
            t *= 0.5
        b = b + t * step
        new = ll(b)
        if abs(new - cur) < tol * (1 + abs(cur)):
            break
        cur = new
    mu = np.exp(np.clip(X @ b, -30, 30))
    H = np.asarray((X.T @ X.multiply(mu[:, None]).tocsr()).todense())
    H[np.diag_indices_from(H)] += 1e-7
    Hi = np.linalg.pinv(H)
    Xe = X.multiply((yv - mu)[:, None]).tocsr()

    def meat(g):
        S = sp.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))),
                          shape=(g.max() + 1, len(g)))
        A = np.asarray((S @ Xe).todense())
        return A.T @ A
    od = pd.factorize(pd.Series(oc).astype(str) + "_" + pd.Series(dc).astype(str))[0]
    V = Hi @ (meat(oc) + meat(dc) - meat(od)) @ Hi
    return b, np.sqrt(np.clip(np.diag(V), 0, None))


OTHER = ["same_pref", "log_x_i", "log_density_j", "log_infra_total_j"]


def fit(tag, with_adj, use_fe):
    fes = [yr_idx] + ([o_idx, d_idx] if use_fe else [])
    keep = drop_sep(fes, y)
    yv = y[keep]
    cols = OTHER + (["adj"] if with_adj else [])
    blocks = [sp.csr_matrix(np.ones((int(keep.sum()), 1))),
              sp.csr_matrix(D.loc[keep, cols].to_numpy(float)),
              dumm(pd.Categorical(bin_idx[keep],
                                  categories=sorted(set(bin_idx[keep]))).codes)]
    names = ["const"] + cols + [f"bin_{b}" for b in sorted(set(bin_idx[keep]))[1:]]
    for idx in fes:
        blocks.append(dumm(pd.factorize(idx[keep])[0]))
        names += ["fe"] * (dumm(pd.factorize(idx[keep])[0]).shape[1])
    X = sp.hstack(blocks).tocsr()
    b, se = ppml(X, yv, pd.factorize(o_idx[keep])[0], pd.factorize(d_idx[keep])[0])
    out = {}
    for nm in names:
        pass
    for i, nm in enumerate(names):
        if nm.startswith("bin_") or nm in cols:
            out[nm] = (b[i], se[i])
    say("")
    say(f"--- {tag} ---  n={int(keep.sum()):,}, k={X.shape[1]}, "
        f"{int((~keep).sum()):,} separated observations dropped")
    ref = sorted(set(bin_idx[keep]))[0]
    say(f"    distance bins, relative to bin {ref} ({EDGES[ref]:.0f}-{EDGES[ref+1]:.0f} km):")
    for nm, (bb, ss) in out.items():
        if not nm.startswith("bin_"):
            continue
        bi = int(nm.split("_")[1])
        lo, hi = bb - 1.96 * ss, bb + 1.96 * ss
        say(f"        bin {bi:>2} (median {BINMID[bi]:>6.1f} km): {bb:>+8.3f} "
            f"[{lo:>+7.3f},{hi:>+7.3f}]   rate ratio {np.exp(bb):>7.4f}")
        recs.append({"spec": tag, "term": nm, "bin": bi, "median_km": BINMID[bi],
                     "coef": round(bb, 4), "se": round(ss, 4), "ci_lo": round(lo, 4),
                     "ci_hi": round(hi, 4), "rate_ratio": round(float(np.exp(bb)), 5)})
    for nm in cols:
        bb, ss = out[nm]
        lo, hi = bb - 1.96 * ss, bb + 1.96 * ss
        say(f"    {nm:>20}: {bb:>+8.3f} [{lo:>+7.3f},{hi:>+7.3f}]  "
            f"p={2*(1-norm.cdf(abs(bb/ss))):.3g}")
        recs.append({"spec": tag, "term": nm, "bin": np.nan, "median_km": np.nan,
                     "coef": round(bb, 4), "se": round(ss, 4), "ci_lo": round(lo, 4),
                     "ci_hi": round(hi, 4), "rate_ratio": np.nan})
    return out


say("")
say("=" * 100)
say("PPML WITH DISTANCE BIN DUMMIES")
say("=" * 100)
A = fit("descriptive, bins + adjacency", True, False)
B = fit("descriptive, bins only (no adjacency)", False, False)
C = fit("origin/destination FE, bins + adjacency", True, True)
E = fit("origin/destination FE, bins only", False, True)

pd.DataFrame(recs).to_csv("outputs/solve/s05_distance_bins.csv", index=False,
                          encoding="utf-8-sig")

# ---------------------------------------------------------------- variance inflation
# Round 41 §3-7. Supplementary A5 (paper B) printed "Variance inflation factors were < 1.6"
# with no producer behind it. These are the VIFs of the reported pooled specification (fit A:
# year fixed effects, the 13 distance bands and adjacency), computed on its design matrix
# without the fixed effects -- the five substantive covariates and the 12 band indicators,
# with an intercept -- as VIF_j = 1 / (1 - R^2_j), the diagonal of the inverse correlation
# matrix. A second pass adds the year indicators, for A5's "(year fixed effects < 2)".
keepV = drop_sep([yr_idx], y)
colsV = OTHER + ["adj"]
binsV = pd.Categorical(bin_idx[keepV], categories=sorted(set(bin_idx[keepV]))).codes
BV = np.eye(binsV.max() + 1)[binsV][:, 1:]
namesV = colsV + [f"bin_{b}" for b in sorted(set(bin_idx[keepV]))[1:]]
yrsV = pd.factorize(yr_idx[keepV])[0]
YV = np.eye(yrsV.max() + 1)[yrsV][:, 1:]


def _vif(M):
    return np.diag(np.linalg.inv(np.corrcoef(M, rowvar=False)))


XV = np.hstack([D.loc[keepV, colsV].to_numpy(float), BV])
v1 = _vif(XV)
v2 = _vif(np.hstack([XV, YV]))
# Round 41-3 ruling 0-1: the band dummies' own VIFs depend on the reference band (the nearest
# band, 0.5% of pairs, gives up to 67.7; the most populous gives up to 1.39), so A5 prints none
# of them. The band factor is summarised by its generalised VIF (Fox and Monette 1992), which
# does not depend on the reference: GVIF = det(R_bands) det(R_rest) / det(R), with R the
# correlation matrix of the design without fixed effects, and GVIF^(1/(2 df)) puts it on the
# scale of a single coefficient's standard error. The reference band is unchanged.
RV = np.corrcoef(XV, rowvar=False)
nsub = len(colsV)
df_band = BV.shape[1]
gvif = float(np.linalg.det(RV[nsub:, nsub:]) * np.linalg.det(RV[:nsub, :nsub]) / np.linalg.det(RV))
gvif_adj = gvif ** (1.0 / (2 * df_band))
sub_i = int(np.argmax(v1[:nsub]))
yr_vif = v2[len(namesV):]
VIF = pd.DataFrame({"term": namesV + [f"year_{k}" for k in range(1, YV.shape[1] + 1)],
                    "vif": list(v1) + [np.nan] * YV.shape[1],
                    "vif_with_year_fe": list(v2)})
VIF = pd.concat([VIF, pd.DataFrame([
    {"term": "VIF, max over the substantive covariates", "vif": float(v1[sub_i]),
     "vif_with_year_fe": float(np.max(v2[:nsub])), "which": colsV[sub_i]},
    {"term": "VIF, max over the year indicators (year indicators added to the design)",
     "vif": np.nan, "vif_with_year_fe": float(np.max(yr_vif)),
     "which": f"year_{int(np.argmax(yr_vif)) + 1}"},
    {"term": "distance-band factor GVIF = det(R_bands) det(R_rest) / det(R)", "vif": gvif,
     "vif_with_year_fe": np.nan, "which": "13 bands, nearest as reference"},
    {"term": "distance-band factor GVIF^(1/(2*df))", "vif": gvif_adj,
     "vif_with_year_fe": np.nan, "which": "13 bands, nearest as reference"},
    {"term": "distance-band factor df", "vif": float(df_band), "vif_with_year_fe": np.nan,
     "which": "13 bands, nearest as reference"}])], ignore_index=True)
VIF.to_csv("outputs/solve/s05_vif.csv", index=False, encoding="utf-8-sig")
say("")
say(f"variance inflation, fit A design without fixed effects (n={int(keepV.sum()):,}): "
    f"substantive max {float(v1[sub_i]):.3f} ({colsV[sub_i]}); year indicators max "
    f"{float(np.max(yr_vif)):.3f}; band factor GVIF {gvif:.3f}, GVIF^(1/(2*{df_band})) "
    f"{gvif_adj:.3f}; band dummies (reference-dependent, not printed) max "
    f"{float(np.max(v1[nsub:])):.3f}")

# ---------------------------------------------------------------- verdicts
say("")
say("=" * 100)
say("VERDICTS")
say("=" * 100)


def curve(o):
    return [(b, o[f"bin_{b}"][0]) for b in range(1, NB_) if f"bin_{b}" in o]


for tag, o in [("descriptive", B), ("origin/dest FE", E)]:
    c = curve(o)
    mono = all(c[i][1] >= c[i + 1][1] - 1e-9 for i in range(len(c) - 1))
    say(f"{tag:>16}: decay monotone across bins = {mono}")
    recs.append({"spec": tag, "term": "monotone", "coef": int(mono)})
say("")
say("adjacency, with distance bins in the model:")
for tag, o in [("descriptive", A), ("origin/dest FE", C)]:
    bb, ss = o["adj"]
    lo, hi = bb - 1.96 * ss, bb + 1.96 * ss
    sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not distinguishable from zero"
    say(f"    {tag:>16}: {bb:+.3f} [{lo:+.3f}, {hi:+.3f}]  -> {sig}")
say("")
say("For comparison, adjacency under the single log(1+dist) term was "
    "+0.178 (ns) descriptive and -0.274 (significant) with fixed effects.")

# fold-change statements for the manuscript
for tag, o in [("descriptive", B), ("origin/dest FE", E)]:
    c = curve(o)
    med_bin = min(range(1, NB_), key=lambda b: abs(BINMID[b] - np.median(posd)))
    if f"bin_{med_bin}" in o:
        f_ = np.exp(-o[f"bin_{med_bin}"][0])
        say(f"{tag:>16}: flow falls {f_:.0f}-fold between the nearest bin "
            f"(median {BINMID[0]:.0f} km) and the bin at the median flow distance "
            f"({BINMID[med_bin]:.0f} km)")
        recs.append({"spec": tag, "term": "fold_nearest_to_median", "coef": round(f_, 1)})

# BAND AGAINST BAND, with nothing else moving -- co-author review round 2, item B-3.
#
# The two fold changes the manuscript printed before this block existed were taken against a
# CONTIGUOUS reference pair, so each carried the adjacency coefficient as well as the band's
# own: about five- and sevenfold at 61 km, two orders of magnitude at 237 km. They answer
# "how much less flow between these two areas than between that adjacent pair", which is a
# second question wearing the first one's clothes. The bare comparison -- same contiguity
# status at both ends, only the distance band different -- is exp(-beta) on the SAME
# coefficients Table S10 prints, and it is what section 3.2 now reports. Emitted here so the
# page can bind to it instead of the reader exponentiating a table cell.
for tag, o in [("descriptive, bins + adjacency", A),
               ("origin/destination FE, bins + adjacency", C)]:
    for b in (7, 11):
        bb, ss = o[f"bin_{b}"]
        lo, hi = bb - 1.96 * ss, bb + 1.96 * ss
        f_, f_lo, f_hi = np.exp(-bb), np.exp(-hi), np.exp(-lo)
        say(f"{tag:>40}: bin {b:>2} (median {BINMID[b]:>6.1f} km) -- flow falls "
            f"{f_:.2f}-fold [{f_lo:.2f}, {f_hi:.2f}] against the reference band "
            f"(median {BINMID[0]:.0f} km), contiguity held")
        recs.append({"spec": tag, "term": f"fold_vs_reference_band_bin_{b}", "bin": b,
                     "median_km": BINMID[b], "coef": round(float(f_), 4),
                     "ci_lo": round(float(f_lo), 4), "ci_hi": round(float(f_hi), 4)})

pd.DataFrame(recs).to_csv("outputs/solve/s05_distance_bins.csv", index=False,
                          encoding="utf-8-sig")

# ---------------------------------------------------------------- figure
SIDE = []


def side(item, value):
    """Record a number that is actually drawn, from the same variable the drawing
    call receives -- never a recomputation."""
    SIDE.append({'figure': 'FigS5_distance_decay', 'item': item,
                 'value': float(value)})

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.right": False,
    "axes.spines.top": False, "axes.linewidth": 0.7, "legend.frameon": False,
    "figure.dpi": 150})
fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
for o, lab, col, mk in [(B, "PPML, descriptive", "#1b4965", "o"),
                        (E, "PPML, origin/destination FE", "#c1440e", "s")]:
    xs = [BINMID[0]] + [BINMID[b] for b in range(1, NB_) if f"bin_{b}" in o]
    ys = [0.0] + [o[f"bin_{b}"][0] for b in range(1, NB_) if f"bin_{b}" in o]
    lo = [0.0] + [o[f"bin_{b}"][0] - 1.96 * o[f"bin_{b}"][1]
                  for b in range(1, NB_) if f"bin_{b}" in o]
    hi = [0.0] + [o[f"bin_{b}"][0] + 1.96 * o[f"bin_{b}"][1]
                  for b in range(1, NB_) if f"bin_{b}" in o]
    _tag = 'descriptive' if lab.endswith('descriptive') else 'FE'
    for _i, (_x, _y, _l, _h) in enumerate(zip(xs, ys, lo, hi)):
        side(f'{_tag}_bin{_i}_km', _x)
        side(f'{_tag}_bin{_i}_coef', _y)
        side(f'{_tag}_bin{_i}_ci_lo', _l)
        side(f'{_tag}_bin{_i}_ci_hi', _h)
    ax.plot(xs, ys, mk + "-", color=col, lw=1.5, ms=4, label=lab)
    ax.fill_between(xs, lo, hi, color=col, alpha=0.13, lw=0)
# the fitted log(1+d) shape, anchored at the reference bin, for comparison
_G = pd.read_csv('outputs/solve/s01_gravity_ppml.csv', encoding='utf-8-sig')
_G = _G[_G.term == 'log_dist'].set_index('spec')['coef']
_b_desc = float(_G[[i for i in _G.index if i.startswith('(a)')][0]])
_b_fe = float(_G[[i for i in _G.index if i.startswith('(b)')][0]])


def _fitlab(b):
    """Render the slope the same way the manuscript does, from the value itself."""
    return '$-' + format(abs(b), '.2f').replace('.', '{\\cdot}') \
        + '\\,\\log(1+d)$'


for beta, col, ls, lab in [(_b_desc, "#1b4965", ":",
                            _fitlab(_b_desc) + ' (fitted, descriptive)'),
                           (_b_fe, "#c1440e", ":",
                            _fitlab(_b_fe) + ' (fitted, FE)')]:
    side('fitted_slope_' + ('descriptive' if col == '#1b4965' else 'FE'), beta)
    xs = np.linspace(max(BINMID[0], 1), max(BINMID), 200)
    ax.plot(xs, beta * (np.log1p(xs) - np.log1p(BINMID[0])), ls, color=col, lw=1.1,
            alpha=0.8, label=lab)
side('adjacent_pair_min_km', adjd.min())
side('adjacent_pair_q75_km', adjd.quantile(0.75))
ax.axvspan(adjd.min(), adjd.quantile(0.75), color="#cccccc", alpha=0.25, zorder=0)
ax.annotate("interquartile range of\nadjacent-pair distances",
            xy=(adjd.quantile(0.4), ax.get_ylim()[0] * 0.55), fontsize=7.5,
            color="#666666", ha="center")
ax.set_xscale("log")
ax.set_xlabel("Centroid distance (km, log scale)")
ax.set_ylabel("Log rate ratio versus the nearest distance bin")
ax.set_title("Distance decay estimated without a functional form", fontsize=10, loc="left")
ax.grid(True, color="#ececec", lw=0.4)
ax.legend(loc="lower left", fontsize=7.5)
for _n, _v in [('xlim_lo', ax.get_xlim()[0]), ('xlim_hi', ax.get_xlim()[1]),
               ('ylim_lo', ax.get_ylim()[0]), ('ylim_hi', ax.get_ylim()[1])]:
    side(_n, _v)
for ext in ["pdf", "png", "svg"]:
    fig.savefig(f"paper/figures/FigS5_distance_decay.{ext}",
                dpi=600 if ext == "png" else None, bbox_inches="tight")
plt.close(fig)
pd.DataFrame(SIDE).to_csv('outputs/audit/s05_figure_sidecar.csv', index=False,
                          encoding='utf-8-sig')
print('sidecar: ' + str(len(SIDE)) + ' drawn values -> outputs/audit/s05_figure_sidecar.csv')

Path("outputs/solve/s05_distance_bins.txt").write_text("\n".join(report) + "\n",
                                                       encoding="utf-8")
print("\nwrote outputs/solve/s05_distance_bins.{csv,txt} and FigS5_distance_decay.*")
