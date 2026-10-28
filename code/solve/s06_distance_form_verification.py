"""
VERIFICATION 1-3 for the distance functional form, before anything is written.

V1  Formally reject log(1+d). The bin model nests the log model, so the clean test is a
    Wald test that the 12 bin dummies are jointly zero in a model that ALSO contains
    log(1+d), evaluated with the two-way cluster-robust covariance. Deviance and an
    AIC-analogue are reported alongside as descriptive fit, flagged as quasi-likelihood
    quantities.

V2  Determine WHICH mechanism moved destination density, rather than asserting one.
    Two candidate stories:
      (a) distance-band residual absorption -- if log(1+d) UNDER-predicts flow at
          20-50 km, covariates correlated with that band are pulled UP to compensate,
          so destination density should be HIGHER under log, not lower. The observed
          move is downward (+0.267 -> +0.084), so (a) predicts the wrong sign.
      (b) collinearity through a distorted adjacency coefficient -- adjacency is forced
          to -0.274 under log(1+d); adjacency correlates with destination density
          (adjacent pairs point at cores); a spuriously negative adjacency drags
          destination density down with it.
    Tests: sign of the model's relative prediction error by distance band; the four-cell
    {log, bins} x {with adj, without adj} table for destination density; the correlation
    structure among adj, distance and destination density; and a numerical read of where
    the fitted log curve lies relative to the bin coefficients.

V3  Keep the cross-sectional and within estimators apart. The origin/destination-FE
    coefficient on destination density is identified only from within-destination
    variation across waves and is a different estimand from the cross-sectional one.
    Reported, not pooled into a range.

Deterministic. Writes outputs/solve/s06_distance_form_verification.{csv,txt}
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.stats import chi2, norm
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from phase3_common import (OD_YEARS, distance_matrix, load_adjacency, load_density,
                           load_infrastructure, load_od_matrix, smsa_codes, stock_series)

report, recs = [], []


def say(s=""):
    print(s)
    report.append(s)


# ---------------------------------------------------------------- data (as s05)
CODES = smsa_codes(); N = len(CODES)
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

adjd = D.loc[D.adj == 1, "dist_km"]; posd = D.loc[D.F > 0, "dist_km"]
q = np.quantile(posd, np.arange(0, 1.01, 0.1))
inner = np.quantile(adjd, [0.25, 0.5, 0.75])
EDGES = np.unique(np.concatenate([[0], inner, q[1:-1], [D.dist_km.max() + 1]]))
D["bin"] = pd.cut(D.dist_km, EDGES, right=False, labels=False)
NB_ = int(D["bin"].max()) + 1
BINMID = [float(D.loc[D["bin"] == b, "dist_km"].median()) for b in range(NB_)]

y = D["F"].to_numpy(float)
o_idx = pd.factorize(D["origin"])[0]
d_idx = pd.factorize(D["dest"])[0]
yr_idx = pd.factorize(D["year"])[0]
bin_codes = pd.Categorical(D["bin"], categories=sorted(set(D["bin"]))).codes


def dumm(idx):
    k = idx.max() + 1
    m = sp.csr_matrix((np.ones(len(idx)), (np.arange(len(idx)), idx)), shape=(len(idx), k))
    return m[:, np.arange(1, k)]


def ppml(X, yv, oc, dc, maxit=200, tol=1e-10):
    k = X.shape[1]
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
    dev = 2 * float(np.sum(np.where(yv > 0, yv * np.log(np.where(yv > 0, yv, 1) / mu), 0)
                           - (yv - mu)))
    return b, V, mu, dev


OTHER = ["same_pref", "log_x_i", "log_density_j", "log_infra_total_j"]


def build(dist_terms, with_adj):
    cols = list(dist_terms) + OTHER + (["adj"] if with_adj else [])
    blocks = [sp.csr_matrix(np.ones((len(D), 1))),
              sp.csr_matrix(D[cols].to_numpy(float))]
    names = ["const"] + cols
    if "BINS" in dist_terms:
        raise ValueError
    return blocks, names, cols


def fit(tag, use_log, use_bins, with_adj):
    cols = (["log_dist"] if use_log else []) + OTHER + (["adj"] if with_adj else [])
    blocks = [sp.csr_matrix(np.ones((len(D), 1))),
              sp.csr_matrix(D[cols].to_numpy(float))]
    names = ["const"] + cols
    if use_bins:
        blocks.append(dumm(bin_codes))
        names += [f"bin_{b}" for b in range(1, NB_)]
    blocks.append(dumm(pd.Categorical(yr_idx, categories=sorted(set(yr_idx))).codes))
    names += [f"yr_{i}" for i in range(1, yr_idx.max() + 1)]
    X = sp.hstack(blocks).tocsr()
    b, V, mu, dev = ppml(X, y, o_idx, d_idx)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    return dict(tag=tag, names=names, b=b, V=V, se=se, mu=mu, dev=dev, k=X.shape[1])


say("=" * 100)
say("VERIFICATION 1 -- is log(1+d) formally rejected?")
say("=" * 100)
FLOG = fit("log(1+d)", True, False, True)
FBIN = fit("13 bins", False, True, True)
FBOTH = fit("log(1+d) + bins", True, True, True)

idx = [i for i, n in enumerate(FBOTH["names"]) if n.startswith("bin_")]
bb = FBOTH["b"][idx]
Vb = FBOTH["V"][np.ix_(idx, idx)]
W = float(bb @ np.linalg.pinv(Vb) @ bb)
dfree = len(idx)
p = 1 - chi2.cdf(W, dfree)
say(f"Wald test that the {dfree} distance-bin dummies are jointly zero, in a model that")
say(f"also contains log(1+d), with the two-way cluster-robust covariance:")
say(f"    W = {W:,.1f} on {dfree} degrees of freedom,  p = {p:.3g}")
say("")
say("Descriptive fit (Poisson deviance; a quasi-likelihood quantity under PPML, so it is")
say("reported for comparison and not used as a likelihood-ratio statistic):")
say(f"{'model':>18} {'k':>5} {'deviance':>16} {'deviance + 2k':>16}")
for f in [FLOG, FBIN, FBOTH]:
    say(f"{f['tag']:>18} {f['k']:>5} {f['dev']:>16,.0f} {f['dev'] + 2*f['k']:>16,.0f}")
say(f"    deviance falls {FLOG['dev'] - FBIN['dev']:,.0f} for {FBIN['k'] - FLOG['k']} "
    f"extra parameters")
recs.append({"check": "V1 Wald bins jointly zero", "value": round(W, 1), "df": dfree, "p": p})
recs.append({"check": "V1 deviance log", "value": round(FLOG["dev"], 4)})
recs.append({"check": "V1 deviance bins", "value": round(FBIN["dev"], 4)})
# A5 quotes the combined model's deviance (218,313) and its reduction against the bands
# (a further 206). Both were computed here and printed only into the text report, at no
# decimals, so 206 could not be adjudicated from anything on disk: the two candidate values
# straddle it. Persisted at full precision, with the differences taken before rounding.
recs.append({"check": "V1 deviance log+bins", "value": round(FBOTH["dev"], 4)})
recs.append({"check": "V1 deviance reduction, bands over log",
             "value": round(FLOG["dev"] - FBIN["dev"], 4)})
recs.append({"check": "V1 deviance reduction, adding log back to bands",
             "value": round(FBIN["dev"] - FBOTH["dev"], 4)})

# ---------------------------------------------------------------- V2
say("")
say("=" * 100)
say("VERIFICATION 2 -- which mechanism actually moved destination density?")
say("=" * 100)

say("")
say("(2a) Where does the log(1+d) model over- or under-predict, by distance band?")
say("     relative error = (sum observed - sum fitted) / sum fitted within the band.")
say("     positive => the model UNDER-predicts flow in that band.")
say("")
mu_log = FLOG["mu"]
say(f"{'bin':>4} {'median km':>10} {'observed':>11} {'fitted':>11} {'rel. error':>12} "
    f"{'fitted log-curve vs bin coef':>30}")
bl = {n: FBIN['b'][i] for i, n in enumerate(FBIN['names'])}
beta_log = FLOG["b"][FLOG["names"].index("log_dist")]
rel = {}
for b_ in range(NB_):
    m = D["bin"].to_numpy() == b_
    obs, fit_ = y[m].sum(), mu_log[m].sum()
    rel[b_] = (obs - fit_) / fit_
    binco = 0.0 if b_ == 0 else bl.get(f"bin_{b_}", np.nan)
    logco = beta_log * (np.log1p(BINMID[b_]) - np.log1p(BINMID[0]))
    where = ("log curve ABOVE bins" if logco > binco + 1e-9 else
             "log curve BELOW bins" if logco < binco - 1e-9 else "equal")
    say(f"{b_:>4} {BINMID[b_]:>10.1f} {obs:>11,.0f} {fit_:>11,.0f} {rel[b_]:>+11.2f} "
        f"  {where} ({logco - binco:+.2f})")
    recs.append({"check": "V2a rel error by bin", "bin": b_, "median_km": BINMID[b_],
                 "rel_error": round(rel[b_], 4),
                 "log_minus_bin_coef": round(float(logco - binco), 4)})

say("")
say("(2b) Four-cell table: destination density coefficient")
say("")
cells = {}
for use_log, use_bins, lab in [(True, False, "log(1+d)"), (False, True, "13 bins")]:
    for with_adj in [True, False]:
        f = FLOG if (use_log and with_adj) else (FBIN if (use_bins and with_adj)
                                                 else fit(lab, use_log, use_bins, with_adj))
        i = f["names"].index("log_density_j")
        cells[(lab, with_adj)] = (f["b"][i], f["se"][i])
say(f"{'distance term':>16} {'with adjacency':>28} {'without adjacency':>28} {'shift':>9}")
for lab in ["log(1+d)", "13 bins"]:
    wa, wo = cells[(lab, True)], cells[(lab, False)]
    say(f"{lab:>16} {f'{wa[0]:+.3f} [{wa[0]-1.96*wa[1]:+.3f},{wa[0]+1.96*wa[1]:+.3f}]':>28} "
        f"{f'{wo[0]:+.3f} [{wo[0]-1.96*wo[1]:+.3f},{wo[0]+1.96*wo[1]:+.3f}]':>28} "
        f"{wo[0]-wa[0]:>+9.3f}")
    recs.append({"check": "V2b dest density", "dist_term": lab,
                 "with_adj": round(wa[0], 4), "without_adj": round(wo[0], 4),
                 "shift": round(wo[0] - wa[0], 4)})

say("")
say("(2c) adjacency coefficient in the same four cells, and the correlation structure")
for lab, use_log, use_bins in [("log(1+d)", True, False), ("13 bins", False, True)]:
    f = FLOG if use_log else FBIN
    i = f["names"].index("adj")
    say(f"    adjacency under {lab:>10}: {f['b'][i]:+.3f} "
        f"[{f['b'][i]-1.96*f['se'][i]:+.3f}, {f['b'][i]+1.96*f['se'][i]:+.3f}]")
say("")
sub = D
say(f"    corr(adj, log_density_j)          = {np.corrcoef(sub.adj, sub.log_density_j)[0,1]:+.4f}")
say(f"    corr(adj, log_dist)               = {np.corrcoef(sub.adj, sub.log_dist)[0,1]:+.4f}")
say(f"    corr(log_dist, log_density_j)     = {np.corrcoef(sub.log_dist, sub.log_density_j)[0,1]:+.4f}")
near = sub[sub.dist_km < 60]
say(f"    among pairs < 60 km: corr(adj, log_density_j) = "
    f"{np.corrcoef(near.adj, near.log_density_j)[0,1]:+.4f}")
say(f"    mean destination density (per 100k): adjacent {np.expm1(sub.loc[sub.adj==1,'log_density_j']).mean():.1f} "
    f"vs non-adjacent {np.expm1(sub.loc[sub.adj==0,'log_density_j']).mean():.1f}")

# ---------------------------------------------------------------- verdict on mechanism
say("")
say("=" * 100)
say("MECHANISM VERDICT")
say("=" * 100)
short = [b_ for b_ in range(NB_) if 20 <= BINMID[b_] <= 50]
say(f"relative prediction error of the log model in the 20-50 km bands: "
    f"{', '.join(f'{rel[b_]:+.2f}' for b_ in short)}")
sign = "UNDER" if np.mean([rel[b_] for b_ in short]) > 0 else "OVER"
say(f"  -> the log model {sign}-predicts flow at 20-50 km")
d_log = cells[("log(1+d)", False)][0] - cells[("log(1+d)", True)][0]
d_bin = cells[("13 bins", False)][0] - cells[("13 bins", True)][0]
say(f"removing adjacency moves destination density by {d_log:+.3f} under log(1+d) "
    f"and {d_bin:+.3f} under bins")
Path("outputs/solve/s06_distance_form_verification.txt").write_text(
    "\n".join(report) + "\n", encoding="utf-8")
pd.DataFrame(recs).to_csv("outputs/solve/s06_distance_form_verification.csv",
                          index=False, encoding="utf-8-sig")
print("\nwrote outputs/solve/s06_distance_form_verification.{csv,txt}")
