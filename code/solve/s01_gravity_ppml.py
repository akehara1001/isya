"""
SOLVE 1 -- replace the positive-flow-conditioned NB-GLM with a zero-inclusive estimator.

The published model conditions on F_ij > 0, which is estimation on a selected sample.
The standard fix for gravity data is Poisson pseudo-maximum-likelihood on the FULL dyad
set with the true zeros included (Silva & Tenreyro 2006). PPML is consistent under
heteroskedasticity, requires no distributional assumption beyond the conditional mean,
and handles zeros natively -- which is why no zero-inflated or hurdle layer is added.

Three specifications, reported side by side:

  (a) PPML, descriptive.  Same covariates as published Eq. (8) plus year fixed effects,
      on all 330 x 329 x 5 = 542,850 directed pairs. Origin stock and destination
      density are identified.
  (b) PPML, structural.   (a) plus origin and destination fixed effects, which absorb
      time-invariant multilateral resistance. Distance, adjacency and same-prefecture
      are identified off within-pair variation; origin stock and destination density
      survive only through their within-unit time variation.
  (b2) PPML, full structural. origin x year and destination x year fixed effects, the
      textbook multilateral-resistance specification. Origin stock and destination
      density are fully absorbed and drop out.
  (c) NB-GLM on positive flows only -- the published model, for comparison.

Inference: two-way cluster-robust on origin and destination (Cameron-Gelbach-Miller),
computed by the sandwich formula V = V_origin + V_dest - V_origin,dest.

PPML with high-dimensional fixed effects is fitted by Fisher scoring on a sparse design
matrix; this is the ordinary Poisson MLE, not an approximation.

Deterministic (no randomness anywhere). Writes outputs/solve/s01_gravity_ppml.csv and .txt
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd
import scipy.sparse as sp
import scipy.linalg as sla
import statsmodels.api as sm
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)

# A clean re-run starts with outputs/ absent; create the directory rather than
# assuming a previous run left it behind.
os.makedirs("outputs/solve", exist_ok=True)
sys.path.insert(0, ROOT)
from phase3_common import (OD_YEARS, distance_matrix, load_adjacency, load_density,
                           load_infrastructure, load_od_matrix, smsa_codes, stock_series)

report = []


def say(s=""):
    print(s)
    report.append(s)


# ============================================================ build the FULL dyad set
CODES = smsa_codes()
N = len(CODES)
dist = distance_matrix(CODES).to_numpy()
adjm = load_adjacency(codes=CODES).to_numpy()
code_arr = np.array(CODES)
off = ~np.eye(N, dtype=bool)
ri, ci = np.where(off)

frames = []
for year in OD_YEARS:
    od = load_od_matrix(year, codes=CODES).to_numpy(float, copy=True)
    np.fill_diagonal(od, 0.0)
    stock = stock_series(year, codes=CODES).reindex(CODES).to_numpy(float)
    dens = load_density(year, codes=CODES).to_numpy(float)   # raises if any is missing
    infra = load_infrastructure(year, codes=CODES).set_index("smsa_code")
    inf = (infra["hospitals"].reindex(CODES).fillna(0.0).to_numpy(float)
           + infra["clinics"].reindex(CODES).fillna(0.0).to_numpy(float))
    frames.append(pd.DataFrame({
        "year": year, "origin": code_arr[ri], "dest": code_arr[ci],
        "F": od[ri, ci],
        "log_dist": np.log1p(dist[ri, ci]),
        "adj": adjm[ri, ci].astype(float),
        "same_pref": (code_arr[ri].astype("<U2") == code_arr[ci].astype("<U2")).astype(float),
        "log_x_i": np.log1p(stock[ri]),
        "log_density_j": np.log1p(dens[ci]),
        "log_infra_total_j": np.log1p(inf[ci]),
    }))
D = pd.concat(frames, ignore_index=True)

say("=" * 100)
say("SOLVE 1 -- zero-inclusive gravity estimation")
say("=" * 100)
say(f"full directed dyad set: {N} x {N-1} x {len(OD_YEARS)} = {len(D):,} rows")
say(f"positive flows : {int((D.F > 0).sum()):,}  ({100*(D.F>0).mean():.2f}%)")
say(f"true zeros     : {int((D.F == 0).sum()):,}  ({100*(D.F==0).mean():.2f}%)")
say(f"total flow volume preserved: {D.F.sum():,.0f}")
say("")
say("zeros by wave:")
for y in OD_YEARS:
    d = D[D.year == y]
    say(f"   {y}: {int((d.F==0).sum()):>7,} zeros of {len(d):>7,}  ({100*(d.F==0).mean():.2f}%)")

PRED = ["log_dist", "adj", "same_pref", "log_x_i", "log_density_j", "log_infra_total_j"]
y = D["F"].to_numpy(float)
o_idx = pd.factorize(D["origin"])[0]
d_idx = pd.factorize(D["dest"])[0]
yr_idx = pd.factorize(D["year"])[0]
oy_idx = pd.factorize(D["origin"].astype(str) + "_" + D["year"].astype(str))[0]
dy_idx = pd.factorize(D["dest"].astype(str) + "_" + D["year"].astype(str))[0]


def dummies(idx, drop_first=True):
    k = idx.max() + 1
    keep = np.arange(1, k) if drop_first else np.arange(k)
    m = sp.csr_matrix((np.ones(len(idx)), (np.arange(len(idx)), idx)), shape=(len(idx), k))
    return m[:, keep]


def build_X(preds, fes):
    blocks = [sp.csr_matrix(np.ones((len(D), 1))),
              sp.csr_matrix(D[preds].to_numpy(float))]
    names = ["const"] + list(preds)
    for nm, idx in fes:
        blocks.append(dummies(idx))
        names += [f"{nm}_{i}" for i in range(1, idx.max() + 1)]
    return sp.hstack(blocks).tocsr(), names


def drop_separated(fe_list):
    """PPML separation: a fixed-effect group whose outcome sums to zero has its
    coefficient driven to -infinity and is not identified. Drop such groups
    iteratively until none remains -- the standard ppmlhdfe pre-step."""
    keep = np.ones(len(y), bool)
    while True:
        drop = np.zeros(len(y), bool)
        for _, idx in fe_list:
            tot = np.bincount(idx[keep], weights=y[keep], minlength=idx.max() + 1)
            bad = np.where(tot == 0)[0]
            if len(bad):
                drop |= keep & np.isin(idx, bad)
        if not drop.any():
            return keep
        keep &= ~drop


def ppml(X, names, keep=None, tol=1e-10, maxit=200):
    """Poisson MLE by Fisher scoring on a sparse design (with step halving)."""
    global y, o_idx, d_idx
    n, k = X.shape
    b = np.zeros(k)
    b[0] = np.log(max(y.mean(), 1e-8))

    def ll(bb):
        e_ = np.clip(X @ bb, -30, 30)
        return float((y * e_ - np.exp(e_)).sum())

    cur = ll(b)
    for it in range(maxit):
        eta = np.clip(X @ b, -30, 30)
        mu = np.exp(eta)
        Xw = X.multiply(mu[:, None]).tocsr()
        H = np.asarray((X.T @ Xw).todense())
        g = X.T @ (y - mu)
        H[np.diag_indices_from(H)] += 1e-7
        step = np.linalg.lstsq(H, g, rcond=None)[0]
        t = 1.0
        for _ in range(30):
            cand = ll(b + t * step)
            if np.isfinite(cand) and cand >= cur - 1e-9:
                break
            t *= 0.5
        b = b + t * step
        new = ll(b)
        if abs(new - cur) < tol * (1 + abs(cur)):
            cur = new
            break
        cur = new
    eta = np.clip(X @ b, -30, 30)
    mu = np.exp(eta)
    Xw = X.multiply(mu[:, None]).tocsr()
    H = np.asarray((X.T @ Xw).todense())
    H[np.diag_indices_from(H)] += 1e-7
    Hinv = np.linalg.pinv(H)
    e = y - mu
    Xe = X.multiply(e[:, None]).tocsr()

    def meat(g):
        ng = g.max() + 1
        S = sp.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))), shape=(ng, len(g)))
        A = np.asarray((S @ Xe).todense())
        return A.T @ A

    oc, dc = o_idx, d_idx
    od_pair = pd.factorize(pd.Series(oc).astype(str) + "_" + pd.Series(dc).astype(str))[0]
    V = Hinv @ (meat(oc) + meat(dc) - meat(od_pair)) @ Hinv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    return b, se, it + 1, mu


def report_spec(tag, preds, fes, note=""):
    global y, o_idx, d_idx, D
    y0, o0, d0, D0 = y, o_idx, d_idx, D
    keep = drop_separated(fes) if fes else np.ones(len(y), bool)
    ndrop = int((~keep).sum())
    if ndrop:
        y = y0[keep]; o_idx = o0[keep]; d_idx = d0[keep]; D = D0[keep].reset_index(drop=True)
        fes = [(nm, pd.factorize(idx[keep])[0]) for nm, idx in fes]
    X, names = build_X(preds, fes)
    b, se, nit, mu = ppml(X, names)
    say("")
    say(f"--- {tag} ---  design {X.shape[0]:,} x {X.shape[1]}, converged in {nit} iterations")
    if ndrop:
        say(f"    {ndrop:,} observations dropped: fixed-effect groups whose flows are all "
            f"zero are separated and not identified (standard PPML pre-step)")
    if note:
        say(f"    {note}")
    say(f"{'term':>22} {'coef':>9} {'SE (2-way)':>11} {'95% CI':>24} {'z':>7} {'p':>10}")
    rows = []
    for t in preds:
        if t not in names:
            continue
        i = names.index(t)
        lo, hi = b[i] - 1.96 * se[i], b[i] + 1.96 * se[i]
        z = b[i] / se[i] if se[i] > 0 else np.nan
        from scipy.stats import norm
        pv = 2 * (1 - norm.cdf(abs(z)))
        say(f"{t:>22} {b[i]:>+9.4f} {se[i]:>11.4f} [{lo:>+9.4f},{hi:>+9.4f}] "
            f"{z:>7.2f} {pv:>10.2e}")
        rows.append({"spec": tag, "term": t, "coef": round(b[i], 4), "se": round(se[i], 4),
                     "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "z": round(z, 2),
                     "p": pv, "sig": bool(lo > 0 or hi < 0), "n_obs": int(X.shape[0])})
    y, o_idx, d_idx, D = y0, o0, d0, D0
    return rows


results = []
results += report_spec(
    "(a) PPML descriptive, year FE", PRED, [("year", yr_idx)],
    "all covariates identified; comparable term-for-term with the published NB-GLM")
results += report_spec(
    "(b) PPML structural, origin + destination + year FE", PRED,
    [("year", yr_idx), ("orig", o_idx), ("dest", d_idx)],
    "unit FE absorb time-invariant multilateral resistance; log_x_i and log_density_j "
    "survive only via within-unit time variation")
results += report_spec(
    "(b2) PPML full structural, origin x year + destination x year FE",
    ["log_dist", "adj", "same_pref"], [("oy", oy_idx), ("dy", dy_idx)],
    "origin stock and destination density are fully absorbed and drop out by construction")

# ---------------------------------------------------------------- (c) published NB
say("")
say("--- (c) NB-GLM on positive flows only (the published model) ---")
Dp = D[D.F > 0].copy()
yd = pd.get_dummies(Dp["year"].astype(str), prefix="year", drop_first=True, dtype=float)
Xp = sm.add_constant(pd.concat([Dp[PRED].astype(float).reset_index(drop=True),
                                yd.reset_index(drop=True)], axis=1), has_constant="add")
tw = np.column_stack([pd.factorize(Dp["origin"])[0], pd.factorize(Dp["dest"])[0]])
nb = sm.GLM(Dp["F"].to_numpy(float), Xp.astype(float),
            family=sm.families.NegativeBinomial(alpha=1.0)).fit(
    maxiter=200, cov_type="cluster", cov_kwds={"groups": tw, "use_correction": True})
say(f"    n = {len(Dp):,} positive flows")
say(f"{'term':>22} {'coef':>9} {'SE (2-way)':>11} {'95% CI':>24}")
for t in PRED:
    lo, hi = nb.conf_int().loc[t]
    say(f"{t:>22} {nb.params[t]:>+9.4f} {nb.bse[t]:>11.4f} [{lo:>+9.4f},{hi:>+9.4f}]")
    results.append({"spec": "(c) NB-GLM positive flows (published)", "term": t,
                    "coef": round(float(nb.params[t]), 4), "se": round(float(nb.bse[t]), 4),
                    "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                    "z": round(float(nb.params[t] / nb.bse[t]), 2),
                    "p": float(nb.pvalues[t]), "sig": bool(lo > 0 or hi < 0)})

R = pd.DataFrame(results)
R.to_csv("outputs/solve/s01_gravity_ppml.csv", index=False, encoding="utf-8-sig")

say("")
say("=" * 100)
say("VERDICTS")
say("=" * 100)
piv = R.pivot_table(index="term", columns="spec", values="coef")
say(piv.to_string())
say("")
for t in PRED:
    sub = R[R.term == t]
    say(f"{t:>22}: " + "  ".join(
        f"{r.spec.split()[0]}{'+' if r.sig and r.coef > 0 else ('-' if r.sig and r.coef < 0 else '0')}"
        for _, r in sub.iterrows()))
say("")
say("legend: + significantly positive, - significantly negative, 0 not distinguishable "
    "from zero (two-way clustered)")

Path("outputs/solve/s01_gravity_ppml.txt").write_text("\n".join(report) + "\n",
                                                      encoding="utf-8")
print("\nwrote outputs/solve/s01_gravity_ppml.{csv,txt}")
