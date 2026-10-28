"""
AUDIT item 5 -- dyadic dependence in the gravity model standard errors.

The 26,011 directed flows are NOT independent: each origin appears up to 329 times
and each destination up to 329 times (and each dyad appears in up to 5 waves).  The
published CIs are conventional GLM CIs, which treat the rows as independent and are
therefore too narrow.

This is a VARIANCE correction to the SAME model, not a new model.  Point estimates are
unchanged by construction; only the standard errors move.  We refit the published G3a
specification with:
    (a) conventional (published) covariance, alpha = 1
    (b) cluster-robust by ORIGIN
    (c) cluster-robust by DESTINATION
    (d) cluster-robust by DYAD (unordered origin-destination pair, which also absorbs
        the repeated-wave dependence within a dyad)
    (e) TWO-WAY cluster-robust by origin AND destination simultaneously
        (Cameron-Gelbach-Miller), which is the theoretically appropriate covariance for
        dyadic data and is what we adopt
and the same set at the data-estimated dispersion alpha_hat.

We report all of them and adopt the widest (most conservative) interval, which is the
two-way one for every term.

Also recomputes the deviance-based pseudo-R^2 at BOTH alpha = 1 (as published) and at
alpha_hat, since the manuscript currently mixes the two.

WHERE THE INFRASTRUCTURE TERM COMES FROM, changed 2026-09-07. It used to be
log1p(hosp_j + clinic_j) read straight out of results/gravity_pooled_positive_pairs.csv.
Those two columns were frozen into that file by gravity_model.py, which is outside the
rebuild closure, and they held the prefecture-total-allocated-by-physician-share numbers
that cc_facilities_pipeline.py has now replaced -- so this script would have gone on
refitting the old variable after every other consumer had moved. The term is now built from
the infrastructure files by wave, exactly as phase3_common.load_infrastructure assigns them.
That the join is right was checked against the frozen columns BEFORE the data changed: on
the superseded files it reproduced hosp_j and clinic_j for all 26,011 rows to 4.5e-13.
The rest of the design matrix still comes from the frozen file and is unchanged.

Writes outputs/audit/a03_gravity_cluster.csv
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd
import statsmodels.api as sm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)

SEED = 20260805          # not used for randomness here; recorded for provenance
PRED = ["log_dist", "adj", "same_pref", "log_x_i", "log_density_j", "log_infra_total_j"]

d = pd.read_csv("results/gravity_pooled_positive_pairs.csv")
d.columns = [c.lstrip("﻿") for c in d.columns]
assert len(d) == 26533, len(d)   # round 40 §3: 26,011 -> 26,533 with the rebuilt flow matrices (round 39)
d["origin"] = d["origin"].apply(lambda c: str(c).zfill(4))
d["dest"] = d["dest"].apply(lambda c: str(c).zfill(4))

# One literal read per survey round, and the wave map that phase3_common applies.
INFRA_SRC = {2017: pd.read_csv("smsa_infrastructure_2017.csv", encoding="utf-8-sig"),
             2020: pd.read_csv("smsa_infrastructure_2020.csv", encoding="utf-8-sig"),
             2023: pd.read_csv("smsa_infrastructure_2023.csv", encoding="utf-8-sig")}
WAVE_SRC = {2016: 2017, 2018: 2017, 2020: 2020, 2022: 2023, 2024: 2023}
_infra = {}
for _wave, _src in WAVE_SRC.items():
    _f = INFRA_SRC[_src].copy()
    _f["smsa_code"] = _f["smsa_code"].apply(lambda c: str(c).zfill(4))
    _f = _f.set_index("smsa_code")
    _infra[_wave] = _f["hospitals"] + _f["clinics"]
total_infra = pd.Series(
    [float(_infra[int(y)].get(j, np.nan)) for y, j in zip(d["year"], d["dest"])],
    index=d.index)
assert total_infra.notna().all(), (
    f"{int(total_infra.isna().sum())} dyads have no destination facility count")
d["log_infra_total_j"] = np.log1p(total_infra)
d["log_density_j"] = np.log1p(pd.to_numeric(d["density_j"], errors="coerce").fillna(0.0))

yd = pd.get_dummies(d["year"].astype(str), prefix="year", drop_first=True, dtype=float)
X = sm.add_constant(pd.concat([d[PRED].astype(float), yd], axis=1),
                    has_constant="add").astype(float)
y = d["F"].to_numpy(float)

print(f"n = {len(d):,};  distinct origins = {d.origin.nunique()};  "
      f"distinct destinations = {d.dest.nunique()}")
print(f"max rows sharing one origin = {d.origin.value_counts().max()};  "
      f"max rows sharing one destination = {d.dest.value_counts().max()}")
dyad = d.apply(lambda r: "-".join(sorted([r.origin, r.dest])), axis=1)
print(f"distinct unordered dyads = {dyad.nunique()};  "
      f"max rows per dyad = {dyad.value_counts().max()}")

# ---- dispersion estimate (same procedure that produced the published alpha_hat) ----
poi = sm.GLM(y, X, family=sm.families.Poisson()).fit(maxiter=200)
mu = poi.fittedvalues
alpha_hat = float(((y - mu) ** 2 - mu).sum() / (mu ** 2).sum())
print(f"\nmethod-of-moments alpha_hat = {alpha_hat:.4f}")


def fit(alpha, cov, groups=None):
    m = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha))
    if cov == "nonrobust":
        return m.fit(maxiter=200)
    return m.fit(maxiter=200, cov_type="cluster",
                 cov_kwds={"groups": np.asarray(groups), "use_correction": True})


rows = []
TWOWAY = np.column_stack([pd.factorize(d["origin"])[0], pd.factorize(d["dest"])[0]])
CLUSTERS = {"nonrobust": None, "cluster_origin": d["origin"].to_numpy(),
            "cluster_dest": d["dest"].to_numpy(), "cluster_dyad": dyad.to_numpy(),
            "cluster_twoway_origin_dest": TWOWAY}

for alpha, atag in [(1.0, "alpha=1 (published)"), (alpha_hat, f"alpha_hat={alpha_hat:.3f}")]:
    for cname, g in CLUSTERS.items():
        r = fit(alpha, "nonrobust" if g is None else "cluster", g)
        pr2 = 1 - r.deviance / r.null_deviance
        for term in PRED:
            b = r.params[term]; se = r.bse[term]
            lo, hi = r.conf_int().loc[term]
            rows.append({"alpha_setting": atag, "alpha": round(alpha, 4), "cov": cname,
                         # SIX decimals, not four. Table A5 prints the dyad-clustered width
                         # as 0.108; at 4 dp the stored width is exactly 0.1075, which is
                         # the tie, so the file could not say whether the page rounded the
                         # true value or re-rounded an already-rounded one. A result file
                         # must not be the reason a printed digit is unresolvable.
                         "term": term, "coef": round(float(b), 6), "se": round(float(se), 6),
                         "ci_lo": round(float(lo), 6), "ci_hi": round(float(hi), 6),
                         "ci_width": round(float(hi - lo), 6),
                         "p": float(r.pvalues[term]),
                         "pseudo_R2": round(float(pr2), 6),
                         "n_clusters": (np.nan if g is None else
                                        (int(np.asarray(g).shape[0]) if np.asarray(g).ndim == 2
                                         else int(pd.Series(g).nunique())))})

out = pd.DataFrame(rows)
out.to_csv("outputs/audit/a03_gravity_cluster.csv", index=False)

print("\n" + "=" * 100)
print("DESTINATION DENSITY coefficient under each covariance")
print("=" * 100)
dd = out[out["term"] == "log_density_j"]
print(dd[["alpha_setting", "cov", "coef", "se", "ci_lo", "ci_hi", "ci_width", "p", "n_clusters"]]
      .to_string(index=False))

print("\n" + "=" * 100)
print("ALL terms, published alpha=1: conventional vs each clustering")
print("=" * 100)
piv = out[out["alpha"] == 1.0].pivot_table(index="term", columns="cov",
                                        values=["coef", "se", "ci_lo", "ci_hi"])
print(piv.to_string())

print("\n" + "=" * 100)
print("pseudo-R^2 by dispersion setting (point estimate is unaffected by the covariance)")
print("=" * 100)
for atag in out["alpha_setting"].unique():
    sub = out[(out["alpha_setting"] == atag) & (out["cov"] == "nonrobust")]
    print(f"  {atag}: pseudo-R2 = {sub['pseudo_R2'].iloc[0]:.4f}")

print("\n" + "=" * 100)
print("WIDEST (most conservative) interval per term, across all clusterings, per alpha")
print("=" * 100)
for atag in out["alpha_setting"].unique():
    sub = out[out["alpha_setting"] == atag]
    for term in PRED:
        t = sub[sub["term"] == term].sort_values("ci_width").iloc[-1]
        sig = "sig" if (t.ci_lo > 0 or t.ci_hi < 0) else "NOT SIG"
        print(f"  {atag:>22} {term:>20}: {t.coef:+.4f} [{t.ci_lo:+.4f}, {t.ci_hi:+.4f}] "
              f"via {t['cov']:>15}  {sig}")
