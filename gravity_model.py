#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimate service-mobility gravity models from Table 24 OD matrices."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedPoisson
from statsmodels.stats.outliers_influence import variance_inflation_factor

from phase3_common import (
    OD_YEARS,
    RESULTS_DIR,
    distance_matrix,
    ensure_results_dir,
    load_adjacency,
    load_centroids,
    load_infrastructure,
    load_od_matrix,
    load_population,
    load_density,
    save_figure,
    save_json,
    smsa_codes,
    stock_series,
)


BASE_PREDICTORS = [
    "log_dist",
    "adj",
    "same_pref",
    "log_x_i",
    "log_x_j",
    "log_hosp_j",
    "log_pop_j",
    "log_density_j",
]

V2_MODELS = {
    "G2a_NegBin_dest_stock_population": ["log_dist", "adj", "same_pref", "log_x_i", "log_hosp_j", "log_x_j", "log_pop_j"],
    "G2b_NegBin_dest_density_only": ["log_dist", "adj", "same_pref", "log_x_i", "log_hosp_j", "log_density_j"],
    "G2c_NegBin_dest_stock_only": ["log_dist", "adj", "same_pref", "log_x_i", "log_hosp_j", "log_x_j"],
}


def build_year_dataset(
    year: int,
    codes: list[str],
    dist: pd.DataFrame,
    adjacency: pd.DataFrame,
    positive_only: bool,
    sample_size: int | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build one year of OD-pair model data."""

    od = load_od_matrix(year, codes=codes)
    values = od.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(values, 0.0)
    n = len(codes)
    if positive_only:
        row_idx, col_idx = np.where(values > 0)
    else:
        row_idx, col_idx = np.where(~np.eye(n, dtype=bool))
        if sample_size is not None and len(row_idx) > sample_size:
            if rng is None:
                rng = np.random.default_rng(20260619 + year)
            selected = rng.choice(len(row_idx), size=sample_size, replace=False)
            row_idx = row_idx[selected]
            col_idx = col_idx[selected]

    stock = stock_series(year, codes=codes)
    infra = load_infrastructure(year, codes=codes).set_index("smsa_code")
    pop = load_population(year, codes=codes)
    density = load_density(year, codes=codes)

    code_array = np.array(codes)
    origin = code_array[row_idx]
    dest = code_array[col_idx]
    data = pd.DataFrame(
        {
            "year": year,
            "origin": origin,
            "dest": dest,
            "F": values[row_idx, col_idx].astype(float),
            "dist_km": dist.to_numpy()[row_idx, col_idx],
            "adj": adjacency.to_numpy()[row_idx, col_idx],
            "same_pref": (pd.Series(origin).str[:2].to_numpy() == pd.Series(dest).str[:2].to_numpy()).astype(int),
            "x_i": stock.reindex(origin).to_numpy(dtype=float),
            "x_j": stock.reindex(dest).to_numpy(dtype=float),
            "hosp_j": infra["hospitals"].reindex(dest).fillna(0.0).to_numpy(dtype=float),
            "clinic_j": infra["clinics"].reindex(dest).fillna(0.0).to_numpy(dtype=float),
            "pop_j": pop.reindex(dest).fillna(0.0).to_numpy(dtype=float),
            "density_j": density.reindex(dest).fillna(0.0).to_numpy(dtype=float),
        }
    )
    data["log_dist"] = np.log1p(data["dist_km"])
    data["log_x_i"] = np.log1p(data["x_i"])
    data["log_x_j"] = np.log1p(data["x_j"])
    data["log_hosp_j"] = np.log1p(data["hosp_j"])
    data["log_pop_j"] = np.log1p(data["pop_j"])
    data["log_density_j"] = np.log1p(data["density_j"])
    return data


def build_pooled_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build positive-flow and sampled full-matrix pooled datasets."""

    codes = smsa_codes()
    dist = distance_matrix(codes)
    adjacency = load_adjacency(codes=codes)
    rng = np.random.default_rng(20260619)
    positives = []
    sampled = []
    for year in OD_YEARS:
        positives.append(build_year_dataset(year, codes, dist, adjacency, positive_only=True))
        sampled.append(build_year_dataset(year, codes, dist, adjacency, positive_only=False, sample_size=50_000, rng=rng))
    return pd.concat(positives, ignore_index=True), pd.concat(sampled, ignore_index=True)


def design_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Create the regression design matrix with year fixed effects."""

    year_dummies = pd.get_dummies(data["year"].astype(str), prefix="year", drop_first=True, dtype=float)
    x = pd.concat([data[BASE_PREDICTORS].astype(float), year_dummies], axis=1)
    return sm.add_constant(x, has_constant="add").astype(float)


def design_matrix_for_predictors(data: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """Create a design matrix for a specific predictor set plus year fixed effects."""

    year_dummies = pd.get_dummies(data["year"].astype(str), prefix="year", drop_first=True, dtype=float)
    x = pd.concat([data[predictors].astype(float), year_dummies], axis=1)
    return sm.add_constant(x, has_constant="add").astype(float)


def calculate_vif(exog: pd.DataFrame) -> dict[str, float | None]:
    """Calculate variance inflation factors for non-constant regressors."""

    clean = exog.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    vif: dict[str, float | None] = {}
    if len(clean) == 0:
        return {col: None for col in exog.columns if col != "const"}
    values = clean.to_numpy(dtype=float)
    for idx, col in enumerate(clean.columns):
        if col == "const":
            continue
        try:
            score = float(variance_inflation_factor(values, idx))
            vif[col] = score if np.isfinite(score) else None
        except Exception:
            vif[col] = None
    return vif


def summarize_result(result, predictors: list[str]) -> dict:
    """Extract coefficients, standard errors, p-values, and fit metrics."""

    params = pd.Series(result.params, index=getattr(result.params, "index", None))
    if params.index.isnull().any() or len(params.index) != len(result.params):
        names = getattr(result.model, "exog_names", [f"param_{i}" for i in range(len(result.params))])
        params = pd.Series(result.params, index=names)
    bse = pd.Series(getattr(result, "bse", np.full(len(params), np.nan)), index=params.index)
    pvalues = pd.Series(getattr(result, "pvalues", np.full(len(params), np.nan)), index=params.index)
    coeffs = {}
    for name in predictors:
        coeffs[name] = {
            "coef": float(params.get(name, np.nan)),
            "se": float(bse.get(name, np.nan)),
            "p": float(pvalues.get(name, np.nan)),
        }
    pseudo = getattr(result, "rsquared", np.nan)
    if not np.isfinite(pseudo):
        deviance = getattr(result, "deviance", np.nan)
        null_deviance = getattr(result, "null_deviance", np.nan)
        if np.isfinite(deviance) and np.isfinite(null_deviance) and null_deviance != 0:
            pseudo = 1.0 - deviance / null_deviance
        elif hasattr(result, "prsquared"):
            pseudo = result.prsquared
    return {
        "coefficients": coeffs,
        "aic": float(getattr(result, "aic", np.nan)),
        "bic": float(getattr(result, "bic", getattr(result, "bic_llf", np.nan))),
        "pseudo_R2": float(pseudo) if np.isfinite(pseudo) else None,
        "nobs": int(getattr(result, "nobs", len(result.model.endog))),
        "converged": bool(getattr(result, "converged", getattr(result, "mle_retvals", {}).get("converged", True))),
    }


def fit_models(positive: pd.DataFrame, sampled: pd.DataFrame) -> tuple[dict, dict]:
    """Fit gravity models and return result summaries plus live result objects."""

    outputs: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    x_pos = design_matrix(positive)
    y_log = np.log1p(positive["F"].to_numpy(dtype=float))
    y_count = positive["F"].to_numpy(dtype=float)

    ols = sm.OLS(y_log, x_pos).fit(cov_type="HC1")
    outputs["G1_OLS_log_positive"] = summarize_result(ols, list(x_pos.columns))
    fitted["G1_OLS_log_positive"] = ols

    poisson = sm.GLM(y_count, x_pos, family=sm.families.Poisson()).fit(maxiter=100)
    outputs["G1_GLM_Poisson_positive"] = summarize_result(poisson, list(x_pos.columns))
    fitted["G1_GLM_Poisson_positive"] = poisson

    nb = sm.GLM(y_count, x_pos, family=sm.families.NegativeBinomial(alpha=1.0)).fit(maxiter=100)
    outputs["G2_GLM_NegativeBinomial_positive"] = summarize_result(nb, list(x_pos.columns))
    fitted["G2_GLM_NegativeBinomial_positive"] = nb

    x_full = design_matrix(sampled)
    y_full = sampled["F"].to_numpy(dtype=float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zip_model = ZeroInflatedPoisson(y_full, x_full, exog_infl=np.ones((len(x_full), 1)), inflation="logit")
            zip_result = zip_model.fit(method="bfgs", maxiter=60, disp=0)
        outputs["G3_ZeroInflatedPoisson_sampled"] = summarize_result(zip_result, list(x_full.columns))
        fitted["G3_ZeroInflatedPoisson_sampled"] = zip_result
    except Exception as exc:
        outputs["G3_ZeroInflatedPoisson_sampled"] = {"error": str(exc), "nobs": int(len(sampled))}
    return outputs, fitted


def fit_v2_negbin_models(positive: pd.DataFrame) -> tuple[dict, dict]:
    """Fit collinearity-safe Negative Binomial gravity variants."""

    outputs: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    y_count = positive["F"].to_numpy(dtype=float)
    for model_name, predictors in V2_MODELS.items():
        x = design_matrix_for_predictors(positive, predictors)
        nb = sm.GLM(y_count, x, family=sm.families.NegativeBinomial(alpha=1.0)).fit(maxiter=150)
        summary = summarize_result(nb, list(x.columns))
        summary["predictor_set"] = predictors
        summary["vif"] = calculate_vif(x)
        outputs[model_name] = summary
        fitted[model_name] = nb
    return outputs, fitted


def plot_coefficients(results: dict) -> None:
    """Plot coefficient comparison for base gravity predictors."""

    rows = []
    for model_name, result in results.items():
        coeffs = result.get("coefficients", {})
        for pred in BASE_PREDICTORS:
            if pred in coeffs and np.isfinite(coeffs[pred]["coef"]):
                rows.append({"model": model_name.replace("_positive", "").replace("_sampled", ""), "predictor": pred, "coef": coeffs[pred]["coef"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    plt.figure(figsize=(14, 8))
    pivot = df.pivot(index="predictor", columns="model", values="coef").reindex(BASE_PREDICTORS)
    pivot.plot(kind="bar", ax=plt.gca())
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Coefficient")
    plt.xlabel("Predictor")
    plt.title("Gravity Model Coefficient Comparison")
    plt.xticks(rotation=35, ha="right")
    plt.legend(fontsize=8)
    save_figure(RESULTS_DIR / "gravity_coefficients_comparison.png")


def write_vif_comparison(results: dict) -> None:
    """Write a plain-text VIF comparison table for v2 gravity models."""

    all_terms = []
    for result in results.values():
        all_terms.extend(result.get("vif", {}).keys())
    terms = sorted(set(all_terms), key=lambda x: (x.startswith("year_"), x))
    lines = ["Gravity v2 VIF comparison", ""]
    header = ["term"] + list(results.keys())
    lines.append("\t".join(header))
    for term in terms:
        row = [term]
        for model_name in results:
            value = results[model_name].get("vif", {}).get(term)
            row.append("" if value is None else f"{value:.3f}")
        lines.append("\t".join(row))
    (RESULTS_DIR / "gravity_vif_comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def plot_v2_coefficients(results: dict) -> None:
    """Plot v2 Negative Binomial coefficient comparison."""

    display_terms = ["log_dist", "adj", "same_pref", "log_x_i", "log_hosp_j", "log_x_j", "log_pop_j", "log_density_j"]
    rows = []
    for model_name, result in results.items():
        coeffs = result.get("coefficients", {})
        for term in display_terms:
            if term in coeffs and np.isfinite(coeffs[term]["coef"]):
                rows.append({"model": model_name.replace("G2", "G2 "), "term": term, "coef": coeffs[term]["coef"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    pivot = df.pivot(index="term", columns="model", values="coef").reindex(display_terms)
    plt.figure(figsize=(14, 8))
    pivot.plot(kind="bar", ax=plt.gca())
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Predictor")
    plt.ylabel("Coefficient")
    plt.title("Gravity v2 Negative Binomial Coefficients")
    plt.xticks(rotation=35, ha="right")
    plt.legend(fontsize=8)
    save_figure(RESULTS_DIR / "gravity_coefficients_v2.png")


def plot_flow_map_2024() -> None:
    """Draw the top 50 2024 OD flows as arrows on SMSA centroids."""

    codes = smsa_codes()
    matrix = load_od_matrix(2024, codes=codes)
    values = matrix.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(values, 0.0)
    flows = []
    for row_idx, col_idx in zip(*np.where(values > 0)):
        flows.append({"origin": codes[row_idx], "dest": codes[col_idx], "flow": values[row_idx, col_idx]})
    top = pd.DataFrame(flows).sort_values("flow", ascending=False).head(50)
    centroids = load_centroids(codes).set_index("smsa_code")
    stock = stock_series(2024, codes=codes)
    sizes = 8 + 130 * np.sqrt(stock.reindex(codes).to_numpy(dtype=float) / stock.max())
    color_map = {"13": "#c1121f", "27": "#1d4ed8", "23": "#15803d"}

    plt.figure(figsize=(12, 10))
    plt.scatter(
        centroids.loc[codes, "longitude"],
        centroids.loc[codes, "latitude"],
        s=sizes,
        color="#c7c7c7",
        alpha=0.72,
        edgecolors="white",
        linewidths=0.2,
        zorder=1,
    )
    max_flow = float(top["flow"].max()) if len(top) else 1.0
    for row in top.itertuples():
        if row.origin not in centroids.index or row.dest not in centroids.index:
            continue
        o = centroids.loc[row.origin]
        d = centroids.loc[row.dest]
        color = color_map.get(row.origin[:2], "#6b7280")
        width = 0.5 + 4.0 * (row.flow / max_flow)
        plt.annotate(
            "",
            xy=(d["longitude"], d["latitude"]),
            xytext=(o["longitude"], o["latitude"]),
            arrowprops=dict(arrowstyle="->", color=color, lw=width, alpha=0.62),
            zorder=2,
        )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Top 50 Service Mobility OD Flows, 2024")
    handles = [
        plt.Line2D([0], [0], color="#c1121f", lw=3, label="Tokyo origin"),
        plt.Line2D([0], [0], color="#1d4ed8", lw=3, label="Osaka origin"),
        plt.Line2D([0], [0], color="#15803d", lw=3, label="Nagoya/Aichi origin"),
        plt.Line2D([0], [0], color="#6b7280", lw=3, label="Other origin"),
    ]
    plt.legend(handles=handles, loc="lower left")
    save_figure(RESULTS_DIR / "gravity_flow_map_2024.png")


def plot_distance_decay(positive: pd.DataFrame, ols_result) -> None:
    """Plot observed distance decay with an OLS fitted curve."""

    sample = positive.sample(n=min(50_000, len(positive)), random_state=20260619)
    x = design_matrix(positive)
    medians = x.median()
    dist_grid = np.linspace(0, float(np.nanpercentile(positive["dist_km"], 99)), 200)
    curve = pd.DataFrame([medians] * len(dist_grid))
    curve["const"] = 1.0
    curve["log_dist"] = np.log1p(dist_grid)
    curve = curve.reindex(columns=x.columns, fill_value=0.0)
    fitted = ols_result.predict(curve)

    plt.figure(figsize=(12, 8))
    plt.scatter(sample["dist_km"], np.log1p(sample["F"]), s=6, alpha=0.12, color="#2a6f97")
    plt.plot(dist_grid, fitted, color="#b23a48", linewidth=2.5)
    plt.xlabel("Distance (km)")
    plt.ylabel("log(F_ij + 1)")
    plt.title("Gravity Distance Decay")
    save_figure(RESULTS_DIR / "gravity_distance_decay.png")


def plot_residual_map(positive: pd.DataFrame, ols_result) -> None:
    """Plot the top 20 absolute OD residuals as arrows on centroid coordinates."""

    x = design_matrix(positive)
    data = positive.copy()
    data["residual"] = np.log1p(data["F"]) - ols_result.predict(x)
    top = data.reindex(data["residual"].abs().sort_values(ascending=False).head(20).index)
    centroids = load_centroids().set_index("smsa_code")

    plt.figure(figsize=(12, 10))
    plt.scatter(centroids["longitude"], centroids["latitude"], s=8, color="#b7b7a4", alpha=0.6)
    for row in top.itertuples():
        if row.origin not in centroids.index or row.dest not in centroids.index:
            continue
        o = centroids.loc[row.origin]
        d = centroids.loc[row.dest]
        color = "#b23a48" if row.residual > 0 else "#2a6f97"
        width = 0.6 + min(abs(row.residual), 4.0) * 0.35
        plt.annotate(
            "",
            xy=(d["longitude"], d["latitude"]),
            xytext=(o["longitude"], o["latitude"]),
            arrowprops=dict(arrowstyle="->", color=color, lw=width, alpha=0.75),
        )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Top 20 Gravity Residual OD Pairs")
    save_figure(RESULTS_DIR / "gravity_residual_map.png")


def main() -> int:
    """Run gravity-model estimation and write outputs."""

    ensure_results_dir()
    positive, sampled = build_pooled_datasets()
    # Written with a literal path (round 39). Through RESULTS_DIR, defined in phase3_common rather
    # than here, n08 and depgraph saw no producer for this file, so a03 -- in the rebuild closure
    # -- read it as a raw input: it stayed as built on 2026-06-20, from the area-code-matched flow
    # matrices and the census denominators before round 38. With the producer visible, n08 puts
    # this script in the closure.
    positive.to_csv("results/gravity_pooled_positive_pairs.csv", index=False, encoding="utf-8-sig")
    results, fitted = fit_models(positive, sampled)
    save_json(
        RESULTS_DIR / "gravity_model_results.json",
        {
            "data": {
                "positive_rows": int(len(positive)),
                "sampled_full_rows": int(len(sampled)),
                "years": OD_YEARS,
            },
            "models": results,
        },
    )
    plot_coefficients(results)
    if "G1_OLS_log_positive" in fitted:
        plot_distance_decay(positive, fitted["G1_OLS_log_positive"])
        plot_residual_map(positive, fitted["G1_OLS_log_positive"])
    v2_results, _v2_fitted = fit_v2_negbin_models(positive)
    save_json(
        RESULTS_DIR / "gravity_model_results_v2.json",
        {
            "data": {
                "positive_rows": int(len(positive)),
                "years": OD_YEARS,
                "model_family": "GLM Negative Binomial",
            },
            "models": v2_results,
        },
    )
    write_vif_comparison(v2_results)
    plot_v2_coefficients(v2_results)
    plot_flow_map_2024()
    print(f"Gravity models complete: {len(positive)} positive rows, {len(sampled)} sampled full rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
