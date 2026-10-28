#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute effective physician supply from primary stock and OD net inflow."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase3_common import (
    OD_YEARS,
    RESULTS_DIR,
    ensure_results_dir,
    gini,
    load_centroids,
    load_smsa_master,
    load_od_matrix,
    load_population,
    matrix_flows,
    save_figure,
    smsa_codes,
    stock_series,
)


THETA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]


def effective_supply_for_year(year: int, theta: float, codes: list[str]) -> pd.DataFrame:
    """Compute effective supply for one year and one secondary-work weight."""

    matrix = load_od_matrix(year, codes=codes)
    inflow, outflow, net, _offdiag = matrix_flows(matrix)
    primary = stock_series(year, codes=codes)
    pop = load_population(year, codes=codes).replace(0, np.nan)
    effective = primary + theta * net
    effective = effective.clip(lower=0.0)
    primary_density = primary / pop * 100000.0
    effective_density = effective / pop * 100000.0
    return pd.DataFrame(
        {
            "smsa_code": codes,
            "year": year,
            "primary_stock": primary.reindex(codes).to_numpy(dtype=float),
            "secondary_inflow": inflow.reindex(codes).to_numpy(dtype=float),
            "secondary_outflow": outflow.reindex(codes).to_numpy(dtype=float),
            "net_inflow": net.reindex(codes).to_numpy(dtype=float),
            "effective_supply": effective.reindex(codes).to_numpy(dtype=float),
            "primary_density": primary_density.reindex(codes).to_numpy(dtype=float),
            "effective_density": effective_density.reindex(codes).to_numpy(dtype=float),
            "population": pop.reindex(codes).to_numpy(dtype=float),
            "theta": theta,
        }
    )


def build_panel(theta: float = 0.5) -> pd.DataFrame:
    """Build the effective-supply panel for all OD years."""

    codes = smsa_codes()
    frames = [effective_supply_for_year(year, theta, codes) for year in OD_YEARS]
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.rename(
        columns={
            "effective_supply": "effective_supply_theta05",
            "effective_density": "effective_density_theta05",
        }
    )
    return panel


def gini_by_year(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute primary and effective density Gini by year."""

    rows = []
    for year, group in panel.groupby("year"):
        rows.append(
            {
                "year": int(year),
                "gini_primary_density": gini(group["primary_density"]),
                "gini_effective_density_theta05": gini(group["effective_density_theta05"]),
                "gini_primary_stock": gini(group["primary_stock"]),
                "gini_effective_supply_theta05": gini(group["effective_supply_theta05"]),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_table(codes: list[str]) -> pd.DataFrame:
    """Compute Gini sensitivity to theta for every OD year."""

    rows = []
    for theta in THETA_VALUES:
        for year in OD_YEARS:
            df = effective_supply_for_year(year, theta, codes)
            rows.append(
                {
                    "theta": theta,
                    "year": year,
                    "gini_effective_density": gini(df["effective_density"]),
                    "gini_effective_supply": gini(df["effective_supply"]),
                }
            )
    return pd.DataFrame(rows)


def plot_gini_comparison(gini_df: pd.DataFrame) -> None:
    """Plot primary versus effective density Gini over time."""

    plt.figure(figsize=(12, 8))
    plt.plot(gini_df["year"], gini_df["gini_primary_density"], marker="o", label="Primary density")
    plt.plot(gini_df["year"], gini_df["gini_effective_density_theta05"], marker="o", label="Effective density (theta=0.5)")
    plt.xlabel("Year")
    plt.ylabel("Gini coefficient")
    plt.title("Primary vs Effective Physician Density Inequality")
    plt.legend()
    plt.grid(alpha=0.25)
    save_figure(RESULTS_DIR / "gini_comparison.png")


def plot_gini_sensitivity(sens: pd.DataFrame) -> None:
    """Plot 2024 Gini sensitivity to theta."""

    df = sens[sens["year"] == 2024].copy()
    plt.figure(figsize=(12, 8))
    plt.plot(df["theta"], df["gini_effective_density"], marker="o", color="#2a6f97")
    plt.xlabel("Theta")
    plt.ylabel("Gini coefficient")
    plt.title("2024 Effective Density Gini Sensitivity")
    plt.grid(alpha=0.25)
    save_figure(RESULTS_DIR / "gini_sensitivity.png")


def plot_effective_supply_map(panel: pd.DataFrame) -> None:
    """Plot 2024 network boost and exporter patterns on centroids."""

    df = panel[panel["year"] == 2024].copy()
    df["network_boost_pct"] = (df["effective_supply_theta05"] - df["primary_stock"]) / df["primary_stock"].replace(0, np.nan) * 100.0
    centroids = load_centroids(df["smsa_code"].tolist())
    df = df.merge(centroids[["smsa_code", "longitude", "latitude"]], on="smsa_code", how="left")
    sizes = 15 + 180 * np.sqrt(df["effective_supply_theta05"].clip(lower=0) / df["effective_supply_theta05"].max())

    plt.figure(figsize=(12, 10))
    sc = plt.scatter(
        df["longitude"],
        df["latitude"],
        c=df["network_boost_pct"].clip(-50, 50),
        s=sizes,
        cmap="coolwarm",
        alpha=0.85,
        edgecolors="black",
        linewidths=0.15,
    )
    plt.colorbar(sc, label="Network boost vs primary stock (%)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Effective Supply Network Boost, 2024")
    save_figure(RESULTS_DIR / "effective_supply_map_2024.png")


def write_outlier_smsa_check(panel: pd.DataFrame) -> None:
    """Write a diagnostic report for low 2024 effective-density SMSAs."""

    master = load_smsa_master()
    df = panel[panel["year"] == 2024].copy()
    df = df.merge(master[["smsa_code", "smsa_name", "pref_code"]], on="smsa_code", how="left")
    outliers = df[df["effective_density_theta05"] < 100].sort_values("effective_density_theta05")
    pop_median = df["population"].median()
    pop_q01 = df["population"].quantile(0.01)
    pop_q99 = df["population"].quantile(0.99)
    lines = [
        "SMSA outlier check: effective_density_theta05 < 100 in 2024",
        f"outlier_count: {len(outliers)}",
        f"population_median: {pop_median:.3f}",
        f"population_p01: {pop_q01:.3f}",
        f"population_p99: {pop_q99:.3f}",
        "",
    ]
    if outliers.empty:
        lines.append("No SMSAs below threshold.")
    else:
        header = [
            "smsa_code",
            "smsa_name",
            "pref_code",
            "effective_density",
            "primary_density",
            "population",
            "primary_stock",
            "net_inflow",
            "population_flag",
        ]
        lines.append("\t".join(header))
        for row in outliers.itertuples():
            if row.population < pop_q01:
                flag = "very_low_population"
            elif row.population > pop_q99:
                flag = "very_high_population"
            else:
                flag = "within_panel_range"
            lines.append(
                "\t".join(
                    [
                        str(row.smsa_code),
                        str(row.smsa_name),
                        str(row.pref_code),
                        f"{row.effective_density_theta05:.3f}",
                        f"{row.primary_density:.3f}",
                        f"{row.population:.3f}",
                        f"{row.primary_stock:.3f}",
                        f"{row.net_inflow:.3f}",
                        flag,
                    ]
                )
            )
    (RESULTS_DIR / "outlier_smsa_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def plot_network_vs_stock_gini(gini_df: pd.DataFrame) -> None:
    """Plot stock Gini and effective-density Gini on twin axes."""

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.plot(gini_df["year"], gini_df["gini_primary_stock"], marker="o", color="#2a6f97", label="Stock Gini")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Gini of primary stock", color="#2a6f97")
    ax1.tick_params(axis="y", labelcolor="#2a6f97")
    ax2 = ax1.twinx()
    ax2.plot(
        gini_df["year"],
        gini_df["gini_effective_density_theta05"],
        marker="s",
        color="#b23a48",
        label="Effective density Gini",
    )
    ax2.set_ylabel("Gini of effective density", color="#b23a48")
    ax2.tick_params(axis="y", labelcolor="#b23a48")
    plt.title("Stock Inequality vs Per-Capita Effective Supply Inequality")
    fig.tight_layout()
    plt.savefig(RESULTS_DIR / "network_vs_stock_gini.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_effective_vs_primary_scatter(panel: pd.DataFrame) -> None:
    """Plot 2024 primary density against effective density."""

    df = panel[panel["year"] == 2024].copy()
    sizes = 18 + 180 * np.sqrt(df["population"] / df["population"].max())
    vmax = max(abs(df["net_inflow"].quantile(0.02)), abs(df["net_inflow"].quantile(0.98)))
    vmax = max(vmax, 1.0)
    lo = min(df["primary_density"].min(), df["effective_density_theta05"].min())
    hi = max(df["primary_density"].max(), df["effective_density_theta05"].max())
    plt.figure(figsize=(12, 8))
    sc = plt.scatter(
        df["primary_density"],
        df["effective_density_theta05"],
        c=df["net_inflow"].clip(-vmax, vmax),
        s=sizes,
        cmap="coolwarm",
        alpha=0.75,
        edgecolors="black",
        linewidths=0.2,
    )
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    plt.colorbar(sc, label="Net inflow")
    plt.xlabel("Primary density")
    plt.ylabel("Effective density (theta=0.5)")
    plt.title("Primary vs Effective Physician Density, 2024")
    plt.grid(alpha=0.25)
    save_figure(RESULTS_DIR / "effective_vs_primary_scatter_2024.png")


def main() -> int:
    """Run effective-supply calculations and write outputs."""

    ensure_results_dir()
    codes = smsa_codes()
    panel = build_panel(theta=0.5)
    # Written with a literal path (round 39). Through RESULTS_DIR, which is defined in phase3_common
    # rather than here, n08 and depgraph saw no producer for this file, so c13 -- in the rebuild
    # closure -- read it as a raw input, and it stayed as built on 2026-06-20 from the area-code-
    # matched flow matrices. With the producer visible, n08 puts this script in the closure.
    panel.to_csv("results/effective_supply_panel.csv", index=False, encoding="utf-8-sig")
    gini_df = gini_by_year(panel)
    gini_df.to_csv(RESULTS_DIR / "effective_supply_gini_by_year.csv", index=False, encoding="utf-8-sig")
    sens = sensitivity_table(codes)
    sens.to_csv(RESULTS_DIR / "gini_sensitivity.csv", index=False, encoding="utf-8-sig")
    plot_gini_comparison(gini_df)
    plot_gini_sensitivity(sens)
    plot_effective_supply_map(panel)
    write_outlier_smsa_check(panel)
    plot_network_vs_stock_gini(gini_df)
    plot_effective_vs_primary_scatter(panel)
    print("Effective supply complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
