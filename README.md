# isya

Code, input data and result tables for a study of how physicians are distributed across Japan's 330 secondary
medical areas from 2006 to 2024. The physician counts are carried onto the fixed 2024 areas and set against census
populations carried onto the same areas, and the study measures how unequal physician density is and how
physicians who also work in a second area carry care between areas.

## What is here

- 39 scripts that run the analysis, at the top level and in `code/audit/` and `code/solve/`, and 9
  modules they import. The folder names are historical: every script here is a step of the analysis.
- 81 input files (see "Data sources" below).
- 117 result tables written by the scripts: in `outputs/`, `results/`, `census_based/`, `census_age/` and
  `data/crosswalk/`, and at the top level (`od_matrix_*.csv`, `od_network_metrics_*.csv`,
  `smsa_infrastructure_*.csv`).
- `CHANGELOG.md`: each change to an input that moved a published estimate, with the values before and after.
- `MANIFEST.sha256`: the SHA-256 of every file.
- `LICENSE`, and `.gitattributes`, which keeps every file byte for byte in every clone.

Figures are not included. Running the scripts draws them into `paper/figures/`, `outputs/validation/` and `results/`.
Comments in the scripts refer to rounds of revision recorded in an internal work record, which is not part of this
repository.

## Data named in the Data availability statements

| In the statement | Files |
|---|---|
| the physician panel on the fixed 2024 areas (A; B) | `data/crosswalk/physician_panel_smsa_repaired.csv` |
| the census-based area population and density tables (A; B) | `census_age/age_circle_YYYY.csv` (4 files, 2005–2020), `census_based/census_year_population_YYYY.csv` (5 files, 2005–2025), `census_based/smsa_density_census_YYYY.csv` (10 files, 2006–2024), `census_based/smsa_population_census_YYYY.csv` (10 files, 2006–2024) |
| the origin–destination service-flow matrices (B) | `od_matrix_YYYY.csv` (5 files, 2016–2024) |
| medical infrastructure (B) | `smsa_infrastructure_YYYY.csv` (3 files, 2017–2023) |
| area centroids (B) | `smsa_centroids_final.csv` |
| the adjacency matrix (B) | `adjacency_final.csv` |
| the secondary medical area boundary polygons (National Land Numerical Information, dataset A38) (A; B) | `exploratory/sma_boundaries.gpkg` |
| the raw census inputs retrieved from the e-Stat application programming interface (A; B) | `census_age/age_muni_YYYY.csv` (4 files, 2005–2020), `census_raw/census_YYYY_municipality.csv` (4 files, 2005–2025), `census_raw/census_2020_municipality_kakutei.csv`, `data/reference/census_area_meta_YYYY.csv` (5 files, 2005–2025), `data/reference/census_official_totals.csv` |
| the analysis result tables (A; B) | `census_based/gini_census_2006_2024.csv`, `od_network_metrics_YYYY.csv` (5 files, 2016–2024), `outputs/audit/` (37 files), `outputs/solve/` (15 files), `outputs/validation/` (4 files), `results/` (8 files) |
| the treatment variants and the municipality noise diagnostics are in the repository (outputs/solve/s02_scale_invariance.csv, s11_municipality_noise.csv) (A supplementary) | `outputs/solve/s02_scale_invariance.csv`, `outputs/solve/s11_municipality_noise.csv` |
| The change log in the code repository (A supplementary; B supplementary) | `CHANGELOG.md` |
| All data and the finished result tables underlying this study (first sentence; no narrower phrase names these) (A; B) | `data/crosswalk/crosswalk_sma_vintage_to_2024.csv`, `data/crosswalk/municipality_code_changes_YYYY_YYYY.csv` (2 files, 2005–2024), `data/crosswalk/unmatched_municipalities.csv`, `data/crosswalk/vintage_composition_vs2024.csv`, `data/crosswalk/vintage_sma_municipalities.csv`, `data/crosswalk/vintage_sma_summary.csv` |

## How to run

The code is written in Python (version 3.13.9).

1. Install the packages the scripts import:

       python -m pip install pandas numpy scipy statsmodels geopandas shapely matplotlib openpyxl

2. Check that the files are as published: `sha256sum -c MANIFEST.sha256`.
3. From the repository root, run the scripts in this order. They need no network access and no API key; the whole
   run takes about half an hour.

       python cc_build_code_changes.py
       python code/audit/b01_vintage_extract.py
       python code/audit/b03_build_crosswalk.py
       python cc_align_underserved.py
       python cc_build_census_age.py
       python cc_build_od_matrix.py
       python cc_census_pipeline.py
       python effective_supply.py
       python cc_facilities_pipeline.py
       python cc_validation_tests.py
       python cc_figs_partC.py
       python code/audit/j02_tables_s1_t2.py
       python cc_figures.py
       python cc_fill_panel.py
       python cc_needweight_2023.py
       python code/audit/a02_scale_truncation_waves.py
       python gravity_model.py
       python code/audit/a03_gravity_cluster.py
       python code/audit/a06_provision_consumption_gap.py
       python code/audit/a10_scaled_inflow_density.py
       python code/audit/a11_age_prefecture.py
       python code/audit/b06_recompute_all.py
       python code/audit/c01_round3_numbers.py
       python code/audit/c13_effective_supply_decomp.py
       python code/audit/d01_regenerate_figures.py
       python code/audit/h05_moran_repaired.py
       python code/audit/h06_decile_profile.py
       python code/audit/i02_theta_surface.py
       python code/solve/s01_gravity_ppml.py
       python code/solve/s02_scale_invariance.py
       python code/solve/s05_distance_bins.py
       python code/solve/s12_area_weighted_gini.py
       python code/audit/i05_emit_census_coverage.py
       python code/audit/i06_emit_age_shares.py
       python code/audit/q01_supp_figures.py
       python code/solve/s06_distance_form_verification.py
       python code/solve/s11_municipality_noise.py
       python code/solve/s13_growth_correlation.py
       python code/solve/s07_mechanism_2x2.py

4. Check the result: `sha256sum -c MANIFEST.sha256` again. Every result table is written again with the same bytes;
   the one file reported as changed is `physician_panel_smsa.xlsx`, which `cc_fill_panel.py` saves again with a
   new modification time (its cells do not change).

What each script writes:

| Step | Script | Writes |
|---:|---|---|
| 1 | `cc_build_code_changes.py` | `data/crosswalk/municipality_code_changes_YYYY_YYYY.csv` (2 files, 2005–2024) |
| 2 | `code/audit/b01_vintage_extract.py` | `data/crosswalk/vintage_composition_vs2024.csv`, `data/crosswalk/vintage_sma_municipalities.csv`, `data/crosswalk/vintage_sma_summary.csv`, `outputs/audit/b01_muni_coverage.csv`, `outputs/audit/b01_vintage_extract.txt` |
| 3 | `code/audit/b03_build_crosswalk.py` | `data/crosswalk/crosswalk_sma_vintage_to_2024.csv`, `data/crosswalk/physician_panel_smsa_repaired.csv`, `data/crosswalk/unmatched_municipalities.csv`, `outputs/audit/b03_allocation.csv`, `outputs/audit/b03_build_crosswalk.txt` |
| 4 | `cc_align_underserved.py` | `outputs/validation/underserved_by_circle.csv` |
| 5 | `cc_build_census_age.py` | `census_age/age_circle_YYYY.csv` (4 files, 2005–2020) |
| 6 | `cc_build_od_matrix.py` | `od_matrix_YYYY.csv` (5 files, 2016–2024), `od_network_metrics_YYYY.csv` (5 files, 2016–2024) |
| 7 | `cc_census_pipeline.py` | `census_based/census_year_population_YYYY.csv` (5 files, 2005–2025), `census_based/gini_census_2006_2024.csv`, `census_based/smsa_density_census_YYYY.csv` (10 files, 2006–2024), `census_based/smsa_population_census_YYYY.csv` (10 files, 2006–2024) |
| 8 | `effective_supply.py` | `results/effective_supply_gini_by_year.csv`, `results/effective_supply_panel.csv`, `results/gini_sensitivity.csv`, `results/outlier_smsa_check.txt`; figures in `results/` (5) |
| 9 | `cc_facilities_pipeline.py` | `smsa_infrastructure_YYYY.csv` (3 files, 2017–2023) |
| 10 | `cc_validation_tests.py` | `outputs/validation/validation_results.json`; figures in `outputs/validation/` (8) |
| 11 | `cc_figs_partC.py` | `outputs/audit/cc_figs_partC_sidecar.csv`, `outputs/validation/fig_validation_data.csv`, `outputs/validation/fig_validation_data_pref.csv`; figures in `paper/figures/` (8) |
| 12 | `code/audit/j02_tables_s1_t2.py` | `outputs/audit/j02_table_2.csv`, `outputs/audit/j02_table_s1.csv` |
| 13 | `cc_figures.py` | `outputs/audit/cc_figures_sidecar.csv`; figures in `paper/figures/` (6) |
| 14 | `cc_fill_panel.py` | saves `physician_panel_smsa.xlsx` again |
| 15 | `cc_needweight_2023.py` | `outputs/audit/cc_needweight_2023.csv` |
| 16 | `code/audit/a02_scale_truncation_waves.py` | `outputs/audit/a02_item15_simplex.csv`, `outputs/audit/a02_item2_scale_invariance.csv`, `outputs/audit/a02_item3_truncation.csv`, `outputs/audit/a02_item4_flow_waves.csv`, `outputs/audit/a02_items_2_3_4_15.txt` |
| 17 | `gravity_model.py` | `results/gravity_model_results.json`, `results/gravity_model_results_v2.json`, `results/gravity_pooled_positive_pairs.csv`, `results/gravity_vif_comparison.txt`; figures in `results/` (5) |
| 18 | `code/audit/a03_gravity_cluster.py` | `outputs/audit/a03_gravity_cluster.csv` |
| 19 | `code/audit/a06_provision_consumption_gap.py` | `outputs/audit/a06_provision_consumption_gap.csv` |
| 20 | `code/audit/a10_scaled_inflow_density.py` | `outputs/audit/a10_scaled_inflow_density.csv` |
| 21 | `code/audit/a11_age_prefecture.py` | `outputs/audit/a11_age_prefecture_S14.csv`, `outputs/audit/a11_age_prefecture_by_pref.csv` |
| 22 | `code/audit/b06_recompute_all.py` | `outputs/audit/b06_recompute_after_crosswalk.csv`, `outputs/audit/b06_recompute_all.txt` |
| 23 | `code/audit/c01_round3_numbers.py` | `outputs/audit/c01_round3_numbers.csv`, `outputs/audit/c01_round3_numbers.txt` |
| 24 | `code/audit/c13_effective_supply_decomp.py` | `outputs/audit/c13_effective_supply_decomp.csv` |
| 25 | `code/audit/d01_regenerate_figures.py` | `outputs/audit/d01_figure_sidecar.csv`; figures in `paper/figures/` (6) |
| 26 | `code/audit/h05_moran_repaired.py` | `outputs/audit/h05_moran_repaired.csv`, `outputs/audit/h05_moran_repaired.txt` |
| 27 | `code/audit/h06_decile_profile.py` | `outputs/audit/h06_decile_profile.csv` |
| 28 | `code/audit/i02_theta_surface.py` | `outputs/audit/i02_theta_surface.csv`, `outputs/audit/i02_theta_surface.txt` |
| 29 | `code/solve/s01_gravity_ppml.py` | `outputs/solve/s01_gravity_ppml.csv`, `outputs/solve/s01_gravity_ppml.txt` |
| 30 | `code/solve/s02_scale_invariance.py` | `outputs/solve/s02_scale_invariance.csv`, `outputs/solve/s02_scale_invariance.txt` |
| 31 | `code/solve/s05_distance_bins.py` | `outputs/audit/s05_figure_sidecar.csv`, `outputs/solve/s05_distance_bins.csv`, `outputs/solve/s05_distance_bins.txt`, `outputs/solve/s05_vif.csv`; figures in `paper/figures/` (3) |
| 32 | `code/solve/s12_area_weighted_gini.py` | `outputs/solve/s12_area_weighted_gini.csv`, `outputs/solve/s12_area_weighted_gini.txt` |
| 33 | `code/audit/i05_emit_census_coverage.py` | `outputs/audit/i05_census_coverage.csv`, `outputs/audit/i05_census_coverage_national.csv` |
| 34 | `code/audit/i06_emit_age_shares.py` | `outputs/audit/i06_age_min_share.csv`, `outputs/audit/i06_age_shares.csv`, `outputs/audit/i06_elderly_share.csv` |
| 35 | `code/audit/q01_supp_figures.py` | `outputs/audit/q01_supp_figures_sidecar.csv`; figures in `paper/figures/` (4) |
| 36 | `code/solve/s06_distance_form_verification.py` | `outputs/solve/s06_distance_form_verification.csv`, `outputs/solve/s06_distance_form_verification.txt` |
| 37 | `code/solve/s11_municipality_noise.py` | `outputs/solve/s11_municipality_noise.csv`, `outputs/solve/s11_municipality_noise.txt` |
| 38 | `code/solve/s13_growth_correlation.py` | `outputs/solve/s13_growth_correlation.csv`, `outputs/solve/s13_growth_correlation.txt` |
| 39 | `code/solve/s07_mechanism_2x2.py` | `outputs/solve/s07_mechanism_2x2.csv`, `outputs/solve/s07_mechanism_2x2.txt` |

## Data sources

The inputs were taken from the following statistics. The e-Stat tables were downloaded from the Portal Site of
Official Statistics of Japan (e-Stat) or retrieved through its API.

| Statistic | Ministry | Tables | Files |
|---|---|---|---|
| 「国勢調査」 Population Census | 総務省統計局 Statistics Bureau, Ministry of Internal Affairs and Communications | statsDataId 0000033787, 0003038587, 0003038591, 0003149040, 0003410379, 0003412418, 0004019309, 0004019348, 0004065881 | `census_age/age_muni_YYYY.csv` (4 files, 2005–2020), `census_raw/census_YYYY_municipality.csv` (4 files, 2005–2025), `census_raw/census_2020_municipality_kakutei.csv`, `data/reference/census_area_meta_YYYY.csv` (5 files, 2005–2025), `data/reference/census_official_totals.csv` |
| 「統計LOD」標準地域コード Statistical LOD, standard area codes | 総務省統計局 Statistics Bureau | SPARQL endpoint https://data.e-stat.go.jp/lod/sparql/alldata/query (municipal code changes) | `data/reference/lod_sac_events.csv`, `data/reference/lod_sac_versions.csv` |
| 「医師・歯科医師・薬剤師統計」（2016年までは「医師・歯科医師・薬剤師調査」） Statistics of Physicians, Dentists and Pharmacists (to 2016, Survey of Physicians, Dentists and Pharmacists) | 厚生労働省 Ministry of Health, Labour and Welfare | statistics code 00450026; statInfId 000012664634, 000031655395, 000031889159, 000032179785, 000040155792, 000040383756, 000040383775 | `data/raw/external/physician_age/estat_000012664634_2010_table29_physicians_age_pref.csv`, `data/raw/external/physician_age/estat_000040383756_2024_table4_physicians_age_pref.csv`, `data/raw/external/physician_residence/phys_juusho_juugyou_2024.csv`, `data/reference/secondary_workplace_by_smsa_YYYY.csv` (5 files, 2016–2024), `smsa_YYYY.csv` (10 files, 2006–2024), `table24_YYYY.csv` (5 files, 2016–2024) |
| 「患者調査」 Patient Survey | 厚生労働省 Ministry of Health, Labour and Welfare | statsDataId 0004002324, 0004002360, 0004002361, 0004002362, 0004026106, 0004026138, 0004026139, 0004026140 | `census_age/juryoritsu_2020.csv`, `data/raw/external/patient_survey/` (7 files), `data/raw/external/patient_survey/meta/` (7 files) |
| 「医療施設調査」 Survey of Medical Institutions | 厚生労働省 Ministry of Health, Labour and Welfare | statistics code 00450021; statsDataId 0002013963, 0002013964, 0003300509, 0003300510, 0004024929, 0004024930 | `facilities_2024.csv`, `facilities_raw/facilities_YYYY_municipality.csv` (3 files, 2017–2023) |
| 「無医地区等調査」 Survey of Areas without Physicians | 厚生労働省 Ministry of Health, Labour and Welfare | statInfId 000031947999, 000031948291, 000040082884 | `data/raw/external/underserved_areas/H26_2014_table1_sma_underserved_000031947999.xlsx`, `data/raw/external/underserved_areas/R1_2019_table2_sma_underserved_000031948291.xlsx`, `data/raw/external/underserved_areas/R4_2022_table2_sma_underserved_000040082884.xlsx` |
| 「国土数値情報（医療圏データ）」 National Land Numerical Information, medical areas (A38, 2020) | 国土交通省 Ministry of Land, Infrastructure, Transport and Tourism | A38-20 (https://nlftp.mlit.go.jp/ksj/) | `adjacency_final.csv`, `exploratory/sma_boundaries.gpkg`, `smsa_centroids_final.csv` |
| Derived and declared | made in this repository's earlier steps, or written by hand | the panel before and after its one-time repair, made from the physician statistics; the earlier physician-share populations, kept only for the old-versus-new comparison; declarations the census crosswalk and the age tables require (each row names its public source) | `data/crosswalk/census_explicit_zero_units.csv`, `data/crosswalk/census_split_apportionment.csv`, `data/superseded/physician_panel_smsa__PRE_REPAIR.csv`, `fill_log.csv`, `physician_panel_smsa.xlsx`, `smsa_population_YYYY.csv` (10 files, 2006–2024), `split/census_age_tables.csv` |

## Terms of use

- The code (`*.py`) is under the MIT License (`LICENSE`).
- The data and the result tables made in this repository are under the Creative Commons Attribution 4.0
  International License (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/).
- The input data taken from e-Stat follow the e-Stat terms of use, which comply with the Government of Japan
  Standard Terms of Use (Version 2.0) (政府標準利用規約（第2.0版）) and are compatible with CC BY 4.0. The National
  Land Numerical Information medical-area data (A38) are published by the Ministry of Land, Infrastructure,
  Transport and Tourism as open data under CC BY 4.0.
- The result tables were made in this repository from the statistics above. They are not official statistics, and
  no ministry is responsible for them.

Source attribution, as the terms ask:

出典：政府統計の総合窓口(e-Stat)（https://www.e-stat.go.jp/）

出典：「国勢調査」（総務省統計局）

出典：「統計LOD」（総務省統計局）

出典：「医師・歯科医師・薬剤師統計」「医師・歯科医師・薬剤師調査」（厚生労働省）

出典：「患者調査」（厚生労働省）

出典：「医療施設調査」（厚生労働省）

出典：「無医地区等調査」（厚生労働省）

出典：「国土数値情報（医療圏データ）」（国土交通省）

The result tables, and the derived inputs listed under "Derived and declared" above:

「国勢調査」（総務省統計局）、「統計LOD」（総務省統計局）、「医師・歯科医師・薬剤師統計」「医師・歯科医師・薬剤師調査」（厚生労働省）、「患者調査」（厚生労働省）、「医療施設調査」（厚生労働省）、「無医地区等調査」（厚生労働省）、「国土数値情報（医療圏データ）」（国土交通省）を加工して作成

The raw census inputs were retrieved through the e-Stat API, whose terms ask for this notice:
このサービスは、政府統計総合窓口(e-Stat)のAPI機能を使用していますが、サービスの内容は国によって保証されたものではありません。
