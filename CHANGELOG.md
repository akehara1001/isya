# Change log

Changes to inputs that moved published estimates of the two papers built from this repository: paper A
(the decomposition of physician-density inequality) and paper B (secondary-workplace flows). Each entry
gives the date, the cause, the scripts of this repository that make the change and the files that carry
the values, the outputs affected and the main values before and after. Values are as printed in the
versions named; table numbers are those of the current version.

## 2026-09-07: destination facility count rebuilt (paper B)

- Cause: the destination facility term of the gravity model was not a facility count. Each prefecture's
  published facility total had been split across its areas in proportion to their physician counts, so
  within a prefecture the term was a fixed multiple of the destination physician stock.
- Change: hospitals plus general clinics from the municipality tables of the Survey of Medical
  Institutions (static rounds 2017, 2020 and 2023), carried onto the fixed 2024 areas; each flow wave
  takes the nearest round.
- Scripts and files: `cc_facilities_pipeline.py` builds `smsa_infrastructure_2017.csv`,
  `smsa_infrastructure_2020.csv` and `smsa_infrastructure_2023.csv` from `facilities_raw/`;
  `phase3_common.py` gives each flow wave the nearest round. The refitted estimates are in
  `outputs/solve/s05_distance_bins.csv` (PPML and fixed-effects columns),
  `outputs/solve/s06_distance_form_verification.csv` (Wald statistic, deviance) and
  `outputs/audit/a03_gravity_cluster.csv` (alpha).
- Outputs affected: every estimate of paper B's gravity model (main text section 3.2; Tables S2 and S3;
  Fig. S1; Supplementary A5).
- Values, version of 4 Sep 2026 -> version of 23 Sep 2026:
  - facility count, PPML column: +0.237 -> +0.391
  - facility count, fixed-effects column: +0.061 -> -0.214
  - destination density, PPML column: +0.238 -> +0.167
  - Wald statistic for the 12 distance bands: 436.0 -> 526.4
  - Poisson deviance with log(1+dist): 266,807 -> 262,361
  - method-of-moments alpha: 0.375 -> 0.346

## 2026-10-05 to 2026-10-06: area populations, flow matrices, 2025 census anchor and physician counts

Four changes were made on these two days. They reached the manuscripts together and were first issued in
the version of 6 Oct 2026; the version issued before them is that of 23 Sep 2026 (paper A was reissued on
5 Oct 2026 with the same values). Values that only one change moved are given under that change; the
combined values follow at the end.

### Census-based area populations repaired (both papers)

- Date: 2026-10-05 to 2026-10-06.
- Cause: census municipalities whose codes did not match the 2024 map were back-filled by scaling, and the
  back-filled area populations summed to 95.6% (2005) and 99.0% (2010) of the published national totals.
- Change: every census municipality is carried to the 2024 map through an explicit table of municipal code
  changes; in every census year 2005-2025 the 330 areas and the 47 prefectures equal the published totals,
  with a difference of zero persons.
- Scripts and files: `cc_build_code_changes.py` builds the table of municipal code changes
  (`data/crosswalk/municipality_code_changes_2005_2010.csv`, `municipality_code_changes_2010_2024.csv`)
  from `data/reference/lod_sac_events.csv` and `lod_sac_versions.csv`; `census_crosswalk.py` carries each
  census year's municipalities through it, with the declarations in `data/crosswalk/`;
  `cc_census_pipeline.py` and `cc_build_census_age.py` build the area populations (`census_based/`,
  `census_age/age_circle_*.csv`), and no script back-fills. The totals per census year are in
  `outputs/audit/i05_census_coverage_national.csv`; the density Gini in
  `census_based/gini_census_2006_2024.csv` and `outputs/audit/c01_round3_numbers.csv`.
- Outputs affected: every output that uses area population or density, in both papers.
- Values: density Gini 2006 0.2473 -> 0.1933 and 2010 0.2214 -> 0.1983 (version of 23 Sep 2026 -> paper
  A supplement of 6 Oct 2026 with this change and the next two applied, before the fourth; neither of the
  next two moves the 2006 or 2010 value).

### Flow matrices rebuilt from the municipalities of Table 24 (paper B, and effective supply in paper A)

- Date: 2026-10-06.
- Cause: the origin-destination matrices were built by matching the area codes of Table 24 of the Survey
  of Physicians, Dentists and Pharmacists, so flows recorded under codes that had since changed were
  dropped or assigned to the wrong area.
- Change: each flow is carried from its municipality to the fixed 2024 areas through the same table of
  municipal code changes, and every year's matrix reconciles to the published table.
- Scripts and files: `cc_build_od_matrix.py` builds `od_matrix_2016.csv` to `od_matrix_2024.csv` from
  `table24_2016.csv` to `table24_2024.csv` (read by `mhlw_tables.py`) through `census_crosswalk.py`; the
  published tables it reconciles to are `data/reference/secondary_workplace_by_smsa_*.csv`.
  `effective_supply.py` writes the effective-supply results (`results/effective_supply_panel.csv`); the
  count of positive flows is in `outputs/audit/a03_gravity_cluster.csv` and
  `outputs/solve/s01_gravity_ppml.txt`.
- Outputs affected: every output built on the flows: paper B throughout, and the effective-supply results
  of paper A (Table S8).
- Values: positive flows in the negative binomial 26,011 -> 26,533 (version of 23 Sep 2026 -> version of
  6 Oct 2026). Most of the movement of the paper B coefficients listed below came from this change.

### 2025 census anchor taken from the final count (both papers)

- Date: 2026-10-06.
- Cause: the 2025 anchor was the preliminary count (released 2026-05-29); the final count was released on
  2026-09-29.
- Change: the final count replaces the preliminary one.
- Scripts and files: `census_inputs.py` reads `census_raw/census_2025_municipality.csv`, the final
  count; `data/reference/census_area_meta_2025.csv` and `data/reference/census_official_totals.csv` carry
  its areas and published totals. The 2025 national total is in
  `outputs/audit/i05_census_coverage_national.csv`.
- Outputs affected: the populations of 2022 and 2024, which are interpolated between the 2020 and 2025
  censuses, and every output of those two waves.
- Values: 2025 national total 123,049,524 -> 122,972,528 (version of 23 Sep 2026 -> version of 6 Oct
  2026).

### Physician counts of municipalities that no longer exist followed in full (paper A)

- Date: 2026-10-06.
- Cause: physicians counted under a municipality code that no longer exists in 2024 were apportioned by
  rule.
- Change: they are followed in full to the 2024 successor through the table of municipal code changes.
- Scripts and files: `code/audit/b03_build_crosswalk.py` builds the physician panel on the 2024 areas
  (`data/crosswalk/physician_panel_smsa_repaired.csv`; the shares assigned by each rule in
  `outputs/audit/b03_allocation.csv`). The Gini values are in `outputs/audit/c01_round3_numbers.csv`.
- Outputs affected: physician counts of the 2006, 2008 and 2010 waves in the areas concerned (later waves
  are unchanged), and every result of paper A that uses those waves.
- Values: count Gini 2006 0.5826 -> 0.5832; density Gini 2006 0.1933 -> 0.1930 and 2010 0.1983 -> 0.1985
  (paper A supplement of 6 Oct 2026 with the first three changes applied -> with all four).

### Combined, version of 23 Sep 2026 -> version of 6 Oct 2026 (unchanged in v14 of 24 Oct 2026)

The current values are in `outputs/audit/c01_round3_numbers.csv` (density Gini and its change),
`outputs/audit/h06_decile_profile.csv` (Table S10), `outputs/solve/s05_distance_bins.csv` (PPML and
fixed-effects coefficients), `outputs/solve/s06_distance_form_verification.csv` (Wald statistic,
deviance) and `outputs/audit/a03_gravity_cluster.csv` (alpha).

Paper A:
- density Gini 2006: 0.2473 -> 0.1930; 2010: 0.2214 -> 0.1985
- change in the density Gini 2010-2024: -0.031 [-0.068, -0.004] -> -0.008 [-0.017, +0.000]
- Table S10: largest change in a decile's share of the national physician stock 0.63 -> 0.70 percentage
  points; densest decile's share of the national population +1.56 -> +0.82; sparsest decile's -0.39 -> -0.35

Paper B:
- adjacency, PPML column: +0.735 -> +0.748
- same prefecture, PPML column: +1.068 -> +1.061
- destination density, PPML column: +0.167 -> +0.156
- facility count, PPML column: +0.391 -> +0.405
- facility count, fixed-effects column: -0.214 [-0.517, +0.090] -> -0.313 [-0.565, -0.061]
- Wald statistic for the 12 distance bands: 526.4 -> 570.6
- Poisson deviance with log(1+dist): 262,361 -> 262,239
- method-of-moments alpha: 0.346 -> 0.345
