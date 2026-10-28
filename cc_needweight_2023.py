"""
Sensitivity: need-weighted denominator rebuilt with NATIONAL age-specific
utilisation rates from the 2023 Patient Survey (in place of 2020), to bound the
concern that 2020 rates are pandemic-distorted. Rates come from the already-
downloaded prefecture table (ps2023_pref_rate, national row, both sexes,
inpatient total + outpatient total); its age brackets (0-4, 5-14, ..., 85+) are
mapped onto the 19 five-year groups by assigning each bracket's per-capita rate
to its constituent groups. Reports the need-Gini series and 2010->2024
decomposition under 2023 weights next to the 2020-weight baseline.
"""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path
from phase2_common import read_csv_rows, parse_code_name

GROUPS = ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44",
          "45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84","85-89","90+"]
BRACKET_TO_GROUPS = {  # cat01 code -> five-year groups covered
    "2": ["0-4"], "3": ["5-9","10-14"], "4": ["15-19","20-24"],
    "5": ["25-29","30-34"], "6": ["35-39","40-44"], "7": ["45-49","50-54"],
    "8": ["55-59","60-64"], "9": ["65-69","70-74"], "13": ["75-79","80-84"],
    "14": ["85-89","90+"]}

# ---- extract 2023 national rates: inpatient(total) + outpatient(total) ----
vals = json.load(open("data/raw/external/patient_survey/ps2023_pref_rate_0004026106.json",
                      encoding="utf-8"))["values"]
meta = json.load(open("data/raw/external/patient_survey/meta/0004026106.json", encoding="utf-8"))
out_code = None
for k, lab in meta["cat04"]["codes"].items():
    if "外来" in str(lab) and "総数" in str(lab) and "初診" not in str(lab) and "再来" not in str(lab):
        out_code = k
print("outpatient-total cat04 code:", out_code, meta["cat04"]["codes"].get(out_code))
acc = {}
for v in vals:
    if v.get("@cat03") != "1" or v.get("@cat02") != "1":     # 全国, 性総数
        continue
    if v.get("@cat04") not in ("1", out_code):               # 入院(総数) or 外来(総数)
        continue
    a = v.get("@cat01")
    if a not in BRACKET_TO_GROUPS:
        continue
    x = float(str(v["$"]).replace(",", "")) if v["$"] not in ("-", "", "…") else np.nan
    acc[a] = acc.get(a, 0.0) + x
W23 = np.empty(len(GROUPS))
for code, gs in BRACKET_TO_GROUPS.items():
    for g in gs:
        W23[GROUPS.index(g)] = acc[code]
W20 = pd.read_csv("census_age/juryoritsu_2020.csv").set_index("age_group")["juryoritsu_per100k"].reindex(GROUPS).to_numpy(float)
print("rates by group (2020 vs 2023):")
for g, a, b in zip(GROUPS, W20, W23):
    print(f"  {g:>6}: 2020={a:8.0f}  2023={b:8.0f}")

# ---- same machinery as cc_decomp_ci ----
# THE REPAIRED PANEL. This read the pre-repair panel -- then at the repo root, now at
# data/superseded/ -- and was never re-pointed after the crosswalk repair. The 2010 need-weighted Gini it produced, 0.2738,
# is exactly the value in the `published` column of c01_round3_numbers.csv; the repaired
# value is 0.2363. Both rate-year arms used the same panel, so the comparison this script
# exists to make was internally consistent, but it was made on a footing the paper no longer
# stands on. Found 2026-08-21 by asking what -0.0651 was, which matched no window in Table S5.
panel = pd.read_csv("data/crosswalk/physician_panel_smsa_repaired.csv",
                    dtype={"code4": str})
panel["code4"] = panel["code4"].str.zfill(4)
CODES = panel["code4"].tolist()

def build_map():
    m = {}; cur = None
    for row in read_csv_rows(Path("smsa_2024.csv")):
        if not row:
            continue
        code, name, kind = parse_code_name(row[0])
        if kind == "smsa" and code:
            cur = str(code).zfill(4)
        elif kind == "municipality" and code and cur:
            m[str(code).zfill(5)] = cur
    return m

MAP = build_map()
def stock(y):
    return panel.set_index("code4")[str(y)].reindex(CODES).astype(float).to_numpy()
def total_pop(y):
    df = pd.read_csv(f"census_based/smsa_population_census_{y}.csv")
    df["code4"] = df["smsa_code"].apply(lambda c: str(c).zfill(4))
    return df.set_index("code4")["population"].reindex(CODES).astype(float).to_numpy()
def sma_age_shares(cy):
    # Round 38 (2026-10-06): the area age table, built once by cc_build_census_age.py through
    # the code-change tables. Summing census_age/age_muni_ over the 2024 map's codes, as this
    # did until then, lost every municipality whose code had changed since the census --
    # 熊本市 in 2010, the seven Hamamatsu wards in 2010-2020 -- and gave those areas the age
    # mix of what was left.
    t = pd.read_csv(f"census_age/age_circle_{cy}.csv", dtype={"code4": str}, encoding="utf-8")
    mat = t.set_index("code4").reindex(CODES)[GROUPS].to_numpy(float)
    tot = mat.sum(1, keepdims=True)
    return np.divide(mat, tot, out=np.full_like(mat, np.nan), where=tot > 0)
SH = {cy: sma_age_shares(cy) for cy in [2010, 2015, 2020]}
def shares_for_year(ay):
    if ay <= 2015:
        v = SH[2010] + (SH[2015] - SH[2010]) * ((ay - 2010) / 5.0)
    else:
        v = SH[2015] + (SH[2020] - SH[2015]) * ((ay - 2015) / 5.0)
    v = np.clip(v, 0, None); rs = v.sum(1, keepdims=True); rs[rs == 0] = np.nan
    return v / rs
from needbase import fraction_from_sums, assert_unity


def _area_sums(cy):
    """Age-stated persons and published population per area, from the area age table.
    The census-edition guard is taken over the same areas as the shares (round 38)."""
    t = pd.read_csv(f"census_age/age_circle_{cy}.csv", dtype={"code4": str},
                    encoding="utf-8").set_index("code4").reindex(CODES)
    return t[GROUPS].to_numpy(float).sum(1), t["population"].to_numpy(float)



# The age-stated fraction, computed as a GUARD on the census edition and NOT applied to
# the base -- see needbase.py. assert_unity is what earns the computation: at the 不詳補完
# anchors it must be exactly 1, so a re-fetch of a different edition fails here.
GFRAC = {cy: fraction_from_sums(*_area_sums(cy)) for cy in [2010, 2015, 2020]}
assert_unity(GFRAC, "circle")


def need_pop(ay, W):
    # PRO RATA: the age shares multiply the PUBLISHED total, distributing the unstated-age
    # residents in the age-stated proportions. The manuscript's Eq. (needweight); see
    # needbase.py for why the age-stated base was reverted.
    raw = total_pop(ay)
    nm = (shares_for_year(ay) * W).sum(1)
    return raw * nm / (np.nansum(raw * nm) / np.nansum(raw))
def gini(v):
    vv = np.sort(v[~np.isnan(v)]); n = vv.size; ii = np.arange(1, n + 1)
    return (2 * (vv * ii).sum()) / (n * vv.sum()) - (n + 1) / n

print("\nneed-weighted density Gini and 2010->2024 decomposition, 2020 vs 2023 rates:")
# This result is asserted in Supplementary A2 -- "results are essentially unchanged, so the
# choice of rate year is immaterial" -- and until 2026-08-21 it was printed to a terminal and
# nowhere else. That is the s07 family: a claim on the page whose only evidence was a console
# session nobody kept. Persisted now, at full precision (see the rounding policy in
# code/audit/q08_rounding_exposure.py: files carry the value, printing does the rounding).
_rows = []
for tag, W in [("2020", W20), ("2023", W23)]:
    x0, x1 = stock(2010), stock(2024)
    P0, P1 = need_pop(2010, W), need_pop(2024, W)
    g00 = gini(x0 / P0 * 1e5); g11 = gini(x1 / P1 * 1e5)
    g10 = gini(x1 / P0 * 1e5); g01 = gini(x0 / P1 * 1e5)
    tot = g11 - g00; A = g10 - g00; B = g01 - g00; I = tot - A - B
    print(f"  rates {tag}: G2010={g00:.4f} G2024={g11:.4f} decline={tot:+.4f}  "
          f"A={A:+.4f} ({100*A/tot:.1f}%)  B={B:+.4f} ({100*B/tot:.1f}%)  I={I:+.4f} ({100*I/tot:.1f}%)")
    _rows.append({"rate_year": tag, "gini_2010": g00, "gini_2024": g11, "change": tot,
                  "A_physician": A, "B_denominator": B, "I_interaction": I,
                  "A_pct": 100 * A / tot, "B_pct": 100 * B / tot, "I_pct": 100 * I / tot})
import pandas as _pd
_pd.DataFrame(_rows).to_csv("outputs/audit/cc_needweight_2023.csv", index=False,
                            encoding="utf-8-sig")
print("\nwrote outputs/audit/cc_needweight_2023.csv")
