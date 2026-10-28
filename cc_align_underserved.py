"""
PHASE 2: align the three 無医地区等調査 rounds (2014/2019/2022, keyed by
prefecture+SMA NAME, no code) to the fixed 330-circle 2024 basis.

Method (no fuzzy matching):
  1. Normalise names (compact + drop ・ + strip medical-region suffixes).
  2. Match each round to its NEAREST-ERA vintage file (smsa_2022 / smsa_2020 /
     smsa_2014) by (pref, name) -> that era's SMA code. Same-era matching avoids
     guessing boundary renames.
  3. Bridge era-code -> fixed 2024 circle by MAJORITY municipality overlap
     (build_map on both vintages). This dissolves Akita 8->3, Kagawa 5->3,
     Kumamoto 上益城 merge, etc. exactly as the analysis panel does.
  4. A tiny explicit ALIAS map covers survey-only splits / renames absent from
     the vintage (each documented with evidence). NO fuzzy matching.
  5. Aggregate counts (sum) into 2024 circles. Circles never appearing = TRUE
     structural zero (0 underserved areas), filled 0 (not NaN).

ROUND 39 (2026-10-06, internal work record). Two things changed, the mapping did not.
  * The bridge carries each era municipality to the 2024 map by the round 38 code-change tables
    (census_crosswalk.resolve_at, from 31 December of the vintage's year) instead of matching
    municipality codes as they stand, and it requires EVERY municipality of an era area to land in
    the same 2024 circle (dominance exactly 1). Before, municipalities whose code had changed
    (the Hamamatsu wards in the 2020 and 2022 vintages) were left out of the vote. On these three
    rounds every era area was already 100% dominant, so no row moves; the file written is the same.
  * The national totals are asserted against the published counts: 2014 1,057 (the H26 table's
    own 全国計, which is 637 無医地区 + 420 準無医地区 -- the H26 SMA table counts both under one
    column, while the 2019 and 2022 columns count 無医地区 alone), 2019 590, 2022 557.

Reads (read-only): smsa_2024.csv, smsa_2022.csv, smsa_2020.csv, smsa_2014.csv,
  data/raw/external/underserved_areas/*.xlsx, physician_panel_smsa_repaired.csv.
Writes: outputs/validation/underserved_by_circle.csv + a match report to stdout.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd, openpyxl
from phase2_common import read_csv_rows, parse_code_name, compact

UD = "data/raw/external/underserved_areas/"
OUT = "outputs/validation"; os.makedirs(OUT, exist_ok=True)
SUF = ["保健医療圏", "保健医療", "医療圏域", "医療圏", "圏域", "地域", "地区"]


def norm(s):
    x = compact(s).replace("・", "").replace("･", "")
    for suf in SUF:
        if x.endswith(suf) and len(x) > len(suf):
            x = x[:-len(suf)]
    return x


def muni_map(fn):
    m = {}; cur = None
    for r in read_csv_rows(Path(fn)):
        if not r:
            continue
        code, name, kind = parse_code_name(r[0])
        if kind == "smsa" and code:
            cur = code
        elif kind == "municipality" and code and cur:
            m[code] = cur
    return m


def vintage_names(fn):
    pref = {}; d = {}
    for r in read_csv_rows(Path(fn)):
        if not r:
            continue
        code, name, kind = parse_code_name(r[0])
        if kind == "pref" and code:
            pref[code] = compact(name)
        elif kind == "smsa" and code:
            d[(pref.get(code[:2], "?"), norm(name))] = code
    return d


M2024 = muni_map("smsa_2024.csv")
# canonical 330 circle codes + names
CODES, NAME2024 = [], {}
_pref = {}
for r in read_csv_rows(Path("smsa_2024.csv")):
    if not r:
        continue
    code, name, kind = parse_code_name(r[0])
    if kind == "pref" and code:
        _pref[code] = compact(name)
    elif kind == "smsa" and code:
        CODES.append(code); NAME2024[code] = (_pref.get(code[:2], "?"), compact(name))
# The panel is used only for its circle-code set, which the repair left identical
# (330 codes, symmetric difference 0). Reading the repaired panel keeps every
# derived file downstream of the repaired tree.
panel_codes = set(pd.read_csv("data/crosswalk/physician_panel_smsa_repaired.csv")["code4"].apply(lambda c: str(c).zfill(4)))
assert set(CODES) == panel_codes, "smsa_2024 circle set != physician panel!"


def bridge_map(vintfile):
    """era SMA code -> (2024 code, dominance fraction). Each era municipality is carried from
    31 December of the vintage's year to the 2024 map by the code-change tables; a municipality
    that cannot be carried stops the run rather than leaving the vote."""
    import census_crosswalk as X
    year = int(vintfile.split("_")[1].split(".")[0])
    date = f"{year}-12-31"
    MV = muni_map(vintfile)
    agg = defaultdict(Counter)
    lost = []
    for muni, cv in MV.items():
        tgt = X.resolve_at(muni, date)
        areas = {M2024[t] for t in tgt if t in M2024}
        if len(areas) != 1 or any(t not in M2024 for t in tgt):
            lost.append((muni, tgt))
            continue
        agg[cv][areas.pop()] += 1
    assert not lost, f"{vintfile}: municipalities the code-change tables cannot place: {lost[:10]}"
    out = {}
    for cv, cnt in agg.items():
        top, n = cnt.most_common(1)[0]
        out[cv] = (top, n / sum(cnt.values()))
    return out


# ---- explicit aliases: (pref, normalised name) -> 2024 code. Survey-only splits
#      / renames absent from the era vintage. Each justified by municipal geography.
ALIAS = {
    ("北海道", "稚内"): "0116",   # 稚内 city split out in 2022 survey; 稚内市 ∈ 宗谷(0116)
    ("兵庫県", "中播磨"): "2812",  # 中播磨(姫路 area) = 播磨姫路(2812) in modern basis
    ("新潟県", "村上"): "1501",    # 村上/岩船 area -> 下越(1501)
    ("福岡県", "甘木朝倉"): "4005", # 甘木・朝倉 -> 朝倉(4005)
    ("秋田県", "由利本荘"): "0502", # old Akita name (由利本荘/にかほ) -> 県央(0502)
}

# Published national counts each round must reproduce after alignment (round 39).
#   2014: the H26 SMA table's own 全国計, 1,057 = 637 無医地区 + 420 準無医地区
#   2019: 590 無医地区 (令和元年度)    2022: 557 無医地区 (令和4年度)
NATIONAL_MUI = {"2014": 637 + 420, "2019": 590, "2022": 557}

ROUNDS = [
    ("2014", "H26_2014_table1_sma_underserved_000031947999.xlsx", "1", "smsa_2014.csv", False),
    ("2019", "R1_2019_table2_sma_underserved_000031948291.xlsx", "②二次医療圏別", "smsa_2020.csv", True),
    ("2022", "R4_2022_table2_sma_underserved_000040082884.xlsx", "②二次医療圏別", "smsa_2022.csv", True),
]


def load_round(f, sheet, has_jun):
    wb = openpyxl.load_workbook(UD + f, data_only=True); ws = wb[sheet]
    out = []
    for r in ws.iter_rows(min_row=4, values_only=True):
        p, s = r[1], r[2]
        if not p or not s:
            continue
        if "合計" in str(s) or "計" == str(s).strip() or "合計" in str(p):
            continue
        mui = r[3]
        jun = r[4] if has_jun else None
        out.append((compact(p), norm(s), _num(mui), _num(jun)))
    return out


def _num(v):
    if v is None:
        return 0
    try:
        return int(float(str(v).replace(",", "")))
    except ValueError:
        return 0


result = {c: {} for c in CODES}
report = {}
for year, f, sheet, vint, has_jun in ROUNDS:
    names = vintage_names(vint)
    bridge = bridge_map(vint)
    rows = load_round(f, sheet, has_jun)
    circ_mui = defaultdict(int); circ_jun = defaultdict(int)
    matched_v = aliased = failed = 0
    fails = []; ambiguous = []; multi = defaultdict(list)
    for pref, nm, mui, jun in rows:
        code2024 = None
        if (pref, nm) in names:
            cv = names[(pref, nm)]
            if cv in bridge:
                code2024, frac = bridge[cv]
                if frac < 0.999:
                    ambiguous.append((pref, nm, cv, code2024, round(frac, 2)))
                matched_v += 1
            else:  # era SMA has no muni overlap (shouldn't happen)
                fails.append((pref, nm, "no-bridge")); failed += 1; continue
        elif (pref, nm) in ALIAS:
            code2024 = ALIAS[(pref, nm)]; aliased += 1
        else:
            fails.append((pref, nm, "no-match")); failed += 1; continue
        circ_mui[code2024] += mui; circ_jun[code2024] += jun
        multi[code2024].append((pref, nm))
    for c in CODES:
        result[c][f"mui_{year}"] = circ_mui.get(c, 0)
        if has_jun:
            result[c][f"jun_{year}"] = circ_jun.get(c, 0)
    covered = [c for c in CODES if circ_mui.get(c, 0) > 0 or circ_jun.get(c, 0) > 0]
    aggregated = {c: v for c, v in multi.items() if len(v) > 1}
    report[year] = dict(rows=len(rows), matched_v=matched_v, aliased=aliased, failed=failed,
                        fails=fails, ambiguous=ambiguous, covered=len(covered),
                        struct0=330 - len(covered), aggregated=aggregated,
                        natl_mui=sum(circ_mui.values()), natl_jun=sum(circ_jun.values()))

# ---- the published national counts, and no bridge short of 100% (round 39) ----
for year, *_ in ROUNDS:
    R = report[year]
    assert R["natl_mui"] == NATIONAL_MUI[year], (
        f"{year}: aligned national 無医地区 total {R['natl_mui']} is not the published "
        f"{NATIONAL_MUI[year]}")
    assert not R["fails"], f"{year}: unmapped survey rows {R['fails']}"
    assert not R["ambiguous"], f"{year}: era areas split across 2024 circles {R['ambiguous']}"

# ---- write circle-level dataset ----
df = pd.DataFrame([{"smsa_code": c, "pref": NAME2024[c][0], "sma": NAME2024[c][1], **result[c]} for c in CODES])
df.to_csv(f"{OUT}/underserved_by_circle.csv", index=False, encoding="utf-8-sig")

# ---- report ----
for year, f, sheet, vint, has_jun in ROUNDS:
    R = report[year]
    print("=" * 90)
    print(f"[{year}] vintage={vint}  rows={R['rows']}")
    print(f"  matched-via-vintage={R['matched_v']}  aliased={R['aliased']}  FAILED={R['failed']}  "
          f"(match rate {(R['matched_v']+R['aliased'])/R['rows']*100:.1f}%)")
    print(f"  distinct 2024 circles covered={R['covered']}  structural-zero circles={R['struct0']}")
    print(f"  national totals after aggregation: 無医地区={R['natl_mui']}" + (f"  準無医={R['natl_jun']}" if has_jun else "  (準無医: n/a)"))
    if R["fails"]:
        print("  !! UNMAPPED (must fix before proceeding):")
        for p, n, why in R["fails"]:
            print(f"       {p} | {n}  [{why}]")
    else:
        print("  no unmapped rows.")
    if R["ambiguous"]:
        print("  ~ bridge ambiguity (<100% muni dominance):")
        for p, n, cv, c24, fr in R["ambiguous"]:
            print(f"       {p}|{n}: era {cv} -> {c24} ({NAME2024[c24][1]}) dominance={fr}")
    if R["aggregated"]:
        print(f"  circles receiving >1 survey row (aggregated, counts summed): {len(R['aggregated'])}")
        for c, members in sorted(R["aggregated"].items()):
            print(f"       {c} {NAME2024[c][1]} <= " + " + ".join(m[1] for m in members))

print("\nsaved", f"{OUT}/underserved_by_circle.csv")
