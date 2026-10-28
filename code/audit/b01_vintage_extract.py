"""
STAGE A -- extract, for every survey wave, the SMA vintage actually used by the
MHLW source file: the SMA codes, their names, their municipality composition, and
the physician count at both SMA and municipality level.

The raw files (smsa_YYYY.csv, cp932) are hierarchical: a prefecture row, then SMA
rows (4-digit code), each followed by its municipality rows (5-digit code). That
structure means each wave carries its OWN SMA-to-municipality mapping, and we can
read it directly rather than inferring it.

Read-only with respect to source data. Writes only to outputs/audit/ and data/crosswalk/.

Outputs:
  data/crosswalk/vintage_sma_municipalities.csv   one row per (year, sma, municipality)
  data/crosswalk/vintage_sma_summary.csv          one row per (year, sma)
  outputs/audit/b01_vintage_extract.txt
"""
import sys, io, os, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from phase2_common import read_csv_rows, parse_code_name, numeric_value

YEARS = [2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
os.makedirs("data/crosswalk", exist_ok=True)
report = []


def say(s=""):
    print(s)
    report.append(s)


def total_column(row):
    """Column 1 is 総数 (all physicians) in every wave of this table."""
    return numeric_value(row[1]) if len(row) > 1 else np.nan


def scan(year):
    """Return (sma_rows, muni_rows) for one wave, preserving the file's hierarchy."""
    sma_rows, muni_rows = [], []
    cur_sma, cur_sma_name, cur_pref = None, None, None
    for row in read_csv_rows(Path(f"smsa_{year}.csv")):
        if not row:
            continue
        code, name, kind = parse_code_name(row[0])
        if kind == "pref" and code:
            cur_pref = code
            cur_sma, cur_sma_name = None, None
        elif kind == "smsa" and code:
            cur_sma = str(code).zfill(4)
            cur_sma_name = re.sub(r"[\s　]+", "", name)
            sma_rows.append({"year": year, "pref": cur_pref, "sma": cur_sma,
                             "sma_name": cur_sma_name, "physicians": total_column(row)})
        elif kind == "municipality" and code and cur_sma:
            muni_rows.append({"year": year, "pref": cur_pref, "sma": cur_sma,
                              "sma_name": cur_sma_name, "muni": str(code).zfill(5),
                              "muni_name": re.sub(r"[\s　]+", "", name),
                              "physicians": total_column(row)})
    return pd.DataFrame(sma_rows), pd.DataFrame(muni_rows)


SMA, MUNI = {}, {}
say("=" * 88)
say("STAGE A-2 -- SMA vintage carried by each MHLW survey file")
say("=" * 88)
say(f"{'year':>5} {'n SMA':>7} {'n muni':>8} {'SMA sum':>10} {'muni sum':>10} "
    f"{'muni/SMA':>9} {'SMA w/o muni':>13}")
for y in YEARS:
    s, m = scan(y)
    SMA[y], MUNI[y] = s, m
    smas_with_muni = set(m["sma"])
    orphan = sorted(set(s["sma"]) - smas_with_muni)
    say(f"{y:>5} {len(s):>7} {len(m):>8} {s.physicians.sum():>10.0f} "
        f"{m.physicians.sum():>10.0f} {m.physicians.sum()/s.physicians.sum():>9.4f} "
        f"{len(orphan):>13}")

allsma = pd.concat(SMA.values(), ignore_index=True)
allmuni = pd.concat(MUNI.values(), ignore_index=True)
allmuni.to_csv("data/crosswalk/vintage_sma_municipalities.csv", index=False,
               encoding="utf-8-sig")
allsma.to_csv("data/crosswalk/vintage_sma_summary.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- code reuse
say("")
say("=" * 88)
say("SMA CODE REUSE -- same 4-digit code, different name, across waves")
say("=" * 88)
piv = allsma.pivot_table(index="sma", columns="year", values="sma_name",
                         aggfunc="first")
changed = []
for code, row in piv.iterrows():
    names = [(y, row[y]) for y in YEARS if y in row.index and isinstance(row[y], str)]
    if not names:
        continue
    distinct = sorted({n for _, n in names})
    if len(distinct) > 1:
        changed.append({"sma": code, "n_distinct_names": len(distinct),
                        "names": " | ".join(f"{y}:{n}" for y, n in names)})
ch = pd.DataFrame(changed)
say(f"codes whose NAME changes across waves: {len(ch)}")
if len(ch):
    for _, r in ch.iterrows():
        say(f"  {r.sma}  {r.names}")

# ---------------------------------------------------------------- composition change
say("")
say("=" * 88)
say("SMA COMPOSITION CHANGE -- municipality set of each code, versus 2024")
say("=" * 88)
comp = {y: MUNI[y].groupby("sma")["muni"].apply(set).to_dict() for y in YEARS}
c24 = comp[2024]
rows = []
for y in YEARS:
    for sma, ms in comp[y].items():
        ref = c24.get(sma, set())
        inter = len(ms & ref)
        rows.append({"year": y, "sma": sma, "n_muni": len(ms), "n_muni_2024": len(ref),
                     "n_shared": inter,
                     "jaccard": inter / len(ms | ref) if (ms | ref) else np.nan,
                     "share_of_own_in_2024": inter / len(ms) if ms else np.nan})
cmp_df = pd.DataFrame(rows)
cmp_df.to_csv("data/crosswalk/vintage_composition_vs2024.csv", index=False)
say(f"{'year':>5} {'SMAs':>6} {'identical to 2024':>18} {'partial':>9} {'disjoint':>9} "
    f"{'code absent in 2024':>20}")
for y in YEARS:
    sub = cmp_df[cmp_df.year == y]
    ident = int((sub.jaccard == 1.0).sum())
    disj = int((sub.n_shared == 0).sum())
    absent = int((sub.n_muni_2024 == 0).sum())
    say(f"{y:>5} {len(sub):>6} {ident:>18} {len(sub)-ident-disj:>9} {disj:>9} {absent:>20}")

say("")
say("Codes whose municipality set is DISJOINT from the same code in 2024 "
    "(i.e. the code refers to a different place):")
for y in YEARS:
    sub = cmp_df[(cmp_df.year == y) & (cmp_df.n_shared == 0) & (cmp_df.n_muni_2024 > 0)]
    if len(sub):
        names = SMA[y].set_index("sma")["sma_name"]
        n24 = SMA[2024].set_index("sma")["sma_name"]
        say(f"  {y}: " + ", ".join(f"{r.sma}({names.get(r.sma,'?')}->2024 is "
                                   f"{n24.get(r.sma,'?')})" for _, r in sub.iterrows()))

# ---------------------------------------------------------------- muni code coverage
say("")
say("=" * 88)
say("MUNICIPALITY CODE COVERAGE against the fixed 2024 municipality->SMA map")
say("=" * 88)
map24 = MUNI[2024].set_index("muni")["sma"].to_dict()
say(f"{'year':>5} {'n muni rows':>12} {'matched in 2024 map':>21} {'unmatched':>10} "
    f"{'physicians unmatched':>21} {'% of wave':>10}")
cov = []
for y in YEARS:
    m = MUNI[y]
    hit = m["muni"].isin(map24)
    unmatched_phys = m.loc[~hit, "physicians"].sum()
    cov.append({"year": y, "n_muni": len(m), "matched": int(hit.sum()),
                "unmatched": int((~hit).sum()), "phys_unmatched": unmatched_phys,
                "pct": 100 * unmatched_phys / m.physicians.sum()})
    say(f"{y:>5} {len(m):>12} {int(hit.sum()):>21} {int((~hit).sum()):>10} "
        f"{unmatched_phys:>21.0f} {100*unmatched_phys/m.physicians.sum():>9.2f}%")
pd.DataFrame(cov).to_csv("outputs/audit/b01_muni_coverage.csv", index=False)

Path("outputs/audit/b01_vintage_extract.txt").write_text("\n".join(report) + "\n",
                                                         encoding="utf-8")
print("\nwrote data/crosswalk/vintage_*.csv and outputs/audit/b01_vintage_extract.txt")
