"""
STAGE B-1 -- rebuild the physician panel on the fixed 2024 SMA definition, by
municipality composition.

The MHLW source tables are hierarchical: every SMA row is followed by its own
municipality rows, and the municipality counts sum exactly to the SMA count in every
wave (verified in b01). So the correct harmonisation is not a majority rule applied
to SMA codes -- it is an exact re-aggregation of municipality counts onto the fixed
2024 municipality-to-SMA map, which is the SAME map the census denominator already
uses. That guarantees numerator and denominator refer to identical municipality sets.

Two allocation rules, in order. Neither apportions a count.

  RULE 1 (exact, municipality level). A municipality code present in the 2024 map is
  assigned to its 2024 SMA directly. Its physicians move with it. No approximation.
  Asserted: the code-change tables leave such a code where it is between survey day
  and the map date, so Rule 1 and Rule 2 cannot disagree about it.

  RULE 2 (the code-change tables). A municipality code absent from the 2024 map belongs
  to a municipality that merged, was absorbed, became a city, or became a ward of a
  designated city before 2024. It is carried from survey day (31 December) to the 2024
  map by the official record of municipal code changes built in round 38
  (data/crosswalk/municipality_code_changes_*.csv, through census_crosswalk.resolve_at,
  the same function that carries the census denominator and the Table 24 flows), every
  step in date order, several steps on one day applied until nothing moves. All its
  physicians go to the one 2024 area its successor codes lie in.

Round 41 (internal work record) replaced the former Rule 2
(pro rata within the vintage SMA, in the proportions of its matched municipalities),
Rule 2b (the vintage SMA code followed forward to a later wave: Sagamihara 1410,
Kumamoto 4301, Isa 4607) and Rule 3 (prefecture pro rata) with the tables. Those rules
were the numerator counterpart of the census back-fill that round 38 withdrew; mergers
are one-to-one successions, so the tables determine every assignment and nothing is left
to apportion. The pro-rata panel is in ../archive/physician_panel_prorata_20261006/.

STOPS, listing the codes, on: a code with no LOD version valid on survey day, a code
whose version on survey day is not a municipality or ward, a code the tables carry
outside the 2024 map, a code whose successors lie in more than one 2024 area (a split),
a Rule-1 code the tables would move, a map code that no municipality of the wave reaches,
and a wave whose assigned total differs from its municipality total.

Deterministic -- no randomness. Read-only on sources; writes the repaired panel to
data/crosswalk/ (NOT over data/superseded/physician_panel_smsa__PRE_REPAIR.csv).

Outputs:
  data/crosswalk/physician_panel_smsa_repaired.csv     the repaired 330 x 10 panel
  data/crosswalk/crosswalk_sma_vintage_to_2024.csv     human-readable mapping table
  data/crosswalk/unmatched_municipalities.csv          every Rule-2 municipality, with the
                                                       2024 codes and the area it reaches
  outputs/audit/b03_allocation.csv                     per wave: exact, followed, total
  outputs/audit/b03_build_crosswalk.txt
"""
import sys, io, os, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import census_crosswalk as X  # noqa: E402
import mhlw_tables as M  # noqa: E402

YEARS = [2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
report = []


def say(s=""):
    print(s)
    report.append(s)


class NumeratorMappingError(AssertionError):
    pass


# Read by name so that n08 orders this script after cc_build_code_changes.py; census_crosswalk
# reads the same files and the assert below requires the two reads to agree.
CHANGE_TABLES = {name: pd.read_csv(f"data/crosswalk/municipality_code_changes_{name}.csv",
                                   dtype=str, encoding="utf-8")
                 for name in ("2005_2010", "2010_2024")}
_mine = pd.concat(CHANGE_TABLES.values(), ignore_index=True)
_theirs, _ = X.changes()
assert _mine.sort_values(list(_mine.columns)).reset_index(drop=True).equals(
    _theirs.sort_values(list(_theirs.columns)).reset_index(drop=True)), (
    "census_crosswalk read different code-change tables from the ones named here")

MUNI = pd.read_csv("data/crosswalk/vintage_sma_municipalities.csv",
                   dtype={"pref": str, "sma": str, "muni": str}, encoding="utf-8-sig")
MUNI["sma"] = MUNI["sma"].str.zfill(4)
MUNI["muni"] = MUNI["muni"].str.zfill(5)
SMAS = pd.read_csv("data/crosswalk/vintage_sma_summary.csv",
                   dtype={"pref": str, "sma": str}, encoding="utf-8-sig")
SMAS["sma"] = SMAS["sma"].str.zfill(4)

# the fixed 2024 target: municipality -> SMA, and the canonical 330 circle list
m24 = MUNI[MUNI.year == 2024]
MAP24 = dict(zip(m24["muni"], m24["sma"]))
CODES24 = sorted(m24["sma"].unique())
NAME24 = dict(zip(m24["sma"], m24["sma_name"]))
assert len(CODES24) == 330, len(CODES24)
fm, _ = X.fixed_map()
assert MAP24 == fm, "the 2024 survey's municipality map differs from census_crosswalk.fixed_map()"
say(f"fixed 2024 target: {len(CODES24)} circles, {len(MAP24)} municipalities "
    "(identical, key for key, to the map the census denominator uses)")

panel_pub = pd.read_csv("data/superseded/physician_panel_smsa__PRE_REPAIR.csv")
panel_pub["code4"] = panel_pub["smsa_code"].apply(lambda c: str(c).zfill(4))
assert sorted(panel_pub["code4"]) == CODES24, "published panel circle set differs from 2024 file"
say("published panel circle set matches the 2024 source file exactly")


def carry(m, year):
    """{municipality code: (2024 codes, 2024 area, rule)} for one wave, or raise."""
    date = M.survey_date(year)
    out, problems = {}, []
    for c, name in zip(m["muni"], m["muni_name"]):
        v = X.version_at(c, date)
        if v is None:
            problems.append(f"{c} {name}: no LOD version valid on {date}")
            continue
        cls = set(v.admin_class.split(";"))
        if "DesignatedCity" in cls or not (cls & X.UNIT_CLASSES):
            problems.append(f"{c} {name}: class {v.admin_class} on {date} is not a municipality or ward")
            continue
        tgt = X.resolve_at(c, date)
        if c in MAP24:
            if tgt != [c]:
                problems.append(f"{c} {name}: in the 2024 map, but the tables move it to {tgt}")
                continue
            out[c] = ([c], MAP24[c], 1)
            continue
        outside = [t for t in tgt if t not in MAP24]
        if outside:
            problems.append(f"{c} {name} -> {tgt}: {outside} not in the 2024 map (untraceable)")
            continue
        areas = sorted({MAP24[t] for t in tgt})
        if len(areas) != 1:
            problems.append(f"{c} {name} -> {tgt} spans areas {areas} (a split)")
            continue
        out[c] = (tgt, areas[0], 2)
    reached = set()
    for tgt, _, _ in out.values():
        reached |= set(tgt)
    unreached = sorted(set(MAP24) - reached)
    if unreached:
        problems.append(f"{len(unreached)} map code(s) reached by no municipality: "
                        + ", ".join(unreached[:30]))
    if problems:
        raise NumeratorMappingError(f"{year}: {len(problems)} problem(s)\n  " + "\n  ".join(problems))
    return out


cw_rows, followed_rows, alloc_rows = [], [], []
repaired = {}

for y in YEARS:
    m = MUNI[MUNI.year == y].copy()
    # A code can be printed twice in one wave: in 2012 栃木市 09203 has a row under 県西 for
    # its former 西方町 part (merged 2011-10-01) and a row under 県南. Both rows carry the same
    # code, so both reach the same 2024 area; they are listed here, not merged or dropped.
    dup = m[m["muni"].duplicated(keep=False)]
    for _, r in dup.iterrows():
        say(f"{y}: code {r['muni']} printed more than once -- {r['muni_name']} under "
            f"{r['sma']} {r['sma_name']}, {r['physicians']:.0f} physicians")
    A = carry(m.drop_duplicates("muni"), y)
    m["target"] = m["muni"].map(lambda c: A[c][1])
    m["rule"] = m["muni"].map(lambda c: A[c][2])
    m["codes_2024"] = m["muni"].map(lambda c: " ".join(A[c][0]))

    acc = pd.Series(0.0, index=CODES24)
    g = m.groupby("target")["physicians"].sum()
    acc.loc[g.index] += g.values
    tot = float(m.physicians.sum())
    if abs(acc.sum() - tot) > 1e-9:
        raise NumeratorMappingError(f"{y}: assigned {acc.sum()} against municipality total {tot}")
    repaired[y] = acc

    f2 = m[m.rule == 2]
    alloc_rows.append({"year": y, "municipalities": len(m), "followed_municipalities": len(f2),
                       "exact": float(m[m.rule == 1].physicians.sum()),
                       "followed": float(f2.physicians.sum()), "apportioned": 0.0,
                       "total": tot})
    for _, r in f2.iterrows():
        followed_rows.append({"year": y, "vintage_sma": r["sma"],
                              "vintage_sma_name": r["sma_name"], "muni": r["muni"],
                              "muni_name": r["muni_name"], "physicians": r["physicians"],
                              "rule": 2, "codes_2024": r["codes_2024"],
                              "allocated_to": r["target"]})

    # --- human-readable crosswalk rows: vintage SMA -> 2024 SMA, every municipality ---
    split = m.groupby(["sma", "target"])["physicians"].sum().rename("phys").reset_index()
    tot_sma = split.groupby("sma")["phys"].transform("sum")
    split["w"] = np.where(tot_sma > 0, split["phys"] / tot_sma, np.nan)
    for _, q in split.iterrows():
        src = SMAS[(SMAS.year == y) & (SMAS.sma == q["sma"])]
        sub = m[(m["sma"] == q["sma"]) & (m["target"] == q["target"])]
        munis = sorted(sub["muni"])
        cw_rows.append({
            "year": y, "vintage_sma_code": q["sma"],
            "vintage_sma_name": (src["sma_name"].iloc[0] if len(src) else ""),
            "target_2024_code": q["target"], "target_2024_name": NAME24.get(q["target"], ""),
            "share_of_vintage_sma_physicians": round(float(q["w"]), 4) if q["w"] == q["w"] else np.nan,
            "physicians_moved": float(q["phys"]),
            "n_municipalities": len(munis),
            "n_followed_by_code_change": int((sub.rule == 2).sum()),
            "evidence_municipalities": " ".join(munis[:40]) + (" ..." if len(munis) > 40 else ""),
            "code_changed": q["sma"] != q["target"],
        })

R = pd.DataFrame(repaired)[YEARS]
R.index.name = "code4"

cw = pd.DataFrame(cw_rows)
cw.to_csv("data/crosswalk/crosswalk_sma_vintage_to_2024.csv", index=False, encoding="utf-8-sig")
fr = pd.DataFrame(followed_rows)
fr.to_csv("data/crosswalk/unmatched_municipalities.csv", index=False, encoding="utf-8-sig")
AL = pd.DataFrame(alloc_rows)
AL.to_csv("outputs/audit/b03_allocation.csv", index=False, encoding="utf-8-sig")

say("")
say("=" * 92)
say("ALLOCATION SUMMARY (no count is apportioned in any wave)")
say("=" * 92)
say(f"{'year':>5} {'exact (Rule 1)':>16} {'code changes (Rule 2)':>22} {'% code changed':>15} "
    f"{'total':>10}")
for _, a in AL.iterrows():
    say(f"{int(a.year):>5} {a.exact:>16.0f} {a.followed:>22.0f} "
        f"{100*a.followed/a.total:>14.2f}% {a.total:>10.0f}")

say("")
say("=" * 92)
say("RULE 2 -- municipalities carried through the code-change tables, by wave")
say("=" * 92)
for y in YEARS:
    sub = fr[fr.year == y] if len(fr) else fr
    if not len(sub):
        say(f"  {y}: none")
        continue
    say(f"  {y}: {len(sub)} municipalities, {sub.physicians.sum():,.0f} physicians")
    for _, r in sub.sort_values("physicians", ascending=False).head(8).iterrows():
        say(f"      {r.muni} {str(r.muni_name)[:10]:<10} ({r.vintage_sma}) -> {r.codes_2024[:40]:<40} "
            f"area {r.allocated_to} {r.physicians:>7.0f}")

say("")
say("=" * 92)
say("CROSSWALK ROWS WHERE THE CODE CHANGES (vintage code != 2024 code)")
say("=" * 92)
chg = cw[cw.code_changed & (cw.physicians_moved > 0)]
say(f"{len(chg)} such rows across all waves; "
    f"{chg[chg.year <= 2016].physicians_moved.sum():,.0f} physicians reassigned in 2006-2016")
for y in YEARS:
    sub = chg[chg.year == y].sort_values("physicians_moved", ascending=False)
    if not len(sub):
        continue
    say(f"  {y}: {len(sub)} rows, {sub.physicians_moved.sum():,.0f} physicians move code")
    for _, r in sub.head(12).iterrows():
        say(f"      {r.vintage_sma_code} {str(r.vintage_sma_name)[:14]:<14} -> "
            f"{r.target_2024_code} {str(r.target_2024_name)[:14]:<14} "
            f"{r.physicians_moved:>8.0f} ({100*r.share_of_vintage_sma_physicians:>5.1f}% of source)")

out = R.copy()
out.insert(0, "smsa_name", [NAME24[c] for c in out.index])
out.round(4).to_csv("data/crosswalk/physician_panel_smsa_repaired.csv", encoding="utf-8-sig")

say("")
say("=" * 92)
say("REPAIRED vs PUBLISHED -- national totals and the biggest circle-level moves")
say("=" * 92)
pub = panel_pub.set_index("code4")[[str(y) for y in YEARS]].astype(float)
pub.columns = YEARS
say(f"{'year':>5} {'published total':>16} {'repaired total':>15} {'source total':>13} {'diff':>8}")
for y in YEARS:
    src = MUNI[MUNI.year == y].physicians.sum()
    say(f"{y:>5} {pub[y].sum():>16.0f} {R[y].sum():>15.0f} {src:>13.0f} "
        f"{R[y].sum()-src:>8.2f}")

diff = (R - pub).abs()
say("")
say(f"circle-years changed by more than 50 physicians: {int((diff > 50).sum().sum())}")
say(f"{'circle':>7} {'name':<18} {'year':>5} {'published':>10} {'repaired':>10} {'diff':>9}")
big = [(c, y) for c in R.index for y in YEARS if diff.loc[c, y] > 200]
for c, y in sorted(big, key=lambda t: -diff.loc[t[0], t[1]])[:35]:
    say(f"{c:>7} {str(NAME24[c])[:18]:<18} {y:>5} {pub.loc[c,y]:>10.0f} "
        f"{R.loc[c,y]:>10.1f} {R.loc[c,y]-pub.loc[c,y]:>+9.1f}")

Path("outputs/audit/b03_build_crosswalk.txt").write_text("\n".join(report) + "\n",
                                                         encoding="utf-8")
print("\nwrote data/crosswalk/physician_panel_smsa_repaired.csv and the crosswalk tables")
