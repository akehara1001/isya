"""Census municipality rows -> 2024 codes -> the 330 areas, by the code-change tables. No back-fill.

Round 38 (internal work record). The ONE place where a census year's
municipality codes are carried to the fixed 2024 area map. cc_census_pipeline.py builds the
area populations with it, cc_build_census_age.py builds the area age tables with it, and q28
reconciles what both write.

THE RULE IT ENFORCES (standing rule 1 of round 38). Every code of the fixed 2024 map must, in
every census year, either exist under the same code or be reached from an older code through
the code-change tables. Nothing is scaled up to stand for what failed to match. If a census
unit cannot be carried to the map, or a map code is reached by no census unit, or one unit
would have to be split between two areas, this module RAISES and lists the codes.

WHAT A CENSUS ROW IS. The census tables mix units with totals and re-listings: national and
prefecture rows, 市部/郡部 totals, sub-prefecture and district totals, designated-city totals
above their wards, and (2010, 2015) the former municipalities re-listed beneath the city that
absorbed them. A row is a UNIT when its code has a version in the e-Stat LOD valid on census
day whose class is a city, town, village, ward or special ward, and it is not a designated
city's total. Every other row is classified and counted, never silently dropped. The proof
that the classification is right is arithmetic, not inspection: the units must sum to the
published prefecture totals of data/reference/census_official_totals.csv to the person.

DECLARED EXCEPTIONS, each a file with a public source on every row, and each required to be
used -- a declaration that matches nothing raises, so it cannot outlive what it describes:

  data/crosswalk/census_explicit_zero_units.csv   a unit the census publishes as "-" (no
      population) that the area map still contains. Read as 0 and kept. One row today:
      双葉町 07546 in 2020 (whole-town evacuation).
  data/crosswalk/census_split_apportionment.csv   a unit whose territory went to two areas.
      Its population is divided in the PUBLISHED persons of each part, which must sum to the
      unit exactly; any other column carried along (the age groups) is divided in the same
      proportion. One unit today: 上九一色村 19341 in 2005, 347 to 甲府市 and 1,174 to
      富士河口湖町, from the 2010 census's 2005 population regrouped onto 2010 boundaries.
"""
import glob
import os

import numpy as np
import pandas as pd

from census_inputs import municipality_csv

CENSUS_YEARS = (2005, 2010, 2015, 2020, 2025)
MAP_VINTAGE = 2024
MAP_DATE = "2024-10-01"
CHANGES_GLOB = "data/crosswalk/municipality_code_changes_*.csv"
EXPLICIT_ZERO = "data/crosswalk/census_explicit_zero_units.csv"
SPLITS = "data/crosswalk/census_split_apportionment.csv"
VERSIONS = "data/reference/lod_sac_versions.csv"
REFERENCE = "data/reference/census_official_totals.csv"
UNIT_CLASSES = {"City", "Town", "Village", "Ward", "SpecialWard", "CoreCity", "SpecialCity"}


class CrosswalkError(AssertionError):
    pass


def census_date(cy):
    return f"{int(cy)}-10-01"


def area_meta(cy):
    return pd.read_csv(f"data/reference/census_area_meta_{cy}.csv", dtype=str,
                       encoding="utf-8").fillna("")


_CACHE = {}


def _versions():
    if "V" not in _CACHE:
        V = pd.read_csv(VERSIONS, dtype=str, encoding="utf-8").fillna("")
        _CACHE["V"] = V
    return _CACHE["V"]


def version_at(code, date):
    """The LOD version of `code` valid on `date`, or None. Valid means issued on or before the
    date and not yet ended; a version that begins and ends on the same day never is."""
    if "BYCODE" not in _CACHE:
        _CACHE["BYCODE"] = {c: g for c, g in _versions().groupby("code")}
    V = _CACHE["BYCODE"].get(code)
    if V is None:
        return None
    v = V[(V.issued <= date) & ((V.valid_until == "") | (V.valid_until > date))]
    if len(v) > 1:
        raise CrosswalkError(f"{code}: {len(v)} LOD versions valid on {date}")
    return None if v.empty else v.iloc[0]


def fixed_map():
    """{2024 municipality code: 4-digit area code}, and {code: name}, from smsa_2024.csv."""
    if "MAP" not in _CACHE:
        from phase2_common import read_csv_rows, parse_code_name, DATA_DIR
        m, names, cur = {}, {}, None
        for row in read_csv_rows(DATA_DIR / f"smsa_{MAP_VINTAGE}.csv"):
            if not row:
                continue
            code, name, kind = parse_code_name(row[0])
            if kind == "smsa" and code:
                cur = str(code).zfill(4)
            elif kind == "municipality" and code and cur:
                m[str(code).zfill(5)] = cur
                names[str(code).zfill(5)] = name
        _CACHE["MAP"] = (m, names)
    return _CACHE["MAP"]


def changes():
    files = sorted(glob.glob(CHANGES_GLOB))
    if not files:
        raise CrosswalkError(f"no code-change table at {CHANGES_GLOB}")
    D = pd.concat([pd.read_csv(f, dtype=str, encoding="utf-8") for f in files], ignore_index=True)
    return D, [f.replace("\\", "/") for f in files]


def explicit_zero(cy):
    if not os.path.exists(EXPLICIT_ZERO):
        return {}
    Z = pd.read_csv(EXPLICIT_ZERO, dtype=str, encoding="utf-8")
    Z = Z[Z.census_year.astype(int) == int(cy)]
    return dict(zip(Z.area_code.str.zfill(5), Z.name))


def declared_splits(cy):
    """{census code: DataFrame of its declared parts} for year cy."""
    if not os.path.exists(SPLITS):
        return {}
    S = pd.read_csv(SPLITS, dtype=str, encoding="utf-8")
    S = S[S.census_year.astype(int) == int(cy)].copy()
    S["persons"] = S["persons"].astype(int)
    return {c: g for c, g in S.groupby(S.old_code.str.zfill(5))}


def resolve(code, cy, C=None):
    """The 2024 codes a census-year code's territory lies in on MAP_DATE, by applying every
    change after census day in date order. Several links on one day are applied until nothing
    moves (熊本市 43201 -> 43100 -> its five wards, all on 2012-04-01)."""
    return resolve_at(code, census_date(cy), C)


def resolve_at(code, date, C=None):
    """resolve() for a code as it stood on any date, not only a census day (round 39: the
    MHLW surveys are taken on 31 December). Changes dated after `date` and on or before MAP_DATE
    are applied, so a change that took effect on the survey day itself is already in its codes."""
    key = ("STEPS", str(date))
    if key not in _CACHE:
        if C is None:
            C, _ = changes()
        lo, hi = str(date), MAP_DATE
        W = C[(C.effective_date > lo) & (C.effective_date <= hi)]
        steps = []
        for d in sorted(W.effective_date.unique()):
            day = W[W.effective_date == d]
            links = {}
            for o, n in zip(day.old_code, day.new_code):
                links.setdefault(o, set()).add(n)
            steps.append((d, links))
        _CACHE[key] = steps
    cur = {code}
    for d, links in _CACHE[key]:
        for _ in range(10):
            nxt = set()
            for c in cur:
                nxt |= links.get(c, {c})
            if nxt == cur:
                break
            cur = nxt
        else:
            raise CrosswalkError(f"{code}: links on {d} do not settle")
    return sorted(cur)


def classify(cy):
    """Every row of census_raw's file for year cy, with its category. Raises on a row it
    cannot place."""
    date = census_date(cy)
    raw = pd.read_csv(municipality_csv(cy), dtype=str, encoding="utf-8")
    raw["area_code"] = raw["area_code"].str.zfill(5)
    raw["population"] = pd.to_numeric(raw["population"].astype(str).str.replace(",", ""),
                                      errors="coerce")
    meta = area_meta(cy).set_index("area_code")
    cats = []
    for c in raw["area_code"]:
        name = meta.name.get(c, "").strip().replace("　", "")
        level = meta.level.get(c, "")
        if not c.isdigit():
            cats.append("re-listed former municipality (suffixed code)")
        elif c == "00000":
            cats.append("national")
        elif c.endswith("000"):
            cats.append("prefecture")
        elif name.endswith("市部") or name.endswith("郡部"):
            cats.append("市部/郡部 total")
        else:
            v = version_at(c, date)
            if v is not None:
                cls = set(v.admin_class.split(";"))
                if "DesignatedCity" in cls:
                    cats.append("designated-city total")
                elif cls & UNIT_CLASSES:
                    cats.append("unit")
                else:
                    cats.append("aggregate: " + "/".join(sorted(cls)))
            elif name.startswith("(旧") or level == "7":
                cats.append("re-listed former municipality")
            else:
                raise CrosswalkError(f"{cy}: census row {c} ({name!r}, level {level}) has no LOD "
                                     f"version valid on {date} and is not a re-listing")
    raw["category"] = cats
    return raw


def unit_table(cy, values=None):
    """The census units of year cy with the area each lands in, one row per unit -- or one row
    per declared part, for a unit split between areas.

    Columns: census_code, population, codes_2024, area, moved, share (the fraction of the
    census unit this row carries; 1 except for a declared split).
    values: optional DataFrame indexed by census-year code whose columns are carried along,
    multiplied by share (the age table); default is the population column of census_raw.
    Raises if a unit has no value, reaches a code outside the map, would span two areas
    without a declaration, or if a map code is reached by no unit, or a declaration is unused."""
    m, names = fixed_map()
    C, _ = changes()
    R = classify(cy)
    U = R[R.category == "unit"].copy()
    zeros = explicit_zero(cy)
    splits = declared_splits(cy)
    used_zero, used_split = set(), set()
    problems = []
    rows = []
    for _, r in U.iterrows():
        c = r.area_code
        tgt = resolve(c, cy, C)
        out = [t for t in tgt if t not in m]
        if out:
            problems.append(f"{c} -> {tgt}: {out} not in the {MAP_VINTAGE} map")
            continue
        pop = r.population
        if not np.isfinite(pop):
            if c in zeros:
                pop = 0.0
                used_zero.add(c)
            else:
                problems.append(f"{c}: no population and not declared in {EXPLICIT_ZERO}")
                continue
        areas = sorted({m[t] for t in tgt})
        if len(areas) == 1:
            rows.append({"census_code": c, "population": pop, "codes_2024": " ".join(tgt),
                         "area": areas[0], "moved": tgt != [c], "share": 1.0})
            continue
        if c not in splits:
            problems.append(f"{c} -> {tgt} spans areas {areas}; a split needs a declared "
                            f"apportionment in {SPLITS}")
            continue
        P = splits[c]
        used_split.add(c)
        bad = sorted(set(P.new_code.str.zfill(5)) - set(tgt))
        if bad:
            problems.append(f"{c}: declared parts {bad} are not where the tables send it ({tgt})")
            continue
        if int(P.persons.sum()) != int(round(pop)):
            problems.append(f"{c}: declared parts sum to {int(P.persons.sum()):,}, the census "
                            f"unit has {pop:,.0f}")
            continue
        by_area = {}
        for nc, persons in zip(P.new_code.str.zfill(5), P.persons):
            by_area.setdefault(m[nc], []).append((nc, persons))
        if sorted(by_area) != areas:
            problems.append(f"{c}: declared parts reach areas {sorted(by_area)}, the tables "
                            f"reach {areas}")
            continue
        for a in areas:
            n = sum(x for _, x in by_area[a])
            rows.append({"census_code": c, "population": float(n),
                         "codes_2024": " ".join(sorted(nc for nc, _ in by_area[a])),
                         "area": a, "moved": True, "share": n / pop})
    T = pd.DataFrame(rows)
    reached = {t for s in T.codes_2024 for t in s.split()} if len(T) else set()
    unreached = sorted(set(m) - reached)
    if unreached:
        problems.append(f"{len(unreached)} map code(s) reached by no census unit: "
                        + ", ".join(f"{u} {names.get(u, '')}" for u in unreached[:40]))
    for c in sorted(set(zeros) - used_zero):
        problems.append(f"{EXPLICIT_ZERO}: {cy} {c} is declared but no census unit needed it")
    for c in sorted(set(splits) - used_split):
        problems.append(f"{SPLITS}: {cy} {c} is declared but no census unit needed it")
    if problems:
        raise CrosswalkError(f"census {cy}: {len(problems)} problem(s)\n  " + "\n  ".join(problems))
    if values is not None:
        V = values.reindex(T.census_code)
        V.index = T.index
        T = pd.concat([T, V.mul(T.share, axis=0)], axis=1)
    return T, R


def reference(cy):
    ref = pd.read_csv(REFERENCE, dtype={"pref_code": str})
    r = ref[ref.census_year == int(cy)].set_index("pref_code")["official_population"].astype(int)
    return r


def reconcile(cy, area_pop):
    """Area sums against the published totals: (national difference, {pref: difference})."""
    r = reference(cy)
    s = pd.Series(area_pop, dtype=float)
    nat = int(round(s.sum())) - int(r["00"])
    by_pref = s.groupby(s.index.str[:2]).sum().round().astype(int)
    diffs = {p: int(by_pref.get(p, 0)) - int(r[p]) for p in r.index if p != "00"
             if int(by_pref.get(p, 0)) != int(r[p])}
    return nat, diffs
