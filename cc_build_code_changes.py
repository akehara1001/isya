"""Build the municipality code-change tables from the frozen public record.

Round 38, stages 2 and 3 (internal work record). No network.

Reads   data/reference/lod_sac_versions.csv, data/reference/lod_sac_events.csv
        (the e-Stat statistical LOD snapshot written by cc_fetch_municipality_changes.py)
Writes  data/crosswalk/municipality_code_changes_2010_2024.csv
        data/crosswalk/municipality_code_changes_2005_2010.csv

ONE ROW PER LINK FROM AN OLD CODE TO A NEW CODE, with the date it took effect, the reason the
LOD gives, and the URI of the LOD event that records it (source_url). Nothing in either table
is typed by hand; every row is read off the record, and a row that cannot name its event
stops the build.

HOW A LINK IS READ OFF THE RECORD. The LOD keeps one version per code per period of validity.
When a code stops being used, its last version ends on the day of the change and points, by
sacs:succeedingMunicipality, at the version(s) that took its territory over:

    岩舟町 09367  --2014-04-05-->  栃木市 09203            (absorption)
    熊本市 43201  --2012-04-01-->  熊本市 43100            (shift to designated city)
    北区   22135  --2024-01-01-->  中央区 22138, 浜名区 22139   (reorganisation of wards)

A designated city's own code is an aggregate, and the area map uses its wards, so the
creation of wards is written as links from the city code to each ward, read off the
establishmentOfWard event:

    熊本市 43100  --2012-04-01-->  中央区 43101 ... 北区 43105

A version that ends but continues under the SAME code (a change of class, name or boundary)
is not a code change and gives no row. Links between aggregates that the census rows never
use (districts, sub-prefectures, 市部/郡部) give no row either; both are counted and printed.

WINDOWS. A link belongs to the table whose window holds its date, start exclusive and end
inclusive, because a census taken on 1 October already uses a code that took effect that day:

    2005_2010   2005-10-01 < date <= 2010-10-01
    2010_2024   2010-10-01 < date <= 2024-10-01

The brief asks for 2005-2010 to be added "to the same table"; it is a second file with the
same columns, and every reader takes the union of municipality_code_changes_*.csv.

    python cc_build_code_changes.py
"""
import csv
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pandas as pd  # noqa: E402

VERSIONS = "data/reference/lod_sac_versions.csv"
EVENTS = "data/reference/lod_sac_events.csv"
OUT_DIR = "data/crosswalk"
WINDOWS = {"2005_2010": ("2005-10-01", "2010-10-01"),
           "2010_2024": ("2010-10-01", "2024-10-01")}
COLUMNS = ["old_code", "old_name", "new_code", "new_name", "effective_date", "change_type",
           "source_url"]

# Classes whose codes can carry population in a census row (or, for a designated city, lead
# to the wards that do). Everything else is an aggregate the area map never uses.
UNIT_CLASSES = {"City", "Town", "Village", "Ward", "SpecialWard", "CoreCity", "SpecialCity",
                "DesignatedCity"}


def split(s):
    return [x for x in str(s).split(";") if x]


def load():
    V = pd.read_csv(VERSIONS, dtype=str, encoding="utf-8").fillna("")
    E = pd.read_csv(EVENTS, dtype=str, encoding="utf-8").fillna("")
    return V.set_index("version_uri"), E.set_index("event_uri")


def display_name(V, uri):
    """A ward's LOD label is the ward alone (中央区); prefix the city it belongs to."""
    v = V.loc[uri]
    if "Ward" in split(v.admin_class) and v.is_part_of:
        city = V.loc[split(v.is_part_of)[-1]].name_ja if split(v.is_part_of)[-1] in V.index else ""
        return city + v.name_ja
    return v.name_ja


def is_unit(v):
    return bool(set(split(v.admin_class)) & UNIT_CLASSES)


def links(V, E):
    """Every old -> new code link in the record, before windowing."""
    rows, skipped = [], {"same_code_continues": 0, "aggregate_class": 0, "no_successor": []}
    for uri, v in V.iterrows():
        t = v.valid_until
        if not t:
            continue
        succ = split(v.succeeding_municipality)
        if not succ:
            if split(v.succeeding_code):
                skipped["same_code_continues"] += 1
            else:
                skipped["no_successor"].append((v.code, v.name_ja, t, uri))
            continue
        if not is_unit(v):
            skipped["aggregate_class"] += 1
            continue
        cand = split(v.resulted_from) + split(v.changed_by)
        for s in succ:
            if s not in V.index:
                raise KeyError(f"{uri} names successor {s}, which is not in the snapshot")
            ev = [e for e in cand if e in E.index and s in split(E.loc[e].resulting)]
            ev = ev or [e for e in cand if e in E.index and e in split(V.loc[s].resulted_from)]
            if not ev:
                raise AssertionError(f"no LOD event records {uri} -> {s}; refusing to write an "
                                     f"unsourced row")
            e = sorted(set(ev), key=lambda x: int(E.loc[x].event_id))[0]
            if V.loc[s].code == v.code:
                skipped["same_code_continues"] += 1
                continue
            rows.append({"old_code": v.code, "old_name": display_name(V, uri),
                         "new_code": V.loc[s].code, "new_name": display_name(V, s),
                         "effective_date": t, "change_type": E.loc[e].reason, "source_url": e})
    # designated city -> its wards, read off each establishmentOfWard event
    for e, ev in E[E.reason == "establishmentOfWard"].iterrows():
        wards = [w for w in split(ev.resulting) if w in V.index]
        assert wards, f"{e}: an establishmentOfWard event with no ward in the snapshot"
        for w in wards:
            cities = [c for c in split(V.loc[w].is_part_of) if c in V.index
                      and "DesignatedCity" in split(V.loc[c].admin_class)
                      and V.loc[c].issued <= ev.date]
            assert cities, f"{e}: ward {w} belongs to no designated city in the snapshot"
            c = sorted(cities, key=lambda x: V.loc[x].issued)[-1]
            rows.append({"old_code": V.loc[c].code, "old_name": display_name(V, c),
                         "new_code": V.loc[w].code, "new_name": display_name(V, w),
                         "effective_date": ev.date, "change_type": ev.reason, "source_url": e})
    return rows, skipped


def main():
    V, E = load()
    rows, skipped = links(V, E)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"LOD snapshot: {len(V):,} code versions, {len(E):,} events")
    print(f"links read off the record: {len(rows):,}")
    print(f"  versions that end but continue under the same code (no row): "
          f"{skipped['same_code_continues']:,}")
    print(f"  links between aggregate classes (no row)                     : "
          f"{skipped['aggregate_class']:,}")
    print(f"  versions that end with no successor of any kind              : "
          f"{len(skipped['no_successor']):,}")
    for x in skipped["no_successor"]:
        print(f"      {x}")
    D = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates()
    for name, (lo, hi) in WINDOWS.items():
        W = D[(D.effective_date > lo) & (D.effective_date <= hi)].sort_values(
            ["effective_date", "old_code", "new_code"]).reset_index(drop=True)
        # Written through os.path.join(OUT_DIR, f"...{name}.csv") in the call itself so that n08
        # sees the producer; cc_census_pipeline.py reads the same f-string by name.
        W.to_csv(os.path.join(OUT_DIR, f"municipality_code_changes_{name}.csv"), index=False,
                 encoding="utf-8", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        path = os.path.join(OUT_DIR, f"municipality_code_changes_{name}.csv")
        print(f"\n{path}: {len(W):,} rows, {lo} < date <= {hi}")
        print("  " + W.change_type.value_counts().to_string().replace("\n", "\n  "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
