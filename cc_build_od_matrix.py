"""Primary x secondary workplace flows on the fixed 2024 areas, built from the MUNICIPALITY rows
and columns of Table 24.

Round 39, stage 2 (internal work record). No network.

WHAT WAS WRONG. download_table24.parse_table24 kept a Table 24 row or column when its
secondary-medical-area CODE was one of the 330 codes of 2024. Areas are redrawn between surveys,
so 22 of 2016's 344 area codes did not exist in 2024 (Yokohama 1401-1403, Nagoya 2301 and
Owari-chubu 2303, Hanshin 2802/2803, Harima 2806/2807, Fukushima 0704/0705, Kagawa 3701 and
3703-3705, Kumamoto 4301/4307, Akita 0504-0508) and every flow touching them was dropped: 5,558
of 46,978 physicians in 2016, 296 to 369 in 2018-2022. Worse, Akita REUSED 0501-0503 for
different areas in 2024 (県北, 県央, 県南 instead of 大館・鹿角, 北秋田, 能代・山本), so the
part that survived was put in the wrong place. Standing rule 6 of round 39: an area code is
never a key across years.

WHAT THIS DOES. Each survey year's Table 24 is read at MUNICIPALITY level only (1,896 or 1,892
rows and columns; the prefecture and area rows are subtotals of them and would double count).
Every municipality code is carried from survey day (31 December) to the 2024 map by the
code-change tables of round 38 (census_crosswalk.resolve_at), and rows and columns are summed
into the 330 areas. It stops, listing the codes, on a municipality that has no valid code on
survey day, that the tables carry outside the map, or whose territory would span two areas,
and on a map code that no municipality reaches.

ASSERTED before anything is written, per year:
  * the 330 x 330 total, diagonal included, equals the Table 24 total at municipality level,
    at area level and at prefecture level, and the published count of physicians with a
    secondary workplace (data/reference/secondary_workplace_by_smsa_{year}.csv);
  * every area's column total equals that published table's municipality rows carried to the
    area by the same tables (standing rule 7); q29 asserts it again over the written files.

Writes {OUT}/od_matrix_{year}.csv (330 x 330, diagonal kept as before; readers zero it) and
{OUT}/od_network_metrics_{year}.csv (same columns as before; primary stock from the repaired
panel, which equals the pre-repair one in every flow wave -- q09 asserts that).
OUT defaults to the project root; round 39 first writes od_v2/ and swaps in after q29 passes.

    python cc_build_od_matrix.py
"""
import io
import os
import sys

if __name__ == "__main__":
    # Only when run directly: q29 imports carry() from here, and re-wrapping stdout on import
    # closes the importer's own wrapper (q17 carries the same guard for the same reason).
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import census_crosswalk as X  # noqa: E402
import mhlw_tables as M  # noqa: E402

OUT = os.environ.get("OD_OUT", ".")
YEARS = (2016, 2018, 2020, 2022, 2024)

# Read by name so that n08 orders this script after cc_build_code_changes.py; census_crosswalk
# reads the same files and main() asserts the two reads agree.
CHANGE_TABLES = {name: pd.read_csv(f"data/crosswalk/municipality_code_changes_{name}.csv",
                                   dtype=str, encoding="utf-8")
                 for name in ("2005_2010", "2010_2024")}
PANEL = pd.read_csv("data/crosswalk/physician_panel_smsa_repaired.csv", dtype={"code4": str},
                    encoding="utf-8-sig")
CODES = PANEL["code4"].tolist()
NAMES = dict(zip(PANEL.code4, PANEL.smsa_name))


class FlowMappingError(AssertionError):
    pass


def carry(codes, year):
    """{survey-year municipality code: 2024 area}, and the codes the tables moved."""
    m, mnames = X.fixed_map()
    date = M.survey_date(year)
    out, moved, problems = {}, {}, []
    for c in codes:
        v = X.version_at(c, date)
        if v is None:
            problems.append(f"{c}: no LOD version valid on {date}")
            continue
        cls = set(v.admin_class.split(";"))
        if "DesignatedCity" in cls or not (cls & X.UNIT_CLASSES):
            problems.append(f"{c}: class {v.admin_class} on {date} is not a municipality or ward")
            continue
        tgt = X.resolve_at(c, date)
        outside = [t for t in tgt if t not in m]
        if outside:
            problems.append(f"{c} -> {tgt}: {outside} not in the {X.MAP_VINTAGE} map")
            continue
        areas = sorted({m[t] for t in tgt})
        if len(areas) != 1:
            problems.append(f"{c} -> {tgt} spans areas {areas}; a flow cannot be apportioned")
            continue
        out[c] = areas[0]
        if tgt != [c]:
            moved[c] = " ".join(tgt)
    reached = set()
    for c in out:
        reached |= set(X.resolve_at(c, date))
    unreached = sorted(set(m) - reached)
    if unreached:
        problems.append(f"{len(unreached)} map code(s) reached by no municipality: "
                        + ", ".join(f"{u} {mnames.get(u, '')}" for u in unreached[:30]))
    if problems:
        raise FlowMappingError(f"Table 24 {year}: {len(problems)} problem(s)\n  " + "\n  ".join(problems))
    return out, moved


def harmonise(year):
    t = M.read_table24(year)
    U, rcodes, ccodes = M.block(t, "municipality", "municipality")
    S, _, _ = M.block(t, "smsa", "smsa")
    P, _, _ = M.block(t, "pref", "pref")
    pub = M.read_secondary_by_area(year)
    published = sum(pub["pref"].values())
    rmap, rmoved = carry(rcodes, year)
    cmap, _ = carry(ccodes, year)
    idx = {c: i for i, c in enumerate(CODES)}
    A = np.zeros((len(rcodes), len(CODES)), dtype=np.int64)
    A[np.arange(len(rcodes)), [idx[rmap[c]] for c in rcodes]] = 1
    B = np.zeros((len(ccodes), len(CODES)), dtype=np.int64)
    B[np.arange(len(ccodes)), [idx[cmap[c]] for c in ccodes]] = 1
    OD = A.T @ U @ B
    tot = int(OD.sum())
    if not (tot == int(U.sum()) == int(S.sum()) == int(P.sum()) == published):
        raise FlowMappingError(f"{year}: totals disagree: areas {tot:,}, Table 24 municipality "
                               f"{int(U.sum()):,}, area {int(S.sum()):,}, prefecture "
                               f"{int(P.sum()):,}, published {published:,}")
    # standing rule 7: column totals against the published table, carried the same way
    pm = pub["municipality"]
    pmap, _ = carry(sorted(pm), year)
    want = pd.Series(0, index=CODES, dtype=np.int64)
    for c, v in pm.items():
        want[pmap[c]] += v
    got = pd.Series(OD.sum(axis=0), index=CODES)
    off = got[got != want]
    if len(off):
        raise FlowMappingError(f"{year}: {len(off)} area column total(s) differ from the "
                               f"published table: " + ", ".join(f"{k} {got[k]} vs {want[k]}"
                                                                for k in off.index[:10]))
    return OD, {"rows": len(rcodes), "cols": len(ccodes), "moved": rmoved, "total": tot,
                "diagonal": int(np.trace(OD)), "areas_in_table": sum(1 for k in t["row_kind"] if k == "smsa"),
                "header_row": t["header_row"]}


def write(year, OD):
    df = pd.DataFrame(OD.astype(float), index=CODES, columns=CODES)
    df.to_csv(f"{OUT}/od_matrix_{year}.csv", encoding="utf-8-sig", index_label="smsa_code",
              lineterminator="\n")
    off = OD.astype(float).copy()
    np.fill_diagonal(off, 0.0)
    out_, in_ = off.sum(axis=1), off.sum(axis=0)
    primary = PANEL.set_index("code4")[str(year)].reindex(CODES).astype(float).to_numpy()
    metrics = pd.DataFrame({
        "smsa_code": CODES, "smsa_name": [NAMES[c] for c in CODES],
        "primary_stock": primary, "secondary_outflow": out_, "secondary_inflow": in_,
        "net_secondary_inflow": in_ - out_, "self_loop": np.diag(OD).astype(float)})
    metrics["effective_supply"] = metrics["primary_stock"] + 0.5 * metrics["net_secondary_inflow"]
    metrics.to_csv(f"{OUT}/od_network_metrics_{year}.csv", index=False, encoding="utf-8-sig",
                   lineterminator="\n")


def main():
    mine = pd.concat(CHANGE_TABLES.values(), ignore_index=True)
    theirs, _ = X.changes()
    assert mine.sort_values(list(mine.columns)).reset_index(drop=True).equals(
        theirs.sort_values(list(theirs.columns)).reset_index(drop=True)), (
        "census_crosswalk read different code-change tables from the ones named here")
    os.makedirs(OUT, exist_ok=True)
    print(f"Table 24 on the fixed 2024 areas -> {OUT}/")
    print(f"{'year':>5} {'hdr':>4} {'areas':>6} {'munis':>6} {'moved':>6} {'total':>8} "
          f"{'diagonal':>9} {'off-diagonal':>13}")
    for year in YEARS:
        OD, info = harmonise(year)
        write(year, OD)
        print(f"{year:>5} {info['header_row'] + 1:>4} {info['areas_in_table']:>6} {info['rows']:>6} "
              f"{len(info['moved']):>6} {info['total']:>8,} {info['diagonal']:>9,} "
              f"{info['total'] - info['diagonal']:>13,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
