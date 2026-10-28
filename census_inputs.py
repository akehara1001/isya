"""Which file holds a census year's municipality population, and why 2020 is not the others.

THE SUPERSEDED-INPUT RULE, applied. On 2026-08-29 the 2020 total population moved from the
preliminary tabulation to the confirmed one. The old file was moved to data/superseded/ and
its path was left EMPTY rather than overwritten, so that a reader which has not been changed
deliberately fails with FileNotFoundError instead of quietly reading a different edition of
the census. That is the same rule the physician panel got: physician_panel_smsa.csv is gone
from the root and q09 asserts it stays gone.

  2005, 2010, 2015   as retrieved by cc_fetch_census.py, unchanged
  2020               DERIVED by cc_derive_census_2020_kakutei.py as the row sums of
                     census_age/age_muni_2020.csv (statsDataId 0004019309, 不詳補完結果,
                     keyed to the confirmed 人口等基本集計). National total 126,146,099.
                     The file it replaces held 126,226,568, the 速報集計 (0003433219,
                     opened 2021-06-25), which publishes no age breakdown at all -- so
                     there was no way to make 2020 internally consistent on that side.
  2025               the CONFIRMED tabulation (人口等基本集計 0004065881, opened 2026-09-29,
                     national 122,972,528), adopted 2026-10-06 in round 40 (ruling 0-7) by
                     the same rule as 2020. Same path as before: it was overwritten in
                     place, so every reader moves with it; the preliminary file (速報集計
                     0004050397, national 123,049,524) is kept with its sha256 in
                     ../archive/census_2025_preliminary_20261006/. The 2024 denominator
                     interpolates onto it at weight 0.8.

WHAT THIS COSTS l01. Six of its seven census rows were `census_raw/census_{cy}_...csv`
classified "OK (parametric) -- resolves for at least one year", so they would have stayed
green with 2020 missing and could never have detected this. Routing the reads through here
removes them from l01's static view and loses nothing it was actually measuring. The
detector that does detect an unrepointed reader is the FileNotFoundError, which is why the
old path is empty rather than overwritten.

THE ADOPTION RULE, written here before the rebuild rather than after it, because a reader
who meets the series later will otherwise take it for row-sum derived throughout.

  THE QUANTITY IS THE PUBLISHED TOTAL POPULATION IN EVERY YEAR. 2005, 2010 and 2015 take
  each year's published municipality total as retrieved. Only 2020's RETRIEVAL PATH
  differs: no confirmed-edition municipality file was published in a form this pipeline
  could retrieve, so its published total is recovered as the row sum of the confirmed
  edition's own age table. That recovery is an IDENTITY, not a modelling choice, and it
  was verified on 2026-08-30 at all three levels of the same table -- 1,895 municipality
  units, 47 prefecture rows and the national row 00000 each summing to 126,146,099, the
  published confirmed national total, with all three exactly +0 against it and -80,469
  against the superseded preliminary.

  So the change of edition is adopted FOR 2020 ONLY. Nothing about 2005, 2010 or 2015
  changes, and the series is not "derived from age tables"; it is published totals, one
  of which had to be reached by a different route.

  A LIMIT OF THAT VERIFICATION, stated so it is not overread: the identity is internal to
  the confirmed edition's age table plus one external anchor, the published national
  total. It does NOT check each municipality against an independently published
  municipality figure, because no such confirmed-edition file exists -- which is the
  reason this derivation exists at all.

  SEPARATE AND NOT CHANGED HERE: paper/provenance_registry.csv:5 still lists the
  superseded 0003433219 among Table S2's identifiers, and DATA_PROVENANCE.md is generated
  from it. Correcting it would desynchronise the registry from tab:census in the .tex,
  which is untouched this round and which h04 does not check either (B-56). Registered
  rather than fixed.
"""
import os

MUNICIPALITY_CSV = {
    2005: "census_raw/census_2005_municipality.csv",
    2010: "census_raw/census_2010_municipality.csv",
    2015: "census_raw/census_2015_municipality.csv",
    2020: "census_raw/census_2020_municipality_kakutei.csv",
    2025: "census_raw/census_2025_municipality.csv",
}

SUPERSEDED = {
    2020: "data/superseded/census_2020_municipality__SOKUHO.csv",
}


def municipality_csv(year):
    """The path for one census year. KeyError, not a silent miss, on an unknown year."""
    return MUNICIPALITY_CSV[int(year)]


def verify():
    """Every declared path resolves, and every superseded path is gone from its old home."""
    missing = [p for p in MUNICIPALITY_CSV.values() if not os.path.exists(p)]
    assert not missing, f"declared census inputs that are not on disk: {missing}"
    revived = [f"census_raw/census_{y}_municipality.csv" for y in SUPERSEDED
               if os.path.exists(f"census_raw/census_{y}_municipality.csv")]
    assert not revived, (
        f"a superseded census file is back at its old path: {revived}. The point of "
        f"emptying it is that a reader which was never repointed fails loudly.")
    return len(MUNICIPALITY_CSV), len(SUPERSEDED)


if __name__ == "__main__":
    n, s = verify()
    print(f"census municipality inputs declared : {n}")
    for y in sorted(MUNICIPALITY_CSV):
        print(f"    {y}  {MUNICIPALITY_CSV[y]}")
    print(f"superseded and left empty           : {s}")
    for y in sorted(SUPERSEDED):
        print(f"    {y}  census_raw/census_{y}_municipality.csv -> {SUPERSEDED[y]}")
