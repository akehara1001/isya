"""Re-aggregate the Survey of Medical Institutions onto the fixed 2024 area map.

Reads facilities_raw/ (written by cc_fetch_facilities.py) and writes
smsa_infrastructure_{2017,2020,2023}.csv -- the destination-infrastructure term of the
gravity model, Eq. 4 of the secondary-workplace flow paper.

WHAT REPLACED WHAT. The files these three supersede carried
method=prefecture_count_allocated_by_smsa_physician_share: each prefecture's published
facility total split across its areas in proportion to their PHYSICIAN counts, because no
area-level table had been found. That makes infra_j an exact multiple of the destination
physician stock within a prefecture, so the term identified supply and not infrastructure.
The superseded files are kept as a record under data/superseded/ and are read by nothing.

THE MAP IS THE POPULATION DENOMINATOR'S MAP. Counts are carried by MUNICIPALITY onto the
fixed 2024 municipality-to-area map -- the same map the companion paper's denominator uses
(its Section 2.6) -- and not by the survey's own area vintage, which had 344 areas in 2017
and 335 in 2020 against 330 in 2024. A municipality code the 2024 map does not know is a
merger or a ward reorganisation between the survey and 2024. Since round 39 every municipality
is carried from survey day (1 October) to the 2024 map by the round 38 code-change tables
(census_crosswalk.resolve_at), and one that cannot be placed in exactly one area stops the run.
Until then a code the 2024 map did not know was placed by its own survey round's AREA CODE --
an area code used as an identity across years, which standing rule 6 of round 39 forbids
(internal work record). The rows concerned were eight in 2017 and seven in 2020
and 2023: the seven pre-2024 wards of Hamamatsu (into 2208) and Nakagawa town, which became
Nakagawa city in 2018 (into 4004). The tables carry them to the same areas, so the three files
written are byte-identical to the ones the area-code route wrote.

WAVE ASSIGNMENT. The survey is triennial and the flow waves are biennial, so each wave takes
the nearest round: 2016 and 2018 <- 2017, 2020 <- 2020, 2022 and 2024 <- 2023. That mapping
is not written here; it is phase3_common.INFRA_YEARS = [2017, 2020, 2023] under
nearest_year(), and this file writes exactly those three years so that the rule and the
files on disk cannot disagree.

VALIDATION, all four asserted rather than reported:
  (i)   each year's national hospital and clinic totals equal the published prefecture
        series in facilities_2024.csv
  (ii)  all 47 prefecture totals reconcile against the same series, for both kinds
  (iii) no area is left with zero facilities
  (iv)  the within-prefecture correlation of the new infra_j with the physician stock is
        positive and NOT indistinguishable from 1 -- the defect being repaired

Writes smsa_infrastructure_2017.csv, smsa_infrastructure_2020.csv,
smsa_infrastructure_2023.csv
"""
import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.getcwd())
from phase2_common import DATA_DIR, parse_code_name, read_csv_rows  # noqa: E402

# The project root is where phase2_common lives (round 39). It used to be this file's own
# directory, but n09 runs every closure member by exec from outputs/audit/_offline_runner.py,
# so __file__ pointed there, the chdir went to outputs/audit/ and smsa_2024.csv was not found:
# this script had failed every offline re-run since it joined the closure on 2026-09-07.
ROOT = str(DATA_DIR)
os.chdir(ROOT)

METHOD = "survey_of_medical_institutions_municipality_reaggregated"
FIXED_MAP_VINTAGE = 2024
ROUNDS = [2017, 2020, 2023]
PREF_NAMES = [
    "北海道", "青森", "岩手", "宮城", "秋田", "山形", "福島", "茨城", "栃木", "群馬",
    "埼玉", "千葉", "東京", "神奈川", "新潟", "富山", "石川", "福井", "山梨", "長野",
    "岐阜", "静岡", "愛知", "三重", "滋賀", "京都", "大阪", "兵庫", "奈良", "和歌山",
    "鳥取", "島根", "岡山", "広島", "山口", "徳島", "香川", "愛媛", "高知", "福岡",
    "佐賀", "長崎", "熊本", "大分", "宮崎", "鹿児島", "沖縄"]


def fixed_map():
    """Return (municipality code -> area code, area code -> area name) for 2024."""

    muni, names, current = {}, {}, None
    for row in read_csv_rows(Path(f"smsa_{FIXED_MAP_VINTAGE}.csv")):
        if not row:
            continue
        code, name, kind = parse_code_name(row[0])
        if kind == "smsa" and code:
            current = code
            names[code] = name
        elif kind == "municipality" and code and current:
            muni[code] = current
    return muni, names


def published_prefecture_totals():
    """Return the published prefecture facility series as {(year, kind, pref): count}."""

    to_code = {n: str(i + 1).zfill(2) for i, n in enumerate(PREF_NAMES)}
    year_col = {2014: 7, 2017: 8, 2020: 9, 2023: 10, 2024: 11}
    out = {}
    for row in read_csv_rows(Path("facilities_2024.csv")):
        if len(row) < 12 or row[0] not in {"hospital", "clinic"}:
            continue
        # The cells are padded to a fixed width with ideographic spaces, and a bare
        # .replace() of the suffixes turns "京　都" into "京". Strip the padding, try the
        # name as printed, and only then drop ONE trailing suffix character.
        name = str(row[2]).replace(" ", "").replace("　", "")
        code = to_code.get(name)
        if not code and name[-1:] in ("県", "府", "都"):
            code = to_code.get(name[:-1])
        if not code:
            continue
        for year, col in year_col.items():
            value = str(row[col]).replace(",", "").strip()
            out[(year, row[0], code)] = float(value) if value else 0.0
    return out


def municipality_counts(year):
    """Read one round's municipality file."""

    kinds = {"muni_code": str, "sma_code_vintage": str}
    if year == 2017:
        return pd.read_csv("facilities_raw/facilities_2017_municipality.csv", dtype=kinds)
    if year == 2020:
        return pd.read_csv("facilities_raw/facilities_2020_municipality.csv", dtype=kinds)
    return pd.read_csv("facilities_raw/facilities_2023_municipality.csv", dtype=kinds)


def reaggregate(year, muni_to_area, area_codes):
    """Carry one round's municipality counts onto the fixed 2024 areas."""

    import census_crosswalk as X
    raw = municipality_counts(year)
    hosp = {c: 0.0 for c in area_codes}
    clin = {c: 0.0 for c in area_codes}
    routed = {"direct": 0, "carried by the code-change tables": 0}
    date = f"{year}-10-01"
    for r in raw.itertuples(index=False):
        code = str(r.muni_code).zfill(5)
        tgt = X.resolve_at(code, date)
        areas = {muni_to_area[t] for t in tgt if t in muni_to_area}
        if len(areas) != 1 or any(t not in muni_to_area for t in tgt):
            raise RuntimeError(
                f"{year}: municipality {code} cannot be carried to exactly one 2024 area "
                f"by the code-change tables ({tgt})")
        area = areas.pop()
        routed["direct" if tgt == [code] else "carried by the code-change tables"] += 1
        hosp[area] += float(r.hospitals)
        clin[area] += float(r.clinics)
    return hosp, clin, routed


def physician_stock(area_codes):
    """Return the repaired physician panel as {wave: Series indexed by area}."""

    panel = pd.read_csv("data/crosswalk/physician_panel_smsa_repaired.csv",
                        encoding="utf-8-sig")
    panel["code4"] = panel["code4"].apply(lambda c: str(c).zfill(4))
    panel = panel.set_index("code4")
    return {w: pd.to_numeric(panel[str(w)], errors="coerce").reindex(area_codes)
            for w in (2016, 2018, 2020, 2022, 2024) if str(w) in panel.columns}


def build(muni_to_area, area_names, area_codes, published):
    """Build and validate one frame per survey round."""

    frames = {}
    for year in ROUNDS:
        hosp, clin, routed = reaggregate(year, muni_to_area, area_codes)
        out = pd.DataFrame({
            "smsa_code": area_codes,
            "smsa_name": [area_names[c] for c in area_codes],
            "hospitals": [hosp[c] for c in area_codes],
            "clinics": [clin[c] for c in area_codes]})
        out["total"] = out["hospitals"] + out["clinics"]
        out["method"] = METHOD
        frames[year] = out

        for kind, col in (("hospital", "hospitals"), ("clinic", "clinics")):
            mine = float(out[col].sum())
            pub = sum(v for (y, k, _), v in published.items() if y == year and k == kind)
            assert abs(mine - pub) < 1e-9, (
                f"{year} {kind}: re-aggregated national total {mine:,.0f} against "
                f"published {pub:,.0f}")
        pref = out.assign(pref=out.smsa_code.str[:2]).groupby("pref")[
            ["hospitals", "clinics"]].sum()
        assert len(pref) == 47, len(pref)
        for kind, col in (("hospital", "hospitals"), ("clinic", "clinics")):
            for code, value in pref[col].items():
                pub = published[(year, kind, code)]
                assert abs(float(value) - pub) < 1e-9, (
                    f"{year} {kind} prefecture {code}: {value:,.0f} against published "
                    f"{pub:,.0f}")
        assert (out["total"] > 0).all(), (
            f"{year}: {int((out['total'] <= 0).sum())} area(s) with no facility at all")
        print(f"{year}: {len(out)} areas, hospitals {out.hospitals.sum():,.0f}, "
              f"clinics {out.clinics.sum():,.0f}; municipalities routed {routed}")
    return frames


def check_not_the_physician_stock(frames, stocks, area_codes):
    """Assert the term is no longer a restatement of the destination physician stock."""

    for wave, stock in stocks.items():
        src = 2017 if wave <= 2018 else (2020 if wave == 2020 else 2023)
        f = frames[src].set_index("smsa_code")
        d = pd.DataFrame(
            {"infra": np.log1p(f["total"].reindex(area_codes).to_numpy(float)),
             "stock": np.log1p(stock.to_numpy(float))}, index=area_codes)
        d["pref"] = [c[:2] for c in area_codes]
        r = np.array([np.corrcoef(g.infra, g.stock)[0, 1]
                      for _, g in d.groupby("pref") if len(g) >= 3])
        assert (r > 0).all(), f"wave {wave}: a within-prefecture correlation is not positive"
        near = int((r > 0.9999).sum())
        assert near < len(r) // 2, (
            f"wave {wave}: {near} of {len(r)} prefectures still have the infrastructure "
            "term indistinguishable from the physician stock; the defect is not repaired")
        print(f"wave {wave} <- {src}: within-prefecture corr(log infra, log stock) "
              f"min {r.min():.3f} median {np.median(r):.3f} max {r.max():.3f}; "
              f"{near}/{len(r)} prefectures above 0.9999")


def main():
    """Build the three infrastructure files and assert the four validations."""

    muni_to_area, area_names = fixed_map()
    area_codes = sorted(set(muni_to_area.values()))
    assert len(area_codes) == 330, len(area_codes)
    published = published_prefecture_totals()
    frames = build(muni_to_area, area_names, area_codes, published)
    check_not_the_physician_stock(frames, physician_stock(area_codes), area_codes)
    frames[2017].to_csv("smsa_infrastructure_2017.csv", index=False, encoding="utf-8-sig")
    frames[2020].to_csv("smsa_infrastructure_2020.csv", index=False, encoding="utf-8-sig")
    frames[2023].to_csv("smsa_infrastructure_2023.csv", index=False, encoding="utf-8-sig")
    print("wrote smsa_infrastructure_2017.csv, smsa_infrastructure_2020.csv, "
          "smsa_infrastructure_2023.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
