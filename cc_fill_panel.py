"""
Fill missing physician-panel cells in physician_panel_smsa.xlsx IN PLACE.
A backup (physician_panel_smsa.backup_before_fill.xlsx) was made beforehand.

- A-class: boundary-merge circles -> sum of old sub-circles from smsa_YYYY.csv
- B-class: brand-new circles -> constant extrapolation from earliest available year
Writes a fill_log.csv documenting every filled cell and its source.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import pandas as pd
from openpyxl import load_workbook
from phase2_common import read_csv_rows, parse_code_name

YEARS = [2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
XLSX = "physician_panel_smsa.xlsx"

# A-class: new circle code -> list of old sub-circle codes to sum
A_GROUPS = {
    "0708": ["0704", "0705"],
    "1412": ["1401", "1402", "1403"],
    "2313": ["2301", "2303"],
    "2811": ["2802", "2803"],
    "2812": ["2806", "2807"],
    "3706": ["3701", "3703"],
    "3707": ["3704", "3705"],
    "4312": ["4301", "4307"],
}
# B-class: brand-new circles -> constant extrapolation from earliest available year
B_CIRCLES = ["0906", "1110", "2312"]


def circle_counts(yr):
    out = {}
    for row in read_csv_rows(Path(f"smsa_{yr}.csv")):
        if not row:
            continue
        code, name, kind = parse_code_name(row[0])
        if kind == "smsa" and code and len(row) > 1:
            v = str(row[1]).strip().replace(",", "")
            try:
                out[code] = float(v)
            except ValueError:
                out[code] = None
    return out


def main():
    allc = {yr: circle_counts(yr) for yr in YEARS}

    df = pd.read_excel(XLSX, sheet_name="panel")
    df["code4"] = df["smsa_code"].apply(lambda c: str(c).zfill(4))
    idx_by_code = {c: i for i, c in enumerate(df["code4"])}
    ycols = [str(y) for y in YEARS]

    log = []

    # ---- A-class ----
    for new, olds in A_GROUPS.items():
        ri = idx_by_code[new]
        for yr in YEARS:
            yc = str(yr)
            if pd.isna(df.at[ri, yc]):
                vals = [allc[yr].get(o) for o in olds]
                if all(v is not None for v in vals):
                    s = sum(vals)
                    df.at[ri, yc] = s
                    log.append({"smsa_code": df.at[ri, "smsa_code"], "code4": new,
                                "name": df.at[ri, "smsa_name"], "year": yr,
                                "filled_value": s, "method": "A_merge",
                                "source": "+".join(f"{o}={allc[yr].get(o):.0f}" for o in olds)})
                else:
                    log.append({"smsa_code": df.at[ri, "smsa_code"], "code4": new,
                                "name": df.at[ri, "smsa_name"], "year": yr,
                                "filled_value": None, "method": "A_merge_FAILED",
                                "source": "missing sub-circle"})

    # ---- B-class: constant extrapolation from earliest available value ----
    for code in B_CIRCLES:
        ri = idx_by_code[code]
        present = [(yr, df.at[ri, str(yr)]) for yr in YEARS if not pd.isna(df.at[ri, str(yr)])]
        if not present:
            continue
        earliest_yr, earliest_val = present[0]
        for yr in YEARS:
            yc = str(yr)
            if pd.isna(df.at[ri, yc]):
                df.at[ri, yc] = earliest_val
                log.append({"smsa_code": df.at[ri, "smsa_code"], "code4": code,
                            "name": df.at[ri, "smsa_name"], "year": yr,
                            "filled_value": earliest_val, "method": "B_const_extrap",
                            "source": f"constant from earliest available {earliest_yr}={earliest_val:.0f}"})

    # ---- write back IN PLACE, preserving structure ----
    # cast year columns: keep ints where whole-numbered
    df_out = df.drop(columns=["code4"])
    wb = load_workbook(XLSX)
    ws = wb["panel"]
    header = [c.value for c in ws[1]]
    col_idx = {name: j + 1 for j, name in enumerate(header)}
    for i in range(len(df_out)):
        excel_row = i + 2
        for yc in ycols:
            val = df_out.at[i, yc]
            if pd.notna(val):
                v = float(val)
                ws.cell(row=excel_row, column=col_idx[yc]).value = int(v) if v == int(v) else v
    wb.save(XLSX)

    logdf = pd.DataFrame(log)

    # THIS SCRIPT IS A ONE-SHOT IN-PLACE REPAIR, and on 2026-08-30 it became a closure
    # member -- h04 started reading fill_log.csv to check tab:fill, so n08 pulled its
    # producer in, and the offline re-run executed it. The panel was already filled, so
    # `log` came back EMPTY, to_csv wrote a bare BOM over fill_log.csv, and the report
    # loop then died on KeyError: 'filled_value'. The 57-row log was destroyed and is
    # untracked, so git could not restore it; it came back from paper_data/fill_log.csv.
    # h04 and q05 failed in the same run as a direct consequence.
    #
    # A closure member must be IDEMPOTENT. Nothing to fill is the correct steady state
    # once the repair has been applied, and the right response to it is to leave the
    # record alone and exit, not to overwrite the record with nothing.
    if logdf.empty:
        import os as _os
        _n = 0
        if _os.path.exists("fill_log.csv"):
            _n = max(0, sum(1 for _ in io.open("fill_log.csv", encoding="utf-8-sig")) - 1)
        print("nothing to fill: the panel already carries every cell this script repairs.")
        print(f"fill_log.csv left untouched ({_n} rows). This is the steady state after "
              f"the repair has been applied, not a failure.")
        assert _n > 0, (
            "the panel needs no filling AND fill_log.csv is empty or missing: the record "
            "of which cells were filled has been lost. Restore it from paper_data/ "
            "before trusting anything that reads it (h04's tab:fill, c08, q05).")
        return

    logdf.to_csv("fill_log.csv", index=False, encoding="utf-8-sig")

    # ---- report ----
    print("=== FILL LOG ===")
    for _, r in logdf.iterrows():
        fv = "FAILED" if r["filled_value"] is None else f"{r['filled_value']:.0f}"
        print(f"  {r['code4']} {r['name']:<12} {r['year']}  -> {fv:>7}  [{r['method']}]  {r['source']}")
    nfill = logdf["filled_value"].notna().sum()
    nfail = (logdf["method"] == "A_merge_FAILED").sum()
    print(f"\nfilled cells: {nfill}   failed: {nfail}")

    # verify no missing remain (except any failed)
    df2 = pd.read_excel(XLSX, sheet_name="panel")
    rem = df2[ycols].isna().sum().sum()
    print(f"remaining missing cells after fill: {rem}")
    print(f"backup kept at: physician_panel_smsa.backup_before_fill.xlsx")
    print(f"fill log written: fill_log.csv")


if __name__ == "__main__":
    main()
