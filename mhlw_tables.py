"""Readers for the MHLW physician-statistics tables that the secondary-practice flows rest on.

Round 39 (internal work record). No network.

  read_table24(year)          table24_{year}.csv: physicians by PRIMARY workplace (rows) and
                              SECONDARY workplace (columns), each axis nested as prefecture (2-digit
                              code), secondary medical area (4-digit) and municipality (5-digit;
                              designated cities appear as their wards only). "-" is 0.
  read_secondary_by_area(y)   data/reference/secondary_workplace_by_smsa_{year}.csv: physicians by
                              SECONDARY workplace, same three levels, 総数 in the first data column.

The header row is FOUND, not assumed: the 2016 survey (医師・歯科医師・薬剤師調査, 閲覧 第10表)
puts it on line 5 under two title lines and a "従たる従業地" label; the statistics of 2018 onwards
(医師・歯科医師・薬剤師統計, 第24表 / 第10表) put it on line 3. The row with the most 4-digit area
codes in its cells is the header. A table whose header holds no area code stops the read.

Survey day is 31 December of the year, which is the date a municipality code is carried from.
"""
import os

import numpy as np

from phase2_common import parse_code_name, read_csv_rows

LEVELS = ("pref", "smsa", "municipality")


def survey_date(year):
    return f"{int(year)}-12-31"


def _num(cell):
    t = str(cell).strip().replace(",", "")
    if t in ("", "-", "－"):
        return 0
    if not t.isdigit():
        raise ValueError(f"not a count: {cell!r}")
    return int(t)


def read_table24(year, path=None):
    """{'header_row', 'col_kind', 'col_code', 'row_kind', 'row_code', 'values'} where values is
    an int matrix over (data rows) x (data columns); rows and columns of every level are kept,
    so callers pick the level they need. Counts of each level are returned for reporting."""
    from pathlib import Path
    rows = read_csv_rows(Path(path or f"table24_{year}.csv"))
    best = max(range(min(20, len(rows))),
               key=lambda i: sum(1 for c in rows[i][1:] if parse_code_name(c)[2] == "smsa"))
    hdr = rows[best]
    if not any(parse_code_name(c)[2] == "smsa" for c in hdr[1:]):
        raise ValueError(f"table24_{year}: no header row with area codes")
    cols = [(j, *parse_code_name(c)[:3:2]) for j, c in enumerate(hdr) if j > 0]
    cols = [(j, code, kind) for j, code, kind in cols if kind in LEVELS]
    data = []
    for r in rows[best + 1:]:
        if not r:
            continue
        code, _, kind = parse_code_name(r[0])
        if kind not in LEVELS:
            continue
        data.append((code, kind, [_num(r[j]) if j < len(r) else 0 for j, _, _ in cols]))
    return {"header_row": best,
            "col_code": [c for _, c, _ in cols], "col_kind": [k for _, _, k in cols],
            "row_code": [c for c, _, _ in data], "row_kind": [k for _, k, _ in data],
            "values": np.array([v for _, _, v in data], dtype=np.int64)}


def block(t, row_level, col_level):
    """The sub-matrix of one row level by one column level, with its row and column codes."""
    ri = [i for i, k in enumerate(t["row_kind"]) if k == row_level]
    ci = [j for j, k in enumerate(t["col_kind"]) if k == col_level]
    return (t["values"][np.ix_(ri, ci)], [t["row_code"][i] for i in ri],
            [t["col_code"][j] for j in ci])


def read_secondary_by_area(year, path=None):
    """{level: {code: 総数}} from the published secondary-workplace table."""
    from pathlib import Path
    p = Path(path or os.path.join("data", "reference", f"secondary_workplace_by_smsa_{year}.csv"))
    out = {k: {} for k in LEVELS}
    for r in read_csv_rows(p):
        if not r or len(r) < 2:
            continue
        code, _, kind = parse_code_name(r[0])
        if kind not in LEVELS:
            continue
        try:
            v = _num(r[1])
        except ValueError:
            continue
        if code in out[kind]:
            raise ValueError(f"{p}: {code} appears twice")
        out[kind][code] = v
    return out
