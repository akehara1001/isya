"""Supplementary Table S14 (paper A): age structure of physicians by prefecture group, 2010 vs 2024.

Round 40 §0-5/§2 (2026-10-06). This is a script from an internal work record, the draft
that produced Table S14 for the Word file of 2026-10-05, moved into the analysis so that the
table and the age-structure item of the main text's 3.4 are produced inside the closure and
bound (h04 compares the table cell by cell, h11 binds the prose). The computation is the
draft's, unchanged: the same inputs, the same pooling, the same terciles, the same bootstrap and
seed. Two things differ from the draft. The inputs are read from data/raw/external/physician_age/
(copies of the two e-Stat files, byte-identical, with provenance rows in the data manifest), and
the prefecture names, which the draft took from physician_per100k_pref.csv and the table does
not print, are not read at all.

Inputs
  data/raw/external/physician_age/estat_000012664634_2010_table29_physicians_age_pref.csv
      平成22年 医師・歯科医師・薬剤師調査 第29表 (e-Stat statInfId 000012664634, cp932)
  data/raw/external/physician_age/estat_000040383756_2024_table4_physicians_age_pref.csv
      令和6年 医師・歯科医師・薬剤師統計 医師 第4表 (statInfId 000040383756, utf-8)
  data/reference/census_official_totals.csv   2010 prefecture population (published census)
  census_based/smsa_population_census_2024.csv 2024 prefecture population: the paper's own
      fixed-map area populations summed by prefecture (0.2 x 2020 + 0.8 x 2025, both confirmed)
Definitions: all physicians (総数) at the (primary) workplace prefecture; shares pooled within a
group (physicians in the age band / all physicians in the group); terciles of the 2010-2024
population change; Spearman rho across the 47 prefectures with a 10 000-resample bootstrap over
prefectures (seed 20261005).

Writes outputs/audit/a11_age_prefecture_by_pref.csv (47 rows) and
outputs/audit/a11_age_prefecture_S14.csv (3 terciles, all 47, and the Spearman row).
"""
import csv
import io
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

AGES = ['24歳以下', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69',
        '70-74', '75-79', '80-84', '85歳以上']


def read_text(path):
    data = open(path, 'rb').read()
    for enc in ('utf-8-sig', 'cp932'):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    raise SystemExit(path)


def parse(path, year):
    rows = list(csv.reader(io.StringIO(read_text(path))))
    out = {}
    for r in rows:
        if year == 2024:
            if len(r) < 20 or r[0] != '総数':
                continue
            if r[2] == '全国':
                name, vals = '全国', r[4:19]
            elif re.match(r'^[０-９]{2}', r[1]) and r[2] == '':
                name, vals = r[1], r[4:19]
            else:
                continue
        else:
            if len(r) < 17:
                continue
            lab = r[0].replace('　', '').replace(' ', '')
            if lab == '全国':
                name, vals = '全国', r[1:16]
            elif re.match(r'^[０-９]{2}', lab):
                name, vals = lab, r[1:16]
            else:
                continue
        vals = [int(v.replace(',', '')) if v.strip() not in ('-', '') else 0 for v in vals]
        total, counts = vals[0], vals[1:15]
        assert abs(sum(counts) - total) <= 2, (year, name, total, sum(counts))
        key = '全国' if name == '全国' else re.sub(r'^([０-９]{2}).*', r'\1', name).translate(
            str.maketrans('０１２３４５６７８９', '0123456789'))
        if key in out:
            continue          # keep the first block (総数); 男 / 女 blocks follow
        out[key] = np.array(counts)
    assert len(out) == 48, len(out)
    return out


d10 = parse("data/raw/external/physician_age/estat_000012664634_2010_table29_physicians_age_pref.csv", 2010)
d24 = parse("data/raw/external/physician_age/estat_000040383756_2024_table4_physicians_age_pref.csv", 2024)
assert d10['全国'].sum() == 295049 and d24['全国'].sum() == 347772

_ref = pd.read_csv("data/reference/census_official_totals.csv", dtype={'pref_code': str})
_r10 = _ref[(_ref.census_year == 2010) & (_ref.pref_code != '00')].set_index('pref_code')['official_population']
_a24 = pd.read_csv("census_based/smsa_population_census_2024.csv")
_a24['pref_code'] = _a24['smsa_code'].astype(str).str.zfill(4).str[:2]
_p24 = _a24.groupby('pref_code')['population'].sum()
pop = pd.DataFrame({'pref_code': _r10.index, '2010': _r10.astype(float).values,
                    '2024': _p24.reindex(_r10.index).values})
assert len(pop) == 47 and pop.notna().all().all()

rows = []
for k in sorted(k for k in d10 if k != '全国'):
    c0, c1 = d10[k], d24[k]
    rows.append(dict(pref_code=k, n10=c0.sum(), n24=c1.sum(),
                     u40_10=c0[:4].sum(), u40_24=c1[:4].sum(), o60_10=c0[8:].sum(), o60_24=c1[8:].sum()))
df = pd.DataFrame(rows).merge(pop[['pref_code', '2010', '2024']], on='pref_code')
df['dpop'] = (df['2024'] / df['2010'] - 1) * 100
df['sh60_10'] = df['o60_10'] / df['n10'] * 100
df['sh60_24'] = df['o60_24'] / df['n24'] * 100
df['d_sh60'] = df['sh60_24'] - df['sh60_10']


def spearman(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


rho = spearman(df['dpop'].values, df['d_sh60'].values)
rng = np.random.default_rng(20261005)
n = len(df)
bs = []
for _ in range(10000):
    i = rng.integers(0, n, n)
    bs.append(spearman(df['dpop'].values[i], df['d_sh60'].values[i]))
lo, hi = np.percentile(bs, [2.5, 97.5])

df['tercile'] = pd.qcut(df['dpop'], 3, labels=['Largest decline', 'Middle', 'Smallest decline or growth'])
df.to_csv("outputs/audit/a11_age_prefecture_by_pref.csv", index=False, encoding="utf-8-sig")
out = []
for name, d in list(df.groupby('tercile', observed=True)) + [('All 47 prefectures', df)]:
    out.append(dict(group=name, prefectures=len(d), dpop_min=d['dpop'].min(), dpop_max=d['dpop'].max(),
                    pop_change=(d['2024'].sum() / d['2010'].sum() - 1) * 100,
                    physicians_2010=d['n10'].sum(), physicians_2024=d['n24'].sum(),
                    physician_change=(d['n24'].sum() / d['n10'].sum() - 1) * 100,
                    share60_2010=d['o60_10'].sum() / d['n10'].sum() * 100,
                    share60_2024=d['o60_24'].sum() / d['n24'].sum() * 100,
                    share40_2010=d['u40_10'].sum() / d['n10'].sum() * 100,
                    share40_2024=d['u40_24'].sum() / d['n24'].sum() * 100,
                    change_n60=(d['o60_24'].sum() / d['o60_10'].sum() - 1) * 100,
                    change_n40=(d['u40_24'].sum() / d['u40_10'].sum() - 1) * 100))
out.append(dict(group='Spearman rho', spearman_rho=rho, spearman_ci_lo=lo, spearman_ci_hi=hi))
t = pd.DataFrame(out)
t.to_csv("outputs/audit/a11_age_prefecture_S14.csv", index=False, encoding="utf-8-sig")
print(f"Spearman rho (47 prefectures, population change 2010-2024 vs change in share aged 60+): "
      f"{rho:.3f} [{lo:.3f}, {hi:.3f}]")
print(t.round(2).to_string())
print("terciles:", {g: sorted(df[df.tercile == g].pref_code) for g in df.tercile.cat.categories})
