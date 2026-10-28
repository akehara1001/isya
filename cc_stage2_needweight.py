"""
STAGE 2: need-weighted denominator. Recompute density Gini + A/B/C decomposition
under three denominators:
  (raw)  resident population         density = stock / pop
  (need) need-weighted population    density = stock / [pop * relative age-need]
  (e65)  >=65 elderly population     density = stock / pop65  (transparent proxy)

Need weight: national age-specific 受療率 (患者調査 2020, 総数=入院+外来, per 100k),
applied to each SMA's age composition (census 2010/2015/2020 aggregated to the fixed
330-circle map, interpolated/extrapolated to each analysis year). Need multiplier is
normalised so national need-pop == national resident pop each year (scale-comparable Gini).

Reads: census_age/ (fetched), census_based/ (validated SMA totals), data/superseded/physician_panel_smsa__PRE_REPAIR.csv,
       smsa_2024.csv (fixed muni->SMA map).  Writes ONLY to exploratory/.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path
from phase2_common import read_csv_rows, parse_code_name

OUT = "exploratory"
YEARS = [2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
CENSUS_AGE_YEARS = [2010, 2015, 2020]
GROUPS = ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44",
          "45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84","85-89","90+"]
ELDERLY = [i for i, g in enumerate(GROUPS) if g in ("65-69","70-74","75-79","80-84","85-89","90+")]
N_BOOT = 10000
SEED = 20260621

panel = pd.read_csv("data/superseded/physician_panel_smsa__PRE_REPAIR.csv")
panel["code4"] = panel["smsa_code"].apply(lambda c: str(c).zfill(4))
CODES = panel["code4"].tolist()


def gini(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = x.size
    if n == 0 or x.sum() == 0:
        return np.nan
    xs = np.sort(x); idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * xs)) / (n * xs.sum()) - (n + 1) / n


def build_map(vintage=2024):
    m = {}; cur = None
    for row in read_csv_rows(Path(f"smsa_{vintage}.csv")):
        if not row:
            continue
        code, name, kind = parse_code_name(row[0])
        if kind == "smsa" and code:
            cur = str(code).zfill(4)
        elif kind == "municipality" and code and cur:
            m[str(code).zfill(5)] = cur
    return m

MAP = build_map(2024)


def stock(y):
    return panel.set_index("code4")[str(y)].reindex(CODES).astype(float)

def total_pop(y):
    df = pd.read_csv(f"census_based/smsa_population_census_{y}.csv")
    df["code4"] = df["smsa_code"].apply(lambda c: str(c).zfill(4))
    return df.set_index("code4")["population"].reindex(CODES).astype(float)

# ---- SMA age SHARES at each census year ----
def sma_age_shares(cy):
    age = pd.read_csv(f"census_age/age_muni_{cy}.csv", dtype={"area_code": str})
    age["area_code"] = age["area_code"].str.zfill(5)
    age = age.set_index("area_code")
    mat = np.zeros((len(CODES), 19)); cidx = {c: i for i, c in enumerate(CODES)}
    covered = {c: 0 for c in CODES}
    for muni, sma in MAP.items():
        if sma in cidx and muni in age.index:
            mat[cidx[sma]] += age.loc[muni, GROUPS].to_numpy(float)
            covered[sma] += 1
    tot = mat.sum(axis=1, keepdims=True)
    shares = np.divide(mat, tot, out=np.full_like(mat, np.nan), where=tot > 0)
    return shares  # (330,19)

CENSUS_SHARES = {cy: sma_age_shares(cy) for cy in CENSUS_AGE_YEARS}


from needbase import unit_fraction, assert_unity

# The census-edition GUARD. THE CONSTRUCTION LIVES IN needbase.py -- see that file for
# what the fraction is, why 2015/2020 must come back exactly 1, and why it is NOT applied
# to the base. All three denominators this module returns stand on the published total.
GFRAC = {cy: unit_fraction(cy, GROUPS, CODES, MAP) for cy in CENSUS_AGE_YEARS}
assert_unity(GFRAC, "circle")

def shares_for_year(ay):
    s2010, s2015, s2020 = CENSUS_SHARES[2010], CENSUS_SHARES[2015], CENSUS_SHARES[2020]
    if ay <= 2015:
        v = s2010 + (s2015 - s2010) * ((ay - 2010) / 5.0)
    else:
        v = s2015 + (s2020 - s2015) * ((ay - 2015) / 5.0)
    v = np.clip(v, 0, None)
    rs = v.sum(axis=1, keepdims=True); rs[rs == 0] = np.nan
    return v / rs

# ---- need weights (patient survey 2020) ----
rate = pd.read_csv("census_age/juryoritsu_2020.csv").set_index("age_group")["juryoritsu_per100k"]
W = rate.reindex(GROUPS).to_numpy(float)  # per-age utilisation rate

def denominators(ay):
    """Return (raw_pop, need_pop, pop65) Series indexed by CODES for analysis year ay."""
    tot = total_pop(ay)
    sh = shares_for_year(ay)                      # (330,19) shares
    raw = tot.to_numpy(float)
    # PRO RATA: the weighted forms stand on the PUBLISHED total, so the age shares
    # distribute the unstated-age residents in the age-stated proportions. The
    # manuscript's Eq. (needweight); see needbase.py for why the age-stated base went.
    base = raw
    need_mult = (sh * W).sum(axis=1)              # per-SMA mean utilisation rate
    # normalise so national need-pop == national resident pop
    natl_rbar = np.nansum(base * need_mult) / np.nansum(base)
    need_pop = base * need_mult / natl_rbar
    pop65 = base * sh[:, ELDERLY].sum(axis=1)
    return (pd.Series(raw, index=CODES), pd.Series(need_pop, index=CODES),
            pd.Series(pop65, index=CODES))

# ---- bootstrap CI on a density vector ----
rng = np.random.default_rng(SEED)
def boot_ci(dens):
    v = np.asarray(dens, float); v = v[~np.isnan(v)]; n = v.size
    idx = rng.integers(0, n, size=(N_BOOT, n)); samp = v[idx]
    s = np.sort(samp, axis=1); ii = np.arange(1, n + 1)
    g = (2 * (s * ii).sum(1)) / (n * s.sum(1)) - (n + 1) / n
    return float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))

# ============ Gini series for each denominator ============
rows = []
for ay in YEARS:
    s = stock(ay)
    raw, need, e65 = denominators(ay)
    d_raw = (s / raw * 1e5).values
    d_need = (s / need * 1e5).values
    d_e65 = (s / e65 * 1e5).values
    lo_n, hi_n = boot_ci(d_need)
    lo_r, hi_r = boot_ci(d_raw)
    rows.append({"year": ay,
                 "gini_raw": round(gini(d_raw), 4), "raw_lo": round(lo_r,4), "raw_hi": round(hi_r,4),
                 "gini_need": round(gini(d_need), 4), "need_lo": round(lo_n,4), "need_hi": round(hi_n,4),
                 "gini_e65": round(gini(d_e65), 4),
                 "natl_elderly_share": round(float(np.nansum(e65)/np.nansum(raw)), 4)})
gdf = pd.DataFrame(rows)
gdf.to_csv(f"{OUT}/S2_gini_by_denominator.csv", index=False)

# ============ A/B/C decomposition (base 2010 -> 2024) ============
def decomp(denom_fn, base=2010, end=2024):
    sb = stock(base); se = stock(end)
    _, _, _ = None, None, None
    db = denom_fn(base); de = denom_fn(end)
    g_base = gini((sb / db * 1e5).values)
    g_end = gini((se / de * 1e5).values)
    cf_A = gini((se / db * 1e5).values)   # vary stock, fix denom@base -> physician redistribution
    cf_B = gini((sb / de * 1e5).values)   # fix stock@base, vary denom -> population/need redistribution
    total = g_end - g_base
    dA = cf_A - g_base
    dB = cf_B - g_base
    inter = total - dA - dB
    return {"g_base": round(g_base,4), "g_end": round(g_end,4), "total": round(total,4),
            "A_phys": round(dA,4), "B_pop": round(dB,4), "interaction": round(inter,4),
            "A_pct": round(100*dA/total,1), "B_pct": round(100*dB/total,1), "int_pct": round(100*inter/total,1)}

den_raw = lambda y: denominators(y)[0]
den_need = lambda y: denominators(y)[1]
den_e65 = lambda y: denominators(y)[2]

dec = {"raw": decomp(den_raw), "need": decomp(den_need), "e65": decomp(den_e65)}
pd.DataFrame(dec).T.to_csv(f"{OUT}/S2_decomposition.csv")

# ============ report ============
print("="*92)
print("STAGE 2: density Gini by denominator (consistent 330 circles)")
print("="*92)
print(f"{'year':>5} {'gini_raw':>9} {'gini_need':>10} {'gini_e65':>9} {'elderly_share':>14}")
for r in rows:
    print(f"{r['year']:>5} {r['gini_raw']:>9.4f} {r['gini_need']:>10.4f} {r['gini_e65']:>9.4f} {r['natl_elderly_share']:>14.4f}")

print("\nNeed-weighted series with bootstrap 95% CI:")
for r in rows:
    print(f"  {r['year']}: need={r['gini_need']:.4f} [{r['need_lo']:.4f},{r['need_hi']:.4f}]   raw={r['gini_raw']:.4f} [{r['raw_lo']:.4f},{r['raw_hi']:.4f}]")

print("\n" + "="*92)
print("A/B/C decomposition 2010 -> 2024 (A=physician redistribution, B=denominator redistribution)")
print("="*92)
print(f"{'denom':>6} {'base2010':>9} {'2024':>7} {'total':>8} {'A_phys':>8} {'B_pop':>8} {'inter':>8}   shares(A/B/int %)")
for k in ["raw","need","e65"]:
    d = dec[k]
    print(f"{k:>6} {d['g_base']:>9.4f} {d['g_end']:>7.4f} {d['total']:>+8.4f} {d['A_phys']:>+8.4f} {d['B_pop']:>+8.4f} {d['interaction']:>+8.4f}   {d['A_pct']}/{d['B_pct']}/{d['int_pct']}")

print("\nValidation vs frozen raw decomp (expect base0.2614 end0.1904 A-0.0189 B-0.0337 int-0.0184):")
print("  raw:", dec["raw"])
print("\nDecline change under need-weighting:")
draw = dec["raw"]["total"]; dneed = dec["need"]["total"]
print(f"  raw total decline {draw:+.4f}  ->  need-weighted total decline {dneed:+.4f}  "
      f"({'ATTENUATED' if abs(dneed)<abs(draw) else 'NOT attenuated'}; need/raw = {dneed/draw:.2f})")

# ============ paired-bootstrap significance of the decline ============
def gini_rows(M):
    S = np.sort(M, axis=1); ii = np.arange(1, M.shape[1] + 1)
    return (2 * (S * ii).sum(1)) / (M.shape[1] * S.sum(1)) - (M.shape[1] + 1) / M.shape[1]

def paired_decline(denom_fn, base, end, seed):
    sb, se = stock(base), stock(end)
    a = (sb / denom_fn(base) * 1e5).values
    b = (se / denom_fn(end) * 1e5).values
    mask = ~(np.isnan(a) | np.isnan(b)); a, b = a[mask], b[mask]; n = a.size
    r = np.random.default_rng(seed); idx = r.integers(0, n, size=(N_BOOT, n))
    dec = gini_rows(b[idx]) - gini_rows(a[idx])
    return dec.mean(), np.percentile(dec, 2.5), np.percentile(dec, 97.5), (dec < 0).mean()

print("\n" + "="*92)
print("Paired-bootstrap decline significance (10,000 joint circle resamples)")
print("="*92)
for name, fn in [("raw", den_raw), ("need", den_need), ("e65", den_e65)]:
    for (b, e) in [(2010, 2024), (2006, 2024)]:
        m, lo, hi, p = paired_decline(fn, b, e, SEED + hash((name, b, e)) % 1000)
        print(f"  {name:>4} {b}->{e}: decline {m:+.4f}  95%CI [{lo:+.4f},{hi:+.4f}]  P(decline)={p:.4f}")

# ============ fully-anchored 2010->2020 decomposition (both census-age endpoints REAL) ============
print("\n" + "="*92)
print("Robustness: decomposition 2010->2020 (census age REAL at both ends, no extrapolation)")
print("="*92)
print(f"{'denom':>6} {'base':>8} {'end':>8} {'total':>8} {'A_phys':>8} {'B_pop':>8} {'inter':>8}   A/B/int %")
for k, fn in [("raw", den_raw), ("need", den_need), ("e65", den_e65)]:
    d = decomp(fn, base=2010, end=2020)
    print(f"{k:>6} {d['g_base']:>8.4f} {d['g_end']:>8.4f} {d['total']:>+8.4f} {d['A_phys']:>+8.4f} {d['B_pop']:>+8.4f} {d['interaction']:>+8.4f}   {d['A_pct']}/{d['B_pct']}/{d['int_pct']}")

print("\nsaved S2_gini_by_denominator.csv, S2_decomposition.csv")
