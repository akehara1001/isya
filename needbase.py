"""THE age-stated fraction: a GUARD on the census edition, not a factor in the base.

WHY THIS EXISTS. Nine live producers implemented the need-weighted denominator
independently -- c01_round3_numbers, s02_scale_invariance, s11_municipality_noise,
b06_recompute_all, cc_decomp_ci, cc_stage2_needweight, a05_decomp_BA_difference,
cc_needweight_2023, a02_scale_truncation_waves -- each with its own copy of
`raw * nm / normaliser`. The family was counted at 2, then 5, then 7, then 9, each time
by grepping harder, so the construction now lives here and c11_need_base_single.py fails
if a producer grows its own copy.

WHAT THE QUANTITY IS. `sma_age_shares` (every producer has one) divides a unit's age
vector by the age-band SUM, so its rows are shares of the AGE-STATED population.
Multiplying those shares by the PUBLISHED total distributes the unstated-age population
pro rata. Whether there is anything to distribute depends on the census edition:

    2005  0000033787   年齢不詳 reported separately, five-year classes kept  -> NOT exhaustive
    2010  0003038591   年齢不詳 reported separately, five-year classes kept  -> NOT exhaustive
    2015  0004019348   不詳補完結果 (published retrospectively with 2020)     -> exhaustive
    2020  0004019309   不詳補完結果                                          -> exhaustive

Measured on this repo's tables, over the municipalities the circle map matches: the
fraction is 0.995850 (2005), 0.992399 (2010), and exactly 1 in 2015 and 2020.

THE FRACTION IS NOT APPLIED TO THE BASE, AND THAT IS A REVERSAL. Between 2026-08-31 and
2026-09-02 every caller multiplied its base by this fraction, which cut the need-weighted
and 65-plus denominators down to the age-stated population. That was wrong on three
counts, each checkable rather than argued (item B-100 of an internal work record):

  1. THE MANUSCRIPT DEFINES THE RULE, and it is pro rata. main.tex:302 and
     supplementary Eq.~(needweight) both print
         P_i^need = P_i * sum_a s_ia r_a / rbar
     with P_i the resident population, and supplementary:309 says the shares "were
     applied to the validated SMA total populations", the persons of unstated age being
     "excluded from every age table alike ... so the shares are of the age-stated
     population in all four census years". Shares of the age-stated population applied
     to the published total IS pro-rata apportionment. Verified numerically: the old
     rule and an explicit apportionment (A_ia = P_i s_ia, then sum_a A_ia r_a) agree to
     2.7e-16 relative.
  2. IT BROKE A PRINTED IDENTITY. supplementary:323 states that any need-vs-resident
     Gini difference "arises solely from between-circle variation in the ratio
     P_i^need/P_i = sum_a s_ia r_a / rbar; a circle whose age composition equals the
     national average is unaffected". Under pro rata that identity holds to 1.7e-16;
     under the fraction it fails by 5.3%. Circle 2801, the closest in the panel to the
     national age composition, prints a need weight of 0.997364 and got 0.988109.
  3. IT WAS NOT ONE RULE FOR FIVE WAVES. The fraction is 1 from the 2015 anchor on, so
     it changed the denominator's DEFINITION for 2006-2014 and left 2016-2024 alone --
     five waves of ten. Worse, `interpolate` then produced fractions like 0.985 for
     2014, a fictional enumeration-completeness rate for a year with no census, when
     completeness is a property of a published table and not of a population.

What the fraction measures is census enumeration completeness, which is a data-collection
artefact; the need weight is meant to carry age composition alone. It stays here, and
every caller still computes it, because `assert_unity` is worth keeping: at an exhaustive
anchor the fraction MUST be 1, so a re-fetch that silently returned a non-補完 edition
would change what `s_ia` means, and that fails loudly instead of moving a Gini quietly.

WHAT IS NOT AFFECTED. The RESIDENT denominator, then and now: every caller's
denom(..., "raw") returns the published total untouched, which is the adoption rule
written in census_inputs.py.

AREA AGE TABLES (round 38, 2026-10-06). The circle-level producers in the rebuild closure --
a02, b06, c01, cc_needweight_2023, s02 (circle scale), s12 and i06 -- no longer sum
census_age/age_muni_{cy}.csv over the 2024 map's codes. That sum silently dropped every
municipality whose code changed between the census and 2024 (熊本市 in 2010, the seven
Hamamatsu wards in 2010-2020, absorbed towns), so their areas carried the age mix of what was
left. They read census_age/age_circle_{cy}.csv instead, which cc_build_census_age.py builds
once through census_crosswalk -- the same code-change tables as the area populations -- and
they take the guard as fraction_from_sums(age-stated, population) over that table's own two
columns. unit_fraction below is unchanged and still serves the municipality-keyed callers
outside the closure; it shares the old blind spot (a municipality absent under its 2024 code
is skipped on both sides), which cannot move a guard that is exactly 1 at the 不詳補完 anchors.

ON ARITHMETIC FIDELITY. `unit_fraction` loops over `muni_to_unit.items()` in the caller's
own dict order, and `interpolate` reproduces each caller's branch structure including the
exact-anchor shortcut, because these figures feed 10,000-draw bootstraps where a
last-ulp difference can flip a tie and move a published probability by 0.0001.
"""
import numpy as np
import pandas as pd

# A fraction outside this band means the two sides were not the same population --
# a coverage mismatch, not a small correction -- so it is clipped and shows up as 1.0
# or 0.5 rather than silently scaling a denominator by an implausible factor.
FLOOR, CEIL = 0.5, 1.0

# Anchors whose census edition is 不詳補完結果, where the fraction must be exactly 1.
EXHAUSTIVE = (2015, 2020)


def muni_published(cy):
    """Published total population by 5-digit municipality code, for census year cy."""
    from census_inputs import municipality_csv
    p = pd.read_csv(municipality_csv(cy), dtype={"area_code": str})
    p["area_code"] = p["area_code"].str.zfill(5)
    return pd.to_numeric(p.set_index("area_code")["population"].astype(str)
                         .str.replace(",", "", regex=False), errors="coerce")


def muni_age_table(cy):
    """The municipality age table for census year cy, indexed by 5-digit code."""
    age = pd.read_csv(f"census_age/age_muni_{cy}.csv", dtype={"area_code": str})
    age["area_code"] = age["area_code"].str.zfill(5)
    return age.set_index("area_code")


def unit_fraction(cy, groups, units, muni_to_unit):
    """Age-stated / published per unit, summed over the SAME municipalities on both sides.

    Coverage cancels: a municipality missing from either table is skipped in both sums.
    The loop runs over muni_to_unit.items() so the accumulation order is the caller's.
    """
    age = muni_age_table(cy)
    pv = muni_published(cy)
    a = np.zeros(len(units))
    p = np.zeros(len(units))
    idx = {c: i for i, c in enumerate(units)}
    for muni, unit in muni_to_unit.items():
        i = idx.get(unit)
        if i is None or muni not in age.index or muni not in pv.index:
            continue
        if not np.isfinite(pv.loc[muni]):
            continue
        a[i] += float(age.loc[muni, groups].to_numpy(float).sum())
        p[i] += float(pv.loc[muni])
    # c01 carried this guard and the others did not. It is kept here, on the loop-based
    # path only: a municipality absent from either table is skipped above, so a NaN here
    # means a NaN INSIDE a row that is present, which is a data fault and not a coverage
    # gap. fraction_from_sums stays tolerant, because s02 and s11 reindex to a basis that
    # legitimately produces NaN for municipalities the age table does not carry.
    if not np.isfinite(a).all():
        raise AssertionError(f"{cy}: non-finite age-stated sum in a matched municipality")
    return fraction_from_sums(a, p)


def fraction_from_sums(a, p):
    """The ratio itself, for callers that build the two sums their own way."""
    a = np.asarray(a, float)
    p = np.asarray(p, float)
    ok = np.isfinite(a) & np.isfinite(p) & (p > 0)
    f = np.where(ok, np.divide(a, np.where(p > 0, p, 1.0)), 1.0)
    return np.clip(f, FLOOR, CEIL)


def assert_unity(frac, label="", anchors=EXHAUSTIVE, tol=1e-9):
    """At an exhaustive anchor the fraction must be 1 in every unit."""
    for cy in anchors:
        if cy not in frac:
            continue
        dev = float(np.nanmax(np.abs(np.asarray(frac[cy], float) - 1.0)))
        if dev > tol:
            raise AssertionError(
                f"{cy}{' ' + label if label else ''}: the age table is 不詳補完結果, so "
                f"its age-stated fraction must be 1 everywhere; max deviation {dev:.3g}")


def interpolate(frac, ay, exact_anchor=False, tail="segment"):
    """Piecewise-linear across the anchors, extrapolating on the nearest segment.

    Reproduces every caller's own branch structure, because these figures feed
    10,000-draw bootstraps and the two families differ in ways that are algebraically
    identical but not bitwise:

      exact_anchor  return the stored array unchanged when ay IS an anchor. c01 had that
                    shortcut, the others did not, and a + (b - a) * 1.0 is not bitwise b.
      tail          past the last anchor, "segment" continues the final segment from its
                    LEFT end -- F15 + (F20 - F15) * (ay - 2015) / 5, which is what the
                    three-anchor producers wrote -- while "anchor" measures from the last
                    anchor itself, F20 + (F20 - F15) * (ay - 2020) / 5, which is what c01
                    wrote. Equal on paper; not equal in the last bit.
    """
    if exact_anchor and ay in frac:
        return frac[ay]
    ks = sorted(frac)
    lo = ks[0]
    for k in ks[:-1]:
        if ay > k:
            lo = k
    hi = ks[ks.index(lo) + 1]
    if ay > ks[-1] and tail == "anchor":
        return frac[hi] + (frac[hi] - frac[lo]) * ((ay - hi) / float(hi - lo))
    return frac[lo] + (frac[hi] - frac[lo]) * ((ay - lo) / float(hi - lo))
