"""
Permanent guards against the failure mode that has now produced three defects:

  1. Moran's I 2006-2016 computed on a changing 319-322 circle set (found June)
  2. Table S4's 2016 row computed on 322 circles (found last round)
  3. the 2010 age anchor moving because 141 of 1,892 municipalities are absent from the
     2005 census, so circle shares went NaN and were silently dropped by nansum

All three are the same bug: the number of units participating in a cross-circle aggregate
changed without anyone noticing, because the missing ones were removed implicitly.

These helpers make that impossible to do quietly. They RAISE; they do not warn.
"""
import numpy as np


class UnitCountError(AssertionError):
    """The number of units entering an aggregate is not what the caller declared."""


class MissingValueError(AssertionError):
    """A value that must be present is NaN."""


def assert_units(n_obs, expected, what):
    """Every cross-unit aggregate must declare how many units it expects."""
    n = int(n_obs) if np.isscalar(n_obs) else len(n_obs)
    if n != expected:
        raise UnitCountError(
            f"{what}: {n} units entered the aggregate, expected {expected}. "
            f"A silent change in the unit set is the defect this guard exists to catch; "
            f"if the change is intended, say so explicitly at the call site.")
    return n


def no_missing(a, what, axis=None):
    """Refuse NaN where completeness is part of the claim."""
    arr = np.asarray(a, dtype=float)
    bad = ~np.isfinite(arr)
    if bad.any():
        idx = np.argwhere(bad)
        raise MissingValueError(
            f"{what}: {int(bad.sum())} missing/non-finite value(s), first at index "
            f"{tuple(idx[0])}. Use an explicit fill with a stated assumption, not nansum.")
    return arr


def lerp(a, b, w, what="interpolation"):
    """Linear interpolation with EXPLICIT endpoints.

    At w == 0 return a, at w == 1 return b, WITHOUT touching the other endpoint. The
    algebraic form a + (b - a) * w is identical in exact arithmetic but propagates NaN
    from the endpoint it should be ignoring -- which is exactly how the 2010 anchor moved
    when the 2005 array carried NaN for uncovered circles.
    """
    if w == 0:
        return np.array(a, dtype=float, copy=True)
    if w == 1:
        return np.array(b, dtype=float, copy=True)
    A = no_missing(a, f"{what}: lower endpoint")
    B = no_missing(b, f"{what}: upper endpoint")
    return A + (B - A) * w


def extrap(a, b, w, what="extrapolation"):
    """Forward extrapolation beyond b along the a->b slope. w is measured from b."""
    A = no_missing(a, f"{what}: earlier anchor")
    B = no_missing(b, f"{what}: later anchor")
    return B + (B - A) * w
