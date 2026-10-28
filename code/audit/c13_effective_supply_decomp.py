"""C-1 (co-author review) -- the counterfactual decomposition run on EFFECTIVE SUPPLY.

WHAT IS DECOMPOSED. The same accounting identity the manuscript uses (main text
Eq.~(decomp)): a density-Gini change split into a numerator component A, a denominator
component B and their interaction, under the x-first, P-first and symmetric (Shapley)
attributions, with joint circle-bootstrap intervals. Nothing here is a new statistic; the
only things that change are the NUMERATOR and the WINDOW, both of which c01's decompose()
now takes as arguments (C12).

THE NUMERATOR. Effective supply as the manuscript defines it, S_i = x_i + theta * n_i,
with x_i the primary-workplace count and n_i the net secondary-practice inflow (main text
Eq.~(eff)). THETA = 0.2, and that value is not a choice made here -- it is the one the
manuscript names as the policy instrument's own:

    supplementary.tex:556  "The grid also contains $\\theta=0\\dmid 2$, the weight adopted
    for Japan's official maldistribution index in 2022, under which a physician counts as
    $0\\dmid 8$ at the primary and $0\\dmid 2$ at the secondary workplace [S5, S6]."

The head-count series x_i is carried at the same window as the comparator, so the pair
differs in the numerator alone.

THE WINDOW. 2016--2024, and only that. S_i exists only where the primary-to-secondary
matrix does -- main.tex:231, "defined only for the five waves in which the primary-to-
secondary matrix exists, t in {2016, 2018, 2020, 2022, 2024}; elsewhere S_i = x_i" -- so a
2010 or 2012 base would silently decompose a window in which effective supply IS the head
count. No other window is computed.

INPUTS, and why they are sound for this window. results/effective_supply_panel.csv carries
primary_stock and net_inflow per circle-wave. Its producer sits outside the rebuild closure
and predates the crosswalk repair, so its primary_stock is checked here against
data/crosswalk/physician_panel_smsa_repaired.csv on every wave used; q09 records the same
agreement independently. The denominators come from c01's own denom(), so the resident and
need-weighted bases are the ones the paper uses.

Writes outputs/audit/c13_effective_supply_decomp.csv with one row per
(numerator, denominator, quantity) and the columns declared in COLUMNS below.
"""
import io, os, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "code", "audit"))

THETA = 0.2
BASE, END = 2016, 2024
COLUMNS = ["numerator", "denominator", "scheme", "quantity", "point", "ci_lo", "ci_hi",
           "excludes_0"]

# c01 rewraps sys.stdout at import and the wrapper it replaces is closed when collected,
# so the stream this file is printing to has to be held and restored around the import --
# the same dance s10_ledger.py does around s06.
_KEEP = [sys.stdout]
import c01_round3_numbers as c01                                          # noqa: E402
_KEEP.append(sys.stdout)
sys.stdout = _KEEP[0]
os.chdir(ROOT)

CODES = list(c01.CODES)
REP = c01.PANELS["repaired"]

ES = pd.read_csv("results/effective_supply_panel.csv")
ES["code4"] = ES["smsa_code"].apply(lambda c: str(c).zfill(4))

# --- the numerator's own check: the panel's primary stock IS the repaired panel
for y in (BASE, END):
    e = ES[ES.year == y].set_index("code4")["primary_stock"].astype(float).reindex(CODES)
    r = REP[y].astype(float).reindex(CODES)
    n_bad = int((np.abs(e.to_numpy() - r.to_numpy()) > 1e-9).sum())
    assert n_bad == 0, (
        f"{y}: {n_bad} of {len(CODES)} circles differ between effective_supply_panel's "
        f"primary_stock and the repaired physician panel; the numerator is not the "
        f"paper's own series")
print(f"primary_stock matches the repaired panel in {BASE} and {END}: "
      f"0 of {len(CODES)} circles differ, both waves")


def supply(y):
    """S_i = x_i + theta * n_i for wave y, in CODES order."""
    e = ES[ES.year == y].set_index("code4")
    x = e["primary_stock"].astype(float).reindex(CODES).to_numpy()
    n = e["net_inflow"].astype(float).reindex(CODES).to_numpy()
    assert np.isfinite(x).all() and np.isfinite(n).all(), f"{y}: non-finite supply input"
    return x + THETA * n


def headcount(y):
    return REP[y].astype(float).reindex(CODES).to_numpy()


# net inflow sums to zero across circles by construction; state it rather than assume it
for y in (BASE, END):
    e = ES[ES.year == y].set_index("code4")["net_inflow"].astype(float).reindex(CODES)
    print(f"  {y}: sum of net secondary inflow over the 330 circles = {e.sum():+.6f} "
          f"(zero by construction, so S and x share a national total)")

rows = []
for numlab, f in (("head count x", headcount), (f"effective supply (theta={THETA})", supply)):
    x0, x1 = f(BASE), f(END)
    for kind in ("raw", "need"):
        P0, P1 = c01.denom(BASE, kind), c01.denom(END, kind)
        d = c01.decompose(x0, x1, P0, P1)
        for scheme, akey, bkey in (("x-first", "A_x", "B_P"),
                                   ("P-first", "A_P", "B_x"),
                                   ("Shapley", "A_S", "B_S")):
            for lab, k in ((f"A ({scheme})", akey), (f"B ({scheme})", bkey)):
                lo, hi = d[k + "_ci"]
                rows.append(dict(numerator=numlab, denominator=kind, scheme=scheme,
                                 quantity=lab, point=round(float(d[k]), 6),
                                 ci_lo=round(float(lo), 6), ci_hi=round(float(hi), 6),
                                 excludes_0=bool(lo > 0 or hi < 0)))
        for lab, k, cik in ((f"G({BASE})", "g0", None), (f"G({END})", "g1", None),
                            ("total dG", "tot", "tot_ci"),
                            ("interaction", "I", "I_ci"),
                            ("B-A (x-first)", "BmA_x", "BmA_x_ci"),
                            ("B-A (Shapley)", "BmA_S", "BmA_S_ci")):
            lo, hi = d[cik] if cik else (np.nan, np.nan)
            rows.append(dict(numerator=numlab, denominator=kind, scheme="—",
                             quantity=lab, point=round(float(d[k]), 6),
                             ci_lo=(round(float(lo), 6) if cik else np.nan),
                             ci_hi=(round(float(hi), 6) if cik else np.nan),
                             excludes_0=(bool(lo > 0 or hi < 0) if cik else False)))

OUT = pd.DataFrame(rows)[COLUMNS]
OUT.to_csv("outputs/audit/c13_effective_supply_decomp.csv", index=False,
           encoding="utf-8-sig")
pd.set_option("display.width", 200)
print()
print(f"{BASE}->{END}, 330 circles, both numerators on the same window and the same circles")
print(OUT.to_string(index=False))
print()
print("-> outputs/audit/c13_effective_supply_decomp.csv")
