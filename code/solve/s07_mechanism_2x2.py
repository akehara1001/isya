"""
VERIFICATION 2, continued -- isolate WHICH change moved destination density.

Between the published +0.267 and the PPML +0.084 two things changed at once: the sample
(positive flows -> all dyads) and the distance form (log(1+d) -> 13 bands). Holding the
estimator fixed at PPML, cross the sample against the distance form to see which one does
the work. Supplementary A5 quotes all four cells.

Deterministic. Writes outputs/solve/s07_mechanism_2x2.{csv,txt}

--- 2026-08-21, on the state this file was found in -------------------------------------
This script had never run. It was 18 lines: this docstring, six lines of imports, and a
single exec() that loaded a slice of s06 and nothing else. It had no body -- no fit, no
table, no write. Its mtime is one minute after s06's, so it was begun and abandoned in the
same sitting, and it entered git already in that state (d49ee82 first tracked code/, so
there is no commit that broke it and none that could show the intent).

The exec line carried two independent defects. It is quoted verbatim in
code/audit/fixtures/broken_parses.txt under [s07-exec-line], not here, because it contains
a triple quote and would end this docstring at that point -- the same defect one level up.
k01 loads that fixture and asserts the current extractors survive it.

  1. The second .split() cut the slice back to line 45 -- imports only. The first split
     alone yields 166 lines and carries D, ppml() and fit(), which is what this script
     needs; the second threw all three away.
  2. The .replace turned both ends of s06's 31-line module docstring into '#', which
     comments the opening line and leaves lines 2-30 as code. It raised SyntaxError on an
     apostrophe in "the model's relative prediction error". s06 has exactly one docstring,
     so the replacement count of 4 never matched anything either. This could not have
     worked on the s06 that existed when it was written.

So the four numbers in A5 were not produced here. They were produced somewhere that was
not saved. Reconstructing the cell is the only way to find out whether they hold, and the
all-dyads column is the control: s06 persists +0.084 and +0.254 for those two cells, and
this script asserts against them. If the machinery reproduces s06 exactly, the positive-
flow column it produces comes from the same machinery.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT)

# Everything s06 does before it starts reporting: the dyad frame D, the distance bands,
# ppml() and fit(). One split, at the marker the original docstring named. compile() first
# so that a future edit moving that marker inside a literal fails legibly instead of
# raising SyntaxError from inside exec().
_PRE = (open("code/solve/s06_distance_form_verification.py", encoding="utf-8")
        .read().split('say("=" * 100)')[0])
assert "def ppml" in _PRE and "D = pd.concat" in _PRE and "def fit" in _PRE, \
    "the s06 prefix no longer carries the data build or the estimator"
compile(_PRE, "s06_prefix", "exec")
# The prefix rewraps sys.stdout. That drops the last reference to the wrapper set above,
# and a TextIOWrapper closes its buffer when collected -- the new wrapper then writes to a
# closed stream. Hold both.
_KEEP = [sys.stdout]
exec(_PRE)
_KEEP.append(sys.stdout)

report = []


def say(s=""):
    print(s)
    report.append(s)


# ---------------------------------------------------------------- the 2x2
POS = (D["F"].to_numpy() > 0)
SAMPLES = [("all dyads", np.ones(len(D), bool)), ("positive flows", POS)]
FORMS = [("log(1+d)", True, False), ("13 bins", False, True)]


def fit_sub(mask, use_log, use_bins, with_adj=True):
    """s06's fit(), restricted to a row subset. Dummies are rebuilt inside the subsample so
    no column is all-zero; the estimator, the covariance and the covariate list are s06's."""
    Dm = D[mask]
    cols = (["log_dist"] if use_log else []) + OTHER + (["adj"] if with_adj else [])
    blocks = [sp.csr_matrix(np.ones((len(Dm), 1))),
              sp.csr_matrix(Dm[cols].to_numpy(float))]
    names = ["const"] + cols
    if use_bins:
        bc = pd.Categorical(Dm["bin"], categories=sorted(set(Dm["bin"]))).codes
        blocks.append(dumm(bc))
        names += [f"bin_{b}" for b in range(1, bc.max() + 1)]
    yc = pd.Categorical(Dm["year"], categories=sorted(set(Dm["year"]))).codes
    blocks.append(dumm(yc))
    names += [f"yr_{i}" for i in range(1, yc.max() + 1)]
    X = sp.hstack(blocks).tocsr()
    b, V, mu, dev = ppml(X, y[mask], o_idx[mask], d_idx[mask])
    i = names.index("log_density_j")
    se = float(np.sqrt(max(V[i, i], 0.0)))
    return float(b[i]), se, int(mask.sum()), X.shape[1]


say("=" * 100)
say("Destination-density coefficient, PPML throughout, adjacency included:")
say("sample (rows) x distance form")
say("=" * 100)
say("")
say(f"{'sample':>16} {'distance form':>14} {'rows':>10} {'k':>5} "
    f"{'beta log_density_j':>30}")

cells, recs = {}, []
for slab, mask in SAMPLES:
    for flab, use_log, use_bins in FORMS:
        bhat, se, n, k = fit_sub(mask, use_log, use_bins)
        cells[(slab, flab)] = bhat
        say(f"{slab:>16} {flab:>14} {n:>10,} {k:>5} "
            f"{f'{bhat:+.3f} [{bhat-1.96*se:+.3f},{bhat+1.96*se:+.3f}]':>30}")
        recs.append({"check": "2x2 dest density", "sample": slab, "dist_term": flab,
                     "rows": n, "k": k, "coef": round(bhat, 4), "se": round(se, 4),
                     "ci_lo": round(bhat - 1.96 * se, 4), "ci_hi": round(bhat + 1.96 * se, 4)})

say("")
say("Effect of including the zeros, holding the distance form fixed:")
for flab, _, _ in FORMS:
    mv = cells[("positive flows", flab)] - cells[("all dyads", flab)]
    say(f"    under {flab:>10}: {cells[('positive flows', flab)]:+.3f} on positive flows "
        f"vs {cells[('all dyads', flab)]:+.3f} on all dyads, a move of {abs(mv):.3f}")
    recs.append({"check": "2x2 zero-inclusion move", "dist_term": flab,
                 "coef": round(abs(mv), 4)})

# ---------------------------------------------------------------- control against s06
say("")
say("Control: the all-dyads column is already persisted by s06. Both cells must match.")
S6 = pd.read_csv("outputs/solve/s06_distance_form_verification.csv", encoding="utf-8-sig")
S6 = S6[S6["check"] == "V2b dest density"].set_index("dist_term")["with_adj"]
ok = True
for flab, _, _ in FORMS:
    here, there = cells[("all dyads", flab)], float(S6[flab])
    good = abs(here - there) <= 5e-4
    ok &= good
    say(f"    {flab:>10}: here {here:+.4f}   s06 {there:+.4f}   "
        f"{'ok' if good else 'MISMATCH'}")
    recs.append({"check": "2x2 control vs s06", "dist_term": flab, "coef": round(here, 4),
                 "s06": there, "status": "ok" if good else "MISMATCH"})

Path("outputs/solve/s07_mechanism_2x2.txt").write_text("\n".join(report) + "\n",
                                                       encoding="utf-8")
pd.DataFrame(recs).to_csv("outputs/solve/s07_mechanism_2x2.csv", index=False,
                          encoding="utf-8-sig")
print("\nwrote outputs/solve/s07_mechanism_2x2.{csv,txt}")
assert ok, "the reconstructed 2x2 does not reproduce s06's all-dyads column"
