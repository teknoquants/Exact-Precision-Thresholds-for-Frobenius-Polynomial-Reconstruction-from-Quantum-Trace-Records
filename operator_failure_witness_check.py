"""Validate the failure witnesses of check 4 by direct execution of the operators.
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
For every (g, Q) and operator (A_FE, A_all), the witness record p(L) + e is fed to float Newton
reconstruction followed by rounding. It must FAIL at 1.001 x delta* and SUCCEED at 0.999 x delta*.

Usage:  python3 -I operator_failure_witness_check.py check_results_4_true_thresholds.json
(run from this folder; needs frobenius_threshold_verification.py next to it)
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frobenius_threshold_verification import power_sums, newton_c


def run(L, Q, ks, e_over, d, scale):
    kmax = max(ks); s = power_sums(L, kmax); dd = d * scale
    rec = [float(s[j]) / Q ** ((j + 1) / 2) + (e_over[j] * dd if j < len(e_over) else 0.0) for j in range(kmax)]
    c = newton_c(rec, kmax)
    b = [round(c[k] * Q ** (k / 2)) for k in range(kmax + 1)]
    return any(b[k] != L[k] for k in ks)          # True = wrong output


if __name__ == "__main__":
    rows = json.load(open(sys.argv[1]))["check4_true_thresholds"]
    allok = True
    for r in rows:
        g, Q = r["g"], r["Q"]
        for tag, ks in (("AFE", list(range(1, g + 1))), ("Aall", list(range(1, 2 * g + 1)))):
            w = r.get(tag + "_witness")
            if not w:
                continue
            lo, hi = r[tag + "_true_threshold_bracket"]
            fail_above = run(w["witness_L"], Q, ks, w["witness_e_over_delta"], hi, 1.001)
            ok_below = not run(w["witness_L"], Q, ks, w["witness_e_over_delta"], lo, 0.999)
            allok &= fail_above and ok_below
            print(f"g={g} Q={Q:3d} {tag:5s} delta*={hi:.6g}  fails at 1.001x: {fail_above}  correct at 0.999x: {ok_below}")
    print("ALL WITNESSES VALID:", allok)
