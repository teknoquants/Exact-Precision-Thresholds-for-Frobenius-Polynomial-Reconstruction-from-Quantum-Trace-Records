# Frobenius threshold validation

Independent checks behind the white paper "Exact Precision Thresholds for Frobenius Polynomial
Reconstruction from Quantum Trace Records" (R. Cooperman, 9 October 2026).
Nothing here imports code from the commentator's package.

| File | Purpose | White paper |
|---|---|---|
| `frobenius_threshold_verification.py` | Checks 1–4 (below) | §3, §5, §9 |
| `operator_failure_witness_check.py` | Runs each failure witness through the operator at 1.001·δ* and 0.999·δ* | §5.1 |
| `check_results_1-3.json` | Output of checks 1–3 | §3, §9 |
| `check_results_4_true_thresholds.json` | Output of check 4 (Table 1) | §5.3 |

Checks in `frobenius_threshold_verification.py`:
1. Exact enumeration of W_g(Q) (trace polynomial, exact factorisation, real-root isolation) and brute-force δ_sep vs δ_cong.
2. Witness pairs of RT Lemma 7.2 in exact rational arithmetic, g = 1..12.
3. Point count of y² = x⁵ − x over F_{3^m}, m = 1..4, with an independently built field.
4. True worst-case thresholds δ* of A_FE and A_all: rigorous sufficiency (exact polynomial expansion) and constructive failure witnesses.

## Run
    pip install numpy scipy sympy
    python3 -I frobenius_threshold_verification.py 1 2 3   # ~40 s, writes frobenius_threshold_verification_123.json
    python3 -I frobenius_threshold_verification.py 4       # ~3 min, writes frobenius_threshold_verification_4.json
    python3 -I operator_failure_witness_check.py check_results_4_true_thresholds.json

Tested with Python 3.13, NumPy 2.5, SciPy 1.18, SymPy 1.14.

## Replay of the commentator's package
    sha256sum -c SHA256SUMS          # in the package root: all 38 files OK
    python -I code/run_all.py        # all result files and generated TeX byte-identical to shipped versions

## License
MIT License, Copyright (c) 2026 Roger Cooperman. See `LICENSE`.
