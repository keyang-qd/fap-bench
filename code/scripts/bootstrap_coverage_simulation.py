"""W7/Q5: actual coverage of the 95% moving-block bootstrap interval at n=60.

The objection is concrete: n/L is ~10 effective blocks at L=6, and block-bootstrap
coverage degrades when the block count is small. RQ3 compares interval WIDTHS
across two regimes, so systematic under-coverage would matter.

Simulated under the null (true mean zero) on AR(1) series of length 60, across a
range of persistence, since the real contrast series' autocorrelation is not
known exactly.
"""
import numpy as np
from fapbench.evaluation.statistics import moving_block_bootstrap_mean

rng = np.random.default_rng(0)
n, trials = 60, 3000
print(f'n={n}; effective blocks: L=3 -> {n/3:.0f}, L=6 -> {n/6:.0f}, L=12 -> {n/12:.0f}')
print()
print(f"{'rho':>6} {'L=3':>8} {'L=6':>8} {'L=12':>8}")
for rho in (0.0, 0.2, 0.4):
    row = []
    for L in (3, 6, 12):
        hits = 0
        for _ in range(trials):
            e = rng.normal(0, 1.0, n + 200)
            x = np.zeros(n + 200)
            for i in range(1, n + 200):
                x[i] = rho * x[i-1] + e[i]
            ci = moving_block_bootstrap_mean(x[200:], block_length=L, n_bootstrap=600)
            hits += int(ci['ci_lower'] <= 0.0 <= ci['ci_upper'])
        row.append(hits / trials)
    print(f"{rho:>6.1f} " + " ".join(f"{v:>8.3f}" for v in row))
