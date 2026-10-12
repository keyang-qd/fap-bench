# FAP-Bench — Supplement to the Paper

This file holds material that did not fit in the paper under its page limit, so
that a reader who wants the derivation, the audit figures, or the robustness
detail can find it. Section numbers refer to the published (camera-ready) paper.

Where the paper says "in the artifact" or "in the artifact supplement", the
material is here. Numerical outputs backing each section are under
`code/outputs/icaif_publication/`.

---

## S1. Budget matching — full derivation (paper §3.3)

Every arm sees the same amount of training, which is what makes a null credible:
if the federated arms were quietly given less, no comparison would mean anything.
We equalise *expected passes over the full panel*.

In one federated round a participating client runs `E` local epochs over its own
shard, so a round in which a fraction `ρ` of clients participate costs `ρE`
full-panel-equivalent passes. Over `R` rounds the federated budget is `R·E·ρ`,
and centralized and local-only training each receive

```
B = ceil(R · E · ρ)   epochs
```

where

```
ρ = R^-1 · Σ_e |C_e| / K
```

is the **realized** participation rate read off the schedule actually run, not the
requested one. A temporal treatment that drops clients therefore shrinks every
baseline's budget automatically, with no manual adjustment.

At the reference setting (`R=50`, `E=1`, `ρ=1`) every arm gets 50 passes.

This equalises optimisation work and **not** communication, which the paper
reports separately. Hyperparameters, checkpoints and calibrations are selected on
validation data only.

---

## S2. Detectable gap — the Sharpe-difference approximation (paper §3.4)

Equation (3) in the paper treats the reference arm's monthly return standard
deviation `s_ref` as the scale that puts a mean return difference on the
annualized Sharpe scale. This is an approximation: a difference of Sharpe ratios
also involves the comparison arm's variance and its covariance with the
reference.

We check that rather than assume it. Recomputing both 408-month cells under
Jobson–Korkie with Memmel's correction, in the serial-correlation-robust form and
at the same bandwidth:

| Cell | Effect on the reported gap |
|---|---|
| Widened (20 char., 408 mo.) | moves by at most **8%** |
| Reference (7 char., 408 mo.) | moves by at most **8%** for three of four arms |
| Reference — FedAdam | Eq. (3) is **conservative by a wide margin**: 0.51 against 0.35 |

**Every null in the paper survives the substitution.** The per-cell comparison is
in the artifact outputs.

Only the variance comes from the data, never the effect size, so this is a
*reporting* rule rather than post-hoc power: a cell publishes its MDE beside its
results, so a null is read against what it could have seen.

### The same gap for Rank IC (paper Table 2, "IC gap" column)

Rank IC is the rank information coefficient: the correlation, within a month,
between the ranking of stocks by predicted return and their ranking by realized
return. The paper's one resolved method separation is stated on Rank IC, so each
408-month cell also publishes the smallest Rank IC gap it can detect. The
calculation is the one above without the Sharpe scaling: the Newey–West standard
error of the seed-averaged monthly Rank IC difference against Centralized,
multiplied by `z_0.975 + z_0.80 ≈ 2.80`. Seeds are held fixed.

| Arm (against Centralized) | Reference cell: observed difference | Reference cell: detectable gap | Widened cell: observed difference | Widened cell: detectable gap |
|---|---|---|---|---|
| Local | −0.0036 | 0.0048 | −0.0063 | 0.0052 |
| FedAvg | −0.0027 | 0.0040 | −0.0067 | 0.0061 |
| SCAFFOLD | −0.0023 | 0.0038 | −0.0072 | 0.0061 |
| FedAdam | +0.0052 | 0.0044 | +0.0002 | 0.0057 |

The paper quotes the range across the four arms: 0.0038–0.0048 in the reference
cell and 0.0052–0.0061 in the widened cell. In the widened cell the FedAvg and
SCAFFOLD shortfalls exceed the whole range, narrowly. In the reference cell only
FedAdam's difference exceeds its gap, and the paper does not read that as a
separation because FedAdam's server optimizer was searched and Centralized's was
not. No seed-resampled version of this gap has been computed, unlike for net
Sharpe (S5). The inputs are
`code/outputs/icaif_publication/paper_tables/rolling_rank_ic_vs_centralized_investable.csv`
and `rolling_rank_ic_vs_centralized_tuned20.csv`.

---

## S3. Matched-control audit — realized figures (paper §3.2)

`≃` in Equation (1) means matched to within 2%. Every run audits its own control.

| Channel | Realized match |
|---|---|
| Size (rows retained) | worst violation anywhere across all scenario–seed cells: **1.46% row gap** against a 2% tolerance |
| Covariate shift | exact — one unique affine map per characteristic |
| Feature masking | exact — Jaccard dissimilarity **0.000** |
| Temporal | exact — identical row and column sums |

The tolerance binds only for size, which retains whole firms. The pipeline exits
non-zero on a breach, so a broken control fails the benchmark instead of being
recorded and passed over. The per-cell audit ships with the artifact.

---

## S4. Matched-control constructions, family by family (paper §3.2, Table 1)

The camera-ready paper carries a condensed version of this section in its
"Constructions" paragraph. The fuller statements follow.

**Covariate shift.** The control reuses the treatment's own realized draws. The
treatment gives each client its own location and scale; the control averages
those draws into one map applied everywhere. That leaves the shift the federation
sees as a whole unchanged while removing all the disagreement between clients.

*What this family does and does not test:* since inputs are rank-then-z-scored
and every scale is positive, an affine map cannot reorder a client's own stocks.
It stresses whether clients agree on a scale, not whether they face different
economics.

**Feature availability.** The control gives every client the same private subset,
so each still sees the same *number* of characteristics and only the variation in
*which* ones disappears. It also drops the step that guarantees every
characteristic reaches someone — that step exists for independent draws, and
applied to a single shared draw it would put the asymmetry back.

**Temporal.** The controls swap edges while preserving degree, so treatment and
control give identical per-round participation and per-client update counts and
differ only in when the absences fall.

**Size.** A size ladder keeping `q_k` of client `k`'s rows is matched by keeping
the pooled fraction `q̄` from every client.

---

## S5. The eight robustness checks (paper §4.2)

The paper names these eight and reports that all passed. Full statements follow.

| Check | Result |
|---|---|
| Placebo floor: each control against itself | \|Δ\| ≤ 0.0039; no system-level effect clears it |
| Known signal through the pipeline | untrained B/M and size reach net Sharpe 0.115/0.120 behind the screen (p > 0.46) |
| Cell provenance | reference cell unselected; widened cell fixed on validation Rank IC before any test-period comparison |
| Seed resampling | bar rises .34–.51 → .72–.77 (reference) and .31–.35 → .71–.88 (widened) |
| Window-level re-test | t falls a fifth to a third over 34 window means; no conclusion changes |
| Checkpoint truncation | Centralized is budget-truncated most (74%) yet wins |
| Control tightness | the tightest-matched treatment is the largest effect |
| Sharpe-difference variance | Jobson–Korkie moves gaps by ≤ 8%; every null survives |

### Seed averaging
Averaging ten seeds makes every arm look better than one run would, so we
calibrate the reference cell twice. Centralized's 0.146 in the reference-cell
table is the Sharpe of a monthly series averaged over ten seeds; the mean of the
ten per-seed Sharpes is 0.070. With seeds held fixed the cell detects
0.34–0.51; resampling seeds as well it needs 0.72–0.77, which is what a single
training run could detect. Every observed gap falls below both, the largest being
0.069.

### Power in the widened cell
Read as the widened-cell table reports them — seeds fixed, month blocks
resampled — the 0.512 and 0.540 shortfalls clear the 0.31–0.35 the cell detects,
at t = 4.9 and 4.8. Resampling seeds as well raises the bar to 0.71–0.88, above
the shortfalls, so a single training run would have had under 80% power to find
an effect this size. The interval still excludes zero under that stricter design,
so seed averaging did not create the shortfall — but an effect found under an
under-powered design is the kind whose magnitude gets overstated, which is why
0.51 is reported as a bound.

### Unit of inference
The 408 months come from 34 refits sharing almost all their training data. We
also test on the 34 per-window means: t falls by a fifth to a third and no
conclusion changes. FedAvg and SCAFFOLD stay below Centralized (t = −4.05 and
−3.86; p ≤ 1.1×10⁻⁴), FedAdam stays indistinguishable from it (p = 0.56), and
FedAdam stays above FedAvg (t = 2.87, p = 0.004, from 6×10⁻⁶).

### Convergence, not aggregation
An arm still improving when its budget ends selects its final checkpoint.
Centralized does so in 74% of the 340 window–seed runs, against 48%, 44% and 36%
for FedAvg, SCAFFOLD and FedAdam. The arm most often cut short by the budget is
the one that wins, which is the opposite of what under-trained federation would
look like. We have not swept rounds, so this remains a cost at this budget and
client rate.

### Selection
Neither the cell nor the winning arm was chosen for its outcome. Width and rate
came from eight candidates scored on *validation* Rank IC alone, and the one we
use ranks third from the bottom of those eight on the gap we report. FedAdam's
server rate was not re-selected on the widened panel either, and re-selecting
would widen that gap rather than close it.

### Ridge aggregates exactly
Ridge sees the data only through `Σ_k X_kᵀX_k` and `Σ_k X_kᵀy_k`, so 44 numbers
per client (176 bytes, against FedAvg's 8.744 MB) reproduce the pooled
coefficients to round-off. The coordination cost vanishing there is an arithmetic
identity, not evidence about model classes. The comparison that still carries
weight is gradient FedAvg on a linear model, which reaches that solution only by
converging.

---

## S6. Multiple-comparison families (paper §4.1)

We correct over two families and never pool them.

**Declared family — 74 matched-control contrasts.** `(5 × 5) + (3 × 4)`
contrast–method pairs on two statistics (standard deviation and worst-client).
Benjamini–Hochberg runs once over this family.

**Descriptive family — method comparisons made inside cells.** Benjamini–Hochberg
runs separately over this family. It holds 98 tests, of which 27 are significant
before correction and 21 remain so after it. The paper writes `q` for a value
that carries its family's correction and `p` for one that does not. The full
list, with both values per test, is
`code/outputs/icaif_publication/paper_tables/descriptive_test_census.csv`.

---

## S7. Local work: fifty rounds of one epoch against ten rounds of five (paper §4.2)

The paper's default spends its training budget as 50 rounds of one local epoch
(a local epoch is one pass by a client over its own data between two rounds of
communication). Short local work keeps clients from drifting apart, which is the
problem the aggregation rules exist to correct, so it could hide differences
among them. Spending the same budget as 10 rounds of five epochs tests that. On
the 60-month cell with the investable universe:

| Rule | Rank IC, 50 × 1 | Rank IC, 10 × 5 | Communication, 50 × 1 (MB) | Communication, 10 × 5 (MB) |
|---|---|---|---|---|
| FedAvg | 0.0058 | 0.0045 | 8.74 | 1.81 |
| FedProx | 0.0058 | 0.0045 | 8.74 | 1.81 |
| SCAFFOLD | 0.0063 | 0.0058 | 16.73 | 3.38 |
| FedAdam | 0.0180 | 0.0099 | 4.44 | 1.44 |
| Spread across the four rules | 0.0122 | 0.0055 | | |

More local work narrows the spread among the rules instead of widening it. This
is descriptive only, because the 60-month cell cannot rank arms (paper Table 2).
Communication is the volume exchanged up to the selected checkpoint. The inputs
are `main_results.csv` and `main_results_10x5.csv` under
`code/outputs/icaif_publication/paper_tables/`.
