# Mathematical Specification — falsesync Phase 2

Project: Aggregation-Induced False Synchrony in Change-Point Analysis.
Status of claims: see `theory_claims_registry.csv`. Proofs live in `proofs/`.
Numeric verification: `simulations/run_phase2_checks.py` → `phase2_theory_checks.csv`,
figures in `outputs/phase2/`.

## 1. Data-generating model

Unit \(i = 1,\dots,N\), calendar time \(t\):

\[
Y_i(t) = \mu_i(t) + \varepsilon_i(t), \qquad
\mu_i(t) = a_i + b_i(t) + A_i\, g_i\!\left(\tfrac{t-\tau_i}{h_i}\right)
\]

- \(a_i\): baseline level; \(b_i(t)\): optional smooth trend
- \(A_i\): amplitude; \(\tau_i\): transition time; \(h_i\): width; \(g_i\): shape
- \(\varepsilon_i\): error process (iid in S1–S6; serially correlated in S7)

Aggregate:

\[
\bar Y_w(t) = \frac{\sum_i w_i(t)\,Y_i(t)}{\sum_i w_i(t)},
\qquad
m_w(t) = E[\bar Y_w(t)]
\]

### Model ladder (analyzed classes)

| Model | \(g\) | \(A,h\) | \(w_i(t)\) | \(\tau_i\) | noise/truncation |
|---|---|---|---|---|---|
| S1 | homogeneous | homog. | 1 | common \(\tau_0\) | iid |
| S2 | homogeneous | homog. | 1 | \(\tau_i \sim F_\tau\) | iid |
| S3 | homogeneous | homog. | \(w_i\) const. | \(F_\tau\), possibly correlated with \(w_i\) | iid |
| S4 | homogeneous | homog. | \(w_i(t)=\mathbf 1\{\text{observed}\}\) | \(F_\tau\) | iid + truncation |
| S5 | homogeneous | \(A_i,h_i\) heterog. | 1 | \(F_\tau\) | iid |
| S6 | homogeneous | homog. | 1 | mixture \(F_\tau=\sum_k \pi_k F_k\) | iid |
| S7 | homogeneous | homog. | 1 | \(F_\tau\) | AR(1) noise |

Under S2 with homogeneous \(g,h\) and \(a_i=a,\ b_i=0\):

\[
m(t) = a + A \int g\!\left(\tfrac{t-\tau}{h}\right) dF_\tau(\tau)
     = a + A\,(g_h * f_\tau)(t)\quad\text{(where density }f_\tau\text{ exists)}
\]

The convolution identity is **prior art** (ERP latency-smearing lineage;
Woody 1967; Nakamura et al. 1991) — used here as a tool, not claimed as novel.

## 2. Estimand framework

Four distinct objects are kept separate:

1. **Structural feature of \(m(t)\)** — the shape the aggregate mean curve has
   (e.g., a CDF, a smooth sigmoid, a bimodal derivative).
2. **Breakpoint functional \(T_M\)** — the population quantity returned by a
   breakpoint-fitting operator \(M\) applied to \(m\): e.g., least-squares
   single break \(T_{\mathrm{LS}} = \arg\min_c \mathrm{SSE}(c)\); max-slope
   \(T_{\mathrm{ms}} = \arg\max_t m'(t)\); inflection \(T_{\mathrm{infl}}\).
3. **Finite-sample estimator** \(\hat T_M\) from observed \(\bar Y_w(t)\).
4. **Unit timing distribution** \(F_\tau\) and its functionals
   (mean/median/mode/dispersion).

Key estimand statement (target of the theory):

\[
\text{aggregate breakpoint} = T_M(F_\tau, g, w, \dots)
\;\not\equiv\; \mathrm{mean/median/mode}(F_\tau)
\]

except in identified special cases (P2, step-transition specializations).

## 3. Regime definitions (operational)

Working definitions; all are statements about *inference-relevant* quantities.

- **SYNCHRONOUS**: \(F_\tau\) concentrated relative to the estimable resolution —
  operationalized via the posterior/likelihood comparison \(M_1\) (common
  \(\tau\)) vs \(M_2\) (distributed \(\tau_i\)), and unit-level \(\hat\tau_i\)
  dispersion small relative to \(\min(h, \text{series length})\).
- **CLUSTERED**: \(F_\tau\) best explained by \(K\ge 2\) components with
  non-negligible weights and separated supports (mixture fit beats single
  component by IC/LOO; per-cluster assignment confident).
- **DIFFUSE_ASYNCHRONOUS**: \(F_\tau\) non-degenerate, no resolvable clusters;
  timing dispersion exceeds what unit-level estimation noise alone produces.
- **NO_TRANSITION**: neither aggregate nor a non-trivial fraction of units
  shows a transition (proportion of units with \(\hat A_i\approx 0\) or
  undetected breaks above a calibrated threshold).

The classification rule must carry uncertainty: bootstrap resamples of units
propagate \(\hat\tau_i\) noise into the regime probabilities.

## 4. Proposition index

- P1 `proofs/P1_step_identity.md` — step \(g\): \(m(t)=a+A F_\tau(t)\); \(m'=f_\tau\).
- P2 `proofs/P2_symmetry.md` — odd-symmetric \(g\) + symmetric \(F\): \(m\)
  symmetric about center; centered functionals return the symmetry center.
- P3 `proofs/P3_dispersion_sharpness.md` — sharpness vs dispersion;
  exact for probit \(g\) + normal \(F\) (Gaussian widening);
  numerical elsewhere; includes a **restriction** (unimodality required).
- P4 `proofs/P4_weighting.md` — constant weights tilt the effective timing
  measure: \(m_w\) is the convolution against the \(w\)-tilted \(F_\tau\).
- P5 `proofs/P5_truncation.md` — time-varying observation weights break the
  single-convolution form; bias characterized for step \(g\).
- P6 `proofs/P6_mixture.md` — two-cluster \(F_\tau\): resolvable vs merged;
  for step \(g\), mixture of two equal-variance normals is bimodal iff
  \(|\mu_1-\mu_2|>2\sigma\); LS break under merge is a \(\pi\)-dependent compromise.
- P7 `proofs/P7_ls_functional.md` — for continuous \(m\), the LS single-break
  location satisfies \(m(\hat c)=(\bar m_L(\hat c)+\bar m_R(\hat c))/2\);
  for step \(g\), \(T_{\mathrm{ms}}=\mathrm{mode}(F_\tau)\).
- P8 `proofs/P8_identifiability.md` — \(F_\tau\) identifiable from \(m\) iff \(g\)
  known with \(\hat g(\omega)\neq 0\) a.e.; unknown \(g\) ⇒ unidentified
  modulo location/shape; unit-level data supplies identification.

## 5. Aggregate-breakpoint characterizations by \((g, F_\tau)\)

| \(g\) | \(F_\tau\) | \(m(t)\) | \(T_{\mathrm{ms}}\) | \(T_{\mathrm{LS}}\) |
|---|---|---|---|---|
| step | any | \(a+AF(t)\) | mode(\(f\)) | solves \(F(\hat c)=(\bar F_L+\bar F_R)/2\) |
| step | symmetric | CDF | center | center |
| ramp (width \(2r\)) | any | \(\frac{1}{2r}\int_{t-2r}^{t} F(s)\,ds + \frac12F(t-2r)\)-type form (P1 extension) | smoothed mode | numeric |
| probit \(\Phi(u)\) | \(N(\mu,\sigma^2)\) | \(a+A\Phi\!\left(\frac{t-\mu}{\sqrt{1+\sigma^2}}\right)\) | \(\mu\) | \(\mu\) |
| logistic | \(N(\mu,\sigma^2)\) | no closed form; ≈ probit form after rescale \(u\cdot\pi/\sqrt3\) | ≈\(\mu\) | ≈\(\mu\) |
| step | two-normal mix | \(\pi\Phi_1+(1-\pi)\Phi_2\) | bimodal iff \(\Delta>2\sigma\) | compromise between clusters |
| any | \(w\)-tilted \(F_w\) | convolution vs \(F_w\) | mode(\(f_w\)) (step) | shifted toward heavy-\(w\) stratum |

## 6. Diagnostic framework (first generation)

`FalseSynchronyResult` fields (implemented in `src/falsesync/diagnostics.py`):
`aggregate_breakpoint`, `aggregate_break_strength`, `unit_breakpoints`,
`unit_breakpoint_uncertainty`, `timing_distribution` (KDE/parametric fit on
\(\hat\tau_i\)), `timing_dispersion` (robust: IQR and NMAD), `cluster_structure`
(1..K mixture), `synchrony_score`, `regime_probabilities`,
`event_aligned_summary`, `interpretation_warning`.

Warning rule (calibrated in Phase 3):
`warn = (aggregate_break_strength >= s_hi) AND (P(SYNCHRONOUS) <= p_lo)`.

## 7. Uncertainty plan

Nonparametric bootstrap over units for \(\hat F_\tau\), dispersion, and regime
probabilities; parametric bootstrap under fitted transition model for unit
\(\hat\tau_i\) uncertainty; block bootstrap when S7 applies. See
`resampling_plan.md`.

## 8. Minimum theory gate (self-assessment)

GATE A (exact step proposition) — P1 PROVED.
GATE B (dispersion/sharpness) — P3 PROVED for probit–normal family
(\(\max m' = A/(\sqrt{2\pi(1+\sigma^2)})\)); numeric elsewhere.
GATE C (weighting/truncation/mixture) — P4 PROVED (tilting identity);
P6 PROVED for the equal-variance two-normal step case.
GATE D (diagnostic + uncertainty) — specified; implementation started.
GATE E (prior-art gate) — no direct competitor for the estimand problem.
