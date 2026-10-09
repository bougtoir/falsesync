# P5 — Truncation / missingness (PROVED structural identity; bias numeric)

**Assumptions.** Model S4: unit \(i\) is observed only on a window
\([L_i,U_i]\); \(w_i(t)=\mathbf 1\{L_i\le t\le U_i\}\). Transition times
\(\tau_i\) and windows may be dependent (e.g., units entering late have later
\(\tau_i\) on average — left truncation).

**Claim.** The observed aggregate is

\[
m_{\mathrm{obs}}(t)=
\frac{E\bigl[\,\mu(t;\tau_i)\,\mathbf 1\{L_i\le t\le U_i\}\,\bigr]}
     {P(L_i\le t\le U_i)},
\]

i.e., a convolution against the **time-varying observed timing measure**
\(F_\tau^{\mathrm{obs}}(\cdot;t)\), not a fixed \(F_\tau\). For step \(g\),

\[
m_{\mathrm{obs}}(t)=a+A\,P(\tau_i\le t\mid L_i\le t\le U_i),
\]

the timing CDF conditioned on being observed at \(t\) — in general
\(\ne F_\tau(t)\).

**Proof.** Direct conditioning under the LLN limit of numerator and
denominator; evaluate for step \(g\). ∎

**Consequences.**
- When the observation window correlates with \(\tau_i\) (left truncation —
  late-entering units transition later; right truncation — early-exiting
  units transition earlier), the aggregate curve is distorted *as a function
  of t*, and no fixed convolution explains it. The breakpoint fitted to
  \(m_{\mathrm{obs}}\) is biased toward the timing of the subpopulation that
  happens to be observed in the critical region.
- Identifiability requires modeling \(P(\text{observed at } t\mid \tau)\) —
  equivalently, the diagnostic needs observation-window metadata, not just the
  aggregate series.

**Numeric characterization (case G):** synthetic panel with left-truncated
entry at \(L_i\approx\tau_i\) (units arrive already transitioned); the LS
breakpoint on \(m_{\mathrm{obs}}\) is pulled early (4.795 vs true center 5.0)
and \(m_{\mathrm{obs}}\) deviates from \(F_\tau\) by up to 0.48 — bias
direction is set by which side of the transition the truncation hits.
Recorded in `phase2_theory_checks.csv` and `counterexamples.md`.
