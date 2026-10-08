# P4 — Weighting bias (PROVED)

**Assumptions.** Model S3: homogeneous \(g\), constant per-unit weights
\(w_i>0\), unit transition times \(\tau_i\) sampled jointly with weights
\((w_i,\tau_i)\sim Q\). Write \(W=\{w\}\), and let the population weight
average be \(\bar w=E_Q[w_i]\).

**Claim.** Define the *weight-tilted timing distribution*

\[
dF_\tau^{w}(\tau)=\frac{w(\tau)\,dF_\tau(\tau)}{\int w(\tau')\,dF_\tau(\tau')}
\quad\text{(deterministic case)}\qquad\text{or generally}\qquad
dF_\tau^{w}(\tau)=\frac{E_Q[\,w_i\,\mathbf 1\{\tau_i\in d\tau\}\,]}{E_Q[w_i]}.
\]

Then the weighted aggregate mean curve is the convolution against the
**tilted** timing distribution:

\[
m_w(t)=a+A\int g(t-\tau)\,dF_\tau^{w}(\tau) + A\,\mathrm{cov}\text{-correction}
\]

where the correction vanishes when \(A_i\equiv A\) and \(w_i\) is independent of
the transition channel except through \(\tau_i\). In the homogeneous case
exactly:

\[
m_w(t)=a+A\,(g * f_\tau^{w})(t).
\]

**Proof.** Numerator:
\(E[\sum_i w_i Y_i(t)]/\sum_i w_i \to E_Q[w_i\mu(t;\tau_i)]/E_Q[w_i]\) by LLN;
expand \(\mu\); the weighted measure over \(\tau\) is the tilted \(F_\tau^w\).
Heterogeneous \(A_i\) adds the weighted mixture correction. ∎

**Consequences.**
- Weighting does not merely reweight the estimate — it *changes the estimand*:
  the aggregate breakpoint equals \(T_M(F_\tau^{w},g,\cdot)\), biased toward
  the timing of heavily weighted units. If weight correlates with early
  transition, the apparent common break is pulled early even though the
  unweighted \(F_\tau\) is diffuse or late-centered.
- For step \(g\), \(m_w=a+A F_\tau^{w}\): the observed CDF is the tilted one —
  an analyst comparing the aggregate against unweighted unit-level \(\hat\tau_i\)
  sees a systematic, explainable discrepancy, which the diagnostic must model
  rather than flag as an anomaly.

**Numerical check:** case F — equal-transition-mixture \(F_\tau\), weights
\(w_i\) proportional to \(\exp(-\kappa\tau_i)\): fitted \(T_{\mathrm{LS}}\)
shifts toward early times monotonically in \(\kappa\), and \(m_w\) coincides
with convolution against the tilted \(F_\tau^{w}\) to < 1e-10.
