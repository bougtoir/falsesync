# P3 — Dispersion vs aggregate sharpness (PROVED_WITH_RESTRICTIONS + numeric)

**Setup.** Model S2, homogeneous \(g\), \(F_\tau\) indexed by a dispersion
parameter \(\sigma\).

## P3a — Exact result for probit \(g\) + Gaussian \(F_\tau\)

**Claim.** If \(g(u)=\Phi(u)\) (standard normal CDF, unit transition width) and
\(\tau\sim N(\mu,\sigma^2)\), then

\[
m(t)=a+A\,\Phi\!\left(\frac{t-\mu}{\sqrt{1+\sigma^2}}\right).
\]

Consequently:
(i) the smoothed aggregate is again a Gaussian CDF with variance \(1+\sigma^2\);
(ii) the maximal slope (a sharpness metric) is
\(\displaystyle \max_t m'(t)=\frac{A}{\sqrt{2\pi\,(1+\sigma^2)}}\),
strictly decreasing in \(\sigma\);
(iii) \(T_{\mathrm{ms}}=T_{\mathrm{LS}}=\mu\).

**Proof.** \(m(t)=a+A\,E_\tau[\Phi(t-\tau)]
=a+A\,P(\tau+Z\le t)\) with \(Z\sim N(0,1)\) independent. \(\tau+Z\sim
N(\mu,1+\sigma^2)\) gives the CDF. (ii) follows from the Gaussian density at
its center; (iii) from P2 symmetry. ∎

**Interpretation.** Even with infinite dispersion noise removed, widening
\(F_\tau\) strictly widens the apparent transition and lowers its peak slope:
\(\sigma\) dispersion produces an aggregate sigmoid of width
\(\sqrt{1+\sigma^2}\), which for large \(\sigma\) is a *diffuse* ramp — but for
moderate \(\sigma\) it is still a sharp-looking smooth transition that a
single-break fit will date at \(\mu\). Sharpness loss is quantified exactly.

## P3b — Step \(g\): sharpness = density concentration

For step \(g\), \(\max m' = A\max f_\tau\) (P1): aggregate sharpness equals the
peak of the timing density. For a mean-preserving spread of a unimodal density
that flattens the peak, sharpness decreases — but this is a statement about the
*density at its mode*, not variance: a point-mass + flat mixture
\(F=(1-\varepsilon)\delta_c+\varepsilon U\) has large variance and nearly
step-sharp \(m'\) at \(c\). **Restriction recorded:** 'dispersion reduces
sharpness' holds within regular unimodal scale families, not under arbitrary
spread — hence PROVED_WITH_RESTRICTIONS, and the mass-spike form is logged in
`counterexamples.md`.

## Numeric verification

- Grid over \(\sigma\in(0,5]\), probit/normal: \(\max m'\) matches
  \(A/\sqrt{2\pi(1+\sigma^2)}\) to < 1e-10.
- Logistic \(g\): \(\max m'\) decreases in \(\sigma\) over the grid (numeric
  observation, no closed form).
- Step \(g\): \(\max m' = A\max f_\tau\) verified exactly; spike-mixture
  counterexample reproduced.
