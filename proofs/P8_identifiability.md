# P8 — Identifiability of \(F_\tau\) (PROVED, two-sided)

**Setup.** Population convolution \(m(t)=a+A\,(g*f)(t)\), where \(f\) is the
timing density (extend by continuity to measures).

## P8a — Known \(g\): \(F_\tau\) identifiable (generically), ill-posed in practice

**Claim.** If \(g\) is known and its Fourier/Laplace transform
\(\hat g(\omega)\neq 0\) a.e., then \(F_\tau\) is identifiable from the
population curve \(m\): \(\hat f = \widehat{m-a}/(A\hat g)\) inverts the
convolution.

For step \(g\) the identifiability is **trivially exact**: \(F_\tau=(m-a)/A\)
(P1) — no inverse needed at all (the deconvolution is differentiation).

For smooth \(g\) the inverse is well-defined but **ill-posed**: the recovery
rate is governed by the decay of \(\hat g\) (logistic/probit kernels give
severely ill-posed deconvolution, matching the classical inverse-problem CP
literature — Neumann 1997 — rather than the easy step case).

**Proof.** Convolution theorem: \(\widehat{m-a}=A\,\hat g\,\hat f\); where
\(\hat g\) is a.e. nonzero, \(\hat f\) is a.e. determined, which determines \(f\)
(by Fourier inversion). Nonuniqueness can only arise on the zero set of
\(\hat g\); for our shapes \(\hat g\neq0\) a.e. ∎

## P8b — Unknown \(g\): unidentified

**Claim.** Without knowing \(g\), the pair \((g,F_\tau)\) is not identifiable
from \(m\) alone: for any location-shift \(s\) and any shape \(g_2\) with
\(g_2 * \delta_s = g * \delta_0\)-type equivalence, \(g*f = g_2 * f_2\) admits
continuum many solutions (translation: \(g(\cdot-s)*f(\cdot+s)=g*f\); more
generally, exchanging a blur kernel between \(g\) and \(f\)).

**Proof.** Take any strictly positive density kernel \(k\) with \(\int k=1\):
set \(g_2=g*k\) and \(f_2\) s.t. \(\hat f_2=\hat f/\hat k\) (exists whenever
\(\hat k\neq0\), e.g., Gaussian \(k\)). Then
\(g_2*f_2=(g*k)*f_2=g*f\) exactly. So the aggregate curve cannot separate
'transition sharpness' from 'timing dispersion' — this is the precise sense in
which unit-level information is *necessary*, not just helpful. ∎

## P8c — Sufficient identification route

**Claim.** \(F_\tau\) is identified without deconvolution once unit-level
transitions are even imperfectly observable: \(\hat\tau_i\) with iid noise
\(e_i\) yields \(F_{\hat\tau}=F_\tau * F_e\), and when the error law \(F_e\) is
known (or estimated via replicated/parametric unit fits), deconvolution of the
*break-time density* recovers \(F_\tau\) — which is precisely the SSRN (2024)
econometric result; our framework credits it and targets the estimand/diagnostic
layer it does not address.

**Status.** PROVED (a,b structural; c recorded as prior art to cite). Manuscript
implication: the paper's inferential novelty is the **aggregate-estimand and
diagnostic** story, not F_\tau recovery — see `deconvolution_assessment.md`.
