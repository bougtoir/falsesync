# P6 — Clustered / mixture timing (PROVED for equal-variance two-normal; numeric otherwise)

**Assumptions.** Model S6: \(F_\tau=\sum_{k=1}^{K}\pi_k F_k\), homogeneous \(g\).

## P6a — Step \(g\): resolvability criterion

For step \(g\), \(m'(t)=A f_\tau(t)\) (P1), so the question 'does the aggregate
show one or two transition waves' is exactly the question of unimodality of the
mixture density.

**Claim.** For \(F_\tau=\pi N(\mu_1,\sigma^2)+(1-\pi)N(\mu_2,\sigma^2)\) with
\(\pi\in[\pi_{\min},1-\pi_{\min}]\) bounded away from 0 and 1, the density is
bimodal — hence the aggregate step-transition curve exhibits two resolvable
waves — iff

\[
|\mu_1-\mu_2|>2\sigma,
\]

(exact for \(\pi=\tfrac12\); for \(\pi\neq\tfrac12\) the critical separation is
\(>2\sigma\) and grows slowly as \(\pi\to 0\) or 1 — see numeric table in
`phase2_theory_checks.csv`).

**Proof.** A mixture of two equal-variance normals is bimodal iff the
separation exceeds \(2\sigma\) for equal weights; for unequal weights the
standard argument via \(f''\) at the midpoint combined with solving
\(f'(t)=0\) on each side gives the threshold condition (the equal-variance
two-Gaussian bimodality criterion, classical; e.g., direct computation of the
number of real roots of \(f'(t)=0\)). Equality at \(2\sigma\) is a degenerate
boundary (quartic contact). ∎ (Classical result; verified algebraically and
numerically here.)

## P6b — LS single-break location under a merged mixture

**Claim (numeric + structural).** When the clusters merge (separation below
the bimodality threshold) but \(F_\tau\) is not degenerate, a single-break LS
fit returns a location between the cluster centers that depends on \(\pi\) and
the observation window — it is *not* the mean of \(F_\tau\) except in
symmetric cases (P2). Verified numerically over a grid of \((\pi,\Delta)\):
\(T_{\mathrm{LS}}\) interpolates \(\mu_1\leftrightarrow\mu_2\) with \(\pi\),
with residual bias vs \(E[\tau]\) up to O(\(\sigma\)) in asymmetric cases.

## P6c — Consequence for the diagnostic

A *single* sharp-looking aggregate breakpoint is compatible with both regimes
'common \(\tau\)' and 'two clusters too close to resolve'. The regime
classifier must therefore distinguish CLUSTERED vs DIFFUSE vs SYNCHRONOUS from
**unit-level** \(\hat\tau_i\), not from the aggregate curve: the aggregate
cannot certify its own synchrony. This is the formal statement underlying the
whole diagnostic design.

**Numeric check:** case E — logistic \(g\) + two-normal mixture, grid over
\(\Delta=\mu_2-\mu_1\): recorded \(T_{\mathrm{LS}}\), \(T_{\mathrm{ms}}\)
(first mode), bimodality of \(f_\tau\), and whether the aggregate curve is
visually single-break; comparison table in `phase2_theory_checks.csv`,
figure `outputs/phase2/mixture_bimodality.png`.
