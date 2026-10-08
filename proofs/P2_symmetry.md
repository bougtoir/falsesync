# P2 — Symmetry result (PROVED)

**Assumptions.** Model S2 with homogeneous \(g\) satisfying the *cdf-symmetry*
property \(g(u)+g(-u)=1\) for all \(u\) (holds for step — in the \(g(0)=1/2\)
convention —, logistic, probit, symmetric ramp). \(F_\tau\) symmetric about \(c\):
\(f_\tau(c+s)=f_\tau(c-s)\) wherever the density exists (or
\(F_\tau(c-s)=1-F_\tau((c+s)^-)\) in general).

**Claim.** The aggregate mean curve is point-symmetric about \((c,\ a+A/2)\):

\[
m(c+s)+m(c-s)=2a+A \qquad\text{for all } s.
\]

Consequently every breakpoint functional that is odd about the center on a
window symmetric around \(c\) returns \(c\): in particular the LS single-break
location \(T_{\mathrm{LS}}=c\) and \(T_{\mathrm{ms}}=c\) when \(f_\tau\) is
unimodal.

**Proof.** With \(m(t)=a+A\int g(t-\tau)dF_\tau(\tau)\),

\[
m(c+s)+m(c-s)-2a
 = A\int\bigl[g(c+s-\tau)+g(c-s-\tau)\bigr]dF_\tau(\tau).
\]

Writing \(\tau=c+v\), the first integral is
\(A\int g(s-v)f_\tau(c+v)\,dv\); the second is
\(A\int g(-s-v)f_\tau(c+v)\,dv\). In the second, substitute \(v\mapsto -v\) and
use \(f_\tau(c-v)=f_\tau(c+v)\): it becomes
\(A\int g(v-s)f_\tau(c+v)\,dv\). Summing,

\[
\frac{m(c+s)+m(c-s)-2a}{A}
=\int\bigl[g(s-v)+g(v-s)\bigr]f_\tau(c+v)\,dv
=\int 1\cdot f_\tau(c+v)\,dv=1,
\]

since \(g(u)+g(-u)=1\). ∎

(The discrete-atom version follows identically: atoms at \(c\pm v\) pair to
contribute \(A\).)

**Consequences.**
- Under joint symmetry the aggregate breakpoint **does** recover the center of
  \(F_\tau\) — so the failure \(T_M\ne\) moment of \(F_\tau\) is driven by
  asymmetry of \(F_\tau\) or \(g\), or by fitting-operator choice, not by
  aggregation per se.
- Symmetric-model sanity checks in the simulation suite verify
  \(T_{\mathrm{LS}}=T_{\mathrm{ms}}=c\) to tolerance.

**Numerical check:** cases C/D — symmetric normal \(F_\tau\) with step and
logistic \(g\): \(|T_{\mathrm{LS}}-\mu|\), \(|T_{\mathrm{ms}}-\mu|\), and
antisymmetry residual \(\max_t|m(c+t)+m(c-t)-(2a+A)|\).
