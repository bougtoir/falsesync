# P1 — Step-transition identity (PROVED)

**Assumptions (Model S2):** homogeneous step shape \(g(u)=\mathbf 1\{u\ge 0\}\),
amplitude \(A\), width \(h\) absorbed into units, equal weights \(w_i(t)=1\),
baseline \(a_i=a\), no trend, unit transition times \(\tau_i \stackrel{\mathrm{iid}}{\sim} F_\tau\),
large \(N\) (population limit).

**Claim.** The aggregate mean curve equals the transition-time CDF up to affine
scaling:

\[
m(t) = a + A\,F_\tau(t),
\]

and wherever \(F_\tau\) has density \(f_\tau\),

\[
m'(t) = A\,f_\tau(t).
\]

**Proof.** Each unit contributes \(\mu_i(t)=a+A\mathbf 1\{t\ge\tau_i\}\), so

\[
m(t)=E[\mu_i(t)]=a+A\,P(\tau_i\le t)=a+A\,F_\tau(t).
\]

Differentiating under the integral where \(f_\tau\) exists gives \(m'=A f_\tau\). ∎

**Consequences.**
- For step transitions the aggregate curve *is* the timing CDF (no smearing
  beyond \(F_\tau\) itself); the aggregate "sharp feature" is exactly the
  concentration of \(f_\tau\).
- A sharp-looking aggregate kink requires \(f_\tau\) peaked: an apparently
  sharp aggregate transition can occur only from concentrated (but possibly
  still non-degenerate) \(\tau_i\).
- \(T_{\mathrm{ms}}(F_\tau,\mathrm{step})=\mathrm{mode}(f_\tau)\) wherever the
  mode exists — the max-slope breakpoint recovers the **mode**, not the mean
  or median, of the timing distribution. For skewed \(F_\tau\) this already
  demonstrates \(T_M\ne\) mean/median/mode.

**Numerical check:** `simulations/run_phase2_checks.py` case A/B/C — population
curve vs \(F_\tau\), max-abs error < 1e-12; \(T_{\mathrm{ms}}\) vs mode for
normal and skewed \(F_\tau\).
