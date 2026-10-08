# P7 — Least-squares breakpoint functional (PROVED FOC + specializations)

**Setup.** Population least-squares single-break piecewise-constant fit on
\(m(t)\), observation window \([0,T]\), uniform time weight:

\[
\hat c = T_{\mathrm{LS}}(m)=\arg\min_{c\in(0,T)}
\Bigl[\int_0^c\bigl(m(t)-\bar m_L\bigr)^2dt
     +\int_c^T\bigl(m(t)-\bar m_R\bigr)^2dt\Bigr],
\]
\(\bar m_L=\frac1c\int_0^c m\,dt,\ \bar m_R=\frac1{T-c}\int_c^T m\,dt.\)

**Claim (FOC).** For continuous \(m\), every interior local optimum satisfies

\[
m(\hat c)=\frac{\bar m_L(\hat c)+\bar m_R(\hat c)}{2},
\]

i.e., the breakpoint sits where the curve equals the **midpoint of the two
segment means** — a global functional of \(m\), not a local property like a
mode or inflection.

**Proof.** Differentiate the objective in \(c\). Boundary terms from the two
integrals give \((m(c)-\bar m_L)^2-(m(c)-\bar m_R)^2\); the terms from the
moving means vanish since \(\int_{\text{seg}}(m-\bar m)=0\) on each segment.
Setting the derivative to zero yields
\((m(c)-\bar m_L)^2=(m(c)-\bar m_R)^2\), and the interior (monotone) branch is
\(m(c)=\tfrac12(\bar m_L+\bar m_R)\). ∎

**Corollary — \(\hat c\) is generally none of mean/median/mode.**
For step \(g\), \(m=a+AF_\tau\), so \(\hat c\) solves
\(F_\tau(\hat c)=\tfrac12(\bar F_L+\bar F_R)\) where the bars are *time-averaged
CDF levels*, functionals depending on the whole window — including the window
endpoints themselves. Same \(F_\tau\), different window ⇒ different \(\hat c\):
the aggregate breakpoint is **window-dependent**, unlike any moment of
\(F_\tau\). Numeric demonstration over skewed and mixture \(F_\tau\) in
`phase2_theory_checks.csv`.

**Max-slope operator.** \(T_{\mathrm{ms}}=\arg\max m'\). For step \(g\),
\(T_{\mathrm{ms}}=\arg\max f_\tau=\mathrm{mode}(F_\tau)\) (P1); for smooth \(g\),
\(T_{\mathrm{ms}}\) is the mode of \(f_\tau * g'\)-type smoothed density —
a *different* functional than \(T_{\mathrm{LS}}\) in general. This already
proves no method-independent 'the aggregate breakpoint' exists: the estimand
is operator-dependent (\(T_M\) notation is load-bearing, not pedantry).

**Numeric check:** for every \((g,F)\) grid case, residual
\(|m(\hat c)-\tfrac12(\bar m_L+\bar m_R)|\) at the LS optimum < 1e-8, and
\(T_{\mathrm{ms}}\) vs \(T_{\mathrm{LS}}\) diverge for skewed/mixture \(F_\tau\)
(table rows `P7_*` in `phase2_theory_checks.csv`).
