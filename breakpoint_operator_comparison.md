# Breakpoint-operator comparison (Phase 2)

Operators compared on the aggregate mean curve \(m(t)\) (population) and on
noisy simulated aggregate series. Rows recorded in
`phase2_theory_checks.csv`; figure `outputs/phase2/operator_comparison.png`.

| Operator | Population functional \(T_M\) | Location vs \(F_\tau\) | Robustness notes |
|---|---|---|---|
| LS single break (piecewise const.) | midpoint of segment means (P7) | ≠ mean/median/mode; window-dependent | standard; the estimand our theory targets |
| Max slope \(T_{\mathrm{ms}}\) | argmax \(m'\) | mode of \(g'\!*\!f_\tau\); = mode(\(F_\tau\)) for step \(g\) | ignores window; diverges from LS for skewed \(F_\tau\) |
| CUSUM-type detector | change in cumulative mean | approximates LS functional; window-dependent | implementation via ruptures `Binseg`/CUSUM |
| Wild binary segmentation (WBS) | single-best CUSUM split | same LS-like estimand when 1 break selected | multi-break-capable; reduces to LS at k=1 |
| Penalized multi-break (PELT/BinSeg) | set of LS breaks | breaks inherit window dependence | used only to count/flag extra breaks |

## Findings

- \(T_{\mathrm{ms}}\) and \(T_{\mathrm{LS}}\) **are different functionals**:
  no method-independent 'aggregate breakpoint' exists — this must be stated
  in the paper's estimand section (P7).
- For symmetric unimodal settings all operators coincide at the center (P2);
  for skewed/mixture \(F_\tau\) they diverge by O(1) in window units —
  empirically shown in the numeric grid.
- Multi-break detectors split a smooth diffuse aggregate into a spurious
  staircase of small breaks when \(F_\tau\) is wide — recorded as an
  operator-level false-synchrony/false-multiplicity artifact to report.
- CUSUM/WBS on the aggregate cannot distinguish CLUSTERED-close from
  SYNCHRONOUS (P6c): classification must use unit-level information.

## Practical spec

Package wrapper `falsesync.changepoints.aggregate_breakpoint(series, t,
method=...)` implements `ls` (native LS), `maxslope` (derivative argmax),
`cusum`, `wbs`, `binseg` (ruptures-backed where available, graceful fallback
to LS where not). Every method returns `(location, strength, extras)` with a
`method` tag so downstream code can never confuse estimands.
