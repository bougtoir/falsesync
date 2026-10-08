# Empirical interpretation (spec §16 language policy)

Permitted language only: diagnostic results, calibrated regime support,
warnings. No claim that a real event did or did not occur.

## Kelmarsh SCADA 2016 (6 turbines, daily Power / learned-PC ratio)

- Aggregate break candidate at day index 246 of the panel window
  (~2016-09-05); aggregate strength 1.24, BELOW the pre-specified
  strong-break criterion (>=3).
- Per-turbine LS breaks: days {244, 244, 252, 246, 278, 324} — spread
  sd ~29 days, deconvolution-corrected ~29 days. Single detected mode,
  but regime support concentrates on NO_TRANSITION (0.98) without
  observation windows and DIFFUSE (0.45) with them.
- Diagnostic verdict: weak-to-moderate evidence of a shared timing
  pattern in the performance-ratio signal, with insufficient aggregate
  break strength for a common-event read. The diagnostic correctly does
  NOT certify synchrony.
- Negative controls: +90-day time-shift moves the shifted unit's break
  to day 90 and mass to CLUSTERED (0.97) — the diagnostic responds to
  timing structure, not identity; bearing-temperature control yields
  NO_TRANSITION (1.0) with a trend-confounding flag; ID shuffle leaves
  the primary verdict unchanged.

## NASA PCoE batteries (4 cells, cycle space)

- Capacity-fade curves in cycle space: unit LS breaks at cycles
  {193, 110, 181, 155} — sd ~32 cycles.
- Aggregate break strength is weak (-1.67) and operator spread is 282
  cycles — no operator-independent aggregate breakpoint exists here.
- Diagnostic verdict: NO_TRANSITION (0.9999). Per the pre-specified plan
  this is a CONTRAST case: cycle indexing already aligns units, the
  degradation curves are smooth, and no false-synchrony claim is
  forced. Useful in the manuscript as the case where aggregation is
  less misleading.

## What these two demonstrations support

- The diagnostic never certified synchrony on real panels where break
  evidence is weak — the false-certification risk observed in
  simulation under informative missingness did not appear, but we
  cannot claim absence of that failure mode from two datasets.
- Kelmarsh is the headline demonstration only if the manuscript
  presents it as "diagnostic workflow on real engineering data", not as
  proof of a specific shared event.
