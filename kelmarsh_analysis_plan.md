# Kelmarsh SCADA analysis plan (pre-specified, §12)

## Data

- `data/raw/kelmarsh/extracted_2016/Turbine_Data_Kelmarsh_{1..6}_2016-01-03_-_2017-01-01_*.csv`
  (10-min SCADA, 366 days, 6 Senvion MM92 turbines, CC-BY-4.0).
- Verified against `data/acquisition_ledger.csv` SHA-256 before use.
- Raw files read-only; all derived products in `outputs/phase3/kelmarsh/`.

## Pre-specified signal (NOT chosen for attractiveness)

**Performance ratio** `r_i(t) = Power(kW) / Potential_power_learned_PC(kW)`
aggregated to **daily means**, restricted to 10-min records where
`Potential_power_learned_PC > 100 kW` (producing regime; avoids the
degenerate low-wind ratio region) and `Power ≥ 0` (excludes NaN and
negative housekeeping states).

Rationale: the learned power curve already adjusts for the wind-speed
distribution per turbine, so `r_i` isolates turbine-level performance
degradation/anomaly from weather-driven variation. Daily aggregation
suppresses diurnal/seasonal structure and gives n≈366 per unit.

Defensive alternatives recorded but not used unless the primary fails:
power normalized by `Potential power default PC`; ambient-temperature
residual. Signal is fixed before any breakpoint output is inspected.

## Workflow

1. Parse 6 turbine files; extract Date/WS/Power/PotPC_learned.
2. Build daily `r_i(t)`; record coverage per unit-day.
3. Unit-level LS break per turbine on `r_i` (365-day grid).
4. Equal-weight aggregate `ȳ(t)` (coverage-weighted; window = unit's
   observed day range).
5. `FalseSynchronyModel.fit` with the simulation-calibrated classifier:
   aggregate breakpoint/strength, per-turbine τ̂_i + uncertainty,
   regime probabilities, operator spread, trend check,
   false-common-event warning.
6. Report ONLY diagnostic language (§16) — no claims of actual
   turbine faults or shared events.

## Negative controls (§13)

- **time-shifted**: turbine 2's series shifted +90 days and wrapped — kills
  any real shared-calendar structure; if the diagnostic still reports
  SYNCHRONOUS, it is mechanically produced.
- **unrelated signal**: nacelle bearing temperature (deg C) daily mean,
  same pipeline — should NOT produce a strong synchronized-break pattern.
- **shuffled identity**: turbine labels permuted — sanity check that the
  classifier does not depend on unit identity.

## Missingness / artifacts

- Days with <30 valid 10-min records or potential<100kW coverage are NaN
  (treated as observation windows, not imputed).
- Curtailment/setpoint effects: `r_i` > 1.15 clipped as sensor/setpoint
  artifact (documented, not deleted silently).
