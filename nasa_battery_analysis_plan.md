# NASA PCoE battery analysis plan (spec §14)

## Data

- `data/raw/nasa_battery.zip` → nested `1. BatteryAgingARC-FY08Q4.zip` →
  `mat_08q4/B0005.mat, B0006.mat, B0007.mat, B0018.mat` (24 °C ambient,
  2 A discharge, shared protocol — a homogeneous cell group).
- Verified SHA-256 per `data/acquisition_ledger.csv`; parsed with
  `scipy.io.loadmat(simplify_cells=True)`.

## Pre-specified signal

- Unit series = discharge **capacity (Ah)** per discharge cycle for each cell.
- `t` = discharge-cycle index (1..N), NOT calendar time — cells were cycled
  on the same protocol, so cycle index is the natural common index.
- Spec note: this design already aligns units naturally (shared protocol in
  cycle space). We do NOT force a false-synchrony claim; the dataset serves
  as a **contrast** — where aggregation is less misleading because the unit
  index is physically meaningful across units.

## Workflow

1. Extract discharge cycles per cell: `(cycle_number, Capacity)`.
2. Trim to common truncation (drop trailing cycles where any cell < 80% SoH
   jitter dominates? NO — keep full per-cell range; windows handle it).
3. Unit-level LS break on capacity vs cycle index (normalized signal =
   capacity / initial capacity).
4. Equal-weight aggregate across 4 cells; `FalseSynchronyModel.fit` with
   the calibrated classifier.
5. Also record calendar-time aggregation variant (time-of-cycle start) as
   a side-check.

## Interpretation policy

Diagnostic-only language (§16). No claim that a battery "failed" — capacity
fade transition is a degradation-shape feature.
