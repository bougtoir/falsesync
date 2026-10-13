# Data

No raw data are redistributed in this repository. `python fetch_data.py`
downloads each file from its original repository, verifies it against the
SHA-256 in `acquisition_ledger.csv`, and extracts the files used by the
analyses into `data/raw/` (git-ignored). Downloaded files remain under the
terms of their original providers.

## Kelmarsh wind farm SCADA data

- Source: C. Plumley and R. Takeuchi, "Kelmarsh wind farm data", Zenodo,
  record 16807551, https://doi.org/10.5281/zenodo.16807551
- License: Creative Commons Attribution 4.0 International (CC-BY-4.0).
- Files used: `Kelmarsh_SCADA_2016_3082.zip` (2016 turbine SCADA data),
  `Kelmarsh_WT_static.csv`, `Kelmarsh_WT_dataSignalMapping.csv`,
  `Kelmarsh_Grid_Meter_4458.zip`.
- Although CC-BY-4.0 permits redistribution with attribution, the files are
  not redistributed here because they are large and permanently available
  from the versioned Zenodo record.

## NASA PCoE Li-ion battery data

- Source: B. Saha and K. Goebel (2007), "Battery Data Set", NASA Prognostics
  Data Repository, NASA Ames Research Center, Moffett Field, CA.
  Download: https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip
  (listed at https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
- Terms: the repository page requests that publications acknowledge the
  repository and the data donors; it does not state an explicit
  redistribution license. The data are therefore not redistributed here.
- Files used: `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat` from the
  FY08Q4 archive inside the download.

## Checksums

`acquisition_ledger.csv` records, for every downloaded file, the source URL,
identifier, access date, license, size in bytes, and SHA-256. `fetch_data.py`
refuses to continue if a checksum does not match.
