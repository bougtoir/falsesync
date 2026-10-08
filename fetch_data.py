"""Re-acquire raw datasets listed in data/acquisition_ledger.csv.

Downloads each recorded file to its ledger local_path and verifies the
SHA-256 recorded at acquisition time. Skips rows marked '(not acquired)'.

Usage: python3 fetch_data.py
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / "data" / "acquisition_ledger.csv"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    led = pd.read_csv(LEDGER)
    failures = []
    for _, row in led.iterrows():
        lp = str(row.get("local_path", ""))
        url = str(row.get("source_url", ""))
        if lp in ("n/a", "nan") or not lp or str(row.get("size_bytes")) in ("n/a", "nan"):
            print(f"skip {row['dataset']} (not acquired)")
            continue
        dest = ROOT / lp
        if dest.exists() and sha256(dest) == str(row["sha256"]):
            print(f"ok    {lp} (already present, sha256 verified)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetch {lp} <- {url}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"FAIL  {lp}: {e}")
            failures.append(lp)
            continue
        got = sha256(dest)
        if got != str(row["sha256"]):
            print(f"FAIL  {lp}: sha256 mismatch {got} != {row['sha256']}")
            failures.append(lp)
        else:
            print(f"ok    {lp} (sha256 verified)")
    if failures:
        print("FAILED:", failures)
        sys.exit(1)
    print("all datasets present and verified")
    extract()


def extract():
    """Unpack the zips into the layouts the analysis scripts expect."""
    import zipfile

    kel = ROOT / "data/raw/kelmarsh"
    kdst = kel / "extracted_2016"
    if not any(kdst.glob("Turbine_Data_*.csv")):
        kdst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(kel / "Kelmarsh_SCADA_2016_3082.zip") as z:
            for m in z.infolist():
                mp = Path(m.filename)
                if mp.is_absolute() or ".." in mp.parts:
                    print(f"skip unsafe zip member {m.filename}")
                    continue
                z.extract(m, kdst)
        print(f"extracted Kelmarsh SCADA -> {kdst}")

    mdst = ROOT / "data/raw/nasa_battery/mat_08q4"
    want = {"B0005", "B0006", "B0007", "B0018"}
    if not all((mdst / f"{c}.mat").exists() for c in want):
        import io
        mdst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ROOT / "data/raw/nasa_battery.zip") as z:
            inner = z.read("5. Battery Data Set/1. BatteryAgingARC-FY08Q4.zip")
        with zipfile.ZipFile(io.BytesIO(inner)) as z2:
            for name in z2.namelist():
                base = Path(name).stem
                if base in want:
                    (mdst / f"{base}.mat").write_bytes(z2.read(name))
        print(f"extracted NASA FY08Q4 cells -> {mdst}")


if __name__ == "__main__":
    main()
