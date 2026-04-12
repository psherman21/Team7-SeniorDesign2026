"""
import_external_gestures.py
----------------------------
Scans an external dataset folder (subfolders 001–025) for single-letter
gesture CSVs, normalizes their columns to match the flex-glove-ml schema,
and writes one CSV per gesture letter to an output folder.

Output filename format: gesture_{LETTER}.csv  (e.g. gesture_A.csv)
If multiple subject folders contain the same letter, their rows are combined.

Configure the paths at the top of this file before running.
"""

import os
import re
import pandas as pd
from pathlib import Path

EXTERNAL_ROOT = "external_data"
OUTPUT_DIR = "data/external"

COLUMN_MAP = {
    "flex_1":  "flex_1",
    "flex_2":  "flex_2",
    "flex_3":  "flex_3",
    "flex_4":  "flex_4",
    "flex_5":  "flex_5",
    "ACCx":    "accel_x",
    "ACCy":    "accel_y",
    "ACCz":    "accel_z",
    "GYRx":    "gyro_x",
    "GYRy":    "gyro_y",
    "GYRz":    "gyro_z",
}

# Final column order written to output CSVs (matches data_logger.py header)
OUTPUT_COLUMNS = [
    "flex_1", "flex_2", "flex_3", "flex_4", "flex_5",
    "accel_x", "accel_y", "accel_z",
    "gyro_x",  "gyro_y",  "gyro_z",
    "label",
]

def is_single_letter_file(filename: str) -> bool:
    """Return True if the file is named exactly one letter, e.g. 'a.csv'."""
    stem = Path(filename).stem          # filename without extension
    ext  = Path(filename).suffix.lower()
    return ext == ".csv" and re.fullmatch(r"[a-zA-Z]", stem) is not None


def normalize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Keep only the columns we need, rename them to the project schema,
    add the label column, and drop any rows with NaN in key columns.
    """
    # Check which expected source columns are actually present
    missing = [col for col in COLUMN_MAP if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df = df[list(COLUMN_MAP.keys())].copy()
    df = df.rename(columns=COLUMN_MAP)
    df["label"] = label.upper()
    df = df[OUTPUT_COLUMNS]
    df = df.dropna()
    return df

def main():
    external_root = Path(EXTERNAL_ROOT)
    output_dir    = Path(OUTPUT_DIR)

    if not external_root.exists():
        print(f"ERROR: EXTERNAL_ROOT not found: {external_root.resolve()}")
        print("Edit the EXTERNAL_ROOT path at the top of this script and try again.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all single-letter CSVs, grouped by letter
    # letter -> list of DataFrames
    letter_frames: dict[str, list[pd.DataFrame]] = {}

    subject_folders = sorted(external_root.iterdir())
    subject_folders = [f for f in subject_folders if f.is_dir()]

    if not subject_folders:
        print(f"No subfolders found in {external_root}")
        return

    print(f"Found {len(subject_folders)} subject folder(s) in {external_root}\n")

    for subject_dir in subject_folders:
        csv_files = [f for f in subject_dir.iterdir() if is_single_letter_file(f.name)]
        if not csv_files:
            continue

        for csv_path in sorted(csv_files):
            letter = csv_path.stem.upper()
            try:
                df_raw = pd.read_csv(csv_path)
                df_norm = normalize(df_raw, letter)
                letter_frames.setdefault(letter, []).append(df_norm)
                print(f"  [{subject_dir.name}] {csv_path.name} -> {len(df_norm)} rows for '{letter}'")
            except Exception as e:
                print(f"  WARNING: Skipped {csv_path} — {e}")

    if not letter_frames:
        print("\nNo single-letter CSV files found. Check your EXTERNAL_ROOT path.")
        return

    # Write one output file per letter
    print(f"\nWriting output to: {output_dir.resolve()}\n")
    total_rows = 0
    for letter in sorted(letter_frames.keys()):
        combined = pd.concat(letter_frames[letter], ignore_index=True)
        out_path  = output_dir / f"gesture_{letter}.csv"
        combined.to_csv(out_path, index=False)
        total_rows += len(combined)
        print(f"  gesture_{letter}.csv  —  {len(combined)} rows  ({len(letter_frames[letter])} subject(s))")

    print(f"\nDone. {len(letter_frames)} gesture file(s), {total_rows} total rows.")
    print(f"Output folder: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
