import argparse
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
OUTPUT = DATA_DIR / "agg_data.csv"


def read_monthly_file(path):
    df = pd.read_csv(path)
    if "ObsTime" not in df.columns:
        df = pd.read_csv(path, skiprows=1)
    year_month = path.stem.replace("467050-", "")
    df["Date"] = pd.to_datetime(year_month + "-" + df["ObsTime"].astype(str).str.zfill(2), errors="coerce")
    return df


def aggregate(data_dir=DATA_DIR, output=OUTPUT):
    files = sorted(
        p for p in Path(data_dir).glob("467050-*.csv") if p.name != output.name
    )
    if not files:
        raise FileNotFoundError(f"No monthly CSV files found in {data_dir}")

    frames = [read_monthly_file(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("Date").reset_index(drop=True)

    columns = ["Date"] + [c for c in df.columns if c != "Date"]
    df = df[columns]
    df.to_csv(output, index=False)
    print(f"Wrote {len(df)} rows and {len(df.columns)} columns to {output}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate English CODIS monthly station CSV files.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    aggregate(args.data_dir, args.output)


if __name__ == "__main__":
    main()
