import argparse
from pathlib import Path

import pandas as pd


def split_xlsx(
    xlsx_path: Path,
    out_dir: Path,
    chunk_rows: int = 50000,
    to_parquet: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load first sheet (adjust as needed)
    df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")

    if to_parquet:
        parquet_path = out_dir / (xlsx_path.stem + ".parquet")
        df.to_parquet(parquet_path, index=False)
        print(f"Wrote: {parquet_path}")
        return

    # Split into CSV chunks for easier handling
    num_rows = len(df)
    part = 0
    for start in range(0, num_rows, chunk_rows):
        end = min(start + chunk_rows, num_rows)
        part += 1
        chunk = df.iloc[start:end]
        out_csv = out_dir / f"{xlsx_path.stem}_part{part:03d}.csv"
        chunk.to_csv(out_csv, index=False)
        print(f"Wrote: {out_csv} ({len(chunk)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split large Excel into CSV chunks or Parquet")
    parser.add_argument("xlsx", type=str, help="Path to dataset.xlsx (large Excel file)")
    parser.add_argument("--out", type=str, default="data/raw/dataset", help="Output directory")
    parser.add_argument("--rows", type=int, default=50000, help="Rows per CSV chunk")
    parser.add_argument("--parquet", action="store_true", help="Write a single Parquet file instead of CSV chunks")
    args = parser.parse_args()

    split_xlsx(Path(args.xlsx), Path(args.out), args.rows, args.parquet)


if __name__ == "__main__":
    main()


