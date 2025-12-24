#!/usr/bin/env python3
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Count number of samples (rows) in a CSV dataset."
    )
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to the CSV file.")
    args = parser.parse_args()

    print(f"Loading: {args.csv}")
    df = pd.read_csv(args.csv)

    num_rows = len(df)
    num_cols = len(df.columns)

    print(f"\nNumber of samples (rows): {num_rows}")
    print(f"Number of columns: {num_cols}")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    main()
