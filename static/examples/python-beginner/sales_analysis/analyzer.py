"""Read a sales CSV and save CSV, JSON and Excel reports."""

import argparse
from pathlib import Path

import pandas as pd
from helpers import calculate_total, format_currency


def analyse(input_file, output_dir):
    table = pd.read_csv(input_file)
    required = {"product", "quantity", "unit_price"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if table.empty:
        raise ValueError("Sales file has no rows")
    if table[list(required)].isna().any().any():
        raise ValueError("Sales rows contain missing values")
    for column in ("quantity", "unit_price"):
        table[column] = pd.to_numeric(table[column], errors="raise")
        if not table[column].between(0, float("inf"), inclusive="left").all():
            raise ValueError(f"{column} must contain finite non-negative numbers")
    if not (table["quantity"] % 1 == 0).all():
        raise ValueError("Quantity must contain whole numbers")
    table["total"] = calculate_total(table["quantity"], table["unit_price"])
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "sales_totals.csv", index=False)
    table.to_json(output_dir / "sales_totals.json", orient="records", indent=2)
    table.to_excel(output_dir / "sales_totals.xlsx", index=False, engine="openpyxl")
    for row in table.itertuples(index=False):
        print(f"{row.product}: {format_currency(row.total)}")
    print(f"Grand total: {format_currency(table['total'].sum())}")
    return table


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data" / "sales.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "output")
    args = parser.parse_args()
    try:
        analyse(args.input, args.output_dir)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"Unable to create sales report: {exc}\n")


if __name__ == "__main__":
    main()
