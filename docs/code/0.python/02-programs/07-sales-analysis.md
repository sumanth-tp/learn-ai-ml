---
id: py-sales-analysis
title: "Practical Python: A Modular Sales Analysis Project"
sidebar_label: "Lab · Modular sales analysis"
sidebar_position: 7
slug: /code/python/sales-analysis
description: "Build a small project with supplied CSV data, reusable helpers, reliable paths and CSV, JSON and Excel exports."
tags: [python, beginner, project, pandas, files, modules]
---

> **Video:** the practical project and helper module sections of [Python for AI - Full Beginner Course](https://www.youtube.com/watch?v=ygXn5nV5qFc).

Keep input data, reusable calculations and generated reports in predictable places.

The video builds a sales analysis script, exports several file types, and extracts helper functions. This lab supplies a small original dataset and a complete implementation. **Added practice** includes input validation, known totals and command-line paths.

## Project files

Download the files below into this structure, or use the matching folder under `static/examples/python-beginner`:

```text
python-beginner/
├── requirements.txt
└── sales_analysis/
    ├── analyzer.py
    ├── helpers.py
    ├── data/
    │   └── sales.csv
    └── output/             # created by the program
```

| File | Purpose |
| --- | --- |
| [requirements.txt](/examples/python-beginner/requirements.txt) | Libraries for both labs |
| [analyzer.py](/examples/python-beginner/sales_analysis/analyzer.py) | Read, check, calculate and export |
| [helpers.py](/examples/python-beginner/sales_analysis/helpers.py) | Reusable calculation and display functions |
| [sales.csv](/examples/python-beginner/sales_analysis/data/sales.csv) | Complete practice input |

Install dependencies in your activated environment:

```bash
python -m pip install -r requirements.txt
```

Pandas uses the additional `openpyxl` package for this lab's Excel export. A pandas installation alone does not necessarily include Excel engines.

## Inspect the input before calculating

The CSV contains a header followed by three rows:

```csv
product,quantity,unit_price
Notebook,3,40
Pen,5,10
Folder,2,25
```

Create `analyzer.py` alongside the `data` folder and start with:

```python
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
table = pd.read_csv(root / "data" / "sales.csv")
print(table.head())
print(table.shape)   # (3, 3)
print(table.dtypes)
```

`shape` gives rows and columns. `head()` previews the data. `dtypes` helps confirm that quantity and price are numeric. In a notebook, choose an explicit root path instead of using `__file__`.

CSV stores delimited text and carries little type information. A column that looks numeric can still contain a typo or missing value. The complete implementation checks required columns, missing values, numeric conversion, non-negative prices and whole-number quantities before writing reports.

## Calculate a column

After loading the table:

```python
table["total"] = table["quantity"] * table["unit_price"]
print(table["total"].tolist())  # [120, 50, 50]
print(table["total"].sum())     # 220
```

Pandas multiplies the two columns row by row. This avoids a Python loop for this calculation. The video's row loop is useful for seeing each function call; column operations are a clearer next step when the operation already supports entire columns.

For real monetary calculations, define currency and rounding rules. This deliberately simple dataset uses exact whole-number amounts. Arbitrary decimal prices may need integer minor units or `decimal.Decimal` rather than binary floats.

## Extract reusable functions

Put the following in `helpers.py`:

```python
def calculate_total(quantity, unit_price):
    """Multiply scalars or aligned pandas columns."""
    return quantity * unit_price

def format_currency(amount):
    """Format the lab's USD amounts for display."""
    return f"${amount:,.2f}"
```

In `analyzer.py`, import the functions by name:

```python
from helpers import calculate_total, format_currency

print(calculate_total(3, 40))  # 120
print(format_currency(120))   # $120.00
```

The same `calculate_total` function also accepts the two aligned pandas columns in this lab. Keep numeric totals as numbers for later calculations; use `format_currency` only when displaying them.

When launching `python sales_analysis/analyzer.py`, Python includes the script's directory in its import search, so it can find the adjacent `helpers.py`. This is a simple script layout. The [modules chapter](./01-modules-and-packages.md) explains how to grow into a proper package and run it with `python -m`.

## Export and read back

With the calculated `table` and `root` above:

```python
output = root / "output"
output.mkdir(parents=True, exist_ok=True)

table.to_csv(output / "sales_totals.csv", index=False)
table.to_json(output / "sales_totals.json", orient="records", indent=2)
table.to_excel(output / "sales_totals.xlsx", index=False, engine="openpyxl")

csv_copy = pd.read_csv(output / "sales_totals.csv")
json_copy = pd.read_json(output / "sales_totals.json")
excel_copy = pd.read_excel(output / "sales_totals.xlsx", engine="openpyxl")
for copy in (csv_copy, json_copy, excel_copy):
    print(copy["total"].sum())  # 220 for each format
```

| Format | Use | Detail to remember |
| --- | --- | --- |
| CSV | Simple tables exchanged between tools | Specify `index=False`; parsing may infer types |
| JSON records | A list of objects for application interchange | `orient="records"` gives one object per row |
| Excel | A workbook for spreadsheet users | Requires an engine; can carry sheets and formatting |
| Plain text | Notes, logs and simple exports | Open with an explicit encoding |
| XML | Structured documents with tags | Use a parser; it is not CSV with different punctuation |
| Parquet | Typed columnar storage for larger datasets | Requires a supported engine such as PyArrow |

The last three are extensions mentioned in the video's file-format discussion; they are not required to run this exercise. For API details, see the [pandas I/O guide](https://pandas.pydata.org/docs/user_guide/io.html).

## Run the finished script

From `python-beginner`:

```bash
python sales_analysis/analyzer.py
```

Expected terminal output:

```text
Notebook: $120.00
Pen: $50.00
Folder: $50.00
Grand total: $220.00
```

The program creates `sales_totals.csv`, `sales_totals.json` and `sales_totals.xlsx` under `sales_analysis/output`. Reruns replace those named outputs. The defaults are anchored to the script location, so you can also run the script by absolute path from a different directory.

To use another dataset or keep another report:

```bash
python sales_analysis/analyzer.py --input sales_analysis/data/sales.csv --output-dir output/sales-check
```

Command-line paths you supply are relative to your terminal's working directory. This is deliberately different from the script-anchored defaults.

## Added practice: prove it handles your data

1. Add `Marker,4,15`. Expected grand total: 280.
2. Rename `unit_price` to `price`. Expect a missing-column message; explain why guessing a column would hide an input contract change.
3. Replace one quantity with `many`, `-1` or `1.5`. Each should fail validation for a different reason.
4. Restore the input, run the script and read all three formats back. Check rows and totals, not just whether files exist.
5. Add a second report that imports `format_currency` without duplicating its implementation.

Move next to the [complete project workflow](../06-engineering/00-project-workflow.md), which adds version control and repeatable environment setup to a project like this.

- [ ] I can separate source code, input data and generated output.
- [ ] I can explain why the default input works from another working directory.
- [ ] I can import a helper without accidentally starting the report.
- [ ] I can check that three different exports preserve the same totals.
