---
title: Pandas Master Cheatsheet
sidebar_position: 3
---

# Pandas Master Cheatsheet

Pandas is the universal data manipulation library for tabular data in Python.
Master `DataFrame` construction, indexing, joins, groupby, and time series —
these cover ~90% of real-world data work.

## DataFrame and Series creation

| Method                                    | Description                                                                                                                                     | Code example                                                                                               |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `pd.DataFrame()`                          | `pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)` — Build from a dict (columns) or list of dicts (rows).               | `import pandas as pd`<br/>`df = pd.DataFrame({"name": ["Ada", "Linus"], "age": [36, 29]})`<br/>`print(df)` |
| `pd.Series()`                             | `pd.Series(data=None, index=None, dtype=None, name=None)` — 1D labeled array. Common output of single-column operations.                        | `s = pd.Series([10, 20, 30], index=["a", "b", "c"], name="score")`                                         |
| `pd.read_csv()`                           | `pd.read_csv(filepath, sep=',', header='infer', usecols=None, dtype=None, parse_dates=False, chunksize=None, nrows=None)` — Read CSV.           | `df = pd.read_csv("data.csv", parse_dates=["timestamp"], dtype={"id": "int64"})`                           |
| `pd.read_parquet()`                       | `pd.read_parquet(path, engine='auto', columns=None, filters=None)` — Read Parquet — columnar, compressed, much faster than CSV.                 | `df = pd.read_parquet("data.parquet")`                                                                     |
| `pd.read_json()`                          | `pd.read_json(path_or_buf, orient=None, lines=False, dtype=None)` — Read JSON. Pass `lines=True` for JSON-lines format.                         | `df = pd.read_json("logs.jsonl", lines=True)`                                                              |
| `pd.read_sql()`                           | `pd.read_sql(sql, con, params=None, parse_dates=None, chunksize=None)` — Read from a database via SQLAlchemy or DB-API.                         | `df = pd.read_sql("SELECT * FROM users", engine)`                                                          |
| `df.to_csv()` / `df.to_parquet()`         | `df.to_csv(path, sep=',', index=True, columns=None, encoding='utf-8')` / `df.to_parquet(path, engine='auto', compression='snappy', index=None)` | `df.to_csv("out.csv", index=False)`<br/>`df.to_parquet("out.parquet")`                                     |
| `df.info()`                               | `df.info(verbose=None, memory_usage=None, show_counts=None)` — Schema, dtypes, non-null counts, memory usage.                                   | `df.info()`                                                                                                |
| `df.head()` / `df.tail()` / `df.sample()` | `df.head(n=5)` / `df.tail(n=5)` / `df.sample(n=None, frac=None, replace=False, random_state=None)` — Inspect first/last/random rows.            | `print(df.head(3))`<br/>`print(df.sample(5, random_state=0))`                                              |
| `df.describe()`                           | `df.describe(percentiles=None, include=None, exclude=None)` — Summary stats. Pass `include='all'` for object columns too.                       | `print(df.describe(include="all"))`                                                                        |
| `df.shape` / `df.columns` / `df.dtypes`   | Attributes (no call). Basic structural metadata.                                                                                                | `print(df.shape, df.columns.tolist(), df.dtypes)`                                                          |

## Selection and indexing

| Method                 | Description                                                                                                             | Code example                                                                |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `df[col]`              | Syntax: `df[column_name]` or `df[[col1, col2]]`. Select column(s) by label.                                             | `ages = df["age"]`<br/>`subset = df[["name", "age"]]`                       |
| `df.loc[]`             | `df.loc[row_labels, col_labels]` — Label-based indexing. Supports booleans, slices (inclusive), lists.                  | `df.loc[df["age"] > 30, ["name", "age"]]`                                   |
| `df.iloc[]`            | `df.iloc[row_positions, col_positions]` — Integer-position-based indexing. Like NumPy.                                  | `df.iloc[0:5, [0, 2]] # first 5 rows, cols 0 and 2`                         |
| Boolean masking        | Syntax: `df[mask]` where `mask` is a boolean Series. Combine with `&`, `\|`, `~` and parentheses.                       | `mask = (df["age"] > 25) & (df["country"] == "IN")`<br/>`adults = df[mask]` |
| `df.at[]` / `df.iat[]` | `df.at[row_label, col_label]` / `df.iat[row_pos, col_pos]` — Fast scalar access.                                        | `print(df.at[0, "name"])`<br/>`df.at[0, "age"] = 37`                        |
| `df.query()`           | `df.query(expr, inplace=False, **kwargs)` — SQL-like string filtering. Reference variables with `@`.                    | `df.query("age > 25 and country == 'IN'")`                                  |
| `df.isin()`            | `df.isin(values)` — Boolean mask of membership in a list, set, dict, or Series.                                         | `df[df["country"].isin(["IN", "US", "DE"])]`                                |
| `df.between()`         | `s.between(left, right, inclusive='both')` — Filter values within a range.                                              | `df[df["age"].between(20, 40)]`                                             |
| `df.filter()`          | `df.filter(items=None, like=None, regex=None, axis=None)` — Select columns by name pattern.                             | `df.filter(regex="^col_")`                                                  |
| Set a column as index  | `df.set_index(keys, drop=True, append=False, inplace=False)` / `df.reset_index(level=None, drop=False, inplace=False)`. | `df = df.set_index("user_id")`<br/>`df = df.reset_index() # undo`           |

## Cleaning and missing values

| Method                                     | Description                                                                                                                       | Code example                                                             |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `df.isna()` / `df.notna()`                 | `df.isna()` / `df.notna()` — Boolean mask of missing values (NaN, NaT, None).                                                     | `mask = df["email"].isna()`<br/>`print(df[mask].shape[0]) # count nulls` |
| `df.dropna()`                              | `df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)` — Drop rows/columns with nulls.                           | `df.dropna(subset=["email"]) # drop rows with null email`                |
| `df.fillna()`                              | `df.fillna(value=None, method=None, axis=None, limit=None)` — Fill with constant or method (`ffill`, `bfill`).                    | `df["age"] = df["age"].fillna(df["age"].median())`                       |
| `df.duplicated()` / `df.drop_duplicates()` | `df.duplicated(subset=None, keep='first')` / `df.drop_duplicates(subset=None, keep='first', inplace=False)`.                      | `df = df.drop_duplicates(subset=["email"], keep="first")`                |
| `df.replace()`                             | `df.replace(to_replace=None, value=None, regex=False, inplace=False)` — Replace specific values.                                  | `df["status"] = df["status"].replace({"N/A": None, "unknown": None})`    |
| `df.astype()`                              | `df.astype(dtype, copy=True, errors='raise')` — Cast columns. Use `errors='ignore'` to skip failures.                             | `df["age"] = df["age"].astype("Int64") # nullable int`                   |
| `pd.to_datetime()`                         | `pd.to_datetime(arg, errors='raise', format=None, unit=None, utc=False)` — Parse to datetime. `errors='coerce'` → NaT on failure. | `df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")`   |
| `pd.to_numeric()`                          | `pd.to_numeric(arg, errors='raise', downcast=None)` — Parse to numbers. Coerce failures to NaN.                                   | `df["price"] = pd.to_numeric(df["price"], errors="coerce")`              |
| `df.rename()`                              | `df.rename(mapper=None, index=None, columns=None, axis=None, inplace=False)` — Rename columns/index.                              | `df = df.rename(columns={"old_name": "new_name"})`                       |
| `df.clip()`                                | `df.clip(lower=None, upper=None, axis=None)` — Cap outliers to a range.                                                           | `df["amount"] = df["amount"].clip(lower=0, upper=10_000)`                |

## Transformations

| Method                   | Description                                                                                                                                                | Code example                                                                                                      |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `df.assign()`            | `df.assign(**kwargs)` — Add/overwrite columns in a chainable way. Values can be callables of `df`.                                                         | `df = df.assign(score_pct=lambda d: d["score"] / 100)`                                                            |
| `df.apply()`             | `df.apply(func, axis=0, raw=False, result_type=None)` — Apply func column-wise (axis=0) or row-wise (axis=1).                                              | `df["full_name"] = df.apply(lambda r: f"{r['first']} {r['last']}", axis=1)`                                       |
| `df.map()` (1.x+)        | `df.map(func, na_action=None)` — Apply element-wise to a DataFrame. Replaces deprecated `applymap`.                                                        | `df_numeric = df.select_dtypes("number").map(np.log1p)`                                                           |
| `s.map()`                | `s.map(arg, na_action=None)` — Map a Series using a dict, Series, or function.                                                                             | `df["country_name"] = df["country_code"].map({"IN": "India", "US": "USA"})`                                       |
| `s.str` accessor         | Accessor with methods like `.str.lower()`, `.str.strip()`, `.str.contains(pat, case=True, na=None)`, `.str.split(pat, expand=False)`, `.str.extract(pat)`. | `df["domain"] = df["email"].str.split("@").str[1]`<br/>`df[df["name"].str.contains("Ada", case=False, na=False)]` |
| `s.dt` accessor          | Accessor for datetime: `.dt.year`, `.dt.month`, `.dt.dayofweek`, `.dt.hour`, `.dt.date`, `.dt.day_name()`.                                                 | `df["year"] = df["created_at"].dt.year`<br/>`df["dow"] = df["created_at"].dt.day_name()`                          |
| `s.cat` accessor         | Categorical methods: `.cat.codes`, `.cat.categories`, `.cat.set_categories(new_cats, ordered=False)`.                                                      | `df["country"] = df["country"].astype("category")`<br/>`print(df["country"].cat.codes)`                           |
| `pd.cut()` / `pd.qcut()` | `pd.cut(x, bins, labels=None, include_lowest=False)` (equal-width) / `pd.qcut(x, q, labels=None, duplicates='raise')` (equal-frequency).                   | `df["age_band"] = pd.cut(df["age"], bins=[0, 18, 35, 60, 100], labels=["minor", "young", "adult", "senior"])`     |
| `df.pipe()`              | `df.pipe(func, *args, **kwargs)` — Chain custom functions cleanly: `df.pipe(f)` = `f(df)`.                                                                 | `df = (df.pipe(clean).pipe(add_features).pipe(filter_active))`                                                    |

## GroupBy and aggregation

| Method                    | Description                                                                                                | Code example                                                                                       |
| ------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `df.groupby()`            | `df.groupby(by=None, axis=0, as_index=True, sort=True, dropna=True)` — Group rows. Lazy until aggregated.  | `g = df.groupby("country")`                                                                        |
| `.agg()` with a dict      | `g.agg(func)` where `func` can be a dict mapping column → aggregation(s).                                  | `df.groupby("country").agg({"age": "mean", "amount": ["sum", "max"]})`                             |
| Multi-column groupby      | `df.groupby([col1, col2])` — Pass a list of column names.                                                  | `df.groupby(["country", "month"])["amount"].sum()`                                                 |
| `.transform()`            | `g.transform(func)` — Returns Series aligned with original. Useful for "normalize within group".           | `df["z_in_group"] = df.groupby("country")["amount"].transform(lambda x: (x - x.mean()) / x.std())` |
| `.filter()` (group-level) | `g.filter(func, dropna=True)` — Keep only groups satisfying a boolean function.                            | `df.groupby("country").filter(lambda g: len(g) >= 10)`                                             |
| `.apply()` (group-level)  | `g.apply(func, *args, **kwargs)` — Most flexible, slowest. Returns Series, DataFrame, or scalar per group. | `df.groupby("country").apply(lambda g: g.nlargest(3, "amount"))`                                   |
| `.size()` vs `.count()`   | `g.size()` is group row count (includes NaN); `g.count()` is non-null count per column.                    | `df.groupby("country").size()`<br/>`df.groupby("country").count()`                                 |
| `.nunique()`              | `g.nunique(dropna=True)` — Distinct value count per group.                                                 | `df.groupby("country")["user_id"].nunique()`                                                       |
| Named aggregations        | `g.agg(out_name=(column, aggfunc), ...)` — Cleaner output column names.                                    | `df.groupby("country").agg(avg_age=("age", "mean"), n_users=("user_id", "nunique"))`               |

## Joining and reshaping

| Method                        | Description                                                                                                                       | Code example                                                                                                    |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `pd.concat()`                 | `pd.concat(objs, axis=0, join='outer', ignore_index=False, keys=None)` — Stack vertically or horizontally.                        | `pd.concat([df1, df2], axis=0, ignore_index=True)`                                                              |
| `pd.merge()` / `df.merge()`   | `pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None, suffixes=('_x', '_y'))` — DB-style join.                | `pd.merge(users, orders, on="user_id", how="left")`                                                             |
| `df.join()`                   | `df.join(other, on=None, how='left', lsuffix='', rsuffix='')` — Index-based join. Use after `set_index`.                          | `df1.set_index("user_id").join(df2.set_index("user_id"), how="left")`                                           |
| `df.pivot()`                  | `df.pivot(index=None, columns=None, values=None)` — Reshape long → wide. Each (index, column) pair must be unique.                | `df.pivot(index="date", columns="metric", values="value")`                                                      |
| `df.pivot_table()`            | `df.pivot_table(values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False)` — Pivot with aggregation. | `df.pivot_table(index="country", columns="month", values="amount", aggfunc="sum", fill_value=0)`                |
| `df.melt()`                   | `pd.melt(frame, id_vars=None, value_vars=None, var_name=None, value_name='value')` — Wide → long. Inverse of `pivot`.             | `pd.melt(df, id_vars=["user_id"], value_vars=["q1_score", "q2_score"], var_name="quarter", value_name="score")` |
| `df.stack()` / `df.unstack()` | `df.stack(level=-1, dropna=True)` / `df.unstack(level=-1, fill_value=None)` — Reshape via the index.                              | `df.set_index(["country", "year"])["amount"].unstack("year")`                                                   |
| `df.explode()`                | `df.explode(column, ignore_index=False)` — Expand a list-valued column into multiple rows.                                        | `df.explode("tags")`                                                                                            |
| `pd.get_dummies()`            | `pd.get_dummies(data, columns=None, prefix=None, drop_first=False, dtype=None)` — One-hot encode.                                 | `pd.get_dummies(df, columns=["country"], drop_first=True)`                                                      |
| `pd.crosstab()`               | `pd.crosstab(index, columns, values=None, aggfunc=None, normalize=False, margins=False)` — Contingency table.                     | `pd.crosstab(df["country"], df["status"], normalize="index")`                                                   |

## Time series

| Method                                 | Description                                                                                                                         | Code example                                                        |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `pd.to_datetime()`                     | `pd.to_datetime(arg, format=None, unit=None, utc=False, errors='raise')` — Parse strings or Unix timestamps.                        | `df["ts"] = pd.to_datetime(df["ts"], unit="s")`                     |
| `pd.date_range()`                      | `pd.date_range(start=None, end=None, periods=None, freq='D', tz=None)` — Regular datetime index.                                    | `idx = pd.date_range("2026-01-01", periods=10, freq="D")`           |
| `df.set_index("ts")`                   | `df.set_index(keys, drop=True)` — Set datetime column as index — unlocks time-series operations.                                    | `df = df.set_index("ts").sort_index()`                              |
| `df.resample()`                        | `df.resample(rule, axis=0, closed=None, label=None)` — Aggregate over time buckets. Frequencies: `D`, `W`, `M`, `Q`, `Y`, `H`, `T`. | `df.resample("D")["amount"].sum()`                                  |
| `df.rolling()`                         | `df.rolling(window, min_periods=None, center=False, win_type=None)` — Moving-window aggregation.                                    | `df["amount_7d"] = df["amount"].rolling("7D").mean()`               |
| `df.expanding()`                       | `df.expanding(min_periods=1, axis=0)` — Cumulative aggregation (window grows over time).                                            | `df["cum_max"] = df["amount"].expanding().max()`                    |
| `df.shift()`                           | `df.shift(periods=1, freq=None, axis=0, fill_value=None)` — Lag a column by N periods.                                              | `df["amount_lag1"] = df["amount"].shift(1)`                         |
| `df.diff()`                            | `df.diff(periods=1, axis=0)` — First difference. Useful for change detection.                                                       | `df["delta"] = df["amount"].diff()`                                 |
| `df.tz_localize()` / `df.tz_convert()` | `df.tz_localize(tz, axis=0, ambiguous='raise')` / `df.tz_convert(tz, axis=0)` — Add or convert timezone.                            | `df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")` |
| `pd.Timedelta`                         | `pd.Timedelta(value=<object>, unit=None, days=0, hours=0, ...)` — Build durations.                                                  | `df[df["ts"] > pd.Timestamp("2026-01-01") - pd.Timedelta(days=7)]`  |

## Performance and memory

| Pattern                                 | Why it matters                                                                                   | Code example                                                                            |
| --------------------------------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| Use `category` dtype                    | `s.astype("category")` — Massive memory savings on columns like country/role.                    | `df["country"] = df["country"].astype("category")`                                      |
| Prefer vectorized ops over `.apply()`   | `apply` is a Python loop. Vector ops are 10-100× faster.                                         | `# slow: df.apply(lambda r: r['a'] + r['b'], axis=1)`<br/>`df["c"] = df["a"] + df["b"]` |
| Read only needed columns                | `pd.read_csv(path, usecols=[...])` cuts memory at read time.                                     | `df = pd.read_csv("big.csv", usecols=["id", "amount", "ts"])`                           |
| Use Parquet over CSV                    | 10× faster, smaller files, preserves dtypes.                                                     | `df.to_parquet("data.parquet")`                                                         |
| `chunksize` for huge CSVs               | `pd.read_csv(path, chunksize=N)` returns an iterator of chunks.                                  | `for chunk in pd.read_csv("huge.csv", chunksize=100_000):`<br/>` process(chunk)`        |
| Use nullable dtypes                     | `Int64`, `boolean`, `string` — First-class missing-value support, no silent float conversions.   | `df["count"] = df["count"].astype("Int64")`                                             |
| Prefer `inplace=False` (assign back)    | Chained assignment is cleaner than scattered `inplace=True` calls.                               | `df = df.dropna().reset_index(drop=True)`                                               |
| Avoid `SettingWithCopyWarning`          | Don't chain `df[cond][col] = val`. Use `df.loc[cond, col] = val`.                                | `df.loc[df["age"] < 0, "age"] = None`                                                   |
| Use `eval()` / `query()` for big frames | `df.eval(expr, inplace=False)` / `df.query(expr)` — Cython/numexpr-backed; faster on large data. | `df.eval("revenue = price * qty", inplace=True)`                                        |

## Common patterns

| Pattern                                      | Code                                                                                                                                                                        |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Train/test split by user                     | `users = df["user_id"].drop_duplicates().sample(frac=0.8, random_state=0)`<br/>`train = df[df["user_id"].isin(users)]`<br/>`test = df[~df["user_id"].isin(users)]`          |
| Top-N per group                              | `df.sort_values(["country", "amount"], ascending=[True, False]).groupby("country").head(3)`                                                                                 |
| Rank within group                            | `df["rank_in_country"] = df.groupby("country")["score"].rank(ascending=False, method="dense")`                                                                              |
| First / last per group                       | `df.sort_values("ts").groupby("user_id").tail(1) # latest event per user`                                                                                                   |
| Cohort analysis (signup month × usage month) | `df["cohort"] = df.groupby("user_id")["ts"].transform("min").dt.to_period("M")`<br/>`df["month"] = df["ts"].dt.to_period("M")`<br/>`pd.crosstab(df["cohort"], df["month"])` |
| Pivot to wide for model features             | `features = df.pivot_table(index="user_id", columns="event_type", values="amount", aggfunc="sum", fill_value=0)`                                                            |
| Forward-fill within group                    | `df["price_ff"] = df.groupby("ticker")["price"].ffill()`                                                                                                                    |
| Encode categorical for ML                    | `df_enc = pd.get_dummies(df[["country", "device"]], drop_first=True)`                                                                                                       |
| Datetime feature engineering                 | `df["hour"] = df["ts"].dt.hour`<br/>`df["is_weekend"] = df["ts"].dt.dayofweek.isin([5, 6])`                                                                                 |
| Save model-ready frame                       | `df.to_parquet("features.parquet", compression="snappy")`                                                                                                                   |
