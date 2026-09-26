---
title: Pandas Interview Questions (100)
sidebar_position: 32
sidebar_label: "Reference · Pandas"
---

# Pandas Interview Questions (100)

:::note Foundation reference
This earlier 100-entry bank remains for prerequisite revision. Its questions were not part of the interview-source audit, and some examples are fragments rather than standalone programmes. Start with the [practical topic bank](01-python-data.md) for sourced scenarios, answered follow-ups and tested code. See [tools and versions](99-tools-versions.md) for current behaviour and [coding labs](11-coding-labs.md) for runnable implementations.
:::

<details>
<summary><strong>1. What is Pandas and what are its main data structures?</strong></summary>

**Answer:** Pandas is a Python library for data manipulation built on NumPy. Its two main structures are **Series** (1D labeled array) and **DataFrame** (2D labeled table).

```python
import pandas as pd

# Series
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s['a'])    # 10

# DataFrame
df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
print(df.dtypes)
print(df.shape)   # (2, 2)
```

**Interview Tip:** Explain that DataFrame columns are Series sharing the same index.
</details>

<details>
<summary><strong>2. How do you load data into a DataFrame?</strong></summary>

**Answer:** Pandas supports loading from CSV, Excel, JSON, SQL, Parquet, and more.

```python
import pandas as pd

df = pd.read_csv('data.csv')
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df = pd.read_json('data.json')
df = pd.read_parquet('data.parquet')

import sqlite3
conn = sqlite3.connect('db.sqlite')
df = pd.read_sql('SELECT * FROM users', conn)

# read_csv options
df = pd.read_csv(
    'data.csv',
    sep=',',
    header=0,
    index_col='id',
    usecols=['name', 'age'],
    dtype={'age': 'int32'},
    nrows=1000,
    skiprows=2,
    na_values=['NA', 'missing']
)
```

**Interview Tip:** Mention `chunksize` for reading large files in chunks.
</details>

<details>
<summary><strong>3. How do you explore a DataFrame?</strong></summary>

**Answer:** Use built-in methods to quickly understand shape, types, and distributions.

```python
import pandas as pd

df = pd.read_csv('data.csv')

df.head(5)           # first 5 rows
df.tail(5)           # last 5 rows
df.shape             # (rows, cols)
df.info()            # dtypes + null counts
df.describe()        # stats for numeric cols
df.dtypes            # column data types
df.columns.tolist()  # column names
df.isnull().sum()    # missing values per col
df.duplicated().sum()# duplicate rows
df.nunique()         # unique values per col
df.value_counts()    # frequency of each value (Series)
df.sample(10)        # random 10 rows
```

**Interview Tip:** Know `info()` vs `describe()` — info shows types/nulls, describe shows stats.
</details>

<details>
<summary><strong>4. How do you select columns and rows?</strong></summary>

**Answer:** Use `[]`, `.loc[]` (label-based), and `.iloc[]` (position-based).

```python
import pandas as pd

df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6], 'C': [7,8,9]})

# Column selection
df['A']            # Series
df[['A', 'B']]     # DataFrame

# loc: label-based
df.loc[0]          # row by index label
df.loc[0:2, 'A']   # rows 0-2, column A
df.loc[:, 'A':'B'] # all rows, cols A to B

# iloc: position-based
df.iloc[0]         # first row
df.iloc[0:2, 0:2]  # rows 0-1, cols 0-1
df.iloc[:, -1]     # last column

# Boolean selection
df[df['A'] > 1]
df[(df['A'] > 1) & (df['B'] < 6)]
```

**Interview Tip:** `.loc` includes the end label; `.iloc` excludes it (like Python slicing).
</details>

<details>
<summary><strong>5. How do you handle missing values?</strong></summary>

**Answer:** Detect with `isnull()`, handle with `dropna()` or `fillna()`.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 2, 3]})

# Detect
df.isnull()
df.isnull().sum()
df.isnull().mean()  # fraction missing

# Drop
df.dropna()                   # drop rows with any NaN
df.dropna(axis=1)             # drop columns with any NaN
df.dropna(how='all')          # drop only if all values NaN
df.dropna(thresh=2)           # keep rows with at least 2 non-NaN

# Fill
df.fillna(0)
df.fillna(df.mean())          # fill with column mean
df.ffill()     # forward fill
df.bfill()     # backward fill
df['A'].fillna(df['A'].median(), inplace=True)

# Interpolate
df.interpolate(method='linear')
```

**Interview Tip:** Explain when to drop vs fill — dropping loses data, filling can introduce bias.
</details>

<details>
<summary><strong>6. How do you filter rows with conditions?</strong></summary>

**Answer:** Boolean masks, `query()`, and `isin()`.

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'NYC']
})

# Boolean mask
df[df['age'] > 25]
df[(df['age'] > 25) & (df['city'] == 'NYC')]
df[(df['age'] < 28) | (df['age'] > 32)]

# isin
df[df['city'].isin(['NYC', 'Chicago'])]
df[~df['city'].isin(['LA'])]  # NOT in

# query (string expression)
df.query('age > 25 and city == "NYC"')
df.query('age > @threshold', threshold=25)

# String filtering
df[df['name'].str.startswith('A')]
df[df['name'].str.contains('li', case=False)]
```

**Interview Tip:** `query()` is more readable for complex conditions; `isin()` is efficient for multi-value checks.
</details>

<details>
<summary><strong>7. How do you sort a DataFrame?</strong></summary>

**Answer:** Use `sort_values()` and `sort_index()`.

```python
import pandas as pd

df = pd.DataFrame({'name': ['Bob','Alice','Charlie'], 'age': [30,25,35], 'score': [85,90,80]})

# Sort by single column
df.sort_values('age')
df.sort_values('age', ascending=False)

# Sort by multiple columns
df.sort_values(['age', 'score'], ascending=[True, False])

# Sort by index
df.sort_index()
df.sort_index(ascending=False)

# nlargest / nsmallest
df.nlargest(2, 'score')    # top 2 by score
df.nsmallest(2, 'age')     # bottom 2 by age

# inplace
df.sort_values('age', inplace=True)
```

**Interview Tip:** `nlargest`/`nsmallest` is faster than sort + head for top-N operations.
</details>

<details>
<summary><strong>8. How do you rename columns and index?</strong></summary>

**Answer:** Use `rename()`, direct assignment, or `set_index()`.

```python
import pandas as pd

df = pd.DataFrame({'A': [1,2], 'B': [3,4]})

# Rename specific columns
df.rename(columns={'A': 'col_a', 'B': 'col_b'}, inplace=True)

# Rename all columns at once
df.columns = ['x', 'y']

# Rename with function
df.rename(columns=str.lower)
df.rename(columns=lambda c: c.strip().replace(' ', '_'))

# Rename index
df.rename(index={0: 'row1', 1: 'row2'})

# Set index
df.set_index('x', inplace=True)
df.reset_index(inplace=True)

# Column cleanup pattern
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
```

**Interview Tip:** Column name cleanup (strip, lower, replace spaces) is a common preprocessing step.
</details>

<details>
<summary><strong>9. How do you add and delete columns?</strong></summary>

**Answer:** Direct assignment, `assign()`, `insert()`, and `drop()`.

```python
import pandas as pd

df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})

# Add column
df['senior'] = df['age'] > 28
df['age_squared'] = df['age'] ** 2

# assign (returns new df, chainable)
df = df.assign(
    decade=df['age'] // 10 * 10,
    name_upper=df['name'].str.upper()
)

# Insert at specific position
df.insert(1, 'id', [1, 2])

# Delete column
df.drop('age_squared', axis=1, inplace=True)
df.drop(columns=['id', 'decade'])

# Delete row
df.drop(index=0)
df.drop(index=[0, 1])

# del
del df['senior']
```

**Interview Tip:** `assign()` is preferred in method chains as it returns a new DataFrame.
</details>

<details>
<summary><strong>10. How do you group and aggregate data?</strong></summary>

**Answer:** `groupby()` splits data into groups for aggregation.

```python
import pandas as pd

df = pd.DataFrame({
    'dept': ['Eng','Eng','HR','HR'],
    'salary': [80000, 90000, 60000, 70000],
    'age': [25, 30, 28, 35]
})

# Basic groupby
df.groupby('dept')['salary'].mean()
df.groupby('dept')['salary'].agg(['mean','max','min','count'])

# Multiple columns
df.groupby('dept').agg({'salary': 'mean', 'age': 'max'})

# Named aggregations
df.groupby('dept').agg(
    avg_salary=('salary', 'mean'),
    max_age=('age', 'max')
)

# Transform (keeps original index, broadcasts result)
df['dept_avg'] = df.groupby('dept')['salary'].transform('mean')

# Filter groups
df.groupby('dept').filter(lambda x: x['salary'].mean() > 65000)
```

**Interview Tip:** Know the difference between `agg` (reduces), `transform` (same shape), and `apply` (custom logic).
</details>

<details>
<summary><strong>11. How do you merge DataFrames?</strong></summary>

**Answer:** `merge()` is SQL-style join; `concat()` stacks DataFrames.

```python
import pandas as pd

df1 = pd.DataFrame({'id': [1,2,3], 'name': ['Alice','Bob','Charlie']})
df2 = pd.DataFrame({'id': [2,3,4], 'salary': [60000,70000,80000]})

# Inner join (default)
pd.merge(df1, df2, on='id')

# Left / Right / Outer joins
pd.merge(df1, df2, on='id', how='left')
pd.merge(df1, df2, on='id', how='right')
pd.merge(df1, df2, on='id', how='outer')

# Different column names
pd.merge(df1, df2, left_on='id', right_on='emp_id')

# Merge on index
pd.merge(df1, df2, left_index=True, right_index=True)

# Concat (stack rows or columns)
pd.concat([df1, df2], axis=0, ignore_index=True)  # stack rows
pd.concat([df1, df2], axis=1)                      # side by side

# join (index-based shortcut)
df1.join(df2.set_index('id'), on='id')
```

**Interview Tip:** `merge()` is for relational joins; `concat()` is for appending rows or columns.
</details>

<details>
<summary><strong>12. How do you pivot and reshape data?</strong></summary>

**Answer:** `pivot_table()`, `melt()`, `stack()`, and `unstack()` reshape data.

```python
import pandas as pd

df = pd.DataFrame({
    'store': ['A','A','B','B'],
    'month': ['Jan','Feb','Jan','Feb'],
    'sales': [100,150,200,250]
})

# Pivot table
pivot = df.pivot_table(values='sales', index='store', columns='month', aggfunc='sum')

# pivot (no aggregation, needs unique index-col pairs)
df.pivot(index='store', columns='month', values='sales')

# Melt (wide → long)
df_wide = pd.DataFrame({'id':[1,2],'jan':[100,200],'feb':[150,250]})
df_long = pd.melt(df_wide, id_vars='id', var_name='month', value_name='sales')

# Stack/Unstack
df.pivot_table(values='sales', index='store', columns='month').stack()
```

**Interview Tip:** `melt` is the inverse of `pivot` — wide to long vs long to wide.
</details>

<details>
<summary><strong>13. How do you apply functions to DataFrames?</strong></summary>

**Answer:** `apply()`, `map()`, `applymap()`/`map()` for element-wise operations.

```python
import pandas as pd

df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})

# apply to each column (axis=0) or row (axis=1)
df.apply(lambda col: col.sum())              # sum each column
df.apply(lambda row: row.sum(), axis=1)      # sum each row

# map (element-wise on Series)
df['A'].map(lambda x: x**2)
df['A'].map({1: 'one', 2: 'two', 3: 'three'})

# applymap / map on DataFrame (element-wise)
df.applymap(lambda x: x*2)    # deprecated → use df.map() in pandas 2.1+
df.map(lambda x: x*2)

# Vectorized string operations
s = pd.Series(['Alice', 'Bob', 'Charlie'])
s.str.upper()
s.str.len()
s.str.split().str[0]

# Custom function
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

df.apply(normalize)
```

**Interview Tip:** Vectorized operations (`+`, `*`, `.str`) are faster than `apply`; use `apply` for complex custom logic.
</details>

<details>
<summary><strong>14. How do you work with string data in Pandas?</strong></summary>

**Answer:** The `.str` accessor provides vectorized string methods.

```python
import pandas as pd

s = pd.Series(['  Alice Smith ', 'BOB JONES', 'charlie brown'])

s.str.strip()
s.str.lower()
s.str.upper()
s.str.title()
s.str.replace('  ', ' ', regex=False)
s.str.split(' ')                         # split into list
s.str.split(' ', expand=True)            # split into columns
s.str.get(0)                             # first element after split
s.str.contains('alice', case=False)
s.str.startswith('A')
s.str.len()
s.str.slice(0, 5)                        # like s.str[:5]
s.str.extract(r'(\w+)\s(\w+)')          # regex capture groups
s.str.findall(r'\d+')                   # find all matches

# Clean a name column
df['name'] = df['name'].str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
```

**Interview Tip:** `.str` methods handle NaN gracefully (return NaN instead of raising).
</details>

<details>
<summary><strong>15. How do you work with datetime data in Pandas?</strong></summary>

**Answer:** Parse with `to_datetime()`, then use the `.dt` accessor.

```python
import pandas as pd

df = pd.DataFrame({'date': ['2024-01-15', '2024-06-30', '2024-12-25']})

# Parse
df['date'] = pd.to_datetime(df['date'])

# dt accessor
df['year']        = df['date'].dt.year
df['month']       = df['date'].dt.month
df['day']         = df['date'].dt.day
df['dayofweek']   = df['date'].dt.dayofweek   # 0=Monday
df['is_weekend']  = df['date'].dt.dayofweek >= 5
df['quarter']     = df['date'].dt.quarter
df['week']        = df['date'].dt.isocalendar().week

# Arithmetic
df['next_week'] = df['date'] + pd.Timedelta(days=7)
df['days_since'] = (pd.Timestamp.today() - df['date']).dt.days

# Resample (time series)
df.set_index('date').resample('M').sum()    # monthly sum
df.set_index('date').resample('W').mean()   # weekly mean

# Date range
pd.date_range('2024-01-01', periods=12, freq='M')
```

**Interview Tip:** Always parse dates at load time (`parse_dates=['date']` in `read_csv`) for efficiency.
</details>

<details>
<summary><strong>16. How do you handle categorical data?</strong></summary>

**Answer:** Use `Categorical` dtype to save memory and enable ordering.

```python
import pandas as pd

df = pd.DataFrame({'size': ['S','M','L','M','S','XL']})

# Convert to category
df['size'] = df['size'].astype('category')
print(df['size'].cat.categories)     # Index(['L', 'M', 'S', 'XL'])
print(df['size'].cat.codes)          # integer codes

# Ordered category
from pandas.api.types import CategoricalDtype
size_order = CategoricalDtype(['S','M','L','XL'], ordered=True)
df['size'] = df['size'].astype(size_order)
df['size'] > 'M'    # True for L and XL

# Memory savings
s_obj = pd.Series(['cat','dog','cat'] * 100000)
s_cat = s_obj.astype('category')
print(s_obj.memory_usage(deep=True))   # large
print(s_cat.memory_usage(deep=True))   # much smaller

# get_dummies
pd.get_dummies(df['size'], prefix='size')
```

**Interview Tip:** Use `category` dtype when a column has few unique values relative to its length.
</details>

<details>
<summary><strong>17. How do you use MultiIndex?</strong></summary>

**Answer:** MultiIndex stores hierarchical index for groupby results and complex data.

```python
import pandas as pd

df = pd.DataFrame({
    'dept': ['Eng','Eng','HR','HR'],
    'level': ['Senior','Junior','Senior','Junior'],
    'salary': [100,70,80,60]
})

# Create MultiIndex via groupby
grouped = df.groupby(['dept','level'])['salary'].mean()
print(grouped)           # MultiIndex Series

# Access
grouped['Eng']
grouped['Eng', 'Senior']
grouped.loc[('Eng', 'Senior')]

# unstack
grouped.unstack('level')    # level becomes columns

# Create manually
idx = pd.MultiIndex.from_tuples([('Eng','Senior'),('Eng','Junior')],
                                 names=['dept','level'])

# Reset
grouped.reset_index()

# xs (cross-section)
grouped.xs('Senior', level='level')
```

**Interview Tip:** Know `unstack()` to convert MultiIndex rows to columns — very common for pivot operations.
</details>

<details>
<summary><strong>18. How do you compute rolling and expanding statistics?</strong></summary>

**Answer:** `rolling()` for fixed window, `expanding()` for cumulative window.

```python
import pandas as pd

df = pd.DataFrame({'sales': [100, 120, 130, 110, 150, 140]})

# Rolling window
df['rolling_mean']  = df['sales'].rolling(window=3).mean()
df['rolling_std']   = df['sales'].rolling(window=3).std()
df['rolling_max']   = df['sales'].rolling(window=3).max()

# Expanding (cumulative)
df['cumulative_mean'] = df['sales'].expanding().mean()
df['cumulative_sum']  = df['sales'].expanding().sum()

# Shift (lag)
df['prev_sales']     = df['sales'].shift(1)    # lag by 1
df['next_sales']     = df['sales'].shift(-1)   # lead by 1

# Diff (change from previous)
df['change']         = df['sales'].diff()
df['pct_change']     = df['sales'].pct_change()

# Exponential weighted mean
df['ewm_mean']       = df['sales'].ewm(span=3).mean()
```

**Interview Tip:** Rolling windows are key for time series feature engineering in ML.
</details>

<details>
<summary><strong>19. How do you deduplicate data?</strong></summary>

**Answer:** Use `duplicated()` to find and `drop_duplicates()` to remove.

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice','Bob','Alice','Bob','Charlie'],
    'age': [25,30,25,31,35]
})

# Find duplicates
df.duplicated()                          # default: all columns, keep='first'
df.duplicated(subset=['name'])           # based on name only
df.duplicated(keep=False)                # mark all duplicates

# Remove duplicates
df.drop_duplicates()
df.drop_duplicates(subset=['name'])
df.drop_duplicates(subset=['name'], keep='last')
df.drop_duplicates(keep=False)           # remove all duplicates

# How many?
print(df.duplicated().sum())

# Identify which rows are duplicate
df[df.duplicated(keep=False)]
```

**Interview Tip:** Always specify `subset` when only certain columns define uniqueness.
</details>

<details>
<summary><strong>20. How do you use `pd.cut` and `pd.qcut` for binning?</strong></summary>

**Answer:** `cut` bins by value ranges; `qcut` bins by quantiles.

```python
import pandas as pd

ages = pd.Series([15, 22, 35, 45, 60, 72, 80])

# cut: fixed bins
bins = [0, 18, 35, 60, 100]
labels = ['teen','young','middle','senior']
df['age_group'] = pd.cut(ages, bins=bins, labels=labels)

# cut with equal-width bins (auto)
pd.cut(ages, bins=4)

# qcut: equal-frequency bins
pd.qcut(ages, q=4, labels=['Q1','Q2','Q3','Q4'])

# With value_counts
pd.cut(ages, bins=4).value_counts()

# Include lowest
pd.cut(ages, bins=[0,18,60,100], include_lowest=True)

# return_bins
result, bins_used = pd.cut(ages, bins=4, retbins=True)
```

**Interview Tip:** `qcut` ensures equal sample sizes per bin; `cut` ensures equal width — choose based on distribution.
</details>

<details>
<summary><strong>21. How do you use `stack` and `unstack`?</strong></summary>

**Answer:** `stack` moves columns to row index; `unstack` moves row index to columns.

```python
import pandas as pd

df = pd.DataFrame({'Jan': [100,200], 'Feb': [150,250]},
                  index=['Store_A', 'Store_B'])

# stack: wide → long (columns → index level)
stacked = df.stack()
# MultiIndex: (Store_A, Jan), (Store_A, Feb) ...

# unstack: long → wide (index level → columns)
stacked.unstack()   # back to original

# Specify level
stacked.unstack(level=0)   # move first index level to columns
stacked.unstack(level=-1)  # move last (default)

# With MultiIndex
df.columns = pd.MultiIndex.from_tuples([('Q1','Jan'),('Q1','Feb')])
df.stack(level=0)
```

**Interview Tip:** `stack/unstack` are for reshaping pivot-like data stored in DataFrames.
</details>

<details>
<summary><strong>22. How do you sample data?</strong></summary>

**Answer:** `sample()` for random rows; `nlargest`/`nsmallest` for sorted top-N.

```python
import pandas as pd

df = pd.DataFrame({'x': range(1000), 'y': range(1000)})

# Random sample
df.sample(n=100)                          # 100 random rows
df.sample(frac=0.1)                       # 10% of data
df.sample(n=100, random_state=42)         # reproducible
df.sample(n=100, replace=True)            # with replacement

# Stratified sample (using groupby)
stratified = df.groupby('category', group_keys=False).apply(
    lambda x: x.sample(min(len(x), 50))
)

# Top / bottom N
df.nlargest(10, 'x')
df.nsmallest(10, 'x')

# Head / tail
df.head(10)
df.tail(10)
```

**Interview Tip:** Use `random_state` for reproducibility in ML experiments.
</details>

<details>
<summary><strong>23. How do you compute correlations?</strong></summary>

**Answer:** `corr()` computes pairwise correlation; supports Pearson, Spearman, Kendall.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1,2,3,4,5],
    'B': [2,4,5,4,5],
    'C': [5,4,3,2,1]
})

# Pearson correlation (default)
df.corr()
df.corr(method='pearson')

# Spearman rank correlation
df.corr(method='spearman')

# Kendall
df.corr(method='kendall')

# Single pair
df['A'].corr(df['B'])

# Covariance
df.cov()

# Correlation with target (ML feature selection)
target = pd.Series([1,0,1,0,1], name='target')
correlations = df.corrwith(target).abs().sort_values(ascending=False)
```

**Interview Tip:** Mention that correlation ≠ causation, and Spearman is more robust to outliers.
</details>

<details>
<summary><strong>24. How do you use `pd.get_dummies` for encoding?</strong></summary>

**Answer:** One-hot encoding converts categorical columns to binary columns.

```python
import pandas as pd

df = pd.DataFrame({'color': ['red','blue','green','red'],'size': ['S','M','L','S']})

# Basic encoding
encoded = pd.get_dummies(df, columns=['color','size'])

# Drop first to avoid multicollinearity
encoded = pd.get_dummies(df, columns=['color'], drop_first=True)

# Prefix
pd.get_dummies(df['color'], prefix='color')

# Boolean vs int
pd.get_dummies(df, dtype=int)   # 1/0 instead of True/False

# With unknown categories at test time
train_dummies = pd.get_dummies(df_train['color'])
test_dummies  = pd.get_dummies(df_test['color']).reindex(
    columns=train_dummies.columns, fill_value=0
)
```

**Interview Tip:** Drop one dummy per category (`drop_first=True`) to avoid the dummy variable trap.
</details>

<details>
<summary><strong>25. How do you export DataFrames?</strong></summary>

**Answer:** Pandas can export to CSV, Excel, JSON, Parquet, SQL, and more.

```python
import pandas as pd

df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})

# CSV
df.to_csv('output.csv', index=False)
df.to_csv('output.csv', sep='\t', encoding='utf-8')

# Excel
df.to_excel('output.xlsx', sheet_name='Sheet1', index=False)

# JSON
df.to_json('output.json', orient='records')
df.to_json('output.json', orient='columns')

# Parquet (columnar, efficient)
df.to_parquet('output.parquet', index=False)

# SQL
import sqlite3
conn = sqlite3.connect('db.sqlite')
df.to_sql('table_name', conn, if_exists='replace', index=False)

# String
csv_string = df.to_csv(index=False)
json_string = df.to_json(orient='records')

# Clipboard
df.to_clipboard()
```

**Interview Tip:** Prefer Parquet for large datasets — faster reads and smaller file size than CSV.
</details>

<details>
<summary><strong>26. What is the difference between `copy()` and view in Pandas?</strong></summary>

**Answer:** In pandas 3, Copy-on-Write makes a derived object behave independently when mutated. Use one `.loc` assignment on the original DataFrame when you intend to change it. Earlier pandas versions had different view/copy warnings; the comments below distinguish that historical behaviour.

```python
import pandas as pd

df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})

# pandas 3: a derived object behaves independently on mutation
subset = df[df['A'] > 1]         # derived DataFrame
subset['B'] = 99                 # pandas 3: changes subset only; older pandas could warn

# Explicit copying is useful when you need an eager independent copy
subset = df[df['A'] > 1].copy()
subset['B'] = 99                 # safe

# Check observable behaviour rather than private internals
assert df.loc[1, "B"] == 5

# One direct assignment on the original DataFrame
df.loc[df['A'] > 1, 'B'] = 99   # directly on original
```

**Interview Tip:** State your pandas version. In pandas 3, chained assignment does not update the original object; mutate the intended frame directly. See [Copy-on-Write examples](99-tools-versions.md).
</details>

<details>
<summary><strong>27. How do you chain operations in Pandas?</strong></summary>

**Answer:** Method chaining with `pipe()` keeps code readable and avoids intermediate variables.

```python
import pandas as pd

df = pd.read_csv('data.csv')

# Without chaining (messy)
df = df.dropna()
df = df.rename(columns={'old': 'new'})
df = df[df['age'] > 18]
df = df.sort_values('name')

# With method chaining
df = (
    pd.read_csv('data.csv')
    .dropna()
    .rename(columns={'old': 'new'})
    .query('age > 18')
    .sort_values('name')
    .reset_index(drop=True)
)

# pipe for custom functions
def clean_names(df):
    df['name'] = df['name'].str.strip().str.title()
    return df

df = (
    pd.read_csv('data.csv')
    .pipe(clean_names)
    .dropna()
)
```

**Interview Tip:** `pipe()` lets you include custom functions in a chain cleanly.
</details>

<details>
<summary><strong>28. How do you compute value counts and frequencies?</strong></summary>

**Answer:** `value_counts()` on Series; `crosstab()` for two variables.

```python
import pandas as pd

df = pd.DataFrame({'dept':['Eng','HR','Eng','Finance','HR','Eng'],
                   'level':['Senior','Junior','Junior','Senior','Senior','Senior']})

# value_counts
df['dept'].value_counts()
df['dept'].value_counts(normalize=True)    # proportions
df['dept'].value_counts(dropna=False)      # include NaN

# crosstab
pd.crosstab(df['dept'], df['level'])
pd.crosstab(df['dept'], df['level'], normalize='index')  # row %
pd.crosstab(df['dept'], df['level'], margins=True)       # totals

# groupby count
df.groupby('dept')['level'].count()
df.groupby('dept').size()               # same result
```

**Interview Tip:** `crosstab` is shorthand for a pivot table with count aggregation.
</details>

<details>
<summary><strong>29. How do you use `where` and `mask`?</strong></summary>

**Answer:** `where` keeps values where condition is True; `mask` replaces where condition is True.

```python
import pandas as pd
import numpy as np

s = pd.Series([10, -5, 30, -20, 50])

# where: keep where True, replace where False
s.where(s > 0)                     # [-5,-20] become NaN
s.where(s > 0, other=0)            # [-5,-20] become 0

# mask: replace where True (opposite of where)
s.mask(s < 0)                      # [-5,-20] become NaN
s.mask(s < 0, other=0)             # [-5,-20] become 0

# On DataFrame
df = pd.DataFrame({'A': [1,2,3], 'B': [4,-5,6]})
df.where(df > 0, other=np.nan)

# Conditional replacement in ML preprocessing
df['income'] = df['income'].where(df['income'] >= 0, other=df['income'].median())
```

**Interview Tip:** `where` and `mask` are vectorized alternatives to `apply` + `if/else`.
</details>

<details>
<summary><strong>30. How do you use `assign` for feature engineering?</strong></summary>

**Answer:** `assign()` creates new columns while returning a copy — ideal for pipelines.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'name': ['Alice','Bob'], 'salary': [50000,80000], 'tax_rate': [0.2,0.3]})

# assign creates new columns
df = df.assign(
    net_salary=df['salary'] * (1 - df['tax_rate']),
    name_len=df['name'].str.len(),
    is_high_earner=df['salary'] > 60000
)

# Can reference newly created columns in same assign
df = df.assign(
    salary_bucket=lambda x: pd.cut(x['salary'], bins=3, labels=['Low','Mid','High'])
)

# Chaining
result = (
    df
    .assign(annual_tax=lambda x: x['salary'] * x['tax_rate'])
    .assign(category=lambda x: np.where(x['salary'] > 60000, 'senior', 'junior'))
    .query('category == "senior"')
)
```

**Interview Tip:** `assign` with `lambda` is safer than direct assignment as it references current state.
</details>

<details>
<summary><strong>31. How do you read large files efficiently?</strong></summary>

**Answer:** Use `chunksize`, optimize `dtype`, and select only needed columns.

```python
import pandas as pd

# Read in chunks
chunks = []
for chunk in pd.read_csv('large.csv', chunksize=10000):
    chunk = chunk[chunk['age'] > 18]
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)

# Optimize dtypes at read time
df = pd.read_csv('data.csv',
    dtype={'id': 'int32', 'age': 'int8', 'status': 'category'},
    usecols=['id', 'age', 'status'],
    parse_dates=['created_at']
)

# Downcast after loading
df['age']  = pd.to_numeric(df['age'], downcast='integer')
df['score']= pd.to_numeric(df['score'], downcast='float')

# Memory check
df.memory_usage(deep=True).sum() / 1e6  # MB

# Use Parquet for repeated reads
df.to_parquet('data.parquet')
df = pd.read_parquet('data.parquet', columns=['id','age'])
```

**Interview Tip:** Specifying `dtype` at read time can reduce memory by 50–70%.
</details>

<details>
<summary><strong>32. How do you use `pd.DataFrame.pipe`?</strong></summary>

**Answer:** `pipe` applies a function to a DataFrame — useful for clean pipelines.

```python
import pandas as pd

def remove_outliers(df, col, n_std=3):
    mean, std = df[col].mean(), df[col].std()
    return df[(df[col] - mean).abs() <= n_std * std]

def encode_categoricals(df, cols):
    return pd.get_dummies(df, columns=cols, drop_first=True)

def scale_features(df, cols):
    df[cols] = (df[cols] - df[cols].mean()) / df[cols].std()
    return df

# Clean pipeline
result = (
    pd.read_csv('data.csv')
    .pipe(remove_outliers, col='salary')
    .pipe(encode_categoricals, cols=['dept','level'])
    .pipe(scale_features, cols=['salary','age'])
)
```

**Interview Tip:** `pipe` makes preprocessing pipelines readable and testable function-by-function.
</details>

<details>
<summary><strong>33. How do you convert data types?</strong></summary>

**Answer:** Use `astype()`, `pd.to_numeric()`, `pd.to_datetime()`, and `pd.to_timedelta()`.

```python
import pandas as pd

df = pd.DataFrame({'age':['25','30','35'], 'score':['1.5','2.3','3.1'], 'date':['2024-01-01','2024-06-15','2024-12-31']})

# astype
df['age']   = df['age'].astype(int)
df['score'] = df['score'].astype(float)
df['age']   = df['age'].astype('int32')    # smaller int

# to_numeric with error handling
df['score'] = pd.to_numeric(df['score'], errors='coerce')   # bad → NaN
df['score'] = pd.to_numeric(df['score'], errors='ignore')   # bad → keep

# to_datetime
df['date']  = pd.to_datetime(df['date'])
df['date']  = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Category
df['dept']  = df['dept'].astype('category')

# Check current types
print(df.dtypes)
```

**Interview Tip:** `errors='coerce'` is invaluable when dirty strings are in numeric columns.
</details>

<details>
<summary><strong>34. How do you compute cumulative operations?</strong></summary>

**Answer:** `cumsum`, `cumprod`, `cummax`, `cummin` work element-wise cumulatively.

```python
import pandas as pd

df = pd.DataFrame({'sales': [100, 120, 90, 150, 110]})

df['cumsum']   = df['sales'].cumsum()     # cumulative sum
df['cumprod']  = df['sales'].cumprod()    # cumulative product
df['cummax']   = df['sales'].cummax()     # running maximum
df['cummin']   = df['sales'].cummin()     # running minimum

# Running count
df['count']    = df['sales'].expanding().count()

# Cumulative % of total
df['cum_pct']  = df['sales'].cumsum() / df['sales'].sum()

# Rank
df['rank']     = df['sales'].rank()
df['rank_pct'] = df['sales'].rank(pct=True)
```

**Interview Tip:** `cumsum` is used heavily in financial and time series analysis.
</details>

<details>
<summary><strong>35. How do you merge on multiple keys?</strong></summary>

**Answer:** Pass a list to `on=` or specify `left_on` and `right_on`.

```python
import pandas as pd

orders = pd.DataFrame({'cust_id':[1,1,2],'product_id':[10,20,10],'qty':[2,1,3]})
prices = pd.DataFrame({'cust_id':[1,2],'product_id':[10,10],'price':[5.0,5.0]})

# Merge on multiple keys
result = pd.merge(orders, prices, on=['cust_id','product_id'])

# Different column names in each df
pd.merge(orders, prices,
         left_on=['cust_id','product_id'],
         right_on=['customer','prod'])

# Suffix for overlapping column names
pd.merge(orders, prices, on='product_id', suffixes=('_order','_price'))

# Validate uniqueness
pd.merge(orders, prices, on='product_id', validate='many_to_one')
# Raises if right key is not unique
```

**Interview Tip:** Use `validate` to catch data quality issues early in pipelines.
</details>

<details>
<summary><strong>36. How does `transform` differ from `agg` in groupby?</strong></summary>

**Answer:** `agg` reduces each group to a scalar; `transform` returns a same-size result aligned to original index.

```python
import pandas as pd

df = pd.DataFrame({'dept':['Eng','Eng','HR','HR'],'salary':[80,90,60,70]})

# agg: reduces to one row per group
df.groupby('dept')['salary'].agg('mean')
# dept
# Eng    85.0
# HR     65.0

# transform: broadcasts back to original shape
df['dept_mean'] = df.groupby('dept')['salary'].transform('mean')
#   dept  salary  dept_mean
#    Eng      80       85.0
#    Eng      90       85.0
#    HR       60       65.0
#    HR       70       65.0

# Use case: normalize within group
df['z_score'] = df.groupby('dept')['salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

**Interview Tip:** `transform` is perfect for adding group-level features while keeping original row count.
</details>

<details>
<summary><strong>37. How do you use `pd.concat` correctly?</strong></summary>

**Answer:** `concat` stacks DataFrames along rows or columns with index control.

```python
import pandas as pd

df1 = pd.DataFrame({'A':[1,2],'B':[3,4]})
df2 = pd.DataFrame({'A':[5,6],'B':[7,8]})
df3 = pd.DataFrame({'C':[9,10],'D':[11,12]})

# Stack rows (axis=0)
pd.concat([df1, df2])                          # keeps original index
pd.concat([df1, df2], ignore_index=True)       # reset index
pd.concat([df1, df2], keys=['df1','df2'])      # hierarchical index

# Stack columns (axis=1)
pd.concat([df1, df3], axis=1)

# Align on index (outer join by default)
df_a = pd.DataFrame({'A':[1,2]}, index=[0,1])
df_b = pd.DataFrame({'B':[3,4]}, index=[1,2])
pd.concat([df_a, df_b], axis=1)               # NaN where indices don't align
pd.concat([df_a, df_b], axis=1, join='inner') # only shared indices

# Don't concat in a loop — use a list
dfs = []
for ... :
    dfs.append(processed_chunk)
final = pd.concat(dfs, ignore_index=True)     # efficient
```

**Interview Tip:** Never call `pd.concat` inside a loop — collect in a list, then concat once.
</details>

<details>
<summary><strong>38. How do you use `explode`?</strong></summary>

**Answer:** `explode` turns list-valued cells into separate rows.

```python
import pandas as pd

df = pd.DataFrame({
    'user': ['Alice', 'Bob'],
    'tags': [['python','ml','data'], ['sql','bi']]
})

# Explode list column into rows
exploded = df.explode('tags')
#   user    tags
#   Alice   python
#   Alice   ml
#   Alice   data
#   Bob     sql
#   Bob     bi

# With reset_index
exploded.reset_index(drop=True)

# Multiple columns (pandas 1.3+)
df['scores'] = [[90,85,92],[70,75]]
df.explode(['tags','scores'])   # explode both in parallel
```

**Interview Tip:** `explode` is essential for working with JSON arrays loaded into Pandas.
</details>

<details>
<summary><strong>39. How do you detect and handle outliers?</strong></summary>

**Answer:** Use IQR, Z-score, or `clip` to identify and handle outliers.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'salary': [50000, 60000, 55000, 1000000, 58000, 62000]})

# IQR method
Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['salary'] < lower) | (df['salary'] > upper)]
clean    = df[(df['salary'] >= lower) & (df['salary'] <= upper)]

# Z-score method
z_scores = (df['salary'] - df['salary'].mean()) / df['salary'].std()
clean = df[z_scores.abs() < 3]

# Clip (cap at boundaries)
df['salary'] = df['salary'].clip(lower=lower, upper=upper)

# Winsorize (cap at percentiles)
df['salary'] = df['salary'].clip(
    lower=df['salary'].quantile(0.05),
    upper=df['salary'].quantile(0.95)
)
```

**Interview Tip:** `clip` is softer than dropping — keeps data but limits extreme values.
</details>

<details>
<summary><strong>40. What is `pd.wide_to_long` used for?</strong></summary>

**Answer:** Converts wide-format time-series/repeated-measurement data to long format.

```python
import pandas as pd

df = pd.DataFrame({
    'id': [1, 2],
    'score_2022': [85, 90],
    'score_2023': [88, 92],
    'grade_2022': ['B', 'A'],
    'grade_2023': ['B+', 'A']
})

# wide_to_long
long_df = pd.wide_to_long(
    df,
    stubnames=['score', 'grade'],
    i='id',
    j='year',
    sep='_'
)
#       score grade
# id year
# 1  2022    85     B
# 1  2023    88    B+
# 2  2022    90     A
# 2  2023    92     A
```

**Interview Tip:** `wide_to_long` is more powerful than `melt` when you have multiple stubnames.
</details>

<details>
<summary><strong>41. How do you use `pd.Series.map` vs `pd.Series.apply`?</strong></summary>

**Answer:** `map` is element-wise for Series; `apply` can work on both Series and DataFrames.

```python
import pandas as pd

s = pd.Series(['cat', 'dog', 'bird', 'cat'])

# map: dictionary lookup or function
s.map({'cat': 0, 'dog': 1, 'bird': 2})     # dictionary
s.map(str.upper)                            # function

# apply: more flexible, same as map for simple cases
s.apply(str.upper)
s.apply(lambda x: x.upper() + '!')

# map vs apply for NaN
s2 = pd.Series(['cat', None, 'dog'])
s2.map({'cat': 0, 'dog': 1})               # NaN stays NaN
s2.apply(lambda x: x.upper() if x else x)  # needs None check

# apply on DataFrame
df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
df.apply(lambda col: col * 2)              # per column
df.apply(lambda row: row.sum(), axis=1)    # per row
```

**Interview Tip:** For dictionary lookups, `map` is faster than `apply`; use vectorized `.str` methods for strings.
</details>

<details>
<summary><strong>42. How do you work with JSON data in Pandas?</strong></summary>

**Answer:** Use `pd.read_json`, `json_normalize` for nested, and `explode` for arrays.

```python
import pandas as pd
from pandas import json_normalize

# Load JSON
df = pd.read_json('data.json')
df = pd.read_json('data.json', orient='records')
df = pd.read_json('data.json', lines=True)        # JSON lines format

# Nested JSON
data = [{'id':1,'name':'Alice','address':{'city':'NYC','zip':'10001'}},
        {'id':2,'name':'Bob',  'address':{'city':'LA', 'zip':'90001'}}]

df = json_normalize(data)
# id  name  address.city  address.zip

# Deeply nested
json_normalize(data, sep='_')

# JSON column in DataFrame
df['tags'] = df['tags'].apply(json.loads)   # parse JSON strings
df.explode('tags')                          # expand arrays
```

**Interview Tip:** `json_normalize` is invaluable for API responses with nested JSON structures.
</details>

<details>
<summary><strong>43. What is `inplace=True` and should you use it?</strong></summary>

**Answer:** `inplace=True` modifies the DataFrame in place and returns `None` — often discouraged in modern Pandas.

```python
import pandas as pd

df = pd.DataFrame({'A': [1, None, 3]})

# With inplace (modifies df, returns None)
df.dropna(inplace=True)

# Without inplace (returns new df, original unchanged)
df = df.dropna()

# Chaining is impossible with inplace
# df.dropna(inplace=True).fillna(0)  # AttributeError: None

# Recommended: reassign
df = (
    df
    .dropna()
    .fillna(0)
    .reset_index(drop=True)
)
```

**Why avoid inplace?**
- Breaks method chaining
- Does not guarantee lower memory use; measure the actual operation
- Mutating a derived object does not update its parent under pandas 3 Copy-on-Write

**Interview Tip:** Prefer clear reassignment for chaining; benchmark if allocation or speed is important.
</details>

<details>
<summary><strong>44. How do you handle time zones in Pandas?</strong></summary>

**Answer:** Use `tz_localize` to assign and `tz_convert` to convert time zones.

```python
import pandas as pd

df = pd.DataFrame({'event': ['login','logout'], 'time': ['2024-01-15 10:00','2024-01-15 11:30']})
df['time'] = pd.to_datetime(df['time'])

# Localize (assign timezone to naive datetime)
df['time_utc'] = df['time'].dt.tz_localize('UTC')

# Convert
df['time_ny']  = df['time_utc'].dt.tz_convert('America/New_York')
df['time_ist'] = df['time_utc'].dt.tz_convert('Asia/Kolkata')

# Strip timezone
df['time_naive'] = df['time_utc'].dt.tz_localize(None)

# Read with timezone
pd.read_csv('data.csv', parse_dates=['time'])
```

**Interview Tip:** Always store datetimes in UTC internally; convert to local time only for display.
</details>

<details>
<summary><strong>45. How do you compare DataFrames?</strong></summary>

**Answer:** Use `equals()`, `compare()`, and element-wise operators.

```python
import pandas as pd

df1 = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
df2 = pd.DataFrame({'A': [1,2,4], 'B': [4,5,7]})

# Check if identical
df1.equals(df2)        # False

# Find differences (pandas 1.1+)
df1.compare(df2)
#     A       B
#   self other self other
# 2    3     4    6     7

# Element-wise comparison
(df1 == df2)           # bool DataFrame
(df1 != df2).any().any()  # any difference

# For testing
pd.testing.assert_frame_equal(df1, df2)    # raises if not equal
pd.testing.assert_series_equal(s1, s2)
```

**Interview Tip:** `df.equals()` handles NaN correctly (NaN == NaN is True); `==` does not.
</details>

<details>
<summary><strong>46. How do you use `pd.crosstab`?</strong></summary>

**Answer:** `crosstab` computes frequency table of two or more categorical variables.

```python
import pandas as pd

df = pd.DataFrame({
    'gender':['M','F','M','F','M'],
    'dept':['Eng','Eng','HR','HR','Finance'],
    'rating':[4,5,3,4,5]
})

# Basic frequency table
pd.crosstab(df['gender'], df['dept'])

# Normalize (proportions)
pd.crosstab(df['gender'], df['dept'], normalize='index')   # row %
pd.crosstab(df['gender'], df['dept'], normalize='columns') # col %
pd.crosstab(df['gender'], df['dept'], normalize='all')     # total %

# With values and aggregation
pd.crosstab(df['gender'], df['dept'], values=df['rating'], aggfunc='mean')

# Margins (row/col totals)
pd.crosstab(df['gender'], df['dept'], margins=True)

# Multiple columns
pd.crosstab(df['gender'], [df['dept'], df['rating']])
```

**Interview Tip:** `crosstab` is a quick EDA tool; equivalent to `groupby + count + unstack`.
</details>

<details>
<summary><strong>47. How do you handle large DataFrames with chunked processing?</strong></summary>

**Answer:** Use `chunksize` in `read_csv` to process data in manageable pieces.

```python
import pandas as pd

# Process in chunks
results = []
for chunk in pd.read_csv('large.csv', chunksize=50000):
    # Process each chunk
    processed = chunk[chunk['age'] > 18].copy()
    processed['income_log'] = processed['income'].apply(lambda x: np.log1p(x))
    results.append(processed)

final = pd.concat(results, ignore_index=True)

# Aggregate across chunks without loading all
total_sales = 0
count = 0
for chunk in pd.read_csv('sales.csv', chunksize=10000):
    total_sales += chunk['amount'].sum()
    count += len(chunk)
avg = total_sales / count

# SQLite for very large data
import sqlite3
conn = sqlite3.connect(':memory:')
for chunk in pd.read_csv('large.csv', chunksize=10000):
    chunk.to_sql('data', conn, if_exists='append', index=False)
result = pd.read_sql('SELECT dept, AVG(salary) FROM data GROUP BY dept', conn)
```

**Interview Tip:** For truly large data, consider Dask or Polars; Pandas chunks work well for 100M–1B rows.
</details>

<details>
<summary><strong>48. How do you compute window functions with `groupby` + `rolling`?</strong></summary>

**Answer:** Combine `groupby` and `rolling` for group-level rolling statistics.

```python
import pandas as pd

df = pd.DataFrame({
    'store': ['A','A','A','B','B','B'],
    'month': [1,2,3,1,2,3],
    'sales': [100,120,110,200,210,190]
})

# Rolling mean per store
df['rolling_avg'] = (
    df.groupby('store')['sales']
    .transform(lambda x: x.rolling(window=2).mean())
)

# Rolling with min_periods
df['rolling_avg'] = (
    df.groupby('store')['sales']
    .transform(lambda x: x.rolling(2, min_periods=1).mean())
)

# Expanding (cumulative) per group
df['cum_avg'] = (
    df.groupby('store')['sales']
    .transform(lambda x: x.expanding().mean())
)

# Rank within group
df['rank_in_group'] = (
    df.groupby('store')['sales']
    .rank(ascending=False)
)
```

**Interview Tip:** Group + rolling is a key feature engineering pattern for time-series ML problems.
</details>

<details>
<summary><strong>49. What are common Pandas performance anti-patterns?</strong></summary>

**Answer:** Avoid loops, chained indexing, and unnecessary `apply`.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'A': range(100000), 'B': range(100000)})

# ❌ SLOW: iterrows loop
for idx, row in df.iterrows():
    df.at[idx, 'C'] = row['A'] + row['B']

# ✅ FAST: vectorized
df['C'] = df['A'] + df['B']

# ❌ SLOW: apply for arithmetic
df['D'] = df['A'].apply(lambda x: x * 2)

# ✅ FAST: vectorized
df['D'] = df['A'] * 2

# ❌ SLOW: string apply
df['name_upper'] = df['name'].apply(str.upper)

# ✅ FAST: .str accessor
df['name_upper'] = df['name'].str.upper()

# ❌ SLOW: concat in loop
result = pd.DataFrame()
for chunk in chunks:
    result = pd.concat([result, chunk])

# ✅ FAST: collect then concat
parts = [chunk for chunk in chunks]
result = pd.concat(parts, ignore_index=True)
```

**Interview Tip:** Rule of thumb — if you're writing a Python loop over Pandas rows, there's almost always a faster vectorized way.
</details>

<details>
<summary><strong>50. How does Pandas integrate with scikit-learn?</strong></summary>

**Answer:** Pandas provides data; sklearn consumes arrays — use pipelines to bridge them.

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

num_cols = X.select_dtypes(include='number').columns.tolist()
cat_cols = X.select_dtypes(include='object').columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

pipe = Pipeline([
    ('prep', preprocessor),
    ('clf',  RandomForestClassifier())
])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

**Interview Tip:** `ColumnTransformer` is the standard way to apply different preprocessing to different column types.
</details>

<details>
<summary><strong>51. How do you use `Styler` for formatting DataFrames?</strong></summary>

```python
df.style.highlight_max(color='lightgreen').highlight_min(color='red').format({'salary': '${:,.0f}'})
```
</details>

<details>
<summary><strong>52. What is `pd.Interval` and `IntervalIndex`?</strong></summary>

Used for half-open intervals, returned by `pd.cut` and useful for binning checks.
</details>

<details>
<summary><strong>53. How do you use `pd.eval` for performance?</strong></summary>

```python
df.eval('C = A + B', inplace=True)  # faster for large DataFrames
```
</details>

<details>
<summary><strong>54. What are Pandas extension types?</strong></summary>

`StringDtype`, `BooleanDtype`, `Int64Dtype` — nullable integer/bool types that distinguish NaN from 0/False.
</details>

<details>
<summary><strong>55. How do you handle sparse DataFrames?</strong></summary>

```python
df_sparse = df.astype(pd.SparseDtype('float64', fill_value=0))
```
Efficient for datasets with many zeros.
</details>

<details>
<summary><strong>56. How do you use `pd.Series.str.extract` for regex?</strong></summary>

```python
df['name'].str.extract(r'(?P<first>\w+)\s(?P<last>\w+)')
```
Returns named capture groups as columns.
</details>

<details>
<summary><strong>57. How do you use `pd.merge_asof`?</strong></summary>

Merges on nearest key — useful for time-series data alignment.
```python
pd.merge_asof(trades, quotes, on='time', by='ticker', direction='backward')
```
</details>

<details>
<summary><strong>58. How do you use `pd.factorize`?</strong></summary>

```python
codes, uniques = pd.factorize(df['category'])  # encode as integers
```
Faster than `LabelEncoder` for large arrays.
</details>

<details>
<summary><strong>59. How do you handle duplicate column names?</strong></summary>

```python
df.columns = pd.io.common.dedup_names(df.columns.tolist(), is_potential_multiindex=False)
```
</details>

<details>
<summary><strong>60. What is `pd.DataFrame.attrs`?</strong></summary>

Stores metadata on the DataFrame without affecting computations.
```python
df.attrs['source'] = 'database_v2'
df.attrs['version'] = '1.0'
```
</details>

<details>
<summary><strong>61. How do you use `pd.option_context`?</strong></summary>

```python
with pd.option_context('display.max_rows', 100, 'display.float_format', '{:.2f}'.format):
    print(df)
```
</details>

<details>
<summary><strong>62. How do you use `pd.Series.between`?</strong></summary>

```python
df[df['age'].between(25, 35)]  # inclusive both ends by default
df[df['age'].between(25, 35, inclusive='left')]
```
</details>

<details>
<summary><strong>63. How do you combine `groupby` and `cumsum`?</strong></summary>

```python
df['running_total'] = df.groupby('store')['sales'].cumsum()
```
</details>

<details>
<summary><strong>64. How do you use `pd.DataFrame.update`?</strong></summary>

Modifies a DataFrame in-place with non-NaN values from another.
```python
df1.update(df2)  # updates df1 where df2 is non-NaN
```
</details>

<details>
<summary><strong>65. How do you pivot without aggregation?</strong></summary>

```python
df.pivot(index='date', columns='variable', values='value')
# Fails if (date, variable) pairs are not unique — use pivot_table instead
```
</details>

<details>
<summary><strong>66. How do you handle memory efficiently for string columns?</strong></summary>

```python
df['text'] = df['text'].astype('string')    # pd.StringDtype
df['cat']  = df['cat'].astype('category')   # repeated strings
```
</details>

<details>
<summary><strong>67. What is `pd.DataFrame.pipe` vs `apply`?</strong></summary>

`pipe` passes the whole DataFrame; `apply` iterates over columns or rows.
</details>

<details>
<summary><strong>68. How do you use `pd.DataFrame.query` with variables?</strong></summary>

```python
threshold = 50000
df.query('salary > @threshold and age < 40')
```
</details>

<details>
<summary><strong>69. How do you reshape with `pd.DataFrame.T`?</strong></summary>

Transpose: rows become columns and vice versa.
```python
df.T   # equivalent to df.transpose()
```
</details>

<details>
<summary><strong>70. How do you detect data drift in Pandas?</strong></summary>

```python
train_stats = X_train.describe()
test_stats  = X_test.describe()
drift = (test_stats.loc['mean'] - train_stats.loc['mean']).abs() / train_stats.loc['std']
```
</details>

<details>
<summary><strong>71. How do you use `pd.DataFrame.corrwith`?</strong></summary>

```python
correlations = X.corrwith(y).abs().sort_values(ascending=False)
```
Fast feature–target correlation for feature selection.
</details>

<details>
<summary><strong>72. How do you aggregate with custom functions in `agg`?</strong></summary>

```python
df.groupby('dept').agg(
    salary_range=('salary', lambda x: x.max() - x.min()),
    top_earner=('name', lambda x: x[x.index == x['salary'].idxmax()].values[0])
)
```
</details>

<details>
<summary><strong>73. How do you use `pd.MultiIndex.from_product`?</strong></summary>

```python
idx = pd.MultiIndex.from_product([['A','B'], [2022,2023]], names=['store','year'])
df = pd.DataFrame({'sales': range(4)}, index=idx)
```
</details>

<details>
<summary><strong>74. How do you read multiple CSV files into one DataFrame?</strong></summary>

```python
import glob
files = glob.glob('data/*.csv')
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
```
</details>

<details>
<summary><strong>75. How do you use `pd.Timestamp` and `pd.Timedelta`?</strong></summary>

```python
now = pd.Timestamp.now()
one_week = pd.Timedelta(days=7)
next_week = now + one_week
```
</details>

<details>
<summary><strong>76. How do you drop columns based on dtype?</strong></summary>

```python
df.select_dtypes(include='number')           # keep only numeric
df.drop(columns=df.select_dtypes('object').columns)  # drop strings
```
</details>

<details>
<summary><strong>77. How do you find the index of the max/min per group?</strong></summary>

```python
df.groupby('dept')['salary'].idxmax()   # index of max salary per dept
df.loc[df.groupby('dept')['salary'].idxmax()]  # actual rows
```
</details>

<details>
<summary><strong>78. How do you use `pd.DataFrame.mask` for conditional replacement?</strong></summary>

```python
df['salary'] = df['salary'].mask(df['salary'] < 0, other=0)  # replace negatives with 0
```
</details>

<details>
<summary><strong>79. How do you chain `.loc` assignments without `SettingWithCopyWarning`?</strong></summary>

Always assign directly on the original:
```python
df.loc[df['age'] > 60, 'category'] = 'senior'
```
</details>

<details>
<summary><strong>80. How do you compute cross-validation folds using Pandas?</strong></summary>

```python
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(df):
    train, val = df.iloc[train_idx], df.iloc[val_idx]
```
</details>

<details>
<summary><strong>81. How do you save and load DataFrames with Pickle?</strong></summary>

```python
df.to_pickle('data.pkl')
df = pd.read_pickle('data.pkl')
# Preserves dtypes exactly; not cross-version safe
```
</details>

<details>
<summary><strong>82. How do you compute pairwise distances?</strong></summary>

```python
from scipy.spatial.distance import cdist
dist_matrix = pd.DataFrame(cdist(X, X, metric='euclidean'), index=df.index, columns=df.index)
```
</details>

<details>
<summary><strong>83. How do you use `pd.Series.rolling` with custom windows?</strong></summary>

```python
df['variable_window'] = df.apply(
    lambda row: df.loc[:row.name, 'sales'].tail(row['window']).mean(), axis=1
)
```
</details>

<details>
<summary><strong>84. How do you profile a DataFrame quickly?</strong></summary>

```python
from pandas_profiling import ProfileReport     # ydata-profiling
report = ProfileReport(df, title='EDA Report')
report.to_file('report.html')
```
</details>

<details>
<summary><strong>85. How do you ensure reproducibility in Pandas workflows?</strong></summary>

- Set `random_state` in `sample`, sklearn, etc.
- Log `df.shape`, `df.dtypes`, and `df.describe()` checkpoints
- Pin Pandas version in `requirements.txt`
</details>

<details>
<summary><strong>86. How do you use `pd.DataFrame.combine_first`?</strong></summary>

Fills NaN in df1 with values from df2:
```python
df_filled = df1.combine_first(df2)
```
</details>

<details>
<summary><strong>87. How do you compute feature importances from a model and display with Pandas?</strong></summary>

```python
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10).plot(kind='bar')
```
</details>

<details>
<summary><strong>88. How do you use `pd.DataFrame.xs` for cross-section?</strong></summary>

```python
df.xs('Senior', level='level')          # extract a level from MultiIndex
df.xs(('Eng', 'Senior'), level=('dept','level'))
```
</details>

<details>
<summary><strong>89. How do you remove leading/trailing whitespace from all string columns?</strong></summary>

```python
df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
```
</details>

<details>
<summary><strong>90. How do you check memory usage per column?</strong></summary>

```python
df.memory_usage(deep=True).sort_values(ascending=False)
(df.memory_usage(deep=True) / 1e6).round(2)  # in MB
```
</details>

<details>
<summary><strong>91. How do you use `pd.Index.difference`, `intersection`, and `union`?</strong></summary>

```python
idx1 = pd.Index(['A','B','C'])
idx2 = pd.Index(['B','C','D'])
idx1.difference(idx2)       # ['A']
idx1.intersection(idx2)     # ['B','C']
idx1.union(idx2)            # ['A','B','C','D']
```
</details>

<details>
<summary><strong>92. How do you forward-fill missing values per group?</strong></summary>

```python
df['value'] = df.groupby('group')['value'].transform(lambda x: x.ffill())
```
</details>

<details>
<summary><strong>93. How do you use Pandas with Arrow/Polars for speed?</strong></summary>

```python
import pyarrow as pa
table = pa.Table.from_pandas(df)         # convert to Arrow
df = table.to_pandas()                   # back to Pandas

import polars as pl
pl_df = pl.from_pandas(df)               # Polars for large data
df = pl_df.to_pandas()
```
</details>

<details>
<summary><strong>94. How do you handle JSON nested arrays with multiple levels?</strong></summary>

```python
from pandas import json_normalize
df = json_normalize(data, record_path=['orders','items'], meta=['user_id','order_id'])
```
</details>

<details>
<summary><strong>95. How do you create a lag feature matrix for time series ML?</strong></summary>

```python
for lag in range(1, 8):
    df[f'lag_{lag}'] = df['value'].shift(lag)
df.dropna(inplace=True)
```
</details>

<details>
<summary><strong>96. How do you aggregate percentiles in groupby?</strong></summary>

```python
df.groupby('dept')['salary'].quantile([0.25, 0.5, 0.75]).unstack()
```
</details>

<details>
<summary><strong>97. How do you detect column types automatically?</strong></summary>

```python
df.dtypes
df.select_dtypes(include='number').columns
df.select_dtypes(include=['object','category']).columns
```
</details>

<details>
<summary><strong>98. How do you convert a DataFrame to a list of dicts?</strong></summary>

```python
df.to_dict('records')   # [{'A':1,'B':4}, {'A':2,'B':5}, ...]
df.to_dict('list')      # {'A':[1,2,3], 'B':[4,5,6]}
df.to_dict('index')     # {0:{'A':1,'B':4}, ...}
```
</details>

<details>
<summary><strong>99. How do you use `pd.Timestamp` for business day calculations?</strong></summary>

```python
from pandas.tseries.offsets import BDay
today = pd.Timestamp.today()
next_bday = today + BDay(1)
five_bdays = today + BDay(5)
```
</details>

<details>
<summary><strong>100. What are the most important Pandas methods to know for ML interviews?</strong></summary>

| Category | Key Methods |
|----------|-------------|
| Loading | `read_csv`, `read_parquet`, `read_sql` |
| EDA | `info`, `describe`, `value_counts`, `isnull` |
| Selection | `loc`, `iloc`, `query`, `isin` |
| Cleaning | `dropna`, `fillna`, `drop_duplicates`, `clip` |
| Engineering | `assign`, `apply`, `map`, `cut`, `qcut` |
| Grouping | `groupby`, `agg`, `transform`, `pivot_table` |
| Joining | `merge`, `concat`, `join` |
| Time | `to_datetime`, `.dt`, `resample`, `rolling` |
| Export | `to_csv`, `to_parquet`, `to_sql` |
| Performance | `astype`, `category`, `vectorize over apply` |
</details>

---

Good luck! 🚀

## Summary in simple points

- Series and DataFrames combine labelled alignment with typed columns. Check schema, index, missingness and row counts after reading data.
- Select with loc/iloc and explicit masks. pandas 3 Copy-on-Write means a chained selection should not be relied on to mutate its parent.
- Use the intended DataFrame in a single assignment; inplace=True is not a general guarantee of lower memory use.
- Handle missing values, categories, strings, extension dtypes and conversions under a documented data contract.
- Groupby agg reduces groups, transform preserves row alignment, and apply permits more general operations with extra complexity.
- Merge with explicit keys and cardinality validation. Concat stacks compatible data; duplicate keys or columns can silently change meaning.
- Pivot, pivot_table, stack, unstack, melt/wide_to_long, explode and transpose reshape data with different uniqueness and aggregation requirements.
- Use assign and pipe for readable transformations. Prefer vectorised operations when they fit instead of repeated row loops or incremental concatenation.
- Date/time work needs timezone, frequency and availability rules. Rolling, expanding, cumulative and lag features must avoid future leakage.
- merge_asof needs sorted time keys and a direction/tolerance contract; event time alone does not establish historical availability.
- Use cut/qcut, value_counts, crosstab, correlations and grouped percentiles with correct denominators and missing-value handling.
- Deduplicate using domain identity. Detect outliers and drift with cohort-aware analysis rather than automatically deleting unusual rows.
- Sampling and cross-validation need saved IDs and suitable time/group boundaries. Fit learned transformations only on training data.
- Read large files in chunks, select columns/dtypes and inspect memory per column. Arrow/Polars integration needs type and execution checks.
- MultiIndex, xs, interval indexes and index set operations help express structured keys; preserve alignment when moving to NumPy or model inputs.
- JSON normalisation, export formats, attrs, styling and display options serve different purposes. DataFrame attributes alone are not durable lineage.
- Use compare, combine_first, update, where and mask according to their alignment/mutation contracts. Treat Pickle as trusted executable input, not a safe interchange format.
