---
title: SQL Interview Questions
sidebar_position: 36
sidebar_label: "Reference · Sql"
---

# SQL Interview Questions

:::note Foundation reference
This earlier 100-entry bank remains for prerequisite revision. Its questions were not part of the interview-source audit, and some examples are fragments rather than standalone programmes. Start with the [practical topic bank](01-python-data.md) for sourced scenarios, answered follow-ups and tested code. See [tools and versions](99-tools-versions.md) for current behaviour and [coding labs](11-coding-labs.md) for runnable implementations.
:::

100 essential SQL interview questions for data science and ML engineering roles.

---

<details>
<summary><strong>1. What is the difference between INNER JOIN, LEFT JOIN, RIGHT JOIN, and FULL OUTER JOIN?</strong></summary>

**Answer:**
- **INNER JOIN**: Returns rows with matching values in both tables
- **LEFT JOIN**: Returns all rows from left table + matched rows from right (NULL where no match)
- **RIGHT JOIN**: Returns all rows from right table + matched rows from left
- **FULL OUTER JOIN**: Returns all rows from both tables (NULL where no match on either side)

```sql
-- Setup
CREATE TABLE employees (id INT, name VARCHAR(50), dept_id INT);
CREATE TABLE departments (id INT, dept_name VARCHAR(50));

INSERT INTO employees VALUES (1,'Alice',1), (2,'Bob',2), (3,'Charlie',NULL);
INSERT INTO departments VALUES (1,'Engineering'), (2,'HR'), (3,'Finance');

-- INNER JOIN: only matched rows
SELECT e.name, d.dept_name
FROM employees e
INNER JOIN departments d ON e.dept_id = d.id;
-- Alice-Engineering, Bob-HR (Charlie excluded, Finance excluded)

-- LEFT JOIN: all employees
SELECT e.name, d.dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.id;
-- Alice-Engineering, Bob-HR, Charlie-NULL

-- RIGHT JOIN: all departments
SELECT e.name, d.dept_name
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.id;
-- Alice-Engineering, Bob-HR, NULL-Finance

-- FULL OUTER JOIN: all rows from both
SELECT e.name, d.dept_name
FROM employees e
FULL OUTER JOIN departments d ON e.dept_id = d.id;
-- Alice-Engineering, Bob-HR, Charlie-NULL, NULL-Finance
```

**Interview Tip:** LEFT JOIN is most common. FULL OUTER JOIN not supported in MySQL (use UNION of LEFT and RIGHT). Self-join: join a table to itself (e.g., employee-manager hierarchy).

</details>

<details>
<summary><strong>2. What is GROUP BY and HAVING?</strong></summary>

**Answer:**
`GROUP BY` aggregates rows sharing the same values into summary rows. `HAVING` filters groups (like WHERE but for aggregated data). WHERE filters before grouping; HAVING filters after.

```sql
CREATE TABLE sales (id INT, product VARCHAR(50), region VARCHAR(20), amount DECIMAL(10,2), sale_date DATE);

INSERT INTO sales VALUES
(1,'Widget','North',100.00,'2024-01-15'),
(2,'Gadget','South',250.00,'2024-01-20'),
(3,'Widget','North',150.00,'2024-02-10'),
(4,'Gadget','North',300.00,'2024-02-15'),
(5,'Widget','South',75.00,'2024-03-01');

-- GROUP BY with aggregation
SELECT product,
       COUNT(*) AS num_sales,
       SUM(amount) AS total_revenue,
       AVG(amount) AS avg_sale,
       MIN(amount) AS min_sale,
       MAX(amount) AS max_sale
FROM sales
GROUP BY product;

-- HAVING: filter groups (after aggregation)
SELECT product, SUM(amount) AS total
FROM sales
GROUP BY product
HAVING SUM(amount) > 300;  -- only products with >300 total

-- WHERE vs HAVING
SELECT region, COUNT(*) AS num_sales
FROM sales
WHERE amount > 100           -- filter individual rows BEFORE grouping
GROUP BY region
HAVING COUNT(*) >= 2;        -- filter groups AFTER grouping

-- GROUP BY multiple columns
SELECT product, region, SUM(amount) AS total
FROM sales
GROUP BY product, region
ORDER BY total DESC;
```

**Interview Tip:** Cannot use column aliases from SELECT in HAVING — use the expression again. GROUP BY with ROLLUP (MySQL/SQL Server) adds subtotals. ALL columns in SELECT must be in GROUP BY or be aggregated.

</details>

<details>
<summary><strong>3. What are window functions?</strong></summary>

**Answer:**
Window functions perform calculations across rows related to the current row without collapsing them into a single row. Syntax: `function() OVER (PARTITION BY ... ORDER BY ...)`.

```sql
CREATE TABLE employee_salaries (
    emp_id INT, name VARCHAR(50), dept VARCHAR(20), salary DECIMAL(10,2)
);
INSERT INTO employee_salaries VALUES
(1,'Alice','Engineering',95000),
(2,'Bob','Engineering',85000),
(3,'Charlie','HR',70000),
(4,'Diana','Engineering',105000),
(5,'Eve','HR',75000);

-- ROW_NUMBER: unique sequential number
SELECT name, dept, salary,
       ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS row_num
FROM employee_salaries;

-- RANK vs DENSE_RANK (ties handled differently)
SELECT name, salary,
       RANK() OVER (ORDER BY salary DESC) AS rank_val,       -- gaps after ties
       DENSE_RANK() OVER (ORDER BY salary DESC) AS dense_rank  -- no gaps
FROM employee_salaries;

-- Running totals and moving averages
SELECT name, dept, salary,
       SUM(salary) OVER (PARTITION BY dept ORDER BY emp_id) AS running_total,
       AVG(salary) OVER (PARTITION BY dept ORDER BY emp_id
                         ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS moving_avg_2
FROM employee_salaries;

-- LAG / LEAD: access previous/next row values
SELECT name, salary,
       LAG(salary, 1) OVER (ORDER BY emp_id) AS prev_salary,
       LEAD(salary, 1) OVER (ORDER BY emp_id) AS next_salary,
       salary - LAG(salary, 1) OVER (ORDER BY emp_id) AS salary_change
FROM employee_salaries;

-- NTILE: divide into buckets
SELECT name, salary,
       NTILE(4) OVER (ORDER BY salary) AS quartile
FROM employee_salaries;

-- FIRST_VALUE / LAST_VALUE
SELECT name, dept, salary,
       FIRST_VALUE(salary) OVER (PARTITION BY dept ORDER BY salary DESC) AS dept_max_salary
FROM employee_salaries;
```

**Interview Tip:** Window functions don't reduce row count (unlike GROUP BY). PARTITION BY is optional (global window if omitted). Frame specification: `ROWS BETWEEN n PRECEDING AND CURRENT ROW` for moving windows.

</details>

<details>
<summary><strong>4. What are subqueries and CTEs?</strong></summary>

**Answer:**
Subqueries are nested queries within another query. CTEs (Common Table Expressions) use `WITH` to create named temporary result sets — more readable and can be referenced multiple times.

```sql
-- Subquery in WHERE
SELECT name, salary
FROM employee_salaries
WHERE salary > (SELECT AVG(salary) FROM employee_salaries);

-- Correlated subquery (references outer query)
SELECT e1.name, e1.dept, e1.salary
FROM employee_salaries e1
WHERE e1.salary = (
    SELECT MAX(e2.salary)
    FROM employee_salaries e2
    WHERE e2.dept = e1.dept  -- references outer query's dept
);

-- Subquery in FROM (derived table)
SELECT dept, avg_salary
FROM (
    SELECT dept, AVG(salary) AS avg_salary
    FROM employee_salaries
    GROUP BY dept
) dept_averages
WHERE avg_salary > 80000;

-- CTE (WITH clause)
WITH dept_stats AS (
    SELECT dept,
           AVG(salary) AS avg_salary,
           COUNT(*) AS num_employees
    FROM employee_salaries
    GROUP BY dept
),
high_paying AS (
    SELECT dept FROM dept_stats WHERE avg_salary > 80000
)
SELECT e.name, e.dept, e.salary, ds.avg_salary
FROM employee_salaries e
JOIN dept_stats ds ON e.dept = ds.dept
WHERE e.dept IN (SELECT dept FROM high_paying);

-- Recursive CTE (for hierarchical data)
WITH RECURSIVE org_chart AS (
    -- Base case: CEO (no manager)
    SELECT emp_id, name, manager_id, 0 AS level
    FROM employees_hierarchy
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive: employees and their level
    SELECT e.emp_id, e.name, e.manager_id, oc.level + 1
    FROM employees_hierarchy e
    JOIN org_chart oc ON e.manager_id = oc.emp_id
)
SELECT * FROM org_chart ORDER BY level, emp_id;
```

**Interview Tip:** CTEs are preferred over subqueries for readability. CTEs can reference themselves (recursive CTEs) for hierarchical data. CTEs are evaluated once per query (unlike correlated subqueries which run per row).

</details>

<details>
<summary><strong>5. How do indexes work and when to use them?</strong></summary>

**Answer:**
Indexes are data structures (B-tree, hash) that allow fast lookups without scanning all rows. Trade-off: faster reads but slower writes, extra storage. Index on columns used in WHERE, JOIN, ORDER BY, GROUP BY.

```sql
-- Create different index types
CREATE TABLE transactions (
    id BIGINT PRIMARY KEY,
    user_id INT,
    amount DECIMAL(10,2),
    status VARCHAR(20),
    created_at TIMESTAMP,
    category VARCHAR(50)
);

-- Single column index
CREATE INDEX idx_user_id ON transactions(user_id);

-- Composite index (order matters!)
-- Good for: WHERE user_id = ? AND created_at BETWEEN ? AND ?
CREATE INDEX idx_user_date ON transactions(user_id, created_at);

-- Partial index (only index relevant rows)
CREATE INDEX idx_pending ON transactions(user_id)
WHERE status = 'pending';

-- Unique index
CREATE UNIQUE INDEX idx_unique_combo ON transactions(user_id, created_at);

-- Covering index (all needed columns in index)
CREATE INDEX idx_covering ON transactions(user_id, status, amount);
-- Query can be satisfied from index alone (no table lookup)

-- Check if query uses index
EXPLAIN SELECT * FROM transactions WHERE user_id = 123 AND created_at > '2024-01-01';

-- When NOT to use indexes:
-- 1. Small tables (full scan is faster)
-- 2. Columns with low cardinality (e.g., boolean, gender)
-- 3. Frequently updated columns
-- 4. Expressions in WHERE: WHERE YEAR(created_at) = 2024 -- can't use index on created_at!

-- Index-friendly queries
-- Use: WHERE created_at BETWEEN '2024-01-01' AND '2024-12-31'
-- Avoid: WHERE YEAR(created_at) = 2024  -- function prevents index use

-- Composite index key order
-- (a, b, c) index can serve: WHERE a=? , WHERE a=? AND b=?, WHERE a=? AND b=? AND c=?
-- Cannot serve: WHERE b=? , WHERE c=? (must start with leftmost column)
```

**Interview Tip:** B-tree is default, good for ranges and equality. Hash indexes only for exact equality (PostgreSQL). Covering index avoids "table heap" lookups. EXPLAIN/EXPLAIN ANALYZE shows query execution plan.

</details>

<details>
<summary><strong>6. What are SQL aggregate functions?</strong></summary>

**Answer:**
Aggregate functions compute a single result from multiple rows: COUNT, SUM, AVG, MIN, MAX, STDDEV, VARIANCE. Used with GROUP BY or as window functions.

```sql
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT user_id) AS unique_users,    -- count distinct values
    COUNT(amount) AS non_null_amounts,           -- excludes NULLs
    SUM(amount) AS total_revenue,
    AVG(amount) AS avg_sale,
    MIN(amount) AS min_sale,
    MAX(amount) AS max_sale,
    STDDEV(amount) AS std_dev,
    VARIANCE(amount) AS variance,
    -- Percentiles (PostgreSQL)
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) AS median,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount) AS p95
FROM sales;

-- Conditional aggregation with CASE WHEN
SELECT
    COUNT(*) AS total,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed,
    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending,
    AVG(CASE WHEN region = 'North' THEN amount END) AS north_avg
FROM sales;

-- Filter aggregation with FILTER (PostgreSQL) or CASE
SELECT
    AVG(amount) FILTER (WHERE region = 'North') AS north_avg,
    AVG(amount) FILTER (WHERE region = 'South') AS south_avg
FROM sales;

-- GROUP_CONCAT / STRING_AGG: aggregate strings
-- MySQL:
SELECT dept, GROUP_CONCAT(name ORDER BY name SEPARATOR ', ') AS members
FROM employee_salaries GROUP BY dept;

-- PostgreSQL:
SELECT dept, STRING_AGG(name, ', ' ORDER BY name) AS members
FROM employee_salaries GROUP BY dept;
```

**Interview Tip:** COUNT(*) counts all rows including NULLs; COUNT(column) excludes NULLs. SUM/AVG also ignore NULLs. DISTINCT can be used with any aggregate: SUM(DISTINCT amount).

</details>

<details>
<summary><strong>7. What are SQL data types and when to use them?</strong></summary>

**Answer:**
Choosing the right data type impacts storage, performance, and data integrity. Key categories: numeric, string, datetime, boolean, and JSON.

```sql
-- Numeric types
CREATE TABLE data_types_demo (
    -- Integers
    small_num SMALLINT,        -- -32768 to 32767 (2 bytes)
    normal_num INT,             -- -2B to 2B (4 bytes)
    big_num BIGINT,             -- -9.2E18 to 9.2E18 (8 bytes)

    -- Exact decimal (no floating point errors)
    price DECIMAL(10, 2),      -- 10 total digits, 2 after decimal
    percentage NUMERIC(5, 4),  -- 99.9999%

    -- Floating point (fast but imprecise)
    measurement FLOAT,         -- 4 bytes, ~7 decimal digits
    precise_float DOUBLE PRECISION,  -- 8 bytes, ~15 decimal digits

    -- Strings
    fixed_code CHAR(10),       -- fixed length, padded with spaces
    name VARCHAR(255),          -- variable length, efficient
    description TEXT,           -- unlimited length

    -- Dates and times
    birth_date DATE,            -- 'YYYY-MM-DD'
    event_time TIME,            -- 'HH:MM:SS'
    created_at TIMESTAMP,       -- 'YYYY-MM-DD HH:MM:SS'
    updated_at TIMESTAMPTZ,     -- with timezone (PostgreSQL)

    -- Boolean
    is_active BOOLEAN,          -- TRUE/FALSE

    -- JSON (PostgreSQL)
    metadata JSONB,             -- binary JSON, indexable

    -- UUID
    user_uuid UUID DEFAULT gen_random_uuid()
);

-- Common pitfalls
-- Use DECIMAL for money, not FLOAT (0.1 + 0.2 != 0.3 in float)
-- Use TIMESTAMP WITH TIME ZONE for cross-timezone applications
-- CHAR vs VARCHAR: use VARCHAR unless truly fixed-length

-- Date operations
SELECT
    CURRENT_DATE,
    CURRENT_TIMESTAMP,
    DATE_TRUNC('month', CURRENT_DATE),     -- first day of month
    EXTRACT(YEAR FROM created_at),          -- extract year
    created_at + INTERVAL '30 days',        -- date arithmetic
    AGE(CURRENT_DATE, birth_date)           -- PostgreSQL: age
FROM data_types_demo LIMIT 1;
```

**Interview Tip:** INT vs BIGINT: use BIGINT for IDs in large systems (INT maxes at 2.1B). DECIMAL for financial data. Use appropriate VARCHAR length — doesn't waste space but does document intent. JSONB in PostgreSQL is indexable and faster than JSON.

</details>

<details>
<summary><strong>8. What is normalization and denormalization?</strong></summary>

**Answer:**
Normalization reduces redundancy by organizing data into related tables. 1NF: atomic values. 2NF: no partial dependencies. 3NF: no transitive dependencies. Denormalization adds redundancy for query performance.

```sql
-- Unnormalized (1NF violation: repeated groups)
CREATE TABLE orders_unnorm (
    order_id INT, customer_name VARCHAR(100), customer_email VARCHAR(100),
    item1 VARCHAR(100), item1_qty INT,
    item2 VARCHAR(100), item2_qty INT  -- repeating groups!
);

-- 1NF: Atomic values, no repeating groups
CREATE TABLE orders_1nf (
    order_id INT, customer_id INT, item_name VARCHAR(100), quantity INT,
    PRIMARY KEY (order_id, item_name)
);

-- 2NF: Remove partial dependency (item_price depends only on item_name, not full PK)
CREATE TABLE orders (order_id INT, customer_id INT, PRIMARY KEY (order_id));
CREATE TABLE order_items (
    order_id INT, item_id INT, quantity INT,
    PRIMARY KEY (order_id, item_id)
);
CREATE TABLE items (item_id INT PRIMARY KEY, item_name VARCHAR(100), price DECIMAL(10,2));

-- 3NF: Remove transitive dependency
-- If customers table has customer_id -> city_id -> city_name,
-- separate city into its own table
CREATE TABLE customers (customer_id INT PRIMARY KEY, name VARCHAR(100), city_id INT);
CREATE TABLE cities (city_id INT PRIMARY KEY, city_name VARCHAR(100), country VARCHAR(50));

-- Denormalization for performance
-- Instead of joining orders -> order_items -> items,
-- store frequently queried data redundantly
CREATE TABLE order_items_denorm (
    order_id INT, item_id INT, quantity INT,
    item_name VARCHAR(100),   -- redundant but avoids join
    unit_price DECIMAL(10,2), -- redundant but captures price at time of sale
    PRIMARY KEY (order_id, item_id)
);

-- Denormalization also includes:
-- Materialized views (precomputed aggregations)
-- Summary tables (pre-aggregated data)
-- Array/JSON columns for one-to-many (PostgreSQL)
```

**Interview Tip:** OLTP systems: normalize for data integrity, fewer anomalies. OLAP/data warehouse: denormalize (star/snowflake schema) for query performance. 3NF is typical goal for production OLTP. Knowing when to denormalize is key.

</details>

<details>
<summary><strong>9. What are transactions and ACID properties?</strong></summary>

**Answer:**
Transactions group SQL operations that must succeed or fail together. ACID: Atomicity (all or nothing), Consistency (valid state to valid state), Isolation (concurrent transactions don't interfere), Durability (committed data persists).

```sql
-- Basic transaction
BEGIN;

UPDATE accounts SET balance = balance - 100 WHERE account_id = 1;
UPDATE accounts SET balance = balance + 100 WHERE account_id = 2;

-- If any statement fails, rollback all changes
-- If all succeed, commit
COMMIT;

-- With error handling (PostgreSQL)
DO $$
BEGIN
    BEGIN;
    UPDATE accounts SET balance = balance - 100 WHERE account_id = 1;

    IF (SELECT balance FROM accounts WHERE account_id = 1) < 0 THEN
        RAISE EXCEPTION 'Insufficient funds';
    END IF;

    UPDATE accounts SET balance = balance + 100 WHERE account_id = 2;
    COMMIT;
EXCEPTION WHEN OTHERS THEN
    ROLLBACK;
    RAISE;
END $$;

-- Savepoints: partial rollback
BEGIN;
INSERT INTO orders (id, amount) VALUES (1, 100);
SAVEPOINT sp1;
INSERT INTO orders (id, amount) VALUES (2, 200);  -- might fail
ROLLBACK TO SAVEPOINT sp1;  -- undo only to savepoint
COMMIT;  -- first insert committed

-- Isolation levels
-- Read Uncommitted: can see dirty reads (uncommitted changes)
-- Read Committed: only see committed data (default in most DBs)
-- Repeatable Read: same row reads same value in transaction
-- Serializable: transactions appear serial (strongest, slowest)

SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
BEGIN;
SELECT balance FROM accounts WHERE id = 1;  -- reads committed data
-- ...
COMMIT;

-- Phenomena by isolation level:
-- Dirty read: Read Uncommitted only
-- Non-repeatable read: avoided from Repeatable Read up
-- Phantom read: avoided only at Serializable
```

**Interview Tip:** Durability is ensured by write-ahead logging (WAL). Isolation levels trade consistency for performance. Deadlocks occur when two transactions wait for each other — databases detect and kill one. MVCC (Multi-Version Concurrency Control) enables high isolation without blocking reads.

</details>

<details>
<summary><strong>10. What are SQL string functions?</strong></summary>

**Answer:**
String functions manipulate text data: concatenation, substring, pattern matching, case conversion, trimming, and regex.

```sql
-- Basic string operations
SELECT
    UPPER('hello world') AS upper_case,        -- 'HELLO WORLD'
    LOWER('Hello World') AS lower_case,        -- 'hello world'
    LENGTH('hello') AS len,                    -- 5
    TRIM('  hello  ') AS trimmed,              -- 'hello'
    LTRIM('  hello  ') AS left_trimmed,        -- 'hello  '
    RTRIM('  hello  ') AS right_trimmed,       -- '  hello'

    -- Concatenation
    CONCAT('hello', ' ', 'world') AS concat_op,  -- 'hello world'
    'hello' || ' ' || 'world' AS pipe_concat,     -- PostgreSQL

    -- Substring
    SUBSTRING('hello world' FROM 7 FOR 5) AS substr_op,  -- 'world'
    LEFT('hello world', 5) AS left_op,           -- 'hello'
    RIGHT('hello world', 5) AS right_op,          -- 'world'

    -- Find and replace
    POSITION('world' IN 'hello world') AS pos,   -- 7
    REPLACE('hello world', 'world', 'SQL') AS replaced,  -- 'hello SQL'

    -- Padding
    LPAD('42', 5, '0') AS left_padded,           -- '00042'
    RPAD('hello', 10, '.') AS right_padded,       -- 'hello.....'

    -- Split (PostgreSQL)
    SPLIT_PART('a,b,c', ',', 2) AS split_part_op; -- 'b'

-- Pattern matching
SELECT name FROM employees
WHERE name LIKE 'A%'       -- starts with A
   OR name LIKE '%son'     -- ends with son
   OR name LIKE '_ohn'     -- 4 chars, ends with ohn
   OR name ILIKE '%alice%'; -- case-insensitive (PostgreSQL)

-- Regex (PostgreSQL)
SELECT name FROM employees
WHERE name ~ '^[A-Z][a-z]+$'   -- capital first, lowercase rest
   OR name ~* 'alice';          -- case-insensitive regex

-- String aggregation
SELECT dept, STRING_AGG(name, ', ' ORDER BY name) AS team_members
FROM employee_salaries GROUP BY dept;

-- Extract from strings
SELECT
    REGEXP_REPLACE(phone, '[^0-9]', '', 'g') AS digits_only,
    REGEXP_MATCH(email, '([^@]+)@(.+)') AS email_parts
FROM users;
```

**Interview Tip:** LIKE is case-sensitive in most databases. Use ILIKE (PostgreSQL) or LOWER(col) LIKE lower_pattern for case-insensitive. Regex is powerful but may not use indexes — prefer LIKE with leading % only if there's no leading wildcard.

</details>

<details>
<summary><strong>11. What is DISTINCT and when to use it?</strong></summary>

DISTINCT eliminates duplicate rows from result set. SELECT DISTINCT col: unique values. SELECT DISTINCT col1, col2: unique combinations. Alternative: GROUP BY for aggregations. DISTINCT is slower than GROUP BY for counting but simpler for listing unique values.
</details>

<details>
<summary><strong>12. What is ORDER BY and NULL handling?</strong></summary>

ORDER BY sorts result set. Default ASC. NULLS FIRST/LAST controls null position. Multiple columns: ORDER BY col1 ASC, col2 DESC. Cannot use column position in parameterized queries. ORDER BY is applied after HAVING and WHERE.
</details>

<details>
<summary><strong>13. What is LIMIT/OFFSET for pagination?</strong></summary>

LIMIT n: return first n rows. OFFSET k: skip k rows. LIMIT 10 OFFSET 20: rows 21-30. Performance degrades for large offsets (must scan all previous rows). Use keyset pagination (WHERE id > last_seen_id LIMIT n) for better performance.
</details>

<details>
<summary><strong>14. What is the difference between WHERE and HAVING?</strong></summary>

WHERE filters rows before aggregation; cannot use aggregate functions. HAVING filters groups after aggregation; can use aggregate functions. WHERE is always faster when possible since it reduces rows before grouping.
</details>

<details>
<summary><strong>15. What is a self-join?</strong></summary>

A table joined to itself. Common use cases: employee-manager relationships, finding pairs with relationships. Requires aliases for the two instances.

```sql
SELECT e.name AS employee, m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;
```
</details>

<details>
<summary><strong>16. What is UNION vs UNION ALL?</strong></summary>

UNION: combines results from two queries, removes duplicates (slower). UNION ALL: combines results, keeps duplicates (faster). Both require same number of columns with compatible types. Use UNION ALL when duplicates don't matter or are impossible.
</details>

<details>
<summary><strong>17. What is INTERSECT and EXCEPT?</strong></summary>

INTERSECT: rows in both queries (set intersection). EXCEPT (or MINUS in Oracle): rows in first query but not second. Less commonly used than UNION. INTERSECT ALL and EXCEPT ALL preserve duplicates.
</details>

<details>
<summary><strong>18. What is a view?</strong></summary>

A virtual table defined by a query. Stored query, not stored data (unless materialized). Provides abstraction, security (limit exposed columns), and simplifies complex queries. Updated through underlying tables (updatable views have restrictions).
</details>

<details>
<summary><strong>19. What is a materialized view?</strong></summary>

Stores the result of a query physically. Faster reads (no query recomputation), but must be refreshed when underlying data changes. REFRESH MATERIALIZED VIEW (PostgreSQL). Excellent for expensive aggregation queries.
</details>

<details>
<summary><strong>20. What are stored procedures and functions?</strong></summary>

Stored procedures: server-side programs that encapsulate logic, can have side effects (INSERT/UPDATE), called with EXECUTE/CALL. Functions: return a value, can be used in SELECT, typically no side effects. More portable in PostgreSQL.
</details>

<details>
<summary><strong>21. What is a trigger?</strong></summary>

SQL code that automatically executes in response to INSERT, UPDATE, or DELETE events on a table. BEFORE or AFTER the event. Use cases: audit logging, data validation, automatic updates (e.g., updated_at timestamp).

```sql
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_timestamp
BEFORE UPDATE ON employees
FOR EACH ROW EXECUTE FUNCTION update_timestamp();
```
</details>

<details>
<summary><strong>22. What is a foreign key constraint?</strong></summary>

Enforces referential integrity: value in child table must exist in parent table. ON DELETE CASCADE: delete child rows when parent deleted. ON DELETE SET NULL: set FK to NULL. ON DELETE RESTRICT: prevent parent deletion if children exist.
</details>

<details>
<summary><strong>23. What is NULL handling in SQL?</strong></summary>

NULL represents unknown/missing value. NULL != NULL (use IS NULL/IS NOT NULL). Arithmetic with NULL returns NULL. COALESCE(a, b, c): returns first non-NULL. NULLIF(a, b): returns NULL if a=b, else a. NULL is excluded from aggregates (COUNT(*) vs COUNT(col)).
</details>

<details>
<summary><strong>24. What is CASE WHEN?</strong></summary>

Conditional expression in SQL. Simple CASE: compares expression to values. Searched CASE: evaluates boolean conditions. Can be used in SELECT, WHERE, ORDER BY, GROUP BY. CASE WHEN condition THEN value ... ELSE default END.

```sql
SELECT name, salary,
       CASE
           WHEN salary > 100000 THEN 'Senior'
           WHEN salary > 70000 THEN 'Mid-level'
           ELSE 'Junior'
       END AS level
FROM employee_salaries;
```
</details>

<details>
<summary><strong>25. What is COALESCE and NULLIF?</strong></summary>

COALESCE(a,b,c,...): returns first non-NULL argument. Useful for default values. NULLIF(x, y): returns NULL if x=y (avoids division by zero: NULLIF(denominator, 0)). Both are standard SQL.
</details>

<details>
<summary><strong>26. What is CAST and type conversion?</strong></summary>

CAST(expression AS type): converts data type. Type-specific: ::type in PostgreSQL (CAST shorthand). Implicit conversion can cause performance issues (prevents index use). Always be explicit when types might mismatch.
</details>

<details>
<summary><strong>27. What is an EXPLAIN plan?</strong></summary>

Shows how the database executes a query: table scan vs index scan, join algorithms, row estimates, cost. EXPLAIN ANALYZE actually runs the query and shows real timing. Use to identify slow operations.
</details>

<details>
<summary><strong>28. What is query optimization?</strong></summary>

Improve query performance: use indexes, avoid SELECT *, avoid functions on indexed columns in WHERE, use EXISTS instead of IN for subqueries, prefer JOINs over correlated subqueries, use LIMIT, batch large operations.
</details>

<details>
<summary><strong>29. What is the N+1 query problem?</strong></summary>

Fetching 1 parent record then N queries for children. Solution: JOIN to fetch everything in one query, or use IN clause. Classic ORM problem. Causes: 1 query for list + N queries for each related record.
</details>

<details>
<summary><strong>30. What is partitioning in SQL?</strong></summary>

Divides large tables into smaller physical partitions while appearing as one table. Range partitioning: by date ranges. Hash partitioning: by hash of column. List partitioning: by specific values. Improves query performance and data management.
</details>

<details>
<summary><strong>31. What is sharding?</strong></summary>

Horizontal partitioning across multiple database servers. Each shard is a subset of data. Enables scale-out for very large datasets. Challenges: cross-shard joins, transactions, rebalancing. Unlike replication (same data on multiple servers).
</details>

<details>
<summary><strong>32. What is database replication?</strong></summary>

Copying data to one or more replica servers. Primary-replica (master-slave): writes to primary, reads from replicas. Used for: high availability, read scaling, disaster recovery. Replication lag: replicas may be slightly behind primary.
</details>

<details>
<summary><strong>33. What is VACUUM in PostgreSQL?</strong></summary>

Reclaims storage from dead tuples (rows updated/deleted). AUTOVACUUM runs automatically. VACUUM FULL: rewrites table, takes exclusive lock. ANALYZE: updates query planner statistics. Run VACUUM ANALYZE after bulk data changes.
</details>

<details>
<summary><strong>34. What is the difference between DELETE, TRUNCATE, and DROP?</strong></summary>

DELETE: removes rows, transaction-safe, fires triggers, slow for large tables. TRUNCATE: removes all rows, fast, not transaction-safe in all DBs, resets sequences. DROP: removes the table entirely. TRUNCATE cannot be rolled back in MySQL.
</details>

<details>
<summary><strong>35. What is WITH TIES in FETCH FIRST?</strong></summary>

FETCH FIRST n ROWS WITH TIES: includes additional rows that tie on the ORDER BY column with the last included row. Useful when you want all rows at the boundary, not just arbitrary n rows.
</details>

<details>
<summary><strong>36. What are lateral joins?</strong></summary>

LATERAL allows a subquery/function in FROM to reference columns from preceding tables. Like a correlated subquery in FROM. PostgreSQL: LATERAL, SQL Server: CROSS APPLY/OUTER APPLY.
</details>

<details>
<summary><strong>37. What is full-text search in SQL?</strong></summary>

PostgreSQL: tsvector + tsquery for lexeme-based search with ranking. MySQL: FULLTEXT indexes with MATCH...AGAINST. More efficient than LIKE '%text%' which can't use indexes. Supports stemming, stop words, ranking by relevance.
</details>

<details>
<summary><strong>38. What is JSONB in PostgreSQL?</strong></summary>

Binary JSON column type. Supports GIN indexing for fast key lookups. Operators: -> (get key), ->> (get as text), @> (contains), ? (key exists). More efficient than JSON for querying. Essential for semi-structured data.

```sql
SELECT data->>'name' AS name,
       data->'address'->>'city' AS city
FROM users_jsonb
WHERE data @> '{"status": "active"}'::jsonb;
```
</details>

<details>
<summary><strong>39. What is GENERATE_SERIES?</strong></summary>

PostgreSQL function that generates a series of values. Useful for: creating date ranges, filling gaps in time series, testing. GENERATE_SERIES(start, stop, step).
</details>

<details>
<summary><strong>40. What is WINDOW FRAME specification?</strong></summary>

Controls which rows are included in window function calculation. ROWS vs RANGE. UNBOUNDED PRECEDING: from start. CURRENT ROW: current row. UNBOUNDED FOLLOWING: to end. Default: RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW.
</details>

<details>
<summary><strong>41. What are analytic functions in SQL?</strong></summary>

Another name for window functions. Compute across a "window" of rows related to current row. Types: ranking (ROW_NUMBER, RANK, DENSE_RANK, NTILE), distribution (PERCENT_RANK, CUME_DIST), value (LAG, LEAD, FIRST_VALUE, LAST_VALUE), aggregate (SUM, AVG over window).
</details>

<details>
<summary><strong>42. How do you find duplicate records?</strong></summary>

Group by columns that should be unique, count, and filter where count > 1. Then use ROW_NUMBER() to keep one and delete the rest.

```sql
-- Find duplicates
SELECT email, COUNT(*) as cnt FROM users GROUP BY email HAVING COUNT(*) > 1;

-- Delete duplicates, keep lowest id
DELETE FROM users WHERE id NOT IN (
    SELECT MIN(id) FROM users GROUP BY email
);
```
</details>

<details>
<summary><strong>43. How do you find the nth highest salary?</strong></summary>

Multiple approaches: subquery with DISTINCT + LIMIT, DENSE_RANK(), or window function.

```sql
-- Nth highest salary (n=2)
SELECT DISTINCT salary FROM employee_salaries
ORDER BY salary DESC LIMIT 1 OFFSET 1;

-- Using DENSE_RANK
SELECT salary FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
    FROM employee_salaries
) ranked WHERE rnk = 2;
```
</details>

<details>
<summary><strong>44. How do you pivot data in SQL?</strong></summary>

Transform rows to columns. CASE WHEN conditional aggregation is the standard approach. PostgreSQL has CROSSTAB. SQL Server has PIVOT operator.

```sql
SELECT product,
    SUM(CASE WHEN region = 'North' THEN amount ELSE 0 END) AS north_total,
    SUM(CASE WHEN region = 'South' THEN amount ELSE 0 END) AS south_total
FROM sales GROUP BY product;
```
</details>

<details>
<summary><strong>45. What is the difference between EXISTS and IN?</strong></summary>

EXISTS: short-circuits on first match, efficient for large subqueries. IN: evaluates all values, can be slow with large sets, doesn't handle NULLs well (NOT IN with NULLs returns empty). Use EXISTS for correlated subqueries and large sets.
</details>

<details>
<summary><strong>46. What is a covering index?</strong></summary>

An index that contains all columns referenced in a query (SELECT and WHERE). The database can satisfy the query entirely from the index without accessing the main table (index-only scan). Dramatically reduces I/O.
</details>

<details>
<summary><strong>47. What is index cardinality?</strong></summary>

Number of distinct values in an index. High cardinality (user_id): very selective, indexes help. Low cardinality (gender, boolean): not selective, index often not used. The query optimizer considers cardinality when choosing indexes.
</details>

<details>
<summary><strong>48. What is a clustered vs non-clustered index?</strong></summary>

Clustered index: determines physical order of data rows — one per table (primary key by default). Non-clustered index: separate structure pointing to row locations — multiple allowed. InnoDB (MySQL): primary key is always clustered. Non-clustered index lookup: two reads (index + row).
</details>

<details>
<summary><strong>49. What is query execution order?</strong></summary>

FROM -> JOIN -> WHERE -> GROUP BY -> HAVING -> SELECT -> DISTINCT -> ORDER BY -> LIMIT. Understanding this order explains why you can't use SELECT aliases in WHERE (WHERE runs before SELECT).
</details>

<details>
<summary><strong>50. What is the difference between CHAR and VARCHAR?</strong></summary>

CHAR(n): fixed length, always n bytes, padded with spaces — fast for fixed-length codes. VARCHAR(n): variable length, stores only actual length + 2 bytes overhead — efficient for variable data. CHAR comparison ignores trailing spaces.
</details>

<details>
<summary><strong>51. What is a database schema?</strong></summary>

Logical structure of a database: tables, views, indexes, stored procedures, constraints. Separates objects into namespaces (PostgreSQL schemas: public, private). Multiple schemas in one database for organization and security.
</details>

<details>
<summary><strong>52. What is referential integrity?</strong></summary>

Ensures that foreign key values reference existing primary key values. Prevents orphaned records. Implemented via FOREIGN KEY constraints. Must be considered when inserting/deleting related records.
</details>

<details>
<summary><strong>53. What is optimistic vs pessimistic locking?</strong></summary>

Pessimistic: lock rows when reading (SELECT FOR UPDATE) — prevents conflicts, reduces concurrency. Optimistic: no locks, check for conflicts at commit (version column) — better concurrency, handles conflicts at write time. MVCC is a form of optimistic concurrency.
</details>

<details>
<summary><strong>54. What is SELECT FOR UPDATE?</strong></summary>

Locks selected rows for the duration of the transaction, preventing other transactions from modifying or locking them. Used when you read-then-write and need to prevent race conditions. SKIP LOCKED: skip rows locked by others (queue processing pattern).
</details>

<details>
<summary><strong>55. What is a deadlock?</strong></summary>

Two transactions wait for each other to release locks. Database detects and kills one transaction. Prevention: acquire locks in consistent order, minimize transaction duration, keep transactions short. SHOW ENGINE INNODB STATUS (MySQL) for deadlock details.
</details>

<details>
<summary><strong>56. What are database constraints?</strong></summary>

NOT NULL, UNIQUE, PRIMARY KEY, FOREIGN KEY, CHECK, DEFAULT. Enforce data integrity at database level (not just application). CHECK constraint: custom condition (CHECK (salary > 0)). Constraints enforced on insert and update.
</details>

<details>
<summary><strong>57. What is a composite primary key?</strong></summary>

Primary key made of multiple columns. Used when no single column uniquely identifies a row. Common in junction/association tables (user_id, product_id). Can impact join performance and foreign key references.
</details>

<details>
<summary><strong>58. What is an auto-increment / sequence?</strong></summary>

Automatically generates unique sequential integers for primary keys. MySQL: AUTO_INCREMENT. PostgreSQL: SERIAL or SEQUENCE. Best practice: use BIGINT for large tables to avoid overflow.
</details>

<details>
<summary><strong>59. What is the difference between SQL and NoSQL?</strong></summary>

SQL: structured, ACID compliant, schema-on-write, vertical scaling, joins supported. NoSQL: flexible schema, horizontal scaling, eventual consistency, various data models (document, key-value, graph, column-family). Use SQL for transactions, NoSQL for scale/flexibility.
</details>

<details>
<summary><strong>60. What is CAP theorem?</strong></summary>

In a distributed system, you can only guarantee two of: Consistency (all nodes see same data), Availability (every request gets a response), Partition tolerance (system works despite network partitions). SQL DBs: CA (single server), distributed SQL: CP. NoSQL often AP.
</details>

<details>
<summary><strong>61. What is eventual consistency?</strong></summary>

Data will be consistent across all nodes eventually, but not necessarily immediately. Used in distributed NoSQL systems. Trade-off: higher availability and performance at the cost of temporary inconsistency.
</details>

<details>
<summary><strong>62. How do you handle hierarchical data in SQL?</strong></summary>

Adjacency list: self-referencing FK (simple, recursive CTEs for traversal). Nested sets: left/right values representing tree structure (fast reads, complex updates). Closure table: all ancestor-descendant pairs (flexible, extra storage).
</details>

<details>
<summary><strong>63. What is a star schema?</strong></summary>

Data warehouse design: central fact table (sales, orders) surrounded by dimension tables (product, customer, time). Few joins needed for typical analytical queries. Denormalized for query performance. Contrast with snowflake schema (normalized dimensions).
</details>

<details>
<summary><strong>64. What is a snowflake schema?</strong></summary>

Extension of star schema where dimension tables are normalized (split into multiple tables). Saves storage, more complex queries. Less common than star schema in practice due to query complexity.
</details>

<details>
<summary><strong>65. What is OLTP vs OLAP?</strong></summary>

OLTP (Online Transaction Processing): many small, fast reads/writes, normalized, ACID, row-oriented. OLAP (Online Analytical Processing): complex analytical queries, large data scans, denormalized, column-oriented (Redshift, BigQuery, Snowflake).
</details>

<details>
<summary><strong>66. What is columnar storage?</strong></summary>

Store data column-by-column instead of row-by-row. Excellent for analytical queries that read few columns across many rows. Better compression (same type values together). Used in: Redshift, BigQuery, Parquet format, ClickHouse.
</details>

<details>
<summary><strong>67. What is partitioning pruning?</strong></summary>

Query optimizer eliminates partitions that can't contain relevant data based on WHERE clause. Requires partition key in WHERE. Dramatically speeds up queries on partitioned tables when querying a specific date range.
</details>

<details>
<summary><strong>68. What is the difference between a hash join and a nested loop join?</strong></summary>

Hash join: build hash table from smaller table, probe with larger — good for large tables without indexes. Nested loop: for each row in outer, scan inner — good when inner is small or indexed. Merge join: both sorted, merged together — good for sorted data.
</details>

<details>
<summary><strong>69. What is a database connection pool?</strong></summary>

Maintaining a pool of pre-established database connections. Applications borrow connections from pool instead of creating new ones each request. Reduces connection overhead. Pool size tuning: too small = waiting, too large = resource exhaustion.
</details>

<details>
<summary><strong>70. What is READ COMMITTED isolation level?</strong></summary>

Most common default (PostgreSQL, SQL Server). Each query within a transaction sees a snapshot of committed data at query start time. Avoids dirty reads. Allows non-repeatable reads: same query within transaction may return different results.
</details>

<details>
<summary><strong>71. What are window function frames?</strong></summary>

ROWS: based on physical row position. RANGE: based on value range relative to current row. GROUPS: based on peer groups. ROWS BETWEEN 2 PRECEDING AND CURRENT ROW: fixed 3-row window. RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW: value-based window.
</details>

<details>
<summary><strong>72. How do you find records in one table not in another?</strong></summary>

Use LEFT JOIN with IS NULL, or NOT EXISTS, or NOT IN. LEFT JOIN IS NULL is often fastest.

```sql
-- LEFT JOIN approach (generally fastest)
SELECT a.id FROM table_a a
LEFT JOIN table_b b ON a.id = b.id
WHERE b.id IS NULL;

-- NOT EXISTS
SELECT id FROM table_a a WHERE NOT EXISTS (SELECT 1 FROM table_b WHERE id = a.id);
```
</details>

<details>
<summary><strong>73. What is MERGE statement (UPSERT)?</strong></summary>

MERGE/UPSERT: insert if not exists, update if exists. PostgreSQL: INSERT ... ON CONFLICT DO UPDATE. MySQL: INSERT ... ON DUPLICATE KEY UPDATE. SQL Server: MERGE statement.

```sql
-- PostgreSQL UPSERT
INSERT INTO products (id, name, price) VALUES (1, 'Widget', 9.99)
ON CONFLICT (id) DO UPDATE SET price = EXCLUDED.price, updated_at = NOW();
```
</details>

<details>
<summary><strong>74. What is the OVER() clause?</strong></summary>

Defines the window for window functions. Empty OVER(): entire result set. PARTITION BY: reset window per group. ORDER BY: sort within window. Frame spec: subset of partition. Multiple OVER() clauses can have different windows in same query.
</details>

<details>
<summary><strong>75. What is a subquery vs JOIN performance?</strong></summary>

JOINs are generally faster than correlated subqueries (run once vs per row). Non-correlated subqueries (run once) can be optimized by the query planner similarly to JOINs. Use EXPLAIN to compare. Modern optimizers often rewrite subqueries as joins automatically.
</details>

<details>
<summary><strong>76. What are common SQL performance anti-patterns?</strong></summary>

SELECT * (fetches unused columns), functions on indexed columns in WHERE, unnecessary DISTINCT, correlated subqueries in SELECT, OR conditions preventing index use, implicit type conversions, not using bind parameters (prevents plan caching).
</details>

<details>
<summary><strong>77. What is a recursive query and when to use it?</strong></summary>

Recursive CTEs solve hierarchical problems: org charts, bill of materials, graph traversal, path finding. Base case + recursive case. Requires UNION ALL between base and recursive parts. Max recursion depth can be set.
</details>

<details>
<summary><strong>78. What is temporal data handling in SQL?</strong></summary>

Bitemporal data: valid time (when fact was true in world) and transaction time (when recorded in DB). SQL:2011 standard: FOR SYSTEM_TIME, PERIOD. PostgreSQL: temporal_tables extension. Common pattern: effective_from/effective_to columns.
</details>

<details>
<summary><strong>79. How do you calculate a running total?</strong></summary>

Window function with SUM and ORDER BY. Default frame is UNBOUNDED PRECEDING to CURRENT ROW.

```sql
SELECT date, amount,
    SUM(amount) OVER (ORDER BY date) AS running_total
FROM transactions;
```
</details>

<details>
<summary><strong>80. What is the GROUPING SETS / CUBE / ROLLUP extension?</strong></summary>

ROLLUP: subtotals for hierarchical groupings. CUBE: all combinations of groupings. GROUPING SETS: specific combinations. More flexible than multiple UNION ALL queries for subtotals.

```sql
SELECT region, product, SUM(amount)
FROM sales
GROUP BY ROLLUP(region, product);  -- adds region subtotals and grand total
```
</details>

<details>
<summary><strong>81. How do you generate sequences of dates?</strong></summary>

PostgreSQL: GENERATE_SERIES with timestamps. Useful for filling gaps in time series data. LEFT JOIN generated series with actual data to show zeros for missing dates.

```sql
SELECT d.dt, COALESCE(SUM(s.amount), 0) AS daily_total
FROM GENERATE_SERIES('2024-01-01'::date, '2024-12-31'::date, '1 day') d(dt)
LEFT JOIN sales s ON s.sale_date = d.dt
GROUP BY d.dt ORDER BY d.dt;
```
</details>

<details>
<summary><strong>82. What is query hints?</strong></summary>

Database-specific directives to override optimizer decisions: force index use, join order, parallelism. SQL Server: WITH (INDEX(...)), NOLOCK. PostgreSQL: SET enable_seqscan = off. Oracle: /*+ INDEX(...) */. Use as last resort after understanding optimizer.
</details>

<details>
<summary><strong>83. What is a materialized path for hierarchies?</strong></summary>

Store full path from root to node as string: "/1/2/5/". LIKE '/1/%' finds all descendants of node 1. Easy to find ancestors and descendants. Complex updates (re-rooting). Used in file systems, categories.
</details>

<details>
<summary><strong>84. What are aggregate window functions?</strong></summary>

Standard aggregate functions (SUM, AVG, COUNT, MIN, MAX) used as window functions. Add OVER() clause. Unlike GROUP BY, don't collapse rows. Powerful for computing group statistics alongside individual rows.
</details>

<details>
<summary><strong>85. How do you calculate year-over-year growth?</strong></summary>

Use LAG window function or self-join on date - 1 year. Calculate percentage change: (current - prior) / prior * 100.

```sql
SELECT year, revenue,
    LAG(revenue) OVER (ORDER BY year) AS prev_year,
    ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY year)) /
          NULLIF(LAG(revenue) OVER (ORDER BY year), 0), 2) AS yoy_growth
FROM annual_revenue;
```
</details>

<details>
<summary><strong>86. What is a lateral join / CROSS APPLY?</strong></summary>

LATERAL (PostgreSQL) / CROSS APPLY (SQL Server): allows subquery to reference columns from preceding FROM items. Like correlated subquery in FROM. Used with table-valued functions, unnesting arrays, top-N per group.
</details>

<details>
<summary><strong>87. How do you detect and handle time zones in SQL?</strong></summary>

Store timestamps in UTC. Use TIMESTAMP WITH TIME ZONE (TIMESTAMPTZ in PostgreSQL). Convert for display: AT TIME ZONE 'America/New_York'. Avoid storing local times without timezone info.
</details>

<details>
<summary><strong>88. What is row-level security (RLS)?</strong></summary>

Database feature to restrict which rows users can access. Policies filter rows automatically on SELECT/INSERT/UPDATE/DELETE. PostgreSQL: CREATE POLICY. Transparent to application code.
</details>

<details>
<summary><strong>89. What is query result caching?</strong></summary>

Some databases (MySQL query cache, deprecated; PgBouncer; application-level) cache identical query results. Modern approach: application-level caching (Redis) with explicit invalidation. Database query caches often cause more problems than they solve.
</details>

<details>
<summary><strong>90. What is approximate counting?</strong></summary>

HyperLogLog for COUNT(DISTINCT): approximate cardinality with low memory. PostgreSQL extension: pg_hyperloglog. Used for analytics on billions of rows where approximate is acceptable (error < 2%). Orders of magnitude faster than exact COUNT(DISTINCT).
</details>

<details>
<summary><strong>91. What is a PostgreSQL extension?</strong></summary>

Modules that add functionality: pg_stat_statements (query statistics), PostGIS (geospatial), pg_trgm (fuzzy text search), pgcrypto (encryption), uuid-ossp (UUID generation), TimescaleDB (time series). CREATE EXTENSION extension_name.
</details>

<details>
<summary><strong>92. What is table inheritance in PostgreSQL?</strong></summary>

Tables can inherit columns from parent tables. SELECT on parent includes child rows (use ONLY to exclude). Used for partitioning (though native partitioning is now preferred). Check constraints inherited.
</details>

<details>
<summary><strong>93. What are array types in PostgreSQL?</strong></summary>

PostgreSQL supports array columns: INT[], TEXT[], etc. Array operators: @> (contains), && (overlap), = ANY (element in array). GIN index for array containment queries. Useful but normalizing to separate table is often cleaner.
</details>

<details>
<summary><strong>94. What is a domain in SQL?</strong></summary>

User-defined type based on existing type with constraints. CREATE DOMAIN email AS VARCHAR(255) CHECK (VALUE ~ '@'). Reusable constraint definition. More semantic than repeating CHECK constraints.
</details>

<details>
<summary><strong>95. What is SKIP LOCKED for queue processing?</strong></summary>

SELECT FOR UPDATE SKIP LOCKED: skip rows locked by other transactions, take next available. Enables concurrent queue workers without blocking each other. Efficient job queue implementation in PostgreSQL.
</details>

<details>
<summary><strong>96. What is logical vs physical replication?</strong></summary>

Logical replication: replicates data changes (INSERT/UPDATE/DELETE), can replicate to different schema version or another database type. Physical replication: replicates WAL at block level, replica is exact copy. Logical is more flexible; physical is simpler and faster.
</details>

<details>
<summary><strong>97. What is vacuuming and why is it important?</strong></summary>

PostgreSQL MVCC keeps old row versions for transactions that might need them. VACUUM removes dead tuples, updates visibility maps. Without regular vacuuming: table bloat, index bloat, slowed queries. AUTOVACUUM handles this automatically.
</details>

<details>
<summary><strong>98. What is connection pooling and PgBouncer?</strong></summary>

PgBouncer: lightweight PostgreSQL connection pooler. Session mode: one server connection per client. Transaction mode: connection returned to pool after each transaction (most common, efficient). Reduces connection overhead for many short transactions.
</details>

<details>
<summary><strong>99. What is WAL (Write-Ahead Log)?</strong></summary>

PostgreSQL records all changes to WAL before applying to data files. Ensures durability (committed changes not lost). Basis for replication and point-in-time recovery. WAL archiving enables continuous backup.
</details>

<details>
<summary><strong>100. How do you design a database for a machine learning feature store?</strong></summary>

Features (user_id, feature_name, value, timestamp). Partitioned by date, indexed on user_id. Materialized views for aggregate features. Time travel queries for point-in-time correctness (no future leakage). Consider columnar storage for analytics. Versioned feature definitions.

```sql
CREATE TABLE feature_store (
    entity_id BIGINT,
    feature_name VARCHAR(100),
    feature_value DOUBLE PRECISION,
    event_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
) PARTITION BY RANGE (event_time);

CREATE INDEX ON feature_store (entity_id, feature_name, event_time DESC);
```
</details>

---

## SQL Quick Reference

| Statement | Use |
|-----------|-----|
| SELECT DISTINCT | Unique values |
| GROUP BY + HAVING | Aggregate and filter groups |
| JOIN types | INNER, LEFT, RIGHT, FULL |
| Window OVER() | Row-level calculations with context |
| CTE (WITH) | Named temporary result sets |
| EXPLAIN | View query execution plan |
| COALESCE | Handle NULLs with defaults |
| CASE WHEN | Conditional logic |

## Summary in simple points

- Joins combine rows by keys; validate cardinality and missing-match behaviour. Self, lateral and anti-joins solve distinct relationship problems.
- WHERE filters rows, GROUP BY forms groups and HAVING filters aggregates. Window functions retain rows while computing partition/order-aware results.
- Choose explicit window frames and deterministic tie-breakers for ranks, running totals, latest records and growth calculations.
- NULL introduces three-valued logic. Use IS NULL, COALESCE, NULLIF and suitable NOT EXISTS patterns deliberately.
- DISTINCT, UNION, UNION ALL, INTERSECT and EXCEPT have different duplicate semantics; do not hide a faulty join with DISTINCT.
- CTEs, subqueries, recursive queries, views and materialised views differ in expression, reuse and freshness. Performance depends on the engine and plan.
- Indexes, covering/composite keys and cardinality affect access paths. Inspect EXPLAIN plans, selectivity and actual workload before adding hints.
- Types, casts, strings, dates, arrays, JSON and domain constraints enforce different contracts. SQL dialect support is not interchangeable.
- Primary/foreign keys, checks and uniqueness preserve integrity. Normalisation and denormalisation trade update consistency for read shape.
- Transactions and isolation determine visibility and conflicts. Understand optimistic/pessimistic locking, deadlocks and retries.
- Partitioning, sharding, replication, pruning and storage layout target different scale and availability problems. CAP applies under network partitions.
- OLTP, OLAP, star/snowflake schemas and columnar stores serve different query/update patterns.
- Pagination needs stable ordering. Avoid N+1 calls, unbounded scans, accidental casts and connection-pool exhaustion.
- UPSERT/MERGE and queue claims require concurrency-aware keys and transaction semantics. SKIP LOCKED is an engine-specific queue tool, not a universal correctness guarantee.
- Hierarchies, pivots, grouping sets, date sequences and temporal joins need explicit boundaries and missing-data rules.
- Row-level security, authorisation and cache scope must preserve tenant identity. Stored procedures/triggers also need auditing and access controls.
- PostgreSQL JSONB, extensions, inheritance, VACUUM, WAL and physical/logical replication are engine-specific operational concepts.
- Feature stores need entity keys, feature versions, event and availability timestamps, freshness and reproducible historical joins.
