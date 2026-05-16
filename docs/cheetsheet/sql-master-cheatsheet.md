---
title: SQL Master Cheatsheet
sidebar_position: 5
---

# SQL Master Cheatsheet

SQL is the universal data interface. ML engineers use it daily to query training
data, validate features, debug pipelines, and integrate models with
applications. Covers ANSI SQL with Postgres-flavored extensions (most common in
production).

## SELECT basics

| Method             | Description                                                                                          | Code example                                                         |
| ------------------ | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `SELECT col`       | Syntax: `SELECT col1, col2, ... FROM table;`. Pick specific columns. Avoid `SELECT *` in production. | `SELECT id, email FROM users;`                                       |
| `AS`               | Syntax: `SELECT expr AS alias` / `FROM table AS t`. Rename columns or tables.                        | `SELECT email AS user_email, id AS user_id FROM users;`              |
| `DISTINCT`         | Syntax: `SELECT DISTINCT col, ... FROM table;`. De-duplicate rows across the selected columns.       | `SELECT DISTINCT country FROM users;`                                |
| `LIMIT` / `OFFSET` | Syntax: `... LIMIT n [OFFSET k]`. Paginate results.                                                  | `SELECT * FROM users ORDER BY id LIMIT 50 OFFSET 100;`               |
| `ORDER BY`         | Syntax: `ORDER BY col [ASC or DESC], col2 [ASC or DESC], ...`. Sort by one or more columns.          | `SELECT * FROM users ORDER BY country ASC, created_at DESC;` |
| Concatenation      | Operator: ANSI double-pipe. Function: `CONCAT(a, b, ...)` (MySQL/Postgres).                          | `SELECT CONCAT(first_name, ' ', last_name) AS full_name FROM users;` |
| String formatting  | Functions: `TRIM(str)`, `LOWER(str)`, `UPPER(str)`, `SUBSTRING(str FROM start FOR length)`.          | `SELECT TRIM(LOWER(email)) AS email FROM users;`                     |
| Literal values     | Syntax: `SELECT 'literal' AS alias`. Useful for constants and flags.                                 | `SELECT id, email, 'active' AS status FROM users;`                   |

## Filtering with WHERE

| Method                    | Description                                                                                                      | Code example                                                                          |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Comparison operators      | `=`, `<>` (not equal), `<`, `>`, `<=`, `>=`.                                                                     | `SELECT * FROM users WHERE age >= 18;`                                                |
| `AND` / `OR` / `NOT`      | Boolean operators. Use parentheses to control precedence.                                                        | `SELECT * FROM users WHERE country = 'IN' AND (age >= 18 OR is_parent_verified);`     |
| `IN`                      | Syntax: `expr IN (v1, v2, ...)` or `expr IN (subquery)`. Membership test.                                        | `SELECT * FROM orders WHERE status IN ('paid', 'shipped', 'delivered');`              |
| `BETWEEN`                 | Syntax: `expr BETWEEN low AND high`. Inclusive range.                                                            | `SELECT * FROM orders WHERE total BETWEEN 100 AND 500;`                               |
| `LIKE`                    | Syntax: `expr LIKE pattern`. `%` = zero or more chars, `_` = single char.                                        | `SELECT * FROM users WHERE email LIKE '%@example.com';`                               |
| `ILIKE` (Postgres)        | Syntax: `expr ILIKE pattern`. Case-insensitive `LIKE`.                                                           | `SELECT * FROM users WHERE name ILIKE 'ada%';`                                        |
| `IS NULL` / `IS NOT NULL` | Test for nulls. `= NULL` does NOT work.                                                                          | `SELECT * FROM users WHERE phone IS NULL;`                                            |
| `EXISTS`                  | Syntax: `EXISTS (subquery)`. True if the subquery returns any rows. Often faster than `IN` for large subqueries. | `SELECT * FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id);` |
| `COALESCE`                | `COALESCE(v1, v2, v3, ...)` — Returns the first non-null value.                                                  | `SELECT id, COALESCE(nickname, first_name, 'guest') AS display_name FROM users;`      |
| `NULLIF`                  | `NULLIF(v1, v2)` — Returns NULL if `v1 = v2`, otherwise `v1`. Treat sentinels as NULL.                           | `SELECT NULLIF(score, -1) AS clean_score FROM submissions;`                           |

## Aggregations

| Method                                | Description                                                                                                          | Code example                                                                           |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `COUNT()`                             | `COUNT(*)` (all rows) / `COUNT(expr)` (non-null only) / `COUNT(DISTINCT expr)`.                                      | `SELECT COUNT(*) AS total, COUNT(email) AS with_email FROM users;`                     |
| `SUM()` / `AVG()` / `MIN()` / `MAX()` | `SUM(expr)`, `AVG(expr)`, `MIN(expr)`, `MAX(expr)` — Standard aggregates. Ignore NULLs.                              | `SELECT AVG(amount), MAX(amount), SUM(amount) FROM orders;`                            |
| `GROUP BY`                            | Syntax: `GROUP BY col1, col2, ...`. Bucket rows before aggregating.                                                  | `SELECT country, COUNT(*) AS users FROM users GROUP BY country;`                       |
| Multiple group keys                   | Pass multiple columns to `GROUP BY`.                                                                                 | `SELECT country, device, COUNT(*) FROM events GROUP BY country, device;`               |
| `HAVING`                              | Syntax: `HAVING condition`. Filter AFTER aggregation (vs `WHERE` before).                                            | `SELECT country, COUNT(*) AS n FROM users GROUP BY country HAVING COUNT(*) > 100;`     |
| `COUNT(DISTINCT col)`                 | Unique count within a group.                                                                                         | `SELECT day, COUNT(DISTINCT user_id) AS daily_active FROM events GROUP BY day;`        |
| `STRING_AGG` / `GROUP_CONCAT`         | `STRING_AGG(expr, sep ORDER BY ...)` (Postgres) / `GROUP_CONCAT(expr SEPARATOR sep)` (MySQL). Concatenate per group. | `SELECT user_id, STRING_AGG(item, ', ') AS items FROM orders GROUP BY user_id;`        |
| `ARRAY_AGG` (Postgres)                | `ARRAY_AGG(expr ORDER BY ...)` — Aggregate values into an array per group.                                           | `SELECT user_id, ARRAY_AGG(item ORDER BY ts) AS history FROM events GROUP BY user_id;` |
| `PERCENTILE_CONT` (Postgres)          | `PERCENTILE_CONT(fraction) WITHIN GROUP (ORDER BY expr)` — Continuous percentile.                                    | `SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) AS median FROM orders;`    |
| `FILTER` clause (Postgres)            | Syntax: `agg(expr) FILTER (WHERE condition)`. Conditional aggregation cleaner than `SUM(CASE WHEN ...)`.             | `SELECT COUNT(*) FILTER (WHERE status = 'paid') AS paid_count FROM orders;`            |

## Joins

| Method                    | Description                                                                                                | Code example                                                                                                                                    |
| ------------------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `INNER JOIN`              | Syntax: `FROM a INNER JOIN b ON a.col = b.col`. Rows present in both tables.                               | `SELECT u.email, o.total FROM users u INNER JOIN orders o ON u.id = o.user_id;`                                                                 |
| `LEFT JOIN`               | Syntax: `FROM a LEFT JOIN b ON ...`. All rows from left table; NULLs when no match on right.               | `SELECT u.email, o.total FROM users u LEFT JOIN orders o ON u.id = o.user_id;`                                                                  |
| `RIGHT JOIN`              | Syntax: `FROM a RIGHT JOIN b ON ...`. Inverse of `LEFT JOIN` (rare in practice).                           | `SELECT u.email, o.total FROM users u RIGHT JOIN orders o ON u.id = o.user_id;`                                                                 |
| `FULL OUTER JOIN`         | Syntax: `FROM a FULL OUTER JOIN b ON ...`. Union of left and right joins.                                  | `SELECT * FROM a FULL OUTER JOIN b ON a.id = b.id;`                                                                                             |
| `CROSS JOIN`              | Syntax: `FROM a CROSS JOIN b`. Cartesian product. Useful for date dimension joins.                         | `SELECT u.id, d.date FROM users u CROSS JOIN (SELECT generate_series('2026-01-01'::date, '2026-01-31'::date, '1 day') AS date) d;`              |
| Self-join                 | Same syntax as join, with different aliases on the same table.                                             | `SELECT a.name AS manager, b.name AS report FROM employees a JOIN employees b ON b.manager_id = a.id;`                                          |
| Multi-table join          | Chain `JOIN` clauses; each `ON` matches one pair.                                                          | `SELECT * FROM orders o JOIN users u ON o.user_id = u.id JOIN products p ON o.product_id = p.id;`                                               |
| `USING`                   | Syntax: `JOIN b USING (col1, col2)`. Shortcut when join column has same name in both tables.               | `SELECT * FROM users JOIN orders USING (user_id);`                                                                                              |
| `LATERAL JOIN` (Postgres) | Syntax: `LEFT JOIN LATERAL (subquery) alias ON true`. Right-side subquery can reference left-side columns. | `SELECT u.id, recent.* FROM users u LEFT JOIN LATERAL (SELECT * FROM orders o WHERE o.user_id = u.id ORDER BY ts DESC LIMIT 3) recent ON true;` |

## Window functions

| Method                           | Description                                                                                                 | Code example                                                                                                          |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `ROW_NUMBER()`                   | `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)` — Unique row number within a partition.                 | `SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY ts DESC) AS rn FROM events;`                              |
| `RANK()` / `DENSE_RANK()`        | `RANK() OVER (...)` (gaps after ties) / `DENSE_RANK() OVER (...)` (no gaps).                                | `SELECT name, score, DENSE_RANK() OVER (ORDER BY score DESC) AS rank FROM players;`                                   |
| `LAG()` / `LEAD()`               | `LAG(expr, offset=1, default=NULL) OVER (...)` / `LEAD(expr, offset=1, default=NULL) OVER (...)`.           | `SELECT ts, amount, LAG(amount) OVER (PARTITION BY user_id ORDER BY ts) AS prev_amount FROM orders;`                  |
| `SUM() OVER`                     | `SUM(expr) OVER (PARTITION BY ... ORDER BY ...)` — Running total or partition-aware sum.                    | `SELECT user_id, ts, amount, SUM(amount) OVER (PARTITION BY user_id ORDER BY ts) AS running_total FROM orders;`       |
| `AVG() OVER`                     | `AVG(expr) OVER (ORDER BY ... ROWS BETWEEN n PRECEDING AND CURRENT ROW)` — Moving average.                  | `SELECT day, revenue, AVG(revenue) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma_7d FROM daily;` |
| `NTILE(n)`                       | `NTILE(n) OVER (ORDER BY ...)` — Divide rows into N buckets (e.g., quartiles).                              | `SELECT user_id, amount, NTILE(4) OVER (ORDER BY amount) AS quartile FROM orders;`                                    |
| `FIRST_VALUE()` / `LAST_VALUE()` | `FIRST_VALUE(expr) OVER (...)` / `LAST_VALUE(expr) OVER (...)` — First / last value in window frame.        | `SELECT user_id, ts, FIRST_VALUE(page) OVER (PARTITION BY user_id ORDER BY ts) AS landing_page FROM events;`          |
| `PARTITION BY` vs `GROUP BY`     | `PARTITION BY` keeps row count; `GROUP BY` collapses rows.                                                  | Window: per-row + group context. Group: one row per group.                                                            |
| Window frame                     | Syntax: `ROWS BETWEEN n PRECEDING AND CURRENT ROW` / `RANGE BETWEEN ...` / `UNBOUNDED PRECEDING/FOLLOWING`. | `SUM(x) OVER (ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)`                                          |

## Subqueries and CTEs

| Method              | Description                                                                                               | Code example                                                                                                                                                                                                                                                  |
| ------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Subquery in `WHERE` | Syntax: `WHERE col IN (SELECT ...)`. Filter by another query's result.                                    | `SELECT * FROM users WHERE id IN (SELECT user_id FROM orders WHERE total > 1000);`                                                                                                                                                                            |
| Correlated subquery | Subquery references the outer row. Expensive — try to rewrite as JOIN.                                    | `SELECT u.*, (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id) AS order_count FROM users u;`                                                                                                                                                             |
| Subquery in `FROM`  | Syntax: `FROM (SELECT ...) AS alias`. Aliased like a regular table.                                       | `SELECT * FROM (SELECT user_id, SUM(amount) AS total FROM orders GROUP BY user_id) s WHERE s.total > 1000;`                                                                                                                                                   |
| `WITH` (CTE)        | Syntax: `WITH cte_name AS (SELECT ...) SELECT ... FROM cte_name;`. Named subquery.                        | `WITH high_value AS (SELECT user_id FROM orders GROUP BY user_id HAVING SUM(amount) > 1000)`<br/>`SELECT * FROM users WHERE id IN (SELECT user_id FROM high_value);`                                                                                          |
| Multiple CTEs       | Comma-separate multiple CTE definitions in one `WITH`.                                                    | `WITH a AS (SELECT ...), b AS (SELECT ... FROM a), c AS (SELECT ... FROM b)`<br/>`SELECT * FROM c;`                                                                                                                                                           |
| Recursive CTE       | Syntax: `WITH RECURSIVE cte AS (base_query UNION ALL recursive_query) SELECT ...;`. For trees and graphs. | `WITH RECURSIVE descendants AS (`<br/>` SELECT id, parent_id, name FROM categories WHERE id = 1`<br/>` UNION ALL`<br/>` SELECT c.id, c.parent_id, c.name FROM categories c JOIN descendants d ON c.parent_id = d.id`<br/>`)`<br/>`SELECT * FROM descendants;` |

## CASE expressions and conditionals

| Method                       | Description                                                                                              | Code example                                                                                                         |
| ---------------------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `CASE WHEN ... THEN ... END` | Syntax: `CASE WHEN cond1 THEN val1 WHEN cond2 THEN val2 ELSE valN END`. If-else expression.              | `SELECT name, CASE WHEN age < 18 THEN 'minor' WHEN age < 65 THEN 'adult' ELSE 'senior' END AS age_group FROM users;` |
| `CASE` in aggregation        | Wrap with `COUNT(...)` or `SUM(...)` for conditional aggregates.                                         | `SELECT COUNT(CASE WHEN status = 'paid' THEN 1 END) AS paid_count, COUNT(*) AS total FROM orders;`                   |
| `CASE` in `ORDER BY`         | Use `CASE` to define custom sort priorities.                                                             | `SELECT * FROM tasks ORDER BY CASE WHEN priority = 'high' THEN 1 WHEN priority = 'med' THEN 2 ELSE 3 END;`           |
| `GREATEST` / `LEAST`         | `GREATEST(v1, v2, ...)` / `LEAST(v1, v2, ...)` — Max / min across columns (not aggregating across rows). | `SELECT GREATEST(score_q1, score_q2, score_q3) AS best_score FROM students;`                                         |

## DDL: schema operations

| Method                       | Description                                                                                 | Code example                                                                                                                                                                     |
| ---------------------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CREATE TABLE`               | Syntax: `CREATE TABLE name (col1 type [constraints], col2 type, ...);`.                     | `CREATE TABLE users (`<br/>` id BIGSERIAL PRIMARY KEY,`<br/>` email TEXT UNIQUE NOT NULL,`<br/>` age INT CHECK (age >= 0),`<br/>` created_at TIMESTAMPTZ DEFAULT NOW()`<br/>`);` |
| `ALTER TABLE`                | Syntax: `ALTER TABLE name ADD/DROP/ALTER COLUMN ...;`. Modify schema in place.              | `ALTER TABLE users ADD COLUMN country TEXT;`<br/>`ALTER TABLE users DROP COLUMN obsolete_field;`<br/>`ALTER TABLE users ALTER COLUMN age TYPE BIGINT;`                           |
| `DROP TABLE`                 | Syntax: `DROP TABLE [IF EXISTS] name;`. Delete a table.                                     | `DROP TABLE IF EXISTS old_logs;`                                                                                                                                                 |
| `CREATE INDEX`               | Syntax: `CREATE INDEX name ON table (col1 [DESC], col2);`. Speed up lookups.                | `CREATE INDEX idx_users_email ON users (email);`<br/>`CREATE INDEX idx_orders_user_ts ON orders (user_id, ts DESC);`                                                             |
| Unique index                 | Syntax: `CREATE UNIQUE INDEX name ON table (expr);`. Enforces uniqueness.                   | `CREATE UNIQUE INDEX idx_users_email_unique ON users (LOWER(email));`                                                                                                            |
| `CREATE VIEW`                | Syntax: `CREATE VIEW name AS SELECT ...;`. A named query that acts like a table.            | `CREATE VIEW active_users AS SELECT * FROM users WHERE deleted_at IS NULL;`                                                                                                      |
| Materialized view (Postgres) | Syntax: `CREATE MATERIALIZED VIEW name AS SELECT ...;` + `REFRESH MATERIALIZED VIEW name;`. | `CREATE MATERIALIZED VIEW daily_revenue AS SELECT day, SUM(amount) FROM orders GROUP BY day;`<br/>`REFRESH MATERIALIZED VIEW daily_revenue;`                                     |
| Foreign key                  | Syntax: `col TYPE REFERENCES parent(col) ON DELETE CASCADE`, `SET NULL`, or `RESTRICT`.     | `CREATE TABLE orders (`<br/>` id BIGSERIAL PRIMARY KEY,`<br/>` user_id BIGINT REFERENCES users(id) ON DELETE CASCADE`<br/>`);` |

## DML: insert, update, delete

| Method                           | Description                                                                                   | Code example                                                                                                      |
| -------------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `INSERT INTO`                    | Syntax: `INSERT INTO table (col1, col2) VALUES (v1, v2);`. Always specify columns explicitly. | `INSERT INTO users (email, age) VALUES ('ada@example.com', 36);`                                                  |
| Bulk `INSERT`                    | Syntax: `INSERT INTO table (cols) VALUES (...), (...), ...;`. Much faster than per-row.       | `INSERT INTO users (email, age) VALUES ('a@x.com', 30), ('b@x.com', 25), ('c@x.com', 40);`                        |
| `INSERT ... SELECT`              | Syntax: `INSERT INTO target (cols) SELECT cols FROM source WHERE ...;`.                       | `INSERT INTO archived_users (email) SELECT email FROM users WHERE deleted_at IS NOT NULL;`                        |
| `ON CONFLICT` (UPSERT, Postgres) | Syntax: `INSERT ... ON CONFLICT (col) DO UPDATE SET col = EXCLUDED.col;` or `DO NOTHING`.     | `INSERT INTO users (email, age) VALUES ('a@x.com', 30) ON CONFLICT (email) DO UPDATE SET age = EXCLUDED.age;`     |
| `UPDATE`                         | Syntax: `UPDATE table SET col = val WHERE condition;`. **Always include `WHERE`**.            | `UPDATE users SET country = 'IN' WHERE country IS NULL AND ip_country = 'IN';`                                    |
| `UPDATE ... FROM` (Postgres)     | Syntax: `UPDATE target SET col = source.col FROM source WHERE target.id = source.id;`.        | `UPDATE orders SET status = 'shipped' FROM shipments s WHERE orders.id = s.order_id AND s.status = 'dispatched';` |
| `DELETE FROM`                    | Syntax: `DELETE FROM table WHERE condition;`. **Always include `WHERE`**.                     | `DELETE FROM users WHERE deleted_at < NOW() - INTERVAL '1 year';`                                                 |
| `TRUNCATE`                       | Syntax: `TRUNCATE [TABLE] name [CASCADE];`. Empty a table quickly.                            | `TRUNCATE TABLE staging_orders;`                                                                                  |
| `RETURNING` (Postgres)           | Syntax: `INSERT/UPDATE/DELETE ... RETURNING col1, col2, ...;`. Get modified rows back.        | `INSERT INTO users (email) VALUES ('a@x.com') RETURNING id, created_at;`                                          |

## Transactions

| Method                          | Description                                                                                             | Code example                                                                                                                                           |
| ------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `BEGIN` / `COMMIT` / `ROLLBACK` | Syntax: `BEGIN; ... COMMIT;` or `ROLLBACK;`. Group statements atomically.                               | `BEGIN;`<br/>`UPDATE accounts SET balance = balance - 100 WHERE id = 1;`<br/>`UPDATE accounts SET balance = balance + 100 WHERE id = 2;`<br/>`COMMIT;` |
| `SAVEPOINT`                     | Syntax: `SAVEPOINT name; ... ROLLBACK TO SAVEPOINT name;`. Nested transaction.                          | `BEGIN;`<br/>`...`<br/>`SAVEPOINT before_risky;`<br/>`...`<br/>`ROLLBACK TO before_risky;`<br/>`COMMIT;`                                               |
| Isolation level                 | Syntax: `BEGIN ISOLATION LEVEL level;` — `READ COMMITTED` (default), `REPEATABLE READ`, `SERIALIZABLE`. | `BEGIN ISOLATION LEVEL SERIALIZABLE;`                                                                                                                  |
| `SELECT ... FOR UPDATE`         | Syntax: `SELECT ... FROM ... WHERE ... FOR UPDATE;`. Lock rows for update.                              | `SELECT * FROM accounts WHERE id = 1 FOR UPDATE;`                                                                                                      |

## Performance and optimization

| Pattern                                     | Why it matters                                                                                             | Code example                                                      |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `EXPLAIN ANALYZE`                           | Syntax: `EXPLAIN [ANALYZE] [VERBOSE] query;`. See actual query plan + timing.                              | `EXPLAIN ANALYZE SELECT * FROM orders WHERE user_id = 42;`        |
| Index foreign keys                          | `CREATE INDEX ... ON child (parent_id);` — Joins on un-indexed FKs are slow.                               | `CREATE INDEX idx_orders_user_id ON orders (user_id);`            |
| Composite indexes — leftmost prefix matters | Index on (a, b) helps `WHERE a = x` and `WHERE a = x AND b = y`, but not `WHERE b = y`.                    | `CREATE INDEX idx_orders_user_ts ON orders (user_id, ts DESC);`   |
| Avoid functions on indexed columns          | `WHERE LOWER(email) = ...` won't use an index on `email`. Create a functional index or pre-lowercase data. | `CREATE INDEX idx_users_email_lower ON users (LOWER(email));`     |
| Use `EXISTS` over `IN` for large subqueries | `EXISTS` can short-circuit; `IN` materializes the subquery.                                                | `WHERE EXISTS (SELECT 1 FROM ... )`                               |
| Avoid `SELECT *` in production              | Wastes bandwidth, breaks views when schema changes.                                                        | `SELECT id, email FROM users;`                                    |
| Pagination with keyset, not OFFSET          | Large OFFSETs scan and discard rows. Use last-seen ID instead.                                             | `SELECT * FROM users WHERE id > :last_seen ORDER BY id LIMIT 50;` |
| Batch writes                                | Bulk INSERTs are 10-100× faster than per-row inserts.                                                      | `INSERT INTO logs (id, msg) VALUES (1, 'a'), (2, 'b'), (3, 'c');` |
| `VACUUM` and `ANALYZE` (Postgres)           | Syntax: `VACUUM [ANALYZE] [table];`. Keep statistics fresh for the query planner.                          | `VACUUM ANALYZE users;`                                           |

## Common ML/data patterns

| Pattern                          | Code                                                                                                                                                                                                                                        |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Top-N per group                  | `SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY ts DESC) AS rn FROM events) sub WHERE rn <= 3;`                                                                                                                  |
| Distinct counts by day           | `SELECT DATE(ts) AS day, COUNT(DISTINCT user_id) AS DAU FROM events GROUP BY DATE(ts);`                                                                                                                                                     |
| Cohort retention                 | `WITH cohort AS (SELECT user_id, MIN(DATE(ts)) AS cohort_day FROM events GROUP BY user_id)`<br/>`SELECT c.cohort_day, DATE(e.ts) AS active_day, COUNT(DISTINCT e.user_id)`<br/>`FROM cohort c JOIN events e USING (user_id) GROUP BY 1, 2;` |
| Funnel conversion                | `SELECT`<br/>` COUNT(*) FILTER (WHERE event = 'view') AS views,`<br/>` COUNT(*) FILTER (WHERE event = 'click') AS clicks,`<br/>` COUNT(*) FILTER (WHERE event = 'purchase') AS purchases`<br/>`FROM events;`                                |
| Session-ization (gap-based)      | `SELECT user_id, ts,`<br/>` SUM(CASE WHEN ts - LAG(ts) OVER (PARTITION BY user_id ORDER BY ts) > INTERVAL '30 min' THEN 1 ELSE 0 END) OVER (PARTITION BY user_id ORDER BY ts) AS session_id`<br/>`FROM events;`                             |
| Train/test split (deterministic) | `SELECT * FROM data WHERE ABS(HASHTEXT(CAST(id AS TEXT))) % 10 < 8; -- 80/20 split`                                                                                                                                                         |
| Pivot table                      | `SELECT day,`<br/>` SUM(CASE WHEN event = 'view' THEN 1 ELSE 0 END) AS views,`<br/>` SUM(CASE WHEN event = 'click' THEN 1 ELSE 0 END) AS clicks`<br/>`FROM events GROUP BY day;`                                                            |
| Feature aggregation per user     | `SELECT user_id,`<br/>` COUNT(*) AS n_orders,`<br/>` SUM(amount) AS total_spent,`<br/>` AVG(amount) AS avg_order,`<br/>` MAX(ts) AS last_order_ts`<br/>`FROM orders GROUP BY user_id;`                                                      |
| Find duplicates                  | `SELECT email, COUNT(*) FROM users GROUP BY email HAVING COUNT(*) > 1;`                                                                                                                                                                     |
| Reservoir-like random sample     | `SELECT * FROM big_table ORDER BY RANDOM() LIMIT 1000; -- ok for small samples`                                                                                                                                                             |
| Median per group (Postgres)      | `SELECT country, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY income) AS median_income FROM users GROUP BY country;`                                                                                                                         |

## Industry-standard SQL engineering

| Method | Method description | Code example |
|---|---|---|
| Query contract | Every production query should define grain, filters, null policy, and expected row count range. | `-- Grain: one row per user_id per day`<br/>`SELECT user_id, DATE(event_ts) AS day, COUNT(*) AS events`<br/>`FROM events GROUP BY 1, 2;` |
| CTE pipeline | Use CTEs to make transformation stages reviewable. | `WITH raw AS (...) , filtered AS (...) , features AS (...)`<br/>`SELECT * FROM features;` |
| Incremental model | Recompute only changed partitions for large tables. | `DELETE FROM user_features WHERE day >= :start_day;`<br/>`INSERT INTO user_features SELECT ... WHERE day >= :start_day;` |
| Idempotent load | Make reruns safe by replacing the target partition or using merge/upsert. | `MERGE INTO target t USING staging s ON t.id = s.id`<br/>`WHEN MATCHED THEN UPDATE SET amount = s.amount`<br/>`WHEN NOT MATCHED THEN INSERT (id, amount) VALUES (s.id, s.amount);` |
| Primary key test | Assert target grain is unique. | `SELECT user_id, day, COUNT(*)`<br/>`FROM user_features`<br/>`GROUP BY 1, 2 HAVING COUNT(*) > 1;` |
| Freshness test | Detect stale upstream data. | `SELECT MAX(event_ts) AS latest_event_ts FROM events;` |
| Null policy test | Fail when required fields become null. | `SELECT COUNT(*) FROM user_features WHERE user_id IS NULL OR snapshot_day IS NULL;` |
| Referential integrity | Check foreign-key-like relationships even in warehouses. | `SELECT o.user_id FROM orders o LEFT JOIN users u USING (user_id) WHERE u.user_id IS NULL;` |
| Explain plan | Inspect whether query scans, joins, and indexes are acceptable. | `EXPLAIN ANALYZE SELECT * FROM orders WHERE user_id = 42;` |
| Index design | Index high-selectivity filters and join keys, not every column. | `CREATE INDEX idx_orders_user_ts ON orders(user_id, created_at DESC);` |
| Partition pruning | Filter on partition columns directly. | `SELECT * FROM events WHERE event_date BETWEEN DATE '2026-05-01' AND DATE '2026-05-16';` |
| Avoid `SELECT *` | Explicit columns protect downstream schema contracts and reduce scan cost. | `SELECT user_id, event_ts, amount FROM events WHERE event_date = CURRENT_DATE;` |
| Late-arriving data | Reprocess a lookback window to absorb delayed events. | `WHERE event_date >= CURRENT_DATE - INTERVAL '3 days'` |
| Slowly changing dimension | Join facts to dimension version valid at event time. | `JOIN dim_user d ON f.user_id = d.user_id AND f.event_ts >= d.valid_from AND f.event_ts < d.valid_to` |
| Snapshot table | Store point-in-time state for reproducible ML features. | `CREATE TABLE account_snapshot AS SELECT account_id, balance, CURRENT_DATE AS snapshot_day FROM accounts;` |
| Feature leakage guard | Feature timestamps must be earlier than label timestamps. | `WHERE feature_ts < label_ts` |
| Transaction boundary | Group related writes atomically. | `BEGIN;`<br/>`INSERT INTO target SELECT * FROM staging;`<br/>`INSERT INTO audit_log VALUES (...);`<br/>`COMMIT;` |
| Lock awareness | Know whether DDL/DML blocks readers or writers in your database. | `SELECT * FROM pg_locks WHERE NOT granted;` |
| Cost-aware sampling | Avoid `ORDER BY RANDOM()` on huge tables. | `SELECT * FROM big_table TABLESAMPLE SYSTEM (1);` |
| Review checklist | Senior SQL review checks grain, join cardinality, partition filters, null handling, idempotency, and explain plan. | `-- Before merge: run row-count, uniqueness, freshness, and explain-plan checks.` |
