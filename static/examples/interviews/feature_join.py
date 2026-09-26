"""Point-in-time feature lookup using event AND availability timestamps."""
import sqlite3

QUERY = """
WITH candidates AS (
 SELECT p.id, p.entity, p.prediction_time, f.id AS feature_id, f.value,
        ROW_NUMBER() OVER (
          PARTITION BY p.id
          ORDER BY f.event_time DESC, f.available_time DESC, f.id DESC
        ) AS rn
 FROM predictions p LEFT JOIN features f
 ON p.entity = f.entity
 AND f.event_time <= p.prediction_time
 AND f.available_time <= p.prediction_time
)
SELECT id, entity, feature_id, value FROM candidates WHERE rn = 1 ORDER BY id
"""

def point_in_time(predictions, features):
    # Integer UTC epoch seconds; units/time zones fixed by schema contract.
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE predictions (id TEXT PRIMARY KEY, entity TEXT, prediction_time INTEGER)")
        db.execute("CREATE TABLE features (id INTEGER PRIMARY KEY, entity TEXT, event_time INTEGER, available_time INTEGER, value REAL)")
        db.executemany("INSERT INTO predictions VALUES (?, ?, ?)", predictions)
        db.executemany("INSERT INTO features VALUES (?, ?, ?, ?, ?)", features)
        return db.execute(QUERY).fetchall()

if __name__ == "__main__":
    print(point_in_time([("p1", "a", 100), ("p2", "b", 100)],
                        [(1, "a", 90, 95, 4.0), (2, "a", 99, 120, 999.0)]))
