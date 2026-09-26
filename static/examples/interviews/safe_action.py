"""Atomic local effect + receipt. External providers need their own idempotency."""
import hashlib
import json
import sqlite3

class SimulatedCrash(RuntimeError):
    pass

class ActionStore:
    def __init__(self, path):
        self.db = sqlite3.connect(path, isolation_level=None)
        self.db.execute("CREATE TABLE IF NOT EXISTS receipts (key TEXT PRIMARY KEY, digest TEXT, result TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS effects (key TEXT PRIMARY KEY, amount INTEGER)")

    def execute(self, key, payload, crash=None):
        if not key or set(payload) != {"amount"} or type(payload["amount"]) is not int or payload["amount"] <= 0:
            raise ValueError("positive integer amount and nonempty key required")
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            old = self.db.execute("SELECT digest, result FROM receipts WHERE key=?", (key,)).fetchone()
            if old:
                if old[0] != digest:
                    raise ValueError("idempotency key reused with a different payload")
                result = json.loads(old[1])
            else:
                self.db.execute("INSERT INTO effects VALUES (?, ?)", (key, payload["amount"]))
                if crash == "before_commit":
                    raise SimulatedCrash("before commit")
                result = {"action_id": key, "status": "committed"}
                self.db.execute("INSERT INTO receipts VALUES (?, ?, ?)", (key, digest, json.dumps(result)))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        if crash == "after_commit":
            raise SimulatedCrash("commit succeeded, response lost")
        return result

    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM effects").fetchone()[0]

    def close(self):
        self.db.close()

if __name__ == "__main__":
    store = ActionStore(":memory:")
    try:
        store.execute("refund-1", {"amount": 500}, crash="after_commit")
    except SimulatedCrash:
        pass
    print(store.execute("refund-1", {"amount": 500}), "effects", store.count())
    store.close()
