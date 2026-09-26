"""Opaque, signed, query-bound pagination cursors.

A cursor is ``base64(json) + "." + hmac``. The JSON holds the last id seen and
a hash of the filters it was issued for. Signing stops a client from forging a
cursor that jumps into another tenant's id range (the repository still filters
by tenant, but we do not rely on one defence), and binding the filters stops a
cursor from one query being replayed against a different query.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json

from fastmcp.exceptions import ValidationError


class CursorCodec:
    def __init__(self, secret: str) -> None:
        self._key = secret.encode()

    @staticmethod
    def filters_hash(filters: dict) -> str:
        blob = json.dumps(filters, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def _sign(self, payload: bytes) -> str:
        return hmac.new(self._key, payload, hashlib.sha256).hexdigest()[:32]

    def encode(self, last_id: int, filters: dict) -> str:
        payload = json.dumps({"after": last_id, "q": self.filters_hash(filters)}).encode()
        body = base64.urlsafe_b64encode(payload).decode().rstrip("=")
        return f"{body}.{self._sign(payload)}"

    def decode(self, cursor: str, filters: dict) -> int:
        """Return the id to continue after, or raise a client-facing ValidationError."""
        try:
            body, sig = cursor.split(".", 1)
            payload = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
            data = json.loads(payload)
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValidationError("Invalid cursor.") from exc
        if not hmac.compare_digest(sig, self._sign(payload)):
            raise ValidationError("Invalid cursor.")
        if data.get("q") != self.filters_hash(filters):
            raise ValidationError("Cursor does not belong to this query; start again without it.")
        return int(data["after"])
