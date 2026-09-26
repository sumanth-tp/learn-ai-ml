"""Shared-secret bearer verification for the demo HTTP upstreams."""

from __future__ import annotations

import hmac

from fastmcp.server.auth import AccessToken, TokenVerifier


class SharedSecretVerifier(TokenVerifier):
    """Accepts exactly one bearer token (the one the gateway's broker holds)."""

    def __init__(self, expected: str, client_id: str = "mcp-gateway") -> None:
        super().__init__()
        if len(expected) < 16:
            raise ValueError("upstream token must be at least 16 characters")
        self._expected = expected
        self._client_id = client_id

    async def verify_token(self, token: str) -> AccessToken | None:
        if hmac.compare_digest(token.encode(), self._expected.encode()):
            return AccessToken(token=token, client_id=self._client_id, scopes=[])
        return None
