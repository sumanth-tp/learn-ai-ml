"""Mint development tokens that match what the production IdP issues.

In production you never mint tokens here: Entra ID / Okta / Keycloak issue
them, and the server verifies them against the IdP's JWKS
(``HELPDESK_JWT_ALGORITHM=RS256`` + ``HELPDESK_JWT_JWKS_URI``). For local
work and tests, HS256 with a shared secret gives the same claim shape.
"""

from __future__ import annotations

import time

import jwt

from helpdesk_mcp.config import Settings


def mint_token(
    settings: Settings,
    *,
    user: str,
    tenant: str,
    roles: list[str],
    ttl_s: int = 3600,
    audience: str | None = None,
    issuer: str | None = None,
    secret: str | None = None,
) -> str:
    if settings.jwt_algorithm != "HS256":
        raise ValueError(
            "Dev tokens can only be minted with HS256; RS256 tokens come from the IdP."
        )
    now = int(time.time())
    claims = {
        "sub": user,
        "tenant_id": tenant,
        "roles": roles,
        "iss": issuer or settings.jwt_issuer,
        "aud": audience or settings.jwt_audience,
        "iat": now,
        "exp": now + ttl_s,
        "scope": "helpdesk",
    }
    key = secret or settings.jwt_secret.get_secret_value()
    return jwt.encode(claims, key, algorithm="HS256")
