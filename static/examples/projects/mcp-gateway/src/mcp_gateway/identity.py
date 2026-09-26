"""Identity at the front door.

The gateway is an OAuth 2.1 *resource server*: it validates bearer JWTs
issued by the company IdP for audience ``mcp-gateway`` and never issues or
forwards user tokens itself. FastMCP's ``JWTVerifier`` checks signature,
expiry, issuer and audience before any MCP message reaches our middleware;
an invalid token gets HTTP 401 at the transport layer.
"""

from __future__ import annotations

import time
from typing import Any

import jwt
from fastmcp.server.auth import AccessToken
from fastmcp.server.auth.providers.jwt import JWTVerifier
from fastmcp.server.dependencies import get_access_token

from mcp_gateway.config import Settings
from mcp_gateway.policy import Principal


class UnauthenticatedError(PermissionError):
    pass


def build_verifier(settings: Settings) -> JWTVerifier:
    if settings.jwt_algorithm == "HS256":
        assert settings.jwt_secret is not None  # enforced by Settings validation
        return JWTVerifier(
            public_key=settings.jwt_secret.get_secret_value(),
            algorithm="HS256",
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
        )
    return JWTVerifier(
        public_key=settings.jwt_public_key,
        jwks_uri=settings.jwks_uri,
        algorithm=settings.jwt_algorithm,
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
    )


def principal_from_token(token: AccessToken | None, groups_claim: str) -> Principal:
    if token is None:
        raise UnauthenticatedError("no authenticated principal")
    claims: dict[str, Any] = token.claims or {}
    sub = claims.get("sub") or token.client_id
    if not sub:
        raise UnauthenticatedError("token has no subject")
    raw_groups = claims.get(groups_claim, [])
    if isinstance(raw_groups, str):
        raw_groups = raw_groups.split()
    groups = frozenset(str(g) for g in raw_groups if isinstance(g, str | int))
    email = claims.get("email") if isinstance(claims.get("email"), str) else None
    return Principal(subject=str(sub), groups=groups, email=email)


def current_principal(groups_claim: str) -> Principal:
    return principal_from_token(get_access_token(), groups_claim)


def mint_dev_token(
    settings: Settings,
    subject: str,
    groups: list[str],
    *,
    email: str | None = None,
    ttl_seconds: int = 3600,
) -> str:
    """Issue an HS256 token for local development, tests and the demo.

    Production tokens come from the IdP (RS256 via JWKS); this helper refuses
    to run for any other algorithm so it cannot be mistaken for an issuer.
    """
    if settings.jwt_algorithm != "HS256" or settings.jwt_secret is None:
        raise RuntimeError("dev tokens can only be minted in HS256 mode")
    now = int(time.time())
    claims: dict[str, Any] = {
        "sub": subject,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": now,
        "exp": now + ttl_seconds,
        settings.groups_claim: groups,
    }
    if email:
        claims["email"] = email
    return jwt.encode(claims, settings.jwt_secret.get_secret_value(), algorithm="HS256")
