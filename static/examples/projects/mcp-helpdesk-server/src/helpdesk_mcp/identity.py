"""Who is calling, and what are they allowed to do.

Authentication (is this token genuine?) is done by FastMCP's ``JWTVerifier``
before a request reaches a tool. This module does the next two steps:

1. map the verified token's claims to an :class:`Identity`
   (user, tenant, roles), and
2. answer authorisation questions about that identity.

Roles, least to most privileged:

* ``requester``: files tickets and sees only tickets they raised.
* ``agent``: sees every ticket in their tenant, updates, assigns, triages.
* ``admin``: an agent who may also delete tickets.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastmcp.server.dependencies import get_access_token

from helpdesk_mcp.config import Settings
from helpdesk_mcp.errors import PermissionDenied

ROLE_RANK = {"requester": 0, "agent": 1, "admin": 2}


@dataclass(frozen=True)
class Identity:
    user: str
    tenant: str
    roles: frozenset[str]

    @property
    def rank(self) -> int:
        return max((ROLE_RANK.get(r, -1) for r in self.roles), default=-1)

    def has_role(self, minimum: str) -> bool:
        return self.rank >= ROLE_RANK[minimum]

    @property
    def is_staff(self) -> bool:
        return self.has_role("agent")

    def require(self, minimum: str, action: str) -> None:
        """Raise a clean, visible permission error when the caller lacks a role."""
        if not self.has_role(minimum):
            raise PermissionDenied(f"'{action}' needs role '{minimum}' or higher.")


def identity_from_claims(claims: dict) -> Identity:
    """Build an identity from verified JWT claims.

    ``tenant_id`` and ``roles`` are custom claims your IdP adds (in Entra ID,
    Okta or Keycloak these are app roles / claim mappings). A token without a
    tenant is rejected rather than defaulted: a default tenant is how data
    leaks between customers.
    """
    sub = claims.get("sub")
    tenant = claims.get("tenant_id")
    raw_roles = claims.get("roles") or []
    if isinstance(raw_roles, str):
        raw_roles = raw_roles.split()
    if not sub or not tenant:
        raise PermissionDenied("Token is missing the 'sub' or 'tenant_id' claim.")
    roles = frozenset(r for r in raw_roles if r in ROLE_RANK) or frozenset({"requester"})
    return Identity(user=str(sub), tenant=str(tenant), roles=roles)


def current_identity(settings: Settings) -> Identity:
    """Resolve the caller for the request being handled right now."""
    token = get_access_token()
    if token is not None:
        return identity_from_claims(token.claims)
    if settings.auth_mode == "local":
        return Identity(
            user=settings.local_user,
            tenant=settings.local_tenant,
            roles=frozenset(settings.local_roles),
        )
    raise PermissionDenied("Authentication required.")
