"""Domain errors with stable, machine-readable codes.

They subclass FastMCP's ``ToolError``, so FastMCP turns them into a tool
result with ``isError: true`` and our message, even when
``mask_error_details`` hides the text of unexpected exceptions. The ``[code]``
prefix lets an LLM (or a client) branch on the error without parsing prose.
"""

from __future__ import annotations

from fastmcp.exceptions import ToolError


class HelpdeskError(ToolError):
    code = "error"

    def __init__(self, message: str) -> None:
        super().__init__(f"[{self.code}] {message}")


class NotFound(HelpdeskError):
    code = "not_found"


class VersionConflict(HelpdeskError):
    code = "version_conflict"


class IdempotencyConflict(HelpdeskError):
    code = "idempotency_conflict"


class InvalidConfirmToken(HelpdeskError):
    code = "invalid_confirm_token"


class RateLimited(HelpdeskError):
    code = "rate_limited"


class Timeout(HelpdeskError):
    code = "timeout"


class PermissionDenied(HelpdeskError):
    code = "permission_denied"
