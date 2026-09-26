"""Exception hierarchy. The split between domain and transient errors drives retries."""

from __future__ import annotations


class SupportError(Exception):
    """Base class for errors raised by this service."""


class DomainError(SupportError):
    """A business-rule failure. Never retried; the agent explains it to the customer."""


class NotFoundError(DomainError):
    pass


class NotAllowedError(DomainError):
    """The action breaks a policy (window closed, amount too high, not approved)."""


class PermissionDeniedError(DomainError):
    """The tool is not permitted for this intent or this user."""


class TransientError(SupportError):
    """A dependency failed in a way that may succeed on retry (timeout, 5xx, lock)."""


class PermanentGatewayError(SupportError):
    """The refund provider rejected the request (4xx). Retrying will not help."""


class BudgetExceededError(SupportError):
    """The per-request step or token budget ran out."""
