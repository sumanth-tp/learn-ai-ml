"""Tenant-scoped data access. The only module that talks SQL.

Rule: **every** query starts from :meth:`HelpdeskRepository._visible`, which
applies the tenant filter and, for requesters, the "own tickets only" filter.
A ticket the caller may not see is reported as *not found*, never as
*forbidden*, so ids cannot be probed to learn what exists in another tenant.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from helpdesk_mcp.db.models import (
    AuditEvent,
    Comment,
    ConfirmToken,
    IdempotencyRecord,
    KBArticle,
    Ticket,
)
from helpdesk_mcp.errors import (
    IdempotencyConflict,
    InvalidConfirmToken,
    NotFound,
    VersionConflict,
)
from helpdesk_mcp.identity import Identity


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _aware(dt: datetime) -> datetime:
    """SQLite returns naive datetimes; treat them as UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


class HelpdeskRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    # ----------------------------------------------------------------- scoping
    @staticmethod
    def _visible(who: Identity) -> Select[tuple[Ticket]]:
        stmt = select(Ticket).where(Ticket.tenant_id == who.tenant)
        if not who.is_staff:
            stmt = stmt.where(Ticket.requester == who.user)
        return stmt

    async def _load(self, s: AsyncSession, who: Identity, ticket_id: int) -> Ticket:
        stmt = (
            self._visible(who).where(Ticket.id == ticket_id).options(selectinload(Ticket.comments))
        )
        ticket = (await s.execute(stmt)).scalar_one_or_none()
        if ticket is None:
            raise NotFound(f"Ticket {ticket_id} not found.")
        return ticket

    @staticmethod
    def _audit(
        s: AsyncSession,
        who: Identity,
        action: str,
        target_id: int | None,
        request_id: str | None,
        **detail: Any,
    ) -> None:
        s.add(
            AuditEvent(
                tenant_id=who.tenant,
                actor=who.user,
                action=action,
                target_id=target_id,
                request_id=request_id,
                detail=detail,
            )
        )

    # ------------------------------------------------------------------- reads
    async def get_ticket(self, who: Identity, ticket_id: int) -> Ticket:
        async with self._sf() as s:
            # Internal comments are filtered out for requesters at serialisation
            # time (server.to_detail); mutating the relationship here would mark
            # the hidden comments as orphans to be deleted.
            return await self._load(s, who, ticket_id)

    async def search(
        self,
        who: Identity,
        *,
        text: str | None,
        status: str | None,
        priority: str | None,
        category: str | None,
        assignee: str | None,
        after_id: int | None,
        limit: int,
    ) -> tuple[list[Ticket], bool]:
        """Keyset pagination, newest first. Returns (rows, has_more)."""
        stmt = self._visible(who)
        if text:
            like = f"%{text.lower()}%"
            stmt = stmt.where(
                func.lower(Ticket.title).like(like) | func.lower(Ticket.description).like(like)
            )
        if status:
            stmt = stmt.where(Ticket.status == status)
        if priority:
            stmt = stmt.where(Ticket.priority == priority)
        if category:
            stmt = stmt.where(Ticket.category == category)
        if assignee:
            stmt = stmt.where(Ticket.assignee == assignee)
        if after_id is not None:
            stmt = stmt.where(Ticket.id < after_id)
        # Fetch one extra row to learn whether another page exists without COUNT(*).
        stmt = stmt.order_by(Ticket.id.desc()).limit(limit + 1)
        async with self._sf() as s:
            rows = list((await s.execute(stmt)).scalars())
        return rows[:limit], len(rows) > limit

    async def tickets_since(
        self, who: Identity, since: datetime, after_id: int | None, limit: int
    ) -> list[Ticket]:
        stmt = self._visible(who).where(Ticket.created_at >= since)
        if after_id is not None:
            stmt = stmt.where(Ticket.id < after_id)
        stmt = stmt.order_by(Ticket.id.desc()).limit(limit)
        async with self._sf() as s:
            return list((await s.execute(stmt)).scalars())

    # ------------------------------------------------------------------ writes
    async def create_ticket(
        self,
        who: Identity,
        *,
        title: str,
        description: str,
        priority: str,
        category: str,
        idempotency_key: str | None,
        request_id: str | None,
    ) -> tuple[Ticket, bool]:
        """Create a ticket. Returns (ticket, replayed).

        With an idempotency key, a retry of the same request returns the
        ticket created the first time instead of a duplicate. Reusing the key
        for a *different* request is a client bug and is rejected.
        """
        payload = {
            "title": title,
            "description": description,
            "priority": priority,
            "category": category,
        }
        request_hash = _hash(payload)
        if idempotency_key:
            existing = await self._replay(who, idempotency_key, "create_ticket", request_hash)
            if existing is not None:
                return await self.get_ticket(who, existing), True
        try:
            async with self._sf.begin() as s:
                ticket = Ticket(tenant_id=who.tenant, requester=who.user, **payload)
                s.add(ticket)
                await s.flush()
                if idempotency_key:
                    s.add(
                        IdempotencyRecord(
                            tenant_id=who.tenant,
                            user_id=who.user,
                            key=idempotency_key,
                            operation="create_ticket",
                            request_hash=request_hash,
                            resource_id=ticket.id,
                        )
                    )
                self._audit(s, who, "ticket.create", ticket.id, request_id, priority=priority)
        except IntegrityError:
            # A concurrent request with the same key won the race; the whole
            # transaction (ticket included) rolled back. Return the winner's.
            if not idempotency_key:
                raise
            existing = await self._replay(who, idempotency_key, "create_ticket", request_hash)
            if existing is None:
                raise
            return await self.get_ticket(who, existing), True
        return await self.get_ticket(who, ticket.id), False

    async def _replay(
        self, who: Identity, key: str, operation: str, request_hash: str
    ) -> int | None:
        async with self._sf() as s:
            rec = await s.get(IdempotencyRecord, (who.tenant, who.user, key))
        if rec is None:
            return None
        if rec.operation != operation or rec.request_hash != request_hash:
            raise IdempotencyConflict(
                "This idempotency_key was already used for a different request. "
                "Use a new key for a new ticket."
            )
        return rec.resource_id

    async def update_ticket(
        self,
        who: Identity,
        ticket_id: int,
        *,
        changes: dict[str, Any],
        expected_version: int | None,
        request_id: str | None,
    ) -> Ticket:
        async with self._sf.begin() as s:
            ticket = await self._load(s, who, ticket_id)
            if expected_version is not None and ticket.version != expected_version:
                raise VersionConflict(
                    f"Ticket {ticket_id} is at version {ticket.version}, not {expected_version}. "
                    "Re-read it with get_ticket and apply your change again."
                )
            before = {k: getattr(ticket, k) for k in changes}
            for key, value in changes.items():
                setattr(ticket, key, value)
            ticket.version += 1
            self._audit(
                s, who, "ticket.update", ticket_id, request_id, before=before, after=changes
            )
        return await self.get_ticket(who, ticket_id)

    async def add_comment(
        self,
        who: Identity,
        ticket_id: int,
        *,
        body: str,
        internal: bool,
        idempotency_key: str | None,
        request_id: str | None,
    ) -> Comment:
        request_hash = _hash({"ticket_id": ticket_id, "body": body, "internal": internal})
        if idempotency_key:
            existing = await self._replay(who, idempotency_key, "add_comment", request_hash)
            if existing is not None:
                async with self._sf() as s:
                    comment = await s.get(Comment, existing)
                    if comment is not None:
                        return comment
        async with self._sf.begin() as s:
            ticket = await self._load(s, who, ticket_id)
            comment = Comment(
                ticket_id=ticket.id,
                tenant_id=who.tenant,
                author=who.user,
                body=body,
                internal=internal,
            )
            s.add(comment)
            ticket.version += 1
            await s.flush()
            if idempotency_key:
                s.add(
                    IdempotencyRecord(
                        tenant_id=who.tenant,
                        user_id=who.user,
                        key=idempotency_key,
                        operation="add_comment",
                        request_hash=request_hash,
                        resource_id=comment.id,
                    )
                )
            self._audit(s, who, "comment.add", ticket_id, request_id, internal=internal)
        return comment

    async def delete_ticket(self, who: Identity, ticket_id: int, request_id: str | None) -> None:
        async with self._sf.begin() as s:
            ticket = await self._load(s, who, ticket_id)
            title = ticket.title
            await s.delete(ticket)
            self._audit(s, who, "ticket.delete", ticket_id, request_id, title=title)

    # ------------------------------------------------------- confirm tokens
    async def issue_confirm_token(
        self, who: Identity, action: str, target_id: int, ttl_s: int
    ) -> str:
        """Mint a single-use token. Only its hash is stored."""
        token = secrets.token_urlsafe(24)
        async with self._sf.begin() as s:
            s.add(
                ConfirmToken(
                    token_hash=hashlib.sha256(token.encode()).hexdigest(),
                    tenant_id=who.tenant,
                    user_id=who.user,
                    action=action,
                    target_id=target_id,
                    expires_at=datetime.now(UTC) + timedelta(seconds=ttl_s),
                )
            )
        return token

    async def consume_confirm_token(
        self, who: Identity, token: str, action: str, target_id: int
    ) -> None:
        """Validate and burn a token. Wrong user, action, target, expired or reused all fail."""
        digest = hashlib.sha256(token.encode()).hexdigest()
        async with self._sf.begin() as s:
            rec = await s.get(ConfirmToken, digest, with_for_update=True)
            ok = (
                rec is not None
                and rec.tenant_id == who.tenant
                and rec.user_id == who.user
                and rec.action == action
                and rec.target_id == target_id
                and rec.used_at is None
                and _aware(rec.expires_at) > datetime.now(UTC)
            )
            if not ok:
                raise InvalidConfirmToken(
                    "The confirmation token is invalid, expired or already used. "
                    "Call delete_ticket without a token to get a new one."
                )
            assert rec is not None
            rec.used_at = datetime.now(UTC)

    # --------------------------------------------------------------------- KB
    async def list_kb(self, who: Identity) -> list[KBArticle]:
        stmt = select(KBArticle).where(KBArticle.tenant_id == who.tenant).order_by(KBArticle.slug)
        async with self._sf() as s:
            return list((await s.execute(stmt)).scalars())

    async def get_kb(self, who: Identity, slug: str) -> KBArticle:
        stmt = select(KBArticle).where(KBArticle.tenant_id == who.tenant, KBArticle.slug == slug)
        async with self._sf() as s:
            article = (await s.execute(stmt)).scalar_one_or_none()
        if article is None:
            raise NotFound(f"Knowledge-base article '{slug}' not found.")
        return article

    async def kb_for_category(self, who: Identity, category: str) -> list[str]:
        stmt = select(KBArticle.slug).where(
            KBArticle.tenant_id == who.tenant, KBArticle.category == category
        )
        async with self._sf() as s:
            return list((await s.execute(stmt)).scalars())

    async def audit_events(self, tenant: str) -> list[AuditEvent]:
        stmt = select(AuditEvent).where(AuditEvent.tenant_id == tenant).order_by(AuditEvent.id)
        async with self._sf() as s:
            return list((await s.execute(stmt)).scalars())
