"""Seed data: two tenants, knowledge-base articles and a realistic ticket mix.

Idempotent: running it twice does not duplicate anything, so compose can run
it on every start.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from helpdesk_mcp.db.models import KBArticle, Ticket
from helpdesk_mcp.db.session import make_session_factory

KB = [
    (
        "reset-password",
        "Reset your SSO password",
        "access",
        "Go to https://sso.example.internal/reset, verify with MFA, choose a new password. "
        "If MFA is lost, ask the helpdesk for a temporary bypass code.",
    ),
    (
        "vpn-troubleshooting",
        "VPN will not connect",
        "network",
        "1) Check internet access. 2) Restart the VPN client. 3) Make sure the client is "
        "version 5.2 or newer. 4) If the error is 'certificate expired', re-enrol the device.",
    ),
    (
        "outlook-sync",
        "Outlook is not syncing",
        "email",
        "Quit Outlook, delete the OST cache, restart. Check the mailbox is under quota.",
    ),
    (
        "laptop-replacement",
        "Request a replacement laptop",
        "hardware",
        "Raise a hardware ticket with the asset tag. Replacements ship within 2 business days.",
    ),
    (
        "software-install",
        "Installing approved software",
        "software",
        "Use the Self Service portal. Unlisted software needs a licence approval ticket.",
    ),
]

TICKETS = {
    "acme": [
        (
            "alice",
            "Locked out after password change",
            "I changed my password and now SSO "
            "says my account is locked. I cannot log in to anything.",
            "p3",
            "access",
            "open",
        ),
        (
            "bob",
            "VPN down for the whole office",
            "Since 9am nobody in the Leeds office can connect to the VPN. Outage affects everyone.",
            "p1",
            "network",
            "open",
        ),
        (
            "alice",
            "Outlook calendar not updating",
            "My calendar stopped syncing yesterday.",
            "p3",
            "email",
            "in_progress",
        ),
        (
            "carol",
            "Monitor flickers",
            "The external monitor flickers when docked.",
            "p4",
            "hardware",
            "resolved",
        ),
    ],
    "globex": [
        (
            "dave",
            "Need Excel licence",
            "Please install Excel on my new laptop.",
            "p4",
            "software",
            "open",
        ),
    ],
}


async def seed(engine: AsyncEngine) -> dict[str, int]:
    sf = make_session_factory(engine)
    added = {"kb": 0, "tickets": 0}
    async with sf.begin() as s:
        for tenant, tickets in TICKETS.items():
            for slug, title, category, body in KB:
                exists = await s.scalar(
                    select(KBArticle.id).where(
                        KBArticle.tenant_id == tenant, KBArticle.slug == slug
                    )
                )
                if not exists:
                    s.add(
                        KBArticle(
                            tenant_id=tenant, slug=slug, title=title, category=category, body=body
                        )
                    )
                    added["kb"] += 1
            for requester, title, desc, priority, category, status in tickets:
                exists = await s.scalar(
                    select(Ticket.id).where(Ticket.tenant_id == tenant, Ticket.title == title)
                )
                if not exists:
                    s.add(
                        Ticket(
                            tenant_id=tenant,
                            requester=requester,
                            title=title,
                            description=desc,
                            priority=priority,
                            category=category,
                            status=status,
                        )
                    )
                    added["tickets"] += 1
    return added
