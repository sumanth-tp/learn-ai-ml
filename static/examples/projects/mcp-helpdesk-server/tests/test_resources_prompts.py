"""Resources (static and templated) and prompts."""

from __future__ import annotations

import json

import mcp_types as mt
import pytest
from fastmcp import Client
from fastmcp.exceptions import ClientError
from mcp import MCPError


async def test_resource_listing(local_client: Client) -> None:
    static = {str(r.uri) for r in await local_client.list_resources()}
    assert {"helpdesk://me", "helpdesk://kb/articles"} <= static
    templates = {t.uri_template for t in await local_client.list_resource_templates()}
    assert templates == {"helpdesk://tickets/{ticket_id}", "helpdesk://kb/articles/{slug}"}


async def test_me_resource(local_client: Client) -> None:
    [content] = await local_client.read_resource("helpdesk://me")
    assert json.loads(content.text) == {"user": "local-admin", "tenant": "acme", "roles": ["admin"]}


async def test_ticket_by_uri(local_client: Client) -> None:
    [content] = await local_client.read_resource("helpdesk://tickets/1")
    data = json.loads(content.text)
    assert data["id"] == 1 and data["uri"] == "helpdesk://tickets/1"
    assert content.mime_type == "application/json"


async def test_missing_ticket_resource_is_invalid_params(local_client: Client) -> None:
    with pytest.raises((MCPError, ClientError)) as exc:
        await local_client.read_resource("helpdesk://tickets/9999")
    assert "not found" in str(exc.value).lower()


async def test_kb_list_and_article(local_client: Client) -> None:
    [index] = await local_client.read_resource("helpdesk://kb/articles")
    slugs = [a["slug"] for a in json.loads(index.text)]
    assert "vpn-troubleshooting" in slugs
    [article] = await local_client.read_resource("helpdesk://kb/articles/vpn-troubleshooting")
    assert "certificate expired" in json.loads(article.text)["body"]


async def test_prompts_listed_with_arguments(local_client: Client) -> None:
    prompts = {p.name: p for p in await local_client.list_prompts()}
    assert set(prompts) == {"triage_ticket", "incident_summary"}
    assert [a.name for a in prompts["triage_ticket"].arguments] == ["ticket_id"]
    assert not prompts["incident_summary"].arguments[0].required


async def test_triage_prompt_embeds_ticket(local_client: Client) -> None:
    res = await local_client.get_prompt("triage_ticket", {"ticket_id": "2"})
    first, second = res.messages
    assert "suggest_triage" in first.content.text
    assert "expected_version=1" in first.content.text
    assert isinstance(second.content, mt.EmbeddedResource)
    assert "VPN down" in second.content.resource.text


async def test_incident_summary_prompt(local_client: Client) -> None:
    res = await local_client.get_prompt(
        "incident_summary", {"window_hours": "6", "category": "network"}
    )
    text = res.messages[0].content.text
    assert "window_hours=6" in text and "category='network'" in text
    assert "Do not invent numbers" in text
