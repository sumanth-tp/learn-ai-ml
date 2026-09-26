"""Calendar MCP server (streamable HTTP).

A small team calendar persisted to a JSON file. It demonstrates idempotent writes
(``create_event`` with an idempotency key), a destructive tool (``cancel_event``) and
sampling: ``summarise_day`` asks the *host's* LLM to write the summary, so the server
needs no model or API key of its own.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ClientCapabilities, SamplingCapability, SamplingMessage, TextContent
from mcp.types import ToolAnnotations as TA
from pydantic import BaseModel

from demo_servers.common import (
    add_health_route,
    advertise_list_changed,
    serve_http,
    transport_security,
)


class Event(BaseModel):
    id: str
    title: str
    start: datetime
    end: datetime
    attendees: list[str] = []
    idempotency_key: str | None = None


class CalendarStore:
    """JSON-file store. A lock keeps concurrent HTTP sessions from interleaving writes."""

    def __init__(self, path: Path, seed: Path | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            initial = seed.read_text() if seed and seed.exists() else "[]"
            path.write_text(initial)

    def all(self) -> list[Event]:
        return [Event.model_validate(e) for e in json.loads(self.path.read_text())]

    def save(self, events: list[Event]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps([e.model_dump(mode="json") for e in events], indent=2))
        tmp.replace(self.path)

    def add(self, event: Event) -> tuple[Event, bool]:
        with self._lock:
            events = self.all()
            if event.idempotency_key:
                for existing in events:
                    if existing.idempotency_key == event.idempotency_key:
                        return existing, False
            events.append(event)
            self.save(events)
            return event, True

    def remove(self, event_id: str) -> Event | None:
        with self._lock:
            events = self.all()
            keep = [e for e in events if e.id != event_id]
            if len(keep) == len(events):
                return None
            self.save(keep)
            return next(e for e in events if e.id == event_id)


def _parse_day(day: str) -> date:
    try:
        return date.fromisoformat(day)
    except ValueError as exc:
        raise ToolError(f"day must be YYYY-MM-DD, got {day!r}") from exc


def create_server(store_path: Path | None = None, seed_path: Path | None = None) -> FastMCP:
    store = CalendarStore(
        store_path or Path(os.environ.get("CALENDAR_STORE", "data/state/calendar.json")),
        seed_path or Path(os.environ.get("CALENDAR_SEED", "data/calendar_seed.json")),
    )
    mcp = FastMCP(
        "calendar",
        instructions="Team calendar. Times are ISO 8601 in UTC.",
        transport_security=transport_security(),
    )
    advertise_list_changed(mcp)
    add_health_route(mcp)

    def events_on(day: date) -> list[Event]:
        return sorted((e for e in store.all() if e.start.date() == day), key=lambda e: e.start)

    @mcp.tool(annotations=TA(readOnlyHint=True))
    def list_events(day: str) -> list[Event]:
        """List the events on one day (YYYY-MM-DD)."""
        return events_on(_parse_day(day))

    @mcp.tool(annotations=TA(destructiveHint=False, idempotentHint=True))
    def create_event(
        title: str,
        start: datetime,
        duration_minutes: int = 30,
        attendees: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> Event:
        """Create an event. Pass the same idempotency_key on retries to avoid duplicates."""
        if not 5 <= duration_minutes <= 480:
            raise ToolError("duration_minutes must be between 5 and 480")
        event = Event(
            id=f"evt_{uuid.uuid4().hex[:8]}",
            title=title.strip()[:120],
            start=start,
            end=start + timedelta(minutes=duration_minutes),
            attendees=attendees or [],
            idempotency_key=idempotency_key,
        )
        saved, _ = store.add(event)
        return saved

    @mcp.tool(annotations=TA(destructiveHint=True, idempotentHint=True))
    def cancel_event(event_id: str) -> str:
        """Cancel (delete) an event by id. This cannot be undone."""
        removed = store.remove(event_id)
        return f"cancelled {removed.title!r}" if removed else f"no event {event_id!r}"

    @mcp.tool(annotations=TA(readOnlyHint=True))
    def find_free_slot(day: str, duration_minutes: int = 30) -> str:
        """Find the first free slot between 09:00 and 17:00 UTC on a day."""
        d = _parse_day(day)
        cursor = datetime.fromisoformat(f"{d.isoformat()}T09:00:00+00:00")
        close = datetime.fromisoformat(f"{d.isoformat()}T17:00:00+00:00")
        need = timedelta(minutes=duration_minutes)
        for event in events_on(d):
            if event.start - cursor >= need:
                break
            cursor = max(cursor, event.end)
        if close - cursor < need:
            return f"no free {duration_minutes}-minute slot on {day}"
        return cursor.isoformat()

    @mcp.tool(annotations=TA(readOnlyHint=True))
    async def summarise_day(day: str, ctx: Context) -> str:
        """Summarise a day's meetings in two sentences (uses the host's LLM via sampling)."""
        events = events_on(_parse_day(day))
        if not events:
            return f"No events on {day}."
        listing = "\n".join(
            f"- {e.start:%H:%M}-{e.end:%H:%M} {e.title} ({', '.join(e.attendees) or 'no attendees'})"
            for e in events
        )
        can_sample = ctx.session.check_client_capability(
            ClientCapabilities(sampling=SamplingCapability())
        )
        if not can_sample:
            return f"Events on {day}:\n{listing}"
        result = await ctx.session.create_message(
            messages=[
                SamplingMessage(
                    role="user",
                    content=TextContent(type="text", text=f"Summarise this day:\n{listing}"),
                )
            ],
            system_prompt="You write two-sentence summaries of a calendar day.",
            max_tokens=200,
            related_request_id=ctx.request_context.request_id,
        )
        text = result.content.text if isinstance(result.content, TextContent) else ""
        return text or f"Events on {day}:\n{listing}"

    @mcp.resource("calendar://today", mime_type="application/json")
    def today() -> str:
        """Today's events as JSON."""
        return json.dumps([e.model_dump(mode="json") for e in events_on(date.today())])

    @mcp.prompt()
    def plan_meeting(topic: str, attendees: str) -> str:
        """Plan a meeting: find a slot, then create the event."""
        return (
            f"Schedule a 30-minute meeting about {topic!r} with {attendees}. "
            "Find a free slot with calendar__find_free_slot first, then create it with "
            "calendar__create_event using an idempotency_key."
        )

    return mcp


def main() -> None:
    serve_http(create_server(), default_port=8101)


if __name__ == "__main__":
    main()
