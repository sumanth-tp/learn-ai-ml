"""Filesystem-notes MCP server (stdio).

Notes are Markdown files in one folder. The server shows the three things a host
must cope with: a destructive tool (``delete_note``), a tool name that collides with
another server (``search``), and tools that appear at run time (``enable_tag_tools``
sends ``notifications/tools/list_changed``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel

from demo_servers.common import advertise_list_changed, log_level

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
MAX_NOTE_BYTES = 64_000
READ_ONLY = ToolAnnotations(readOnlyHint=True)


class WriteResult(BaseModel):
    name: str
    status: str  # created | updated | unchanged
    sha256: str


class SearchHit(BaseModel):
    name: str
    line: int
    text: str


def create_server(root: Path | None = None) -> FastMCP:
    """Build a notes server over ``root`` (default ``$NOTES_DIR`` or ``data/notes``)."""
    base = (root or Path(os.environ.get("NOTES_DIR", "data/notes"))).resolve()
    base.mkdir(parents=True, exist_ok=True)
    tags_file = base / ".tags.json"

    mcp = FastMCP(
        "notes",
        instructions="Personal notes stored as Markdown files. Names are lowercase slugs.",
        log_level=log_level(),
    )
    advertise_list_changed(mcp)

    def note_path(name: str) -> Path:
        """Map a note name to a file, refusing anything that could escape the folder."""
        if not NAME_RE.match(name):
            raise ToolError(f"invalid note name {name!r}: use lowercase letters, digits, - and _")
        path = (base / f"{name}.md").resolve()
        if not path.is_relative_to(base):
            raise ToolError("note path escapes the notes directory")
        return path

    @mcp.tool(annotations=READ_ONLY)
    def list_notes() -> list[str]:
        """List the names of all notes."""
        return sorted(p.stem for p in base.glob("*.md"))

    @mcp.tool(annotations=READ_ONLY)
    def read_note(name: str) -> str:
        """Return the full Markdown text of one note."""
        path = note_path(name)
        if not path.exists():
            raise ToolError(f"note {name!r} does not exist")
        return path.read_text(encoding="utf-8")

    @mcp.tool(annotations=ToolAnnotations(destructiveHint=False, idempotentHint=True))
    def write_note(name: str, content: str) -> WriteResult:
        """Create or overwrite a note. Writing identical content twice is a no-op."""
        if len(content.encode()) > MAX_NOTE_BYTES:
            raise ToolError(f"note is larger than {MAX_NOTE_BYTES} bytes")
        path = note_path(name)
        digest = hashlib.sha256(content.encode()).hexdigest()
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() == digest:
                return WriteResult(name=name, status="unchanged", sha256=digest)
            status = "updated"
        else:
            status = "created"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)  # atomic rename: a crash never leaves half a note
        return WriteResult(name=name, status=status, sha256=digest)

    @mcp.tool(annotations=ToolAnnotations(destructiveHint=True, idempotentHint=True))
    def delete_note(name: str) -> str:
        """Permanently delete a note."""
        path = note_path(name)
        if not path.exists():
            return f"note {name!r} was already absent"
        path.unlink()
        return f"deleted note {name!r}"

    @mcp.tool(annotations=READ_ONLY)
    def search(query: str, limit: int = 5) -> list[SearchHit]:
        """Case-insensitive substring search across all personal notes."""
        needle = query.lower().strip()
        if not needle:
            raise ToolError("query must not be empty")
        hits: list[SearchHit] = []
        for path in sorted(base.glob("*.md")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if needle in line.lower():
                    hits.append(SearchHit(name=path.stem, line=number, text=line.strip()[:200]))
                    if len(hits) >= limit:
                        return hits
        return hits

    def load_tags() -> dict[str, list[str]]:
        return json.loads(tags_file.read_text()) if tags_file.exists() else {}

    def tag_note(name: str, tag: str) -> list[str]:
        """Add a tag to a note and return the note's tags."""
        if not note_path(name).exists():
            raise ToolError(f"note {name!r} does not exist")
        tags = load_tags()
        tags[name] = sorted(set(tags.get(name, [])) | {tag.lower()})
        tags_file.write_text(json.dumps(tags, indent=2))
        return tags[name]

    def list_tags() -> dict[str, list[str]]:
        """Return every note's tags."""
        return load_tags()

    @mcp.tool()
    async def enable_tag_tools(ctx: Context) -> str:
        """Turn on the tag tools (tag_note, list_tags). Call this before tagging notes."""
        if mcp._tool_manager.get_tool("tag_note") is None:
            mcp.add_tool(tag_note)
            mcp.add_tool(list_tags, annotations=READ_ONLY)
            # Tell the client its cached tool list is stale.
            await ctx.session.send_tool_list_changed()
            return "tag tools enabled: tag_note, list_tags"
        return "tag tools were already enabled"

    @mcp.resource("notes://index", mime_type="application/json")
    def notes_index() -> str:
        """JSON list of note names."""
        return json.dumps(sorted(p.stem for p in base.glob("*.md")))

    @mcp.resource("notes://note/{name}", mime_type="text/markdown")
    def note_resource(name: str) -> str:
        """One note as a resource."""
        return read_note(name)

    @mcp.prompt()
    def daily_review(focus: str = "open actions") -> str:
        """Review the notes and list what needs doing."""
        return (
            f"Review my notes and list the {focus}. Use notes__list_notes and "
            "notes__read_note, then answer as a short checklist."
        )

    return mcp


def main() -> None:
    create_server().run("stdio")


if __name__ == "__main__":
    main()
