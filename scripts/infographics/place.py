"""Place infographic boards into a doc.

    from place import Doc
    d = Doc("docs/projects/ai-security/01-guardrails-and-llm-security.md")
    d.replace_mermaid("*Redrawn from the mentor's whiteboard (0:03 to 0:06).*", d.tag(src, alt, caption))
    d.insert_after("text that ends a paragraph", d.tag(...))
    d.save()

Anchors are matched with flexible whitespace, so line wrapping in the doc doesn't matter. Every
anchor must match exactly once, otherwise nothing is written.
"""

from __future__ import annotations

import re
from pathlib import Path

IMPORT = "import Infographic from '@site/src/components/Infographic';"


def _flex(text: str) -> re.Pattern:
    parts = [re.escape(p) for p in text.split()]
    return re.compile(r"\s+".join(parts))


class Doc:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.s = self.path.read_text(encoding="utf-8")
        self.count = 0

    @staticmethod
    def tag(src: str, alt: str, caption: str = "") -> str:
        for value in (src, alt, caption):
            assert '"' not in value and "{" not in value, value
        cap = f'\n  caption="{caption}"' if caption else ""
        return f'<Infographic\n  src="{src}"\n  alt="{alt}"{cap}\n/>'

    def _splice(self, start: int, end: int, text: str):
        """Replace s[start:end] with text, tidying blank lines only at the seam.

        A global collapse would also shrink blank lines inside code blocks, so the
        tidy is limited to a few characters either side of the edit.
        """
        self.s = self.s[:start] + text + self.s[end:]
        a, b = max(0, start - 4), min(len(self.s), start + len(text) + 4)
        self.s = self.s[:a] + re.sub(r"\n{3,}", "\n\n", self.s[a:b]) + self.s[b:]

    def _one(self, anchor: str) -> re.Match:
        matches = list(_flex(anchor).finditer(self.s))
        if len(matches) != 1:
            raise SystemExit(f"{self.path.name}: anchor matched {len(matches)} times: {anchor[:70]!r}")
        return matches[0]

    def replace_mermaid(self, caption_anchor: str, replacement: str, max_gap: int = 400):
        """Replace a caption line and the Mermaid block right after it."""
        m = self._one(caption_anchor)
        start = self.s.rfind("\n", 0, m.start()) + 1
        j = self.s.find("```mermaid", m.end())
        if j == -1 or j - m.end() > max_gap:
            raise SystemExit(f"{self.path.name}: no mermaid right after {caption_anchor[:60]!r}")
        k = self.s.find("\n```", j + 10)
        k = self.s.find("\n", k + 4)
        end = k + 1 if k != -1 else len(self.s)
        self._splice(start, end, replacement + "\n" if replacement else "")
        self.count += 1

    def replace_mermaid_containing(self, needle: str, replacement: str):
        """Replace the Mermaid block that contains `needle` (and a caption line just above it)."""
        m = self._one(needle)
        j = self.s.rfind("```mermaid", 0, m.start())
        k = self.s.find("\n```", m.end())
        k = self.s.find("\n", k + 4)
        start = j
        before = self.s[:j].rstrip("\n")
        last_line = before[before.rfind("\n") + 1:]
        if last_line.startswith("*") and last_line.endswith("*"):
            start = before.rfind("\n") + 1
        self._splice(start, k + 1, replacement + "\n")
        self.count += 1

    def insert_after(self, anchor: str, block: str):
        m = self._one(anchor)
        end = self.s.find("\n", m.end())
        end = len(self.s) if end == -1 else end + 1
        self._splice(end, end, "\n" + block + "\n")
        self.count += 1

    def ensure_import(self, line: str = IMPORT):
        if line in self.s:
            return
        assert self.s.startswith("---\n")
        end = self.s.index("\n---\n", 4) + 5
        self.s = self.s[:end] + "\n" + line + "\n" + self.s[end:]

    def save(self):
        self.ensure_import()
        self.path.write_text(self.s, encoding="utf-8")
        return self.count
