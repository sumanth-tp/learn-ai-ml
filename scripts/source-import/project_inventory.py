from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
GROUPS = {"theory", "mlops", "code", "scaler", "projects", "cheetsheet", "interviews", "research-papers", "daily", "document-collection"}
MODULE = re.compile(r"^\d{2}-")


def front(text: str) -> dict[str, str]:
    match = re.match(r"---\n(.*?)\n---", text, flags=re.S)
    data = {}
    for line in (match.group(1).splitlines() if match else []):
        if ":" in line:
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip().strip('"')
    return data


def is_project(meta: dict[str, str], text: str) -> bool:
    tags = meta.get("tags", "")
    label = meta.get("sidebar_label", "") + " " + meta.get("title", "")
    return bool(re.search(r"\bproject\b", tags, re.I) or re.search(r"\b(project|capstone)\b", label, re.I))


def topics() -> list[Path]:
    found = []
    for category in sorted(DOCS.rglob("_category_.json")):
        folder = category.parent
        rel = folder.relative_to(DOCS)
        if any(part.startswith("_") for part in rel.parts):
            continue
        if len(rel.parts) == 1 and rel.parts[0] in GROUPS:
            continue
        if MODULE.match(folder.name) or folder.name in ("99-practice", "90-projects"):
            continue
        found.append(folder)
    return found


def pages(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*.md*") if not any(part.startswith("_") for part in p.relative_to(folder).parts))


def slug_of(path: Path, meta: dict[str, str]) -> str:
    if meta.get("slug", "").startswith("/"):
        return "/docs" + meta["slug"]
    parts = [re.sub(r"^\d+[-_. ]+", "", part) for part in path.relative_to(DOCS).with_suffix("").parts]
    if meta.get("id"):
        parts[-1] = meta["id"]
    return "/docs/" + "/".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description="List every topic with its chapter count, its project pages, and the chapters that no project page links to.")
    p.add_argument("--uncovered", action="store_true", help="also list the uncovered chapters for each topic")
    a = p.parse_args()
    nested = {t for t in topics()}
    rows = []
    for topic in topics():
        inner = [t for t in nested if t != topic and topic in t.parents]
        files = [f for f in pages(topic) if not any(t in f.parents for t in inner)]
        chapters, projects = [], []
        for f in files:
            text = f.read_text(encoding="utf-8")
            meta = front(text)
            (projects if is_project(meta, text) else chapters).append((f, meta, text))
        linked = set()
        for _, _, text in projects:
            linked |= set(re.findall(r"\]\((/docs/[^)#\s]+)", text))
        uncovered = [f for f, meta, _ in chapters if slug_of(f, meta) and slug_of(f, meta) not in linked]
        rows.append((topic.relative_to(DOCS), len(chapters), len(projects), uncovered))
    print(f"{'topic':55} {'chapters':>8} {'projects':>8} {'not used by a project':>22}")
    for rel, n_ch, n_pr, uncovered in rows:
        flag = "" if n_pr else "   <- no project"
        print(f"{str(rel):55} {n_ch:8} {n_pr:8} {len(uncovered):22}{flag}")
        if a.uncovered and n_pr:
            for f in uncovered:
                print(f"    - {f.relative_to(ROOT)}")
    missing = sum(1 for r in rows if r[2] == 0)
    print(f"\n{len(rows)} topics, {missing} without any project")


if __name__ == "__main__":
    main()
