from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

CLIENTS = ["android", "ios", "mweb", "web_embedded", "android_vr"]
URL = re.compile(r"https?://[^\s<>\"')\]]+")
LINK_KINDS = [
    ("repo", re.compile(r"github\.com|gitlab\.com|bitbucket\.org|huggingface\.co/spaces", re.I)),
    ("notebook", re.compile(r"colab\.research\.google\.com|kaggle\.com/code|\.ipynb", re.I)),
    ("drive", re.compile(r"drive\.google\.com|docs\.google\.com|dropbox\.com|onedrive", re.I)),
    ("slides", re.compile(r"slides|speakerdeck|slideshare|\.pdf\b", re.I)),
    ("paper", re.compile(r"arxiv\.org|aclanthology|openreview|doi\.org", re.I)),
    ("model-or-data", re.compile(r"huggingface\.co|kaggle\.com/datasets|ollama\.com/library", re.I)),
    ("social", re.compile(r"instagram|twitter\.com|x\.com/|linkedin|facebook|telegram|discord|whatsapp|t\.me/", re.I)),
    ("course-promo", re.compile(r"course|enrol|enroll|discount|coupon|membership|join\b|bit\.ly|tinyurl", re.I)),
]


def video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    raise SystemExit(f"cannot find a video id in {url}")


def watch_url(url: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id(url)}"


def stamp(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def to_seconds(text: str) -> int:
    parts = [int(p) for p in text.split(":")]
    total = 0
    for part in parts:
        total = total * 60 + part
    return total


def yt_dlp(args: list[str], client: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "yt_dlp", "--no-warnings", "--extractor-args", f"youtube:player_client={client}"] + args
    return subprocess.run(cmd, capture_output=True, text=True)


def with_clients(args: list[str], ok, rounds: int, pause: float) -> str | None:
    for attempt in range(rounds * len(CLIENTS)):
        client = CLIENTS[attempt % len(CLIENTS)]
        result = yt_dlp(args, client)
        if ok(result):
            print(f"  ok via {client}")
            return client
        reason = (result.stderr.strip().splitlines() or ["no output"])[-1][:140]
        print(f"  {client} failed: {reason}")
        time.sleep(pause)
    return None


def classify(link: str) -> str:
    for kind, pattern in LINK_KINDS:
        if pattern.search(link):
            return kind
    return "other"


def cmd_meta(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "meta.json"
    args = ["-J", "--skip-download", watch_url(a.url)]
    holder = {}

    def ok(result):
        if result.returncode == 0 and result.stdout.strip().startswith("{"):
            holder["json"] = result.stdout
            return True
        return False

    if not with_clients(args, ok, a.rounds, a.pause):
        raise SystemExit("metadata failed on every client; wait and retry, or rotate networks")
    data = json.loads(holder["json"])
    keep = {k: data.get(k) for k in ("id", "title", "channel", "uploader", "upload_date", "duration", "description", "chapters", "tags", "webpage_url", "language")}
    target.write_text(json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8")
    description = keep.get("description") or ""
    links = sorted(set(m.rstrip(".,;:") for m in URL.findall(description)))
    lines = [
        f"# {keep['title']}",
        "",
        f"- id: {keep['id']}",
        f"- channel: {keep.get('channel') or keep.get('uploader')}",
        f"- uploaded: {keep.get('upload_date')}",
        f"- duration: {stamp(keep.get('duration') or 0)}",
        f"- fetched: {time.strftime('%Y-%m-%d')}",
        "",
        "## Chapters from the description",
        "",
    ]
    chapters = keep.get("chapters") or []
    lines += [f"{n:02d}. [{stamp(c['start_time'])} to {stamp(c['end_time'])}] {c['title']}" for n, c in enumerate(chapters, 1)] or ["none"]
    lines += ["", "## Links in the description", ""]
    lines += [f"- {classify(link)}: {link}" for link in links] or ["none"]
    lines += ["", "## Description, verbatim", "", "```text", description, "```", ""]
    (out / "description.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{keep['title']} | {stamp(keep.get('duration') or 0)} | {len(chapters)} chapters | {len(links)} links")


def cmd_playlist(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    holder = {}

    def ok(result):
        if result.returncode == 0 and result.stdout.strip().startswith("{"):
            holder["json"] = result.stdout
            return True
        return False

    if not with_clients(["-J", "--flat-playlist", a.url], ok, a.rounds, a.pause):
        raise SystemExit("playlist listing failed on every client")
    data = json.loads(holder["json"])
    rows = []
    for n, entry in enumerate(data.get("entries") or [], 1):
        rows.append(f"{n:02d}|{entry.get('id')}|{stamp(entry.get('duration') or 0)}|{entry.get('title')}")
    (out / "playlist.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"{data.get('title')}: {len(rows)} videos -> {out / 'playlist.tsv'}")


def json3_segments(path: Path) -> list[tuple[float, str]]:
    events = json.loads(path.read_text(encoding="utf-8")).get("events", [])
    segs = []
    for e in events:
        if "segs" not in e:
            continue
        text = "".join(s.get("utf8", "") for s in e["segs"]).replace("\n", " ").strip()
        if text:
            segs.append((e.get("tStartMs", 0) / 1000, text))
    return segs


def fetch_with_api(vid: str, langs: list[str], translate: bool) -> tuple[str, list[tuple[float, str]], list[str]]:
    from youtube_transcript_api import YouTubeTranscriptApi

    listing = YouTubeTranscriptApi().list(vid)
    available = [f"{t.language_code}{'(auto)' if t.is_generated else ''}" for t in listing]
    transcript = listing.find_transcript(langs)
    lang = transcript.language_code
    if translate and lang != "en" and transcript.is_translatable:
        transcript = transcript.translate("en")
        lang = f"en-from-{lang}"
    segs = [(s.start, s.text.replace("\n", " ").strip()) for s in transcript.fetch() if s.text.strip()]
    return lang, segs, available


def cmd_transcript(a) -> None:
    out = Path(a.out)
    subs = out / "subs"
    subs.mkdir(parents=True, exist_ok=True)
    vid = video_id(a.url)
    langs = a.langs.split(",")
    wanted = langs + [f"{l}-orig" for l in langs]
    args = ["--skip-download", "--write-auto-subs", "--write-subs", "--sub-langs", ",".join(wanted), "--sub-format", "json3", "-o", str(subs / "%(id)s.%(ext)s"), watch_url(a.url)]

    def ok(result):
        return any(p.stat().st_size > 0 for p in subs.glob(f"{vid}.*.json3"))

    found: dict[str, list[tuple[float, str]]] = {}
    if not a.api_only and with_clients(args, ok, a.rounds, a.pause):
        for path in sorted(subs.glob(f"{vid}.*.json3")):
            found[path.name.split(".")[-2]] = json3_segments(path)
    if not found:
        print("  yt-dlp gave nothing; trying youtube-transcript-api")
        try:
            lang, segs, available = fetch_with_api(vid, langs, a.translate)
        except Exception as error:
            raise SystemExit(f"transcript failed: {type(error).__name__}: {str(error).splitlines()[0][:200]}")
        print(f"  tracks available: {', '.join(available)}")
        (subs / f"{vid}.yta.json").write_text(json.dumps({"lang": lang, "available": available, "segments": segs}, ensure_ascii=False), encoding="utf-8")
        found[lang] = segs
    for lang, segs in found.items():
        lines = [f"[{stamp(t)}] {text}" for t, text in segs]
        (out / f"transcript-{lang}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        last = stamp(segs[-1][0]) if segs else "-"
        print(f"  transcript-{lang}.txt: {len(segs)} segments, last at {last}")


def merge_blocks(segs: list[tuple[float, str]], seconds: int) -> list[tuple[float, str]]:
    blocks, current, start = [], [], None
    for t, text in segs:
        if start is None:
            start = t
        current.append(text)
        if t - start >= seconds:
            blocks.append((start, re.sub(r"\s+", " ", " ".join(current)).strip()))
            current, start = [], None
    if current:
        blocks.append((start, re.sub(r"\s+", " ", " ".join(current)).strip()))
    return blocks


def read_transcript(path: Path) -> list[tuple[float, str]]:
    segs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\[(\d+:\d{2}:\d{2})\]\s*(.*)", line)
        if match:
            segs.append((to_seconds(match.group(1)), match.group(2)))
    return segs


def cmd_blocks(a) -> None:
    out = Path(a.out)
    sources = sorted(out.glob("transcript-*.txt"))
    if not sources:
        raise SystemExit("no transcript-*.txt; run the transcript step first")
    for source in sources:
        lang = source.stem.replace("transcript-", "")
        blocks = merge_blocks(read_transcript(source), a.seconds)
        text = "\n\n".join(f"[{stamp(t)}] {body}" for t, body in blocks) + "\n"
        (out / f"blocks-{lang}.txt").write_text(text, encoding="utf-8")
        ledger = ["| Block | Starts | Covered by heading | Done |", "| --- | --- | --- | --- |"]
        ledger += [f"| B{n:03d} | {stamp(t)} |  |  |" for n, (t, _) in enumerate(blocks, 1)]
        (out / f"ledger-{lang}.md").write_text("\n".join(ledger) + "\n", encoding="utf-8")
        print(f"  blocks-{lang}.txt: {len(blocks)} blocks of about {a.seconds} s; ledger-{lang}.md ready")


def cmd_video(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "video.mp4"
    if target.exists() and target.stat().st_size > 0:
        print(f"  {target} already present")
        return
    for fmt in ["18", "best[height<=720][ext=mp4]", "best[height<=480]"]:
        args = ["-f", fmt, "--http-chunk-size", "10M", "-o", str(target), watch_url(a.url)]
        if with_clients(args, lambda r: target.exists() and target.stat().st_size > 0, 1, a.pause):
            size = target.stat().st_size / 1e6
            print(f"  video.mp4 {size:.1f} MB (format {fmt})")
            return
    raise SystemExit("video download failed; 720p and 1080p often return 403, 360p (format 18) is the reliable one")


def frame_signature(path: Path):
    from PIL import Image
    import numpy as np

    return np.asarray(Image.open(path).convert("L").resize((96, 54)), dtype="float32")


def cmd_frames(a) -> None:
    from PIL import Image, ImageDraw
    import numpy as np

    out = Path(a.out)
    video = out / "video.mp4"
    if not video.exists():
        raise SystemExit("no video.mp4; run the video step first")
    frames = out / "frames"
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps=1/{a.every}", "-q:v", "3", str(frames / "f_%05d.jpg")], check=True)
    files = sorted(frames.glob("f_*.jpg"))
    kept, last = [], None
    rows = ["file\tseconds\ttime\tkept\tdiff"]
    for n, path in enumerate(files):
        seconds = n * a.every
        sig = frame_signature(path)
        diff = float(np.abs(sig - last).mean()) if last is not None else 255.0
        keep = diff >= a.diff
        if keep:
            kept.append((path, seconds))
            last = sig
        rows.append(f"{path.name}\t{seconds}\t{stamp(seconds)}\t{int(keep)}\t{diff:.1f}")
    (frames / "index.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    sheets = out / "sheets"
    if sheets.exists():
        shutil.rmtree(sheets)
    sheets.mkdir()
    cell_w, cell_h = 426, 240
    for s in range(0, len(kept), 9):
        sheet = Image.new("RGB", (cell_w * 3, cell_h * 3), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (path, seconds) in enumerate(kept[s:s + 9]):
            x, y = i % 3 * cell_w, i // 3 * cell_h
            sheet.paste(Image.open(path).convert("RGB").resize((cell_w, cell_h)), (x, y))
            label = f"{stamp(seconds)}  {path.name}"
            draw.rectangle([x, y, x + 8 + 7 * len(label), y + 16], fill="black")
            draw.text((x + 4, y + 2), label, fill="yellow")
        sheet.save(sheets / f"s{s // 9 + 1:03d}.jpg", quality=85)
    print(f"  {len(files)} frames every {a.every} s, {len(kept)} distinct (diff >= {a.diff}), {len(list(sheets.glob('*.jpg')))} contact sheets")


def cmd_grab(a) -> None:
    out = Path(a.out)
    video = out / "video.mp4"
    grabs = out / "grabs"
    grabs.mkdir(exist_ok=True)
    for moment in a.at:
        seconds = to_seconds(moment) if ":" in moment else int(moment)
        target = grabs / f"g_{stamp(seconds).replace(':', '')}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(seconds), "-i", str(video), "-frames:v", "1", str(target)], check=True)
        print(f"  {target}")


def cmd_notebook(a) -> None:
    nb = json.loads(Path(a.path).read_text(encoding="utf-8"))
    for n, cell in enumerate(nb.get("cells", []), 1):
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        print(f"----- cell {n} [{cell.get('cell_type')}]")
        print(source)
        if a.outputs:
            for output in cell.get("outputs", []):
                text = "".join(output.get("text", []) or output.get("data", {}).get("text/plain", []))
                if text.strip():
                    print("..... output")
                    print(text[:1500])


def cmd_all(a) -> None:
    print("meta")
    cmd_meta(a)
    time.sleep(a.pause)
    print("transcript")
    cmd_transcript(a)
    print("blocks")
    cmd_blocks(a)
    if a.frames:
        time.sleep(a.pause)
        print("video")
        cmd_video(a)
        print("frames")
        cmd_frames(a)


def main() -> None:
    p = argparse.ArgumentParser(description="Build a private source pack for one YouTube video: metadata and description, transcript, numbered blocks with a coverage ledger, 360p video, de-duplicated frames and contact sheets. Run with /usr/bin/python3.")
    p.add_argument("--rounds", type=int, default=2, help="passes over the player clients before giving up")
    p.add_argument("--pause", type=float, default=7.0, help="seconds between YouTube requests; never fetch in parallel")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, func, help_text, url=True):
        s = sub.add_parser(name, help=help_text)
        if url:
            s.add_argument("url")
        s.add_argument("out", help="pack folder, under .lecture-import/")
        s.set_defaults(func=func)
        return s

    add("meta", cmd_meta, "title, chapters, links and the verbatim description")
    add("playlist", cmd_playlist, "list a playlist into playlist.tsv")
    t = add("transcript", cmd_transcript, "captions via yt-dlp, falling back to youtube-transcript-api")
    for s in (t,):
        s.add_argument("--langs", default="en,hi")
        s.add_argument("--translate", action="store_true", help="ask YouTube for an English translation when the API path is used")
        s.add_argument("--api-only", action="store_true")
    b = add("blocks", cmd_blocks, "merge captions into numbered blocks and a coverage ledger", url=False)
    b.add_argument("--seconds", type=int, default=40)
    add("video", cmd_video, "download the 360p mp4")
    f = add("frames", cmd_frames, "one frame every N seconds, de-duplicated, 3x3 contact sheets", url=False)
    f.add_argument("--every", type=int, default=10)
    f.add_argument("--diff", type=float, default=6.0, help="mean grey-level change (0 to 255) that counts as a new frame")
    g = add("grab", cmd_grab, "full-size frame at exact moments, e.g. 1:16:05 or 4565", url=False)
    g.add_argument("at", nargs="+")
    n = sub.add_parser("notebook", help="print a notebook's cells in order")
    n.add_argument("path")
    n.add_argument("--outputs", action="store_true")
    n.set_defaults(func=cmd_notebook)
    al = add("all", cmd_all, "meta, transcript and blocks; add --frames for video, frames and sheets")
    al.add_argument("--langs", default="en,hi")
    al.add_argument("--translate", action="store_true")
    al.add_argument("--api-only", action="store_true")
    al.add_argument("--seconds", type=int, default=40)
    al.add_argument("--frames", action="store_true")
    al.add_argument("--every", type=int, default=10)
    al.add_argument("--diff", type=float, default=6.0)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
