"""Infographics for docs/agentic-frontier, chapters 04 and 05.

Run from the repo root:

    python3 scripts/infographics/afr_2.py            # all boards
    python3 scripts/infographics/afr_2.py voice      # boards whose name contains the word
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board, PALETTE, MONO, INK, FAINT, esc

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "afr"
BOARDS = {}
NAMES = {}


def board(name):
    def deco(fn):
        BOARDS[fn.__name__] = fn
        NAMES[fn.__name__] = name
        return fn
    return deco


def raw_text(b, x, y, text, size=12, fill=INK, anchor="middle", weight="400"):
    b.parts.append(
        f'<text xml:space="preserve" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{MONO}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{esc(text)}</text>'
    )


def rect(b, x, y, w, h, fill, stroke, width=1.6, rx=4):
    b.parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    )


def hbar(b, x, y, w, h, frac, color, label, value):
    rect(b, x, y, w, h, "#f1f3f5", "#ced4da", 1, 3)
    if frac > 0:
        rect(b, x, y, max(2, w * frac), h, PALETTE[color]["stroke"], PALETTE[color]["stroke"], 1, 3)
    raw_text(b, x - 10, y + h / 2 + 4, label, 12, INK, "end")
    raw_text(b, x + w + 10, y + h / 2 + 4, value, 12, PALETTE[color]["text"], "start", "700")


@board("voice-cascade-vs-speech-to-speech")
def voice_big_picture():
    b = Board(1240, 740, "Two ways to build a voice agent", "Latency stack from chapter code blocks 2 and 3: measured parts and assumed parts are marked")
    b.group(20, 90, 1200, 215, "Cascade: four stages in a chain, each with its own delay", "blue")
    mic = b.card(40, 150, 130, 100, "microphone", ["audio in", "16,000 samples", "a second"], "grey", size=12)
    vad = b.card(205, 150, 150, 100, "detector", ["Silero VAD:", "talking or not,", "is the turn over"], "teal", size=12)
    asr = b.card(390, 150, 160, 100, "speech to text", ["Whisper tiny.en", "about 0.05 of real", "time, batch"], "blue", size=12)
    llm = b.card(585, 150, 170, 100, "language model", ["SmolLM2 135M:", "first token about", "0.3 s here"], "purple", size=12)
    tts = b.card(790, 150, 170, 100, "text to speech", ["first audio chunk", "assumed 150 ms", "here"], "orange", size=12)
    spk = b.card(995, 150, 205, 100, "speaker", ["the user hears", "the reply, and may", "talk over it"], "grey", size=12)
    for a, c in ((mic, vad), (vad, asr), (asr, llm), (llm, tts), (tts, spk)):
        b.arrow(a.right(), c.left())
    b.group(20, 325, 1200, 150, "Speech to speech: one model reads audio and writes audio", "green")
    m = b.card(40, 375, 130, 80, "microphone", ["audio in"], "grey", size=12)
    s2s = b.card(360, 375, 520, 80, "speech-to-speech model, served over WebRTC or WebSocket", ["detects turns, may be interrupted, can call tools, keeps its own state"], "green", size=12)
    sp = b.card(1070, 375, 130, 80, "speaker", ["audio out"], "grey", size=12)
    b.arrow(m.right(), s2s.left())
    b.arrow(s2s.right(), sp.left())

    b.group(20, 495, 1200, 225, "From the last word to the first audio (cascade, silence rule 300 ms)", "orange")
    parts = [("silence rule", 300, "orange", "chosen"), ("batch speech to text", 420, "blue", "measured"), ("model first token", 307, "purple", "measured"),
             ("text to speech", 150, "yellow", "assumed"), ("network", 80, "grey", "assumed")]
    total = sum(p[1] for p in parts)
    x = 60.0
    width = 1120.0
    for name, ms, color, kind in parts:
        w = width * ms / total
        rect(b, x, 545, w, 54, PALETTE[color]["fill"], PALETTE[color]["stroke"], 2, 4)
        raw_text(b, x + w / 2, 568, f"{ms} ms", 14, PALETTE[color]["text"], "middle", "700")
        raw_text(b, x + w / 2, 588, kind, 11, INK)
        raw_text(b, x + w / 2, 620, name, 11, PALETTE[color]["text"], "middle", "700")
        x += w
    raw_text(b, 620, 668, f"total {total:,} ms, about 1.26 s: the real figure varies with the machine's load", 13, INK, "middle", "700")
    raw_text(b, 620, 698, "a speech-to-speech model removes two stages but its own delay was not measured here", 12, FAINT)
    return b


@board("voice-turn-taking-worked-example")
def voice_worked():
    b = Board(1240, 730, "Where to cut a sentence", "A 3.2 second request with a natural 400 ms pause, replayed with three silence rules (chapter code block 1)")
    b.group(20, 90, 1200, 190, "What the user said", "blue")
    rect(b, 60, 150, 590, 60, PALETTE["blue"]["fill"], PALETTE["blue"]["stroke"], 2, 5)
    raw_text(b, 355, 186, "I would like to book a flight", 15, PALETTE["blue"]["text"], "middle", "700")
    raw_text(b, 355, 232, "1,900 ms", 12, INK)
    rect(b, 650, 150, 130, 60, "#fff5f5", PALETTE["red"]["stroke"], 2, 5)
    raw_text(b, 715, 186, "pause", 14, PALETTE["red"]["text"], "middle", "700")
    raw_text(b, 715, 232, "400 ms", 12, INK)
    rect(b, 780, 150, 360, 60, PALETTE["blue"]["fill"], PALETTE["blue"]["stroke"], 2, 5)
    raw_text(b, 960, 186, "to Lisbon", 15, PALETTE["blue"]["text"], "middle", "700")
    raw_text(b, 960, 232, "900 ms", 12, INK)
    raw_text(b, 620, 262, "a person pauses to think; the detector sees 400 ms of silence", 12, FAINT)

    rows = [["silence rule", "400 ms pause", "turns sent on", "first audio after the last word"],
            ["300 ms", "longer than the rule: cut", "2: the sentence is split", "300 + 400 + 300 + 150 + 80 = 1,230 ms"],
            ["500 ms", "shorter than the rule: kept", "1: the whole request", "500 + 400 + 300 + 150 + 80 = 1,430 ms"],
            ["800 ms", "shorter than the rule: kept", "1: the whole request", "800 + 400 + 300 + 150 + 80 = 1,730 ms"]]
    b.table(30, 305, [170, 280, 270, 440], rows, "teal", size=13, row_h=46)
    b.group(20, 520, 1200, 185, "The same trade-off on real speech: 10 LibriSpeech utterances, 124.8 s, Silero VAD", "orange")
    rows2 = [["silence rule (ms)", "100", "200", "300", "500", "800", "1200"],
             ["segments (10 true)", "26", "21", "19", "14", "12", "11"],
             ["premature cut-offs", "16", "11", "9", "4", "2", "1"]]
    b.table(40, 565, [260, 150, 150, 150, 150, 150, 150], rows2, "orange", size=13, row_h=40)
    return b


@board("voice-barge-in-and-budget")
def voice_results():
    b = Board(1240, 700, "Interrupting the agent, and what the pipeline costs", "Chapter code blocks 3 and 4, Silero VAD 6.2.3, Whisper tiny.en, SmolLM2 135M on CPU")
    b.group(20, 90, 600, 340, "Stopping the agent: speech needed before it stops", "red")
    rows = [["sound", "run", "0 ms", "100 ms", "250 ms", "400 ms"],
            ["real interruption", "1,472 ms", "64", "164", "314", "464"],
            ["backchannel", "224 ms", "64", "164", "never", "never"],
            ["noise burst", "0 ms", "never", "never", "never", "never"]]
    b.table(35, 140, [170, 95, 70, 80, 80, 80], rows, "red", size=12, row_h=46)
    raw_text(b, 320, 360, "milliseconds from the start of the sound to the moment the agent stops", 12, FAINT)
    raw_text(b, 320, 388, "a 32 ms chunk is the detector's unit, so every time is a multiple of 32 plus the rule", 11, FAINT)
    b.group(640, 90, 580, 340, "Speech to text and the model, measured", "blue")
    rows2 = [["clip", "seconds", "real-time factor", "word error rate"],
             ["0", "5.9", "0.05", "0.059"],
             ["1", "4.8", "0.05", "0.100"],
             ["2", "12.5", "0.04", "0.000"],
             ["3", "9.9", "0.04", "0.042"]]
    b.table(655, 140, [100, 130, 170, 150], rows2, "blue", size=12, row_h=42)
    raw_text(b, 930, 395, "timings vary from run to run; word error rates do not", 12, FAINT)
    b.group(20, 450, 1200, 235, "The lesson in three lines", "yellow")
    b.card(40, 495, 380, 170, "end of turn", ["a short silence rule cuts people", "off; a long one adds the same", "delay to every single turn"], "yellow", size=13)
    b.card(430, 495, 380, 170, "barge-in", ["the detector rejects noise, but", "a short real word still counts:", "decide how much speech is a turn"], "yellow", size=13)
    b.card(820, 495, 380, 170, "budget", ["the chain adds up to about", "1.3 s before any speech-to-", "speech model is considered"], "yellow", size=13)
    return b


@board("dspy-from-prompts-to-programs")
def dspy_big_picture():
    b = Board(1240, 740, "DSPy: write the program, let a search write the prompt", "DSPy 3.4.0, documentation read on 8 October 2026")
    b.group(20, 90, 1200, 200, "What you write", "blue")
    sig = b.card(40, 140, 260, 120, "signature", ["message -> queue", "names, types and one", "line of instruction"], "blue", size=12)
    mod = b.card(340, 140, 260, 120, "module", ["Predict, ChainOfThought,", "ReAct and others:", "how to call the model"], "blue", size=12)
    data = b.card(640, 140, 250, 120, "examples", ["a few dozen inputs", "with the right answer", "train, dev and test"], "teal", size=12)
    met = b.card(930, 140, 270, 120, "metric", ["a function that scores", "one prediction:", "right or wrong, 0 to 1"], "teal", size=12)
    b.arrow(sig.right(), mod.left())
    b.group(20, 310, 1200, 230, "What the optimiser does when you call compile", "purple")
    run = b.card(40, 365, 250, 140, "run the program", ["on training inputs,", "with the current", "instructions and demos"], "purple", size=12)
    score = b.card(330, 365, 250, 140, "score each run", ["with the metric; keep", "the traces that passed"], "purple", size=12)
    prop = b.card(620, 365, 250, 140, "propose changes", ["new demonstrations,", "new instruction text,", "or new weights"], "purple", size=12)
    best = b.card(910, 365, 290, 140, "keep the best on dev", ["the compiled program:", "same code, a better", "prompt, saved to a file"], "green", size=12)
    for a_, c_ in ((run, score), (score, prop), (prop, best)):
        b.arrow(a_.right(), c_.left())
    b.arrow(best.bottom(0.5), run.bottom(0.5), via=[(1055, 530), (165, 530)], label="repeat until the budget is spent", label_dy=-4, dashed=True)
    rows = [["optimiser", "what it changes", "needs a second model to write text"],
            ["LabeledFewShot", "adds labelled examples as demonstrations", "no"],
            ["BootstrapFewShot (and ...WithRandomSearch)", "adds demonstrations that the program itself got right", "no"],
            ["MIPROv2", "instructions and demonstrations, searched together", "yes, proposes instructions"],
            ["GEPA", "instruction text, from reflection on failures and written feedback", "yes, a reflection model"]]
    b.table(30, 560, [380, 520, 300], rows, "purple", size=12, row_h=34)
    return b


@board("dspy-worked-example")
def dspy_worked():
    b = Board(1240, 700, "Why one more demonstration fixed one more answer", "A stand-in model that copies the label of the demonstration sharing most words with the message (chapter code blocks 1 and 2)")
    b.group(20, 90, 1200, 190, "Three demonstrations", "blue")
    for i, (t, q) in enumerate((("charged twice soon", "billing"), ("parcel late again", "shipping"), ("app crashes today", "technical"))):
        b.card(40 + i * 390, 140, 360, 100, f'"{t}"', [f"label: {q}"], "blue", size=13)
    rows = [["message", "demo it matches", "shared words", "answer", "truth"],
            ["charged twice please", "charged twice soon", "2", "billing", "billing"],
            ["parcel late today", "parcel late again", "2", "shipping", "shipping"],
            ["tracking stuck soon", "charged twice soon", "1 (below 2)", "billing (the default)", "shipping"],
            ["app crashes again", "app crashes today", "2", "technical", "technical"]]
    b.table(30, 300, [300, 270, 190, 250, 160], rows, "teal", size=13, row_h=40)
    b.group(20, 540, 1200, 140, "Add the fourth demonstration: \"tracking stuck please\" is shipping", "green")
    b.card(40, 590, 560, 70, "now the third message shares two words", ["with \"tracking stuck please\": answer shipping"], "green", size=13)
    b.card(640, 590, 560, 70, "three of four right becomes four of four", ["one demonstration covered one more kind of message"], "green", size=13)
    return b


@board("dspy-results")
def dspy_results():
    b = Board(1240, 740, "What the optimisers did, on a stand-in and on a real small model", "Chapter code blocks 3 and 5. Real model: Qwen2.5-0.5B-Instruct on CPU, 32 held-out messages with unseen phrasing")
    b.group(20, 90, 600, 330, "Stand-in: more demos, more accuracy, longer prompt", "orange")
    rows = [["labelled demos", "dev", "test", "prompt tokens"],
            ["0", "0.250", "0.250", "135"], ["4", "0.375", "0.375", "215"], ["8", "0.458", "0.479", "297"],
            ["12", "0.500", "0.583", "378"], ["16", "0.583", "0.604", "458"], ["24", "0.625", "0.771", "618"]]
    b.table(40, 140, [180, 120, 120, 150], rows, "orange", size=12, row_h=36)
    raw_text(b, 320, 405, "the test messages share their phrases with the demonstrations", 11, FAINT)
    b.group(640, 90, 580, 330, "Bootstrapping from a weak teacher", "red")
    rows2 = [["program", "dev", "test", "queues in demos"],
             ["bootstrapped, up to 4", "0.250", "0.250", "1 of 4"],
             ["bootstrapped, up to 12", "0.333", "0.312", "4 of 4"]]
    b.table(655, 140, [220, 90, 90, 150], rows2, "red", size=12, row_h=40)
    b.text(930, 340, "a teacher that always says billing\nis only ever right about billing,\nso its good traces teach one queue", 12, INK)
    b.group(20, 440, 1200, 280, "Real model: Qwen2.5-0.5B-Instruct through the same DSPy code", "green")
    rows3 = [["program", "correct of 32", "what it answered"],
             ["zero-shot", "3", "mostly billing, often with a stray > in front"],
             ["8 labelled demonstrations", "11", "billing 8, account 16, technical 8: never shipping"],
             ["bootstrapped 4 + 4", "15", "billing 25, account 7: it leans on one label"]]
    b.table(40, 490, [330, 190, 640], rows3, "green", size=13, row_h=44)
    raw_text(b, 620, 700, "demonstrations fix the format first; a model this small still cannot tell the queues apart reliably", 12, FAINT)
    return b


def main(names):
    todo = names or list(BOARDS)
    for name in todo:
        keys = [k for k, v in NAMES.items() if k == name or v == name or name in v]
        for key in keys:
            path = BOARDS[key]().save(OUT / f"{NAMES[key]}.svg")
            print(path.relative_to(OUT.parents[2]))


if __name__ == "__main__":
    main(sys.argv[1:])
