"""Infographics for docs/theory/speech.

Run from the repo root:

    python3 scripts/infographics/speech_1.py              # all boards
    python3 scripts/infographics/speech_1.py pause        # boards whose name ends with the argument
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board, PALETTE, MONO, INK, FAINT, esc

OUT = Path(__file__).resolve().parents[2] / "static" / "img" / "speech"
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


def line(b, x1, y1, x2, y2, stroke=INK, width=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    b.parts.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
        f'stroke-width="{width}"{d} stroke-linecap="round"/>'
    )


def rect(b, x, y, w, h, fill, stroke=None, opacity=1.0):
    s = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
    b.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" fill-opacity="{opacity}"{s}/>')


@board("audio-to-features")
def audio_to_features():
    b = Board(1240, 560, "From a waveform to the numbers a recogniser reads", "One real clip, 4.815 s at 16,000 samples per second, through the same steps Whisper uses")
    steps = [
        ("1. samples", ["77,040 numbers", "16,000 per second", "each in -1 to 1"], "blue"),
        ("2. frames", ["400 samples = 25 ms", "a new frame every", "160 samples = 10 ms"], "teal"),
        ("3. window", ["multiply by a Hann", "bump so the edges", "fade to zero"], "green"),
        ("4. FFT", ["201 frequency bins", "0 to 8,000 Hz", "40 Hz apart"], "yellow"),
        ("5. mel bank", ["80 triangles, wide", "at the top, narrow", "at the bottom"], "orange"),
        ("6. log + scale", ["log10, clamp 8 below", "the peak, then", "(x + 4) / 4"], "purple"),
    ]
    x = 20
    cards = []
    for title, lines, color in steps:
        cards.append(b.card(x, 120, 180, 120, title, lines, color, size=12, title_size=14))
        x += 204
    for a, c in zip(cards, cards[1:]):
        b.arrow(a.right(), c.left())
    b.card(20, 290, 590, 120, "What comes out", ["80 numbers per frame, 481 frames for 4.815 s", "a table of 80 rows by 481 columns", "padded with silence to 3,000 columns (30 s)", "for Whisper's fixed-size input"], "green", size=13, align="left")
    b.card(630, 290, 590, 120, "Checked against the library", ["our numpy and scipy version", "against WhisperFeatureExtractor:", "largest difference 8.0e-06", "mean difference 1.3e-07"], "blue", size=13, align="left")
    b.group(20, 440, 1200, 80, "The three choices that matter", "grey")
    raw_text(b, 220, 490, "window length: time against pitch detail", 13, INK)
    raw_text(b, 620, 490, "hop: how often we look", 13, INK)
    raw_text(b, 1010, 490, "mel bands: how much pitch detail to keep", 13, INK)
    return b


@board("audio-worked-example")
def worked_example():
    b = Board(1240, 600, "Worked example: three tiny calculations", "Every number below is reproduced by the first and third code blocks")
    b.group(20, 90, 400, 490, "Eight samples, by hand", "blue")
    b.table(40, 135, [90, 80, 80, 90], [
        ["n", "x[n]", "n", "x[n]"],
        ["0", "0", "4", "0"],
        ["1", "1", "5", "1"],
        ["2", "0", "6", "0"],
        ["3", "-1", "7", "-1"],
    ], "blue", size=13, row_h=30)
    raw_text(b, 220, 320, "Pattern repeats every 4 samples,", 13, INK)
    raw_text(b, 220, 340, "so it fits 2 cycles in 8 samples.", 13, INK)
    raw_text(b, 220, 375, "Bin k = 2 multiplies by a 2-cycle wave:", 13, INK)
    raw_text(b, 220, 400, "|X[2]| = 4", 18, "#1864ab", weight="700")
    raw_text(b, 220, 435, "All other bins add to zero.", 13, INK)
    raw_text(b, 220, 460, "numpy agrees: [0, 0, 4, 0, 0]", 13, "#2b8a3e", weight="700")
    b.card(40, 495, 360, 70, "Idea", ["a bin measures how well the", "signal matches one cycle count"], "yellow", size=12)

    b.group(440, 90, 380, 490, "Aliasing: a 700 Hz tone", "red")
    b.table(460, 135, [100, 100, 140], [
        ["rate", "Nyquist", "seen at"],
        ["16000", "8000", "700 Hz"],
        ["4000", "2000", "700 Hz"],
        ["1000", "500", "300 Hz"],
        ["800", "400", "100 Hz"],
    ], "red", size=13, row_h=32)
    raw_text(b, 630, 340, "Above Nyquist the tone folds back:", 13, INK)
    raw_text(b, 630, 365, "1000 - 700 = 300", 15, "#c92a2a", weight="700")
    raw_text(b, 630, 392, "800 - 700 = 100", 15, "#c92a2a", weight="700")
    raw_text(b, 630, 430, "A 7000 Hz tone at 8000 Hz", 13, INK)
    raw_text(b, 630, 450, "lands at 8000 - 7000 = 1000 Hz.", 13, INK)
    b.card(460, 495, 340, 70, "Rule", ["keep the rate above twice the", "highest frequency you care about"], "yellow", size=12)

    b.group(840, 90, 380, 490, "Mel: where the filters sit", "orange")
    b.table(860, 135, [120, 110, 110], [
        ["Hz", "mel", "share"],
        ["0 to 1000", "0 to 15", "33%"],
        ["1000 to 4000", "15 to 35.2", "45%"],
        ["4000 to 8000", "35.2 to 45.2", "22%"],
    ], "orange", size=13, row_h=34)
    raw_text(b, 1030, 320, "below 1000 Hz: mel = Hz / 66.7", 13, INK)
    raw_text(b, 1030, 340, "above: 15 + ln(Hz / 1000) / 0.0688", 13, INK)
    raw_text(b, 1030, 385, "One third of the scale sits", 13, INK)
    raw_text(b, 1030, 405, "under 1 kHz, so 26 of the", 13, INK)
    raw_text(b, 1030, 425, "80 filters are packed there.", 13, INK)
    raw_text(b, 1030, 460, "(15 / 45.24) x 80 = 26.5", 13, "#c2410c", weight="700")
    b.card(860, 495, 340, 70, "Idea", ["ears resolve low pitch finely", "and high pitch coarsely"], "yellow", size=12)
    return b


@board("window-tradeoff")
def window_tradeoff():
    b = Board(1160, 560, "A short window sees time, a long window sees pitch", "Printed by the window experiment: two tones near 1000 Hz, and a sweep of 3000 Hz per second")
    b.group(20, 95, 560, 300, "Smallest gap that still shows two peaks", "blue")
    b.table(40, 140, [100, 130, 130, 160], [
        ["window", "samples", "bin width", "gap resolved"],
        ["5 ms", "80", "200 Hz", "150 Hz"],
        ["10 ms", "160", "100 Hz", "75 Hz"],
        ["25 ms", "400", "40 Hz", "30 Hz"],
        ["50 ms", "800", "20 Hz", "15 Hz"],
        ["100 ms", "1600", "10 Hz", "10 Hz"],
    ], "blue", size=14, row_h=36)
    raw_text(b, 300, 375, "Longer window, narrower bins, finer pitch.", 13, INK)
    b.group(600, 95, 540, 300, "Width of the ridge for a fast sweep", "orange")
    b.table(620, 140, [130, 160, 210], [
        ["window", "frames in 1 s", "ridge width"],
        ["5 ms", "399", "600 Hz"],
        ["25 ms", "79", "120 Hz"],
        ["100 ms", "19", "150 Hz"],
        ["400 ms", "4", "602 Hz"],
    ], "orange", size=14, row_h=36)
    raw_text(b, 870, 345, "Too short blurs the pitch.", 13, INK)
    raw_text(b, 870, 365, "Too long blurs the movement.", 13, INK)
    b.card(20, 425, 360, 110, "Short window", ["many frames, quick changes", "visible, pitch fuzzy"], "teal", size=13)
    b.card(400, 425, 360, 110, "25 ms, the speech default", ["short enough to follow quick", "changes, long enough to see pitch"], "green", size=13)
    b.card(780, 425, 360, 110, "Long window", ["sharp pitch lines, but the", "sweep smears across time"], "purple", size=13)
    return b


@board("asr-three-designs")
def asr_three_designs():
    b = Board(1240, 640, "Three ways to turn sound into text", "CTC labels every frame, attention models write one token at a time, Whisper is the second kind trained on 680,000 hours")
    b.group(20, 95, 380, 520, "CTC", "blue")
    c1 = b.card(45, 140, 330, 60, "encoder", ["one output per audio frame"], "blue", size=12)
    c2 = b.card(45, 230, 330, 70, "letter probabilities", ["each frame: a, b, ... or blank"], "blue", size=12)
    c3 = b.card(45, 330, 330, 70, "collapse", ["merge repeats, drop blanks:", "a a _ b b  ->  a b"], "blue", size=12)
    b.arrow(c1.bottom(), c2.top())
    b.arrow(c2.bottom(), c3.top())
    b.card(45, 430, 330, 70, "Strength", ["one fast pass, no alignment labels", "needed to train"], "green", size=12)
    b.card(45, 520, 330, 75, "How it fails", ["frames are scored independently, so", "spelling can be plausible but wrong"], "red", size=12)

    b.group(420, 95, 400, 520, "Attention encoder-decoder", "orange")
    d1 = b.card(445, 140, 350, 60, "encoder", ["a state for every audio frame"], "orange", size=12)
    d2 = b.card(445, 230, 350, 70, "decoder with attention", ["each step looks over all frames", "and picks where to listen"], "orange", size=12)
    d3 = b.card(445, 330, 350, 70, "next token", ["written left to right, using", "the tokens already written"], "orange", size=12)
    b.arrow(d1.bottom(), d2.top())
    b.arrow(d2.bottom(), d3.top())
    b.card(445, 430, 350, 70, "Strength", ["learns the language too: fluent,", "punctuated text"], "green", size=12)
    b.card(445, 520, 350, 75, "How it fails", ["can loop, skip words or write fluent", "text the audio never contained"], "red", size=12)

    b.group(840, 95, 380, 520, "Whisper (tiny.en)", "purple")
    e1 = b.card(865, 140, 330, 60, "80 x 3000 log-mel", ["30 s window, padded"], "purple", size=12)
    e2 = b.card(865, 230, 330, 70, "encoder 1500 x 384", ["two conv layers, stride 2,", "then Transformer blocks"], "purple", size=12)
    e3 = b.card(865, 330, 330, 70, "decoder", ["4 layers, cross-attention to", "the 1500 frames"], "purple", size=12)
    b.arrow(e1.bottom(), e2.top())
    b.arrow(e2.bottom(), e3.top())
    b.card(865, 430, 330, 70, "Printed here", ["37.8 M parameters, Apache 2.0", "real-time factor 0.041 on CPU"], "green", size=12)
    b.card(865, 520, 330, 75, "How it fails", ["at 0 dB noise it wrote a fluent", "sentence that is not in the audio"], "red", size=12)
    return b


@board("ctc-worked-example")
def ctc_worked_example():
    b = Board(1240, 600, "Worked example: how CTC scores the word ab", "Three frames, three symbols (blank, a, b). The code prints the same 0.4680")
    b.group(20, 90, 380, 300, "Step 1: what the network says", "blue")
    b.table(40, 135, [100, 80, 80, 80], [
        ["frame", "_", "a", "b"],
        ["1", "0.1", "0.6", "0.3"],
        ["2", "0.2", "0.3", "0.5"],
        ["3", "0.3", "0.1", "0.6"],
    ], "blue", size=14, row_h=36)
    raw_text(b, 210, 305, "Each row sums to 1.", 13, INK)
    raw_text(b, 210, 328, "Greedy: pick the best per frame.", 13, INK)
    b.group(420, 90, 400, 300, "Step 2: paths that collapse to ab", "orange")
    b.table(440, 135, [100, 150, 110], [
        ["path", "product", "probability"],
        ["a a b", "0.6 x 0.3 x 0.6", "0.1080"],
        ["a b b", "0.6 x 0.5 x 0.6", "0.1800"],
        ["a b _", "0.6 x 0.5 x 0.3", "0.0900"],
        ["a _ b", "0.6 x 0.2 x 0.6", "0.0720"],
        ["_ a b", "0.1 x 0.3 x 0.6", "0.0180"],
    ], "orange", size=13, row_h=34)
    raw_text(b, 620, 365, "Repeats merge, then blanks go.", 13, INK)
    b.group(840, 90, 380, 300, "Step 3: add them up", "green")
    raw_text(b, 1030, 150, "0.1080 + 0.1800 + 0.0900", 14, INK)
    raw_text(b, 1030, 175, "+ 0.0720 + 0.0180", 14, INK)
    raw_text(b, 1030, 215, "P(ab) = 0.4680", 20, "#2b8a3e", weight="700")
    raw_text(b, 1030, 260, "loss = -ln 0.4680 = 0.7593", 15, INK, weight="700")
    raw_text(b, 1030, 300, "torch ctc_loss: 0.7593", 14, "#1864ab", weight="700")
    raw_text(b, 1030, 325, "forward algorithm: 0.4680", 14, "#1864ab", weight="700")
    b.card(20, 420, 590, 160, "Why it works", ["Training does not need to know which frame holds which letter.", "It adds up every alignment that spells the right answer", "and pushes that total up. The forward algorithm does the", "sum with a small table instead of listing 3 x 3 x 3 paths."], "yellow", size=13, align="left")
    b.card(630, 420, 590, 160, "Greedy decoding", ["Best symbol per frame: a, b, b.", "Collapse: ab. Correct here, and its path probability", "0.1800 is only 38 per cent of the total 0.4680.", "The best path is not always the best word."], "purple", size=13, align="left")
    return b


@board("asr-results")
def asr_results():
    b = Board(1240, 640, "What the real run showed", "LibriSpeech dev-clean sample, CPU, greedy decoding; every figure is printed by the chapter's blocks")
    b.group(20, 95, 590, 250, "The same output, two scores (tiny.en, 73 clips)", "red")
    b.card(45, 140, 260, 110, "as written", ["WER 0.9983", "Mr. against MISTER,", "case, commas"], "red", size=14, title_size=15)
    b.card(325, 140, 260, 110, "normalised both sides", ["WER 0.0898", "same words, same audio"], "green", size=14, title_size=15)
    raw_text(b, 315, 285, "Formatting, not hearing, made the first score 11 times worse.", 13, INK)
    raw_text(b, 315, 310, "Whisper's normaliser also maps Mr to mister.", 13, INK)
    b.group(630, 95, 590, 250, "Bigger model, better score?", "blue")
    b.table(650, 140, [160, 130, 130, 130], [
        ["clips", "tiny.en", "base.en", "gap"],
        ["first 20", "0.0530", "0.0684", "-0.0154"],
        ["all 73", "0.0898", "0.0830", "+0.0068"],
    ], "blue", size=13, row_h=34)
    raw_text(b, 925, 270, "Bootstrap 95% interval of the", 13, INK)
    raw_text(b, 925, 292, "all-73 gap: -0.0104 to +0.0241", 13, INK, weight="700")
    raw_text(b, 925, 316, "It includes zero: no clear winner.", 13, INK)
    b.group(20, 365, 590, 255, "Adding noise (tiny.en, 30 clips)", "orange")
    b.table(40, 410, [100, 110, 360], [
        ["SNR", "WER", "part of clip 1's output"],
        ["clean", "0.0612", "mister quilter is manner"],
        ["20 dB", "0.0917", "mister colter is manner"],
        ["10 dB", "0.1942", "mister colter is matter"],
        ["5 dB", "0.3237", "mister caulford is matter"],
        ["0 dB", "0.7104", "i know there is no secret"],
    ], "orange", size=12, row_h=30)
    b.group(630, 365, 590, 255, "Where the decoder listens", "purple")
    b.table(650, 410, [260, 300], [
        ["layers averaged", "token order vs time (rank corr.)"],
        ["last layer only", "0.642"],
        ["layer 1 only", "1.000"],
        ["all four layers", "0.985"],
    ], "purple", size=13, row_h=34)
    raw_text(b, 925, 575, "Nor 0.64 s ... matter 3.98 s: attention", 13, INK)
    raw_text(b, 925, 597, "sweeps left to right across the clip.", 13, INK)
    return b


@board("tts-two-stages")
def tts_two_stages():
    b = Board(1240, 620, "Text to speech in two stages", "SpeechT5 writes a mel spectrogram, HiFi-GAN turns it into a waveform; figures are from the first test sentence")
    c1 = b.card(20, 120, 210, 120, "1. text", ["The quick brown fox", "jumps over the lazy dog.", "turned into token ids"], "blue", size=12)
    c2 = b.card(270, 120, 250, 120, "2. acoustic model", ["SpeechT5, 144.4 M parameters", "writes 184 frames x 80 mel", "plus a speaker vector (512)", "0.52 to 0.58 s"], "orange", size=12)
    c3 = b.card(560, 120, 250, 120, "3. vocoder", ["HiFi-GAN, 12.7 M parameters", "256 samples per frame", "184 x 256 = 47,104 samples", "0.19 to 0.27 s"], "purple", size=12)
    c4 = b.card(850, 120, 370, 120, "4. waveform", ["2.94 s of 16 kHz audio", "real-time factor 0.24 to 0.29", "(about 0.7 s of work for 2.94 s)"], "green", size=12)
    b.arrow(c1.right(), c2.left())
    b.arrow(c2.right(), c3.left(), label="mel")
    b.arrow(c3.right(), c4.left())
    b.group(20, 290, 590, 320, "Round trip: speak it, then listen to it", "teal")
    b.table(40, 335, [70, 270, 230], [
        ["WER", "typed", "Whisper base.en heard"],
        ["0.000", "The quick brown fox...", "same words"],
        ["0.000", "...two hundred and fifty pounds", "...250 pounds"],
        ["0.333", "at 3 PM on 21 June", "at come on June"],
        ["0.000", "at three in the afternoon on the twenty first of June", "at 3 in the afternoon on the 21st of June"],
    ], "teal", size=11, row_h=42)
    raw_text(b, 315, 580, "Whisper's normaliser maps number words to digits,", 12, INK)
    raw_text(b, 315, 598, "so rows 2 and 4 count as matches.", 12, INK)
    b.card(630, 290, 590, 135, "Where it breaks", ["Digits and abbreviations. Whisper heard", "'3 PM on 21' as 'come on' and lost the 21.", "Spelling the same date out in words scored", "0.000. Audio was not listened to by ear."], "red", size=13, align="left")
    b.card(630, 450, 590, 140, "Why two stages", ["The mel table has 256 times fewer steps than", "the waveform, so the first stage stays small.", "A separate vocoder fills in the fine detail,", "and either part can be swapped alone."], "yellow", size=13, align="left")
    return b


@board("vocoder-comparison")
def vocoder_comparison():
    b = Board(1240, 600, "Two ways to turn a mel spectrogram back into sound", "Same 4.82 s clip, same 80 x 301 mel input; mel error is the relative gap when the output is turned back into mel")
    b.group(20, 95, 600, 290, "Griffin-Lim against HiFi-GAN", "orange")
    b.table(40, 140, [230, 110, 220], [
        ["method", "seconds", "relative mel error"],
        ["Griffin-Lim, 1 pass", "0.01", "0.2541"],
        ["Griffin-Lim, 8 passes", "0.07", "0.1609"],
        ["Griffin-Lim, 32 passes", "0.32", "0.1168"],
        ["Griffin-Lim, 128 passes", "1.07", "0.0929"],
        ["HiFi-GAN, 12.7 M", "0.26", "0.1214"],
    ], "orange", size=13, row_h=36)
    raw_text(b, 320, 365, "Lower mel error is closer to the input spectrogram.", 12, INK)
    b.group(640, 95, 580, 290, "What a recogniser makes of each (12 clips, 186 words)", "blue")
    b.table(660, 140, [210, 150, 180], [
        ["audio", "WER", "S / D / I"],
        ["original", "0.0699", "9 / 4 / 0"],
        ["HiFi-GAN", "0.0591", "7 / 4 / 0"],
        ["Griffin-Lim, 32 passes", "0.0699", "9 / 4 / 0"],
    ], "blue", size=13, row_h=40)
    raw_text(b, 930, 340, "Differences of two words in 186 are noise.", 12, INK)
    raw_text(b, 930, 360, "The recogniser cannot tell the three apart.", 12, INK)
    b.card(20, 415, 600, 160, "Surprise 1", ["Griffin-Lim with 128 passes matches the", "spectrogram better (0.0929) than the neural", "vocoder (0.1214). Matching the spectrogram", "is not the same as sounding natural."], "yellow", size=13, align="left")
    b.card(640, 415, 580, 160, "Surprise 2, and a limit", ["Whisper reads all three equally well. WER does", "not measure naturalness. That needs listening", "tests, which this chapter did not run, so no", "claim about how any of them sounds is made."], "red", size=13, align="left")
    return b


@board("reply-latency")
def reply_latency():
    b = Board(1240, 640, "Where the time goes after you stop talking", "Measured stage times on a laptop CPU plus a 700 ms end-of-turn rule; first audio at 1,570 ms")
    scale = 1100 / 2168
    x = 70
    parts = [("wait", 700, "blue"), ("ASR", 141, "orange"), ("LLM", 267, "green"), ("TTS", 462, "purple")]
    cols = {"blue": "#1c7ed6", "orange": "#e8590c", "green": "#2f9e44", "purple": "#7048e8"}
    raw_text(b, 70, 128, "streamed by sentence: first audio at 1,570 ms", 13, INK, anchor="start", weight="700")
    for name, ms, color in parts:
        w = ms * scale
        rect(b, x, 140, w, 44, PALETTE[color]["fill"], cols[color])
        raw_text(b, x + w / 2, 168, f"{name} {ms}", 12, PALETTE[color]["text"], weight="700")
        x += w
    raw_text(b, 70, 238, "waiting for the whole reply: first audio at 2,168 ms", 13, INK, anchor="start", weight="700")
    x = 70
    for name, ms, color in [("wait", 700, "blue"), ("ASR", 141, "orange"), ("LLM, 2 sentences", 403, "green"), ("TTS, 2 sentences", 924, "purple")]:
        w = ms * scale
        rect(b, x, 250, w, 44, PALETTE[color]["fill"], cols[color])
        raw_text(b, x + w / 2, 278, f"{name} {ms}" if w > 100 else f"{name}", 12, PALETTE[color]["text"], weight="700")
        x += w
    raw_text(b, 70, 322, "0 ms", 11, FAINT, anchor="start")
    raw_text(b, 1170, 322, "2,168 ms", 11, FAINT, anchor="end")
    b.group(20, 350, 600, 270, "Stage times (median of 3 calls, one run)", "teal")
    b.table(40, 395, [260, 310], [
        ["stage", "measured"],
        ["speech to text, 4.82 s of speech", "141 ms (Whisper tiny.en)"],
        ["language model first token", "142 ms, then 88 tokens/s"],
        ["synthesis, 2.08 s of audio", "465 ms, real-time factor 0.22"],
    ], "teal", size=12, row_h=40)
    raw_text(b, 320, 585, "SmolLM2 135M, SpeechT5, 4 CPU threads", 12, FAINT)
    b.group(640, 350, 580, 270, "Pauses inside read speech (73 clips)", "red")
    b.table(660, 395, [190, 190, 170], [
        ["end-of-turn wait", "pauses cut", "per minute"],
        ["200 ms", "74", "10.15"],
        ["300 ms", "50", "6.86"],
        ["500 ms", "14", "1.92"],
        ["700 ms", "6", "0.82"],
        ["1000 ms", "0", "0.00"],
    ], "red", size=12, row_h=32)
    return b


def build(only=None):
    OUT.mkdir(parents=True, exist_ok=True)
    for key, fn in BOARDS.items():
        name = NAMES[key]
        if only and not name.endswith(only):
            continue
        path = fn().save(OUT / f"{name}.svg")
        print("wrote", path)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else None)
