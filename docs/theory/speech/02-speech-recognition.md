---
id: speech-recognition
title: "Speech Recognition: CTC, Attention Models and Whisper"
sidebar_label: "2 · Speech recognition"
sidebar_position: 2
slug: /theory/speech/speech-recognition
description: "How a recogniser turns audio features into words: the alignment problem, CTC worked by hand and checked against torch, attention encoder-decoders, and a real run of Whisper tiny.en and base.en on CPU with word error rate computed by hand and with jiwer."
tags: [speech, asr, ctc, attention, whisper, wer, jiwer, transformers]
---

import Infographic from '@site/src/components/Infographic';
import CtcPathLab from '@site/src/components/viz/CtcPathLab';
import WerLab from '@site/src/components/viz/WerLab';

**In one line.** Speech recognition is the problem of writing letters for audio when nobody tells the model which sound belongs to which letter, and the three answers (CTC, attention and Whisper) differ in how they handle that missing alignment.

:::tip Before you start
- **You should already know** what log-mel features are ([audio basics](/docs/theory/speech/audio-basics)) and, roughly, how an encoder and a decoder work together ([encoder-decoder and sequence-to-sequence](/docs/theory/dnn/encoder-decoder-and-sequence-to-sequence-architecture)).
- **Reading time:** about 50 minutes, plus about four minutes to run all the code and download two small models.
- **After this chapter you can** score a CTC alignment by hand, explain why attention models can hallucinate, compute word error rate by hand and with a library, and say why the same transcript can score 1.00 or 0.09.
:::

:::note Not from a lecture
This chapter was written for this site from the sources under Go deeper. Every number is printed by the code in the chapter. Environment: Python 3.14, torch 2.14.1 (CPU), transformers 5.18.0, jiwer 4.0.0, NumPy 2.5.3, SciPy 1.18.1. Audio is a 73-clip sample of LibriSpeech dev-clean (CC BY 4.0). Models are `openai/whisper-tiny.en` and `openai/whisper-base.en`, both Apache 2.0 per their model cards. Sources were opened on 8 October 2026.
:::

## In 30 seconds

Imagine writing down a sentence while someone speaks quickly in a language you half know. You do not stop the talker to mark where each letter starts. You listen, keep a rough running guess, and write.

A recogniser has the same problem. The audio is 100 frames per second and the text is a handful of letters per second, so the lengths never match and nobody labels which frame is which letter. CTC (connectionist temporal classification) gives every frame a vote and then merges the votes. Attention models read the whole recording and write one word piece at a time. Whisper is an attention model trained on 680,000 hours of audio, small enough that its tiny version runs more than 20 times faster than real time on a laptop CPU.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Alignment | Which stretch of audio produced which letter | "m" lives in frames 31 to 34 |
| Blank | A CTC symbol meaning "no new letter here" | the `_` in `a _ b` |
| Collapse | The CTC rule: merge repeats, then delete blanks | `a a _ b b` becomes `ab` |
| Path | One full choice of symbol for every frame | `a b b` |
| Greedy decoding | Pick the best symbol at every step, ignoring the rest | frames say a, b, b so the text is `ab` |
| Attention | A decoder step that looks across all audio frames and weights them | the word "matter" looks at 3.98 s |
| Token | A word piece the decoder writes | ` Qu`, `il`, `ter` |
| Word error rate (WER) | Word edits needed to fix the guess, divided by the reference length | 3 edits in 6 words is 0.50 |
| Normalisation | Lower-casing and standardising text before scoring | `Mr.` and `MISTER` both become `mister` |

## The idea in plain words

Why is alignment a problem? Take the word "cat" said in 0.3 seconds. At 100 frames per second that is 30 frames. Which of the 30 are "c", which are "a", which are "t"? A person labelling thousands of hours of audio frame by frame is too expensive. CTC lets the model learn from the plain text alone.

Here is the smallest example with numbers. There are three frames and three symbols: blank (`_`), `a` and `b`. The network says, for each frame, how likely each symbol is. We want the probability that the output reads `ab`. The collapse rule lists every way of choosing three symbols that ends as `ab`: `a a b`, `a b b`, `a b _`, `a _ b` and `_ a b`. Each path's probability is the product of its three frame probabilities, and since any one of them gives the right text, we add them.

With the table in the worked example below, `a b b` is 0.6 x 0.5 x 0.6 = 0.1800 and the five paths total 0.4680. In words: the model is rewarded for the answer, not for any one alignment, so it is free to choose the alignment that fits best.

The second family, attention models, drops the per-frame vote. An encoder turns the audio into one summary vector per frame. A decoder then writes the answer one token at a time. For each token it computes a weight over all the audio frames, mixes the frames by those weights and uses the mix, plus the words it has already written, to choose the next token. The weights are the attention. They are not told where to look; training teaches them that "the next word" lives a little to the right of "the previous word".

<Infographic src="/img/speech/asr-three-designs.svg" alt="Three columns. CTC: encoder, letter probabilities per frame, collapse of repeats and blanks, with the failure that frames are scored independently. Attention encoder-decoder: encoder, decoder with attention, next token, with the failure that it can loop or write fluent text the audio never contained. Whisper tiny.en: 80 by 3000 log-mel, a 1500 by 384 encoder, a four layer decoder, 37.8 million parameters, real-time factor 0.041 and a fluent wrong sentence at 0 dB noise." caption="Read the red box at the bottom of each column first: the failures are what separate the designs. The green box above it says what each design buys." />

## Worked example, step by step

The network's output for three frames, rows summing to 1:

| frame | `_` | `a` | `b` |
| --- | --- | --- | --- |
| 1 | 0.1 | 0.6 | 0.3 |
| 2 | 0.2 | 0.3 | 0.5 |
| 3 | 0.3 | 0.1 | 0.6 |

1. **List the paths that spell `ab`.** Collapse each of the 27 possible paths. Five survive: `a a b`, `a b b`, `a b _`, `a _ b`, `_ a b`.
2. **Multiply along each path.** `a a b` is 0.6 x 0.3 x 0.6 = 0.1080. `a b b` is 0.6 x 0.5 x 0.6 = 0.1800. `a b _` is 0.6 x 0.5 x 0.3 = 0.0900. `a _ b` is 0.6 x 0.2 x 0.6 = 0.0720. `_ a b` is 0.1 x 0.3 x 0.6 = 0.0180.
3. **Add them.** 0.1080 + 0.1800 + 0.0900 + 0.0720 + 0.0180 = 0.4680.
4. **Turn it into a loss.** Training minimises the negative log of the total: -ln 0.4680 = 0.7593.
5. **Read the greedy answer.** The best symbol per frame is `a`, `b`, `b`, which collapses to `ab`. That single path has probability 0.1800, only 38 per cent of the total, so the greedy path is not the whole story.

<Infographic src="/img/speech/ctc-worked-example.svg" alt="Three panels. The first is the table of frame probabilities. The second lists the five paths that collapse to ab with their products and probabilities. The third adds them to 0.4680 and a loss of 0.7593, and says torch ctc_loss agrees." caption="Follow the three top panels left to right, then read the two bottom notes. The sum in the green panel is the number the code prints." />

The explicit list has 3 x 3 x 3 = 27 paths. A real clip has 1,500 frames and dozens of symbols, so listing is impossible. The **forward algorithm** gets the same sum with a table: it keeps, for each position in the target (with blanks inserted between letters), the total probability of all paths that have reached it, and updates it frame by frame. Block 1 implements it and gets the same 0.4680.

## How it works

### What does CTC assume, and what does it cost?

CTC treats each frame's output as independent of the others given the audio. That is why it trains fast and decodes in one pass, and also why it spells by sound: it has no memory of the letters it has just written. A CTC model can produce "recognise speech" or "wreck a nice beach" with equal confidence, because nothing inside it knows which is a sentence. Real systems add an external language model, which is a second component to build and to keep in sync.

Two rules about the blank are worth stating. A blank is needed between two identical letters, because `l l` collapses to one `l`: the path for "hello" must be `h e l _ l o`. And many frames are blank, so a CTC model outputs a spiky sequence in which most frames say nothing and a few frames name a letter.

### How do attention models differ?

Listen, Attend and Spell (Chan, Jaitly, Le and Vinyals, 2015) was an early well-known attention speech model. Its abstract describes a listener that encodes filter-bank features and a speller that writes characters, and presents the speller as an improvement over earlier end-to-end CTC models because it makes no independence assumption between output characters. On a subset of Google voice search it reported a word error rate of 14.1 per cent without a dictionary or language model.

The price is that decoding is a loop: one decoder pass per token, which cannot be parallelised. And because the decoder is a language model that conditions on the audio, it can lean on language when the audio is weak. That is the source of fluent wrong output, shown below.

### What is Whisper?

Whisper is an attention encoder-decoder Transformer (Radford and colleagues, 2022). The paper resamples audio to 16,000 Hz, computes an 80-channel log-magnitude mel spectrogram on 25 ms windows with a 10 ms stride, and processes 30-second segments. The encoder begins with two convolution layers, the second of stride two, so 3,000 frames become 1,500 encoder frames, each 20 ms. The decoder writes tokens with cross-attention over those frames. It was trained on 680,000 hours of audio and transcripts collected from the internet, about 65 per cent English. The `.en` models are English only. The model card lists sizes of 39 million parameters (tiny), 74 million (base), 244 million (small), 769 million (medium) and 1,550 million (large).

A short run of special tokens at the start of the decoder's output tells the model what job to do (transcribe or translate, language, timestamps or none). That is how one network covers many tasks. For recordings longer than 30 seconds, tools chop the audio into windows and stitch the pieces, which is where timing errors creep in.

### Why can the same transcript score 1.00 or 0.09?

Word error rate counts the fewest word insertions, deletions and substitutions that turn the hypothesis into the reference, divided by the number of reference words. A reference in upper case with no punctuation, and a hypothesis in sentence case with commas, differ in every word. The Whisper paper states the problem: a system can output a transcript that a person would call correct and still receive a large WER because of formatting differences. It handles this with a text normaliser applied to both sides, and reports WER drops of up to 50 per cent from it on some datasets. Block 3 shows a larger effect on one corpus.

<Infographic src="/img/speech/asr-results.svg" alt="Four panels of results. The same Whisper tiny.en output scores a WER of 0.9983 as written and 0.0898 after normalising both sides. Tiny and base compared on the first 20 and on all 73 clips, with a bootstrap interval for the gap that includes zero. Word error rate by noise level from 0.0612 clean to 0.7104 at 0 dB. A table of how well the decoder's attention follows time in different layers." caption="Start with the top left: same audio, same model, two scores. Then look at the noise table, where the last row is a fluent sentence that is not in the audio." />

## Code you can run

Five blocks. The first two need only numpy, torch and jiwer. The rest download `whisper-tiny.en` (about 150 MB) and `whisper-base.en` (about 290 MB) on first use and read the LibriSpeech sample from the Hugging Face cache. Everything runs on CPU. If the Hub is slow once the models are cached, set `HF_HUB_OFFLINE=1`.

### 1. CTC by hand, by brute force, and against torch

The first block lists the five paths, adds them, implements the forward algorithm, and asks `torch.nn.functional.ctc_loss` for the same number. It then builds a second table where greedy decoding is wrong.

```python
import itertools
import numpy as np
import torch
import torch.nn.functional as F

labels = ["_", "a", "b"]

def collapse(path):
    out, previous = [], None
    for symbol in path:
        if symbol != previous and symbol != "_":
            out.append(symbol)
        previous = symbol
    return "".join(out)

def brute_force(probs, target):
    total, valid = 0.0, []
    for path in itertools.product(range(3), repeat=len(probs)):
        word = "".join(labels[k] for k in path)
        if collapse(word) == target:
            p = float(np.prod([probs[t, k] for t, k in enumerate(path)]))
            valid.append((word, p))
            total += p
    return total, valid

def ctc_forward(probs, target):
    ext = ["_"]
    for ch in target:
        ext += [ch, "_"]
    S, T = len(ext), len(probs)
    alpha = np.zeros((T, S))
    alpha[0, 0] = probs[0, labels.index(ext[0])]
    alpha[0, 1] = probs[0, labels.index(ext[1])]
    for t in range(1, T):
        for s in range(S):
            total = alpha[t - 1, s] + (alpha[t - 1, s - 1] if s >= 1 else 0)
            if s >= 2 and ext[s] != "_" and ext[s] != ext[s - 2]:
                total += alpha[t - 1, s - 2]
            alpha[t, s] = total * probs[t, labels.index(ext[s])]
    return alpha[-1, -1] + alpha[-1, -2]

probs = np.array([[0.1, 0.6, 0.3], [0.2, 0.3, 0.5], [0.3, 0.1, 0.6]])
total, valid = brute_force(probs, "ab")
for word, p in valid:
    print(f"  path {word}  probability {p:.4f}")
print(f"brute force P('ab') = {total:.4f}   loss = {-np.log(total):.4f}")
print(f"forward algorithm   = {ctc_forward(probs, 'ab'):.4f}")
log_probs = torch.log(torch.tensor(probs, dtype=torch.float32)).unsqueeze(1)
loss = F.ctc_loss(log_probs, torch.tensor([[1, 2]]), torch.tensor([3]), torch.tensor([2]), blank=0, reduction="sum")
print(f"torch ctc_loss      = {loss.item():.4f}   exp(-loss) = {np.exp(-loss.item()):.4f}")

trap = np.array([[0.8, 0.2, 0.0], [0.55, 0.45, 0.0], [0.2, 0.0, 0.8]])
path = "".join(labels[k] for k in trap.argmax(axis=1))
print(f"second example: greedy path {path} reads {collapse(path)!r}")
for target in ("b", "ab"):
    print(f"  P({target!r}) = {brute_force(trap, target)[0]:.4f}")
```

**Reading the output.** The five paths and their probabilities are exactly those of the worked example, and they add to 0.4680 with loss 0.7593. The forward algorithm gives 0.4680 without listing paths, and torch's loss returns 0.7593, so `exp(-loss)` is 0.4680 again. Three independent routes agree.

The second table is the surprise. Greedy decoding picks the best symbol per frame: blank, blank, `b`, which reads `b`. But the probability of `b` summed over all its paths is 0.3520, while `ab` is 0.4480. The first frame hints at `a`, and the second is nearly a coin flip between blank and `a`. Frame by frame the blank wins; summed over alignments the `a` is more likely than not. Greedy is fast and often right, and this is the shape of its failure.

**Line by line.**

- `collapse` merges a repeated symbol and drops blanks, in that order. Dropping blanks first would merge the two `l` in "hello".
- `alpha[t, s]` in `ctc_forward` is the total probability of all paths that have reached position `s` of the blank-padded target at frame `t`. The skip term `alpha[t - 1, s - 2]` jumps over a blank, and is only allowed when the two letters differ.
- `F.ctc_loss(..., reduction="sum")` expects log-probabilities shaped time x batch x classes, with `blank=0`. Target `[[1, 2]]` is `a`, `b`.

### 2. Word error rate by hand and with jiwer

WER is edit distance on words. The block writes the table that finds the cheapest edits, traces it back to count each kind, and checks the answer against `jiwer`.

```python
import jiwer

def word_errors(reference, hypothesis):
    r, h = reference.split(), hypothesis.split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j - 1] + cost, d[i - 1][j] + 1, d[i][j - 1] + 1)
    i, j, sub, dele, ins = len(r), len(h), 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            sub += r[i - 1] != h[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return sub, dele, ins, len(r)

reference = "the cat sat on the mat"
hypothesis = "the big cat sit on mat"
s, d, i, n = word_errors(reference, hypothesis)
print(f"by hand : S={s} D={d} I={i} N={n}  WER = ({s}+{d}+{i})/{n} = {(s + d + i) / n:.4f}")
out = jiwer.process_words(reference, hypothesis)
print(f"jiwer   : S={out.substitutions} D={out.deletions} I={out.insertions} N={len(reference.split())}  WER = {out.wer:.4f}")

short = "yes"
long_wrong = "yes please send the report now"
print("reference 'yes' against a six word guess:", round(jiwer.wer(short, long_wrong), 4), "(above 1.0 is possible)")
print("character error rate of the same pair:", round(jiwer.cer(reference, hypothesis), 4))
```

**Reading the output.** The reference has six words. The hypothesis has one inserted word (`big`), one substitution (`sat` to `sit`) and one deletion (`the` before `mat`). That is 3 edits in 6 words, WER 0.5, by hand and by jiwer. Two warnings sit in the last lines. WER can exceed 1: a six-word guess for a one-word reference scores 5.0, because every extra word is an insertion. And character error rate (0.4091) is a different, finer measure of the same pair.

**Line by line.**

- `d[i][j]` is the cheapest way to turn the first `i` reference words into the first `j` hypothesis words, from a substitution (diagonal), a deletion (up) or an insertion (left).
- The `while` loop walks back from the corner, preferring the diagonal, to count which kind of edit was used. Ties can be broken differently by different libraries, so S, D and I counts can differ while the total, and so the WER, cannot.

### 3. Whisper tiny.en and base.en on real speech

Now the real thing: 73 clips, 481 seconds, greedy decoding. We score every transcript twice, as written and after the normaliser that ships with the Whisper tokenizer, and then ask whether the bigger model really wins.

```python
import glob, io, time
import jiwer, numpy as np, pandas as pd, soundfile as sf, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.utils import logging

logging.set_verbosity_error()
torch.set_num_threads(4)
path = glob.glob("/Users/sumanth.tp/.cache/huggingface/hub/datasets--hf-internal-testing--librispeech_asr_dummy/snapshots/*/clean/*.parquet")[0]
frame = pd.read_parquet(path)
waves = [sf.read(io.BytesIO(a["bytes"]), dtype="float32")[0] for a in frame.audio]
keep = [i for i, w in enumerate(waves) if len(w) < 30 * 16000]
waves, refs = [waves[i] for i in keep], [frame.text[i] for i in keep]
seconds = sum(len(w) for w in waves) / 16000
print(f"{len(waves)} clips, {seconds:.1f} s of audio, {sum(len(r.split()) for r in refs)} reference words (LibriSpeech dev-clean, CC BY 4.0)")

def run(name):
    processor = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name).eval()
    hyps, start = [], time.perf_counter()
    for wave in waves:
        features = processor(wave, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(features, num_beams=1, do_sample=False)
        hyps.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return hyps, time.perf_counter() - start, processor, model

errors = {}
for name in ("openai/whisper-tiny.en", "openai/whisper-base.en"):
    hyps, wall, processor, model = run(name)
    norm = processor.tokenizer.normalize
    clean_refs, clean_hyps = [norm(r) for r in refs], [norm(h) for h in hyps]
    errors[name] = np.array([jiwer.process_words(r, h).wer * len(r.split()) for r, h in zip(clean_refs, clean_hyps)])
    print(f"{name}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters, wall {wall:.1f} s, real-time factor {wall / seconds:.3f}")
    print(f"   WER as written {jiwer.wer(refs, hyps):.4f}   after normalising both sides {jiwer.wer(clean_refs, clean_hyps):.4f}")
    if name.endswith("tiny.en"):
        print("   as written:", hyps[0])
        print("   reference :", refs[0])
        print("   normalised:", clean_hyps[0])
words = np.array([len(norm(r).split()) for r in refs])
print(f"{words.sum()} reference words after normalising (the normaliser splits possessives such as quilter's into two words)")
tiny, base = errors["openai/whisper-tiny.en"], errors["openai/whisper-base.en"]
print(f"first 20 clips: tiny {tiny[:20].sum() / words[:20].sum():.4f}, base {base[:20].sum() / words[:20].sum():.4f}")
rng = np.random.default_rng(0)
gaps = []
for _ in range(2000):
    pick = rng.integers(0, len(words), len(words))
    gaps.append((tiny[pick].sum() - base[pick].sum()) / words[pick].sum())
low, high = np.percentile(gaps, [2.5, 97.5])
print(f"tiny minus base WER, bootstrap over clips: {np.mean(gaps):+.4f}, 95% interval {low:+.4f} to {high:+.4f}")
```

**Reading the output.** There are 73 clips (all under 30 seconds) with 481.0 seconds of audio. Run as written, `tiny.en` scores a WER of 0.9983: the reference is upper case with no punctuation and the output is `Mr. Quilter is the apostle of the middle classes, and we are glad...`. After normalising both sides the same transcripts score 0.0898, about 9 per cent. The normaliser also turns `Mr.` into `mister` and splits `quilter's` into `quilter is`; it does the same to the reference, so those cancel.

Speed on this CPU: `tiny.en` has 37.8 million parameters and a real-time factor of 0.041 in this run, meaning 481 seconds of speech took about 20 seconds. `base.en` (72.6 million parameters) took 0.071. Timings vary from run to run; across my runs the tiny model ranged from 0.04 to 0.06.

**What surprised me.** The bigger model did not win on the first 20 clips: tiny scored 0.0530 and base 0.0684. On all 73 clips base is ahead, 0.0830 against 0.0898. A bootstrap over clips puts the gap (tiny minus base) at +0.0069 with a 95 per cent interval from -0.0104 to +0.0241. The interval includes zero, so this sample cannot separate the two models. With a few thousand words, a handful of rare names (Leighton, kaliko) move the score by a point. Report an interval, or evaluate on thousands of words, before claiming one model is better.

**Line by line.**

- `processor.tokenizer.normalize` is Whisper's English text normaliser. Passing both reference and hypothesis through it removes formatting differences while leaving real word errors.
- `jiwer.process_words(r, h).wer * len(r.split())` recovers the number of errors in each clip. Sums of errors, not averages of per-clip WER, are what a corpus WER means: a long clip counts for more.
- The bootstrap redraws the 73 clips with replacement 2,000 times and recomputes the tiny-minus-base gap each time.

### 4. Noise, and words from nothing

Real audio is not clean. This block adds white noise at chosen signal-to-noise ratios (SNR, in decibels: 20 dB is faint noise, 0 dB is noise as loud as the speech) to 30 clips and re-scores `tiny.en`. It also prints the shape of the encoder output, and ends by feeding the model audio that contains no speech at all.

```python
import glob, io
import jiwer, numpy as np, pandas as pd, soundfile as sf, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.utils import logging

logging.set_verbosity_error()
torch.set_num_threads(4)
path = glob.glob("/Users/sumanth.tp/.cache/huggingface/hub/datasets--hf-internal-testing--librispeech_asr_dummy/snapshots/*/clean/*.parquet")[0]
frame = pd.read_parquet(path)
waves = [sf.read(io.BytesIO(a["bytes"]), dtype="float32")[0] for a in frame.audio]
keep = [i for i, w in enumerate(waves) if len(w) < 30 * 16000][:30]
waves = [waves[i] for i in keep]
refs = [frame.text[i] for i in keep]

def add_noise(wave, snr_db, seed):
    noise = np.random.default_rng(seed).normal(size=wave.shape).astype("float32")
    scale = np.sqrt(np.mean(wave**2) / (10 ** (snr_db / 10)) / np.mean(noise**2))
    return wave + scale * noise

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").eval()
features = processor(waves[0], sampling_rate=16000, return_tensors="pt").input_features
with torch.no_grad():
    encoded = model.model.encoder(features).last_hidden_state
    ids = model.generate(features, num_beams=1, do_sample=False)
print("log-mel input", tuple(features.shape), "-> encoder output", tuple(encoded.shape), "(20 ms per encoder frame)")
print("clip 0 lasts", round(len(waves[0]) / 16000, 2), "s and the decoder emitted", ids.shape[1], "tokens")

normalise = processor.tokenizer.normalize
print("SNR (dB)   tiny.en WER   example hypothesis")
for snr in (None, 20, 10, 5, 0):
    hyps = []
    for k, wave in enumerate(waves):
        heard = wave if snr is None else add_noise(wave, snr, k)
        feats = processor(heard, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            out = model.generate(feats, num_beams=1, do_sample=False)
        hyps.append(processor.batch_decode(out, skip_special_tokens=True)[0].strip())
    score = jiwer.wer([normalise(r) for r in refs], [normalise(h) for h in hyps])
    label = "clean" if snr is None else f"{snr:>5}"
    print(f"{label:>8}   {score:10.4f}   {normalise(hyps[1])}")

rng = np.random.default_rng(0)
seconds = 8 * 16000
nothing = {"8 s of digital silence": np.zeros(seconds, dtype="float32"),
           "8 s of faint hiss": (0.001 * rng.normal(size=seconds)).astype("float32"),
           "8 s of a 440 Hz tone": (0.2 * np.sin(2 * np.pi * 440 * np.arange(seconds) / 16000)).astype("float32")}
for label, wave in nothing.items():
    feats = processor(wave, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        text = processor.batch_decode(model.generate(feats, num_beams=1, do_sample=False), skip_special_tokens=True)[0]
    print(f"{label:24s} -> {text!r}")
```

**Reading the output.** The log-mel input is `(1, 80, 3000)` and the encoder output `(1, 1500, 384)`: the stride-two convolution halves 3,000 frames to 1,500, one every 20 ms, each a 384-number summary. The clip lasts 5.86 s and the decoder emitted 22 token ids, including the start and end markers.

Word error rate climbs smoothly while the noise grows: 0.0612 clean, 0.0917 at 20 dB, 0.1942 at 10 dB, 0.3237 at 5 dB. At 0 dB it is 0.7104, and the example clip's transcript is `i know there is no secret to this man but let us come to us and we do not lose matters`, a well-formed English sentence that is not in the audio (the clip says `nor is mister quilter's manner less interesting than his matter`). A CTC model fails differently: it tends to produce broken spellings. An attention decoder with a language model inside can produce a confident invention. A fluent transcript is not evidence that the audio supported it. The last three lines feed it no speech at all: 8 seconds of digital silence and of faint hiss both produce the word ` you`, and a pure 440 Hz tone produces ` A`. A model trained on transcripts of real recordings has learned that something is usually said. The Whisper paper lists this failure, "complete hallucination" where the transcript is unrelated to the audio, among its known problems.

**Line by line.**

- `add_noise` scales the noise so that signal power divided by noise power equals `10 ** (snr_db / 10)`. The seed `k` makes each clip's noise reproducible.
- `model.model.encoder(features).last_hidden_state` runs only the encoder, which is where the 1,500 frames come from.

### 5. Where the decoder looks

The last block asks the attention weights themselves: for each token Whisper writes, which stretch of audio does it attend to? Averaging over heads and over layers gives one time per token.

```python
import glob, io
import numpy as np, pandas as pd, soundfile as sf, torch
from scipy.stats import spearmanr
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.utils import logging

logging.set_verbosity_error()
path = glob.glob("/Users/sumanth.tp/.cache/huggingface/hub/datasets--hf-internal-testing--librispeech_asr_dummy/snapshots/*/clean/*.parquet")[0]
wave = sf.read(io.BytesIO(pd.read_parquet(path).audio[1]["bytes"]), dtype="float32")[0]
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", attn_implementation="eager").eval()
features = processor(wave, sampling_rate=16000, return_tensors="pt").input_features
with torch.no_grad():
    out = model.generate(features, num_beams=1, do_sample=False, output_attentions=True, return_dict_in_generate=True)

steps = out.cross_attentions
tokens = [processor.tokenizer.decode(t) for t in out.sequences[0][-len(steps):]]
frames = int(len(wave) / 16000 / 0.02)
print(f"{len(steps)} decoding steps, {len(steps[0])} decoder layers, attention per step {tuple(steps[0][0].shape)} (batch, heads, queries, encoder frames)")

def attended_seconds(layers):
    seconds = []
    for step in steps:
        stacked = torch.stack([step[i][0][:, -1, :frames] for i in layers])
        seconds.append(int(stacked.mean(dim=(0, 1)).argmax()) * 0.02)
    return seconds

for name, layers in (("last layer only", [3]), ("layer 1 only", [1]), ("all four layers", [0, 1, 2, 3])):
    seconds = attended_seconds(layers)
    print(f"{name:16s} rank correlation of token order with attended time: {spearmanr(range(len(seconds)), seconds).statistic:.3f}")
for token, second in zip(tokens, attended_seconds([0, 1, 2, 3])):
    print(f"   {token!r:20} attends to {second:4.2f} s")
```

**Reading the output.** The tokens follow the audio left to right: ` Nor` at 0.64 s, ` Mr` at 1.14 s, ` manner` at 1.92 s, ` than` at 3.32 s, ` matter` at 3.98 s, and the final period at 4.16 s. The clip is 4.8 s long. Nothing told the decoder to move forward; training discovered it. The rank correlation between token order and attended time is 0.985 across all four layers and 1.000 for layer 1 alone.

The surprise is the last layer. On its own it scores only 0.642: its peaks jump around (` Qu` at 1.62 s, the final end marker at 0.24 s). Averaging all four layers hides the weak one. The attention that tracks time sits in the middle of the decoder, and the last layer does something else. If you want word timings out of Whisper, pick heads by testing, not by assuming.

**Line by line.**

- `attn_implementation="eager"` is needed to return attention weights; the faster kernels do not expose them.
- `step[i][0][:, -1, :frames]` takes layer `i`, the single batch item, the last query position (the token being written) and the encoder frames that cover real audio (the rest is padding).
- `int(...argmax()) * 0.02` converts the attended frame index to seconds at 20 ms per frame.

## Try it yourself

Two labs. The first runs the CTC sum of block 1 in your browser, the second the WER arithmetic of block 2. Both are written in TypeScript and were checked against the Python numbers above.

<CtcPathLab />

**What each control does.**

- **Frame 1, 2, 3: P(a) and P(b)** set the network's output for each frame. Blank gets whatever remains, and `b` cannot exceed `1 - a`.
- The bar chart shows each surviving path's probability. The text under the controls shows the sum, the loss and the greedy decode.

**Try it yourself.**

1. Leave the defaults: five paths, total 0.4680, loss 0.7593, greedy `abb` reading `ab`. These are block 1's numbers.
2. Set frame 1 to a 0.20 and b 0.00, frame 2 to a 0.45 and b 0.00, frame 3 to a 0.00 and b 0.80. Greedy now reads `b`, but P(ab) is 0.4480, above P(b) of 0.3520 in block 1. Why: blank wins each of the first two frames narrowly, but summing over alignments favours `a`.
3. Set frame 1 to a 1.00, frame 2 to a 0.00 and b 1.00, frame 3 to b 1.00. One path holds all the probability, P(ab) is 1.0000 and the loss is 0. A confident network leaves no alignment ambiguity to sum over.

<WerLab />

**What each control does.**

- **Example** loads a pair of sentences. **Reference** and **Hypothesis** are editable.
- **Lower-case and drop punctuation** is a minimal normaliser. It is simpler than Whisper's, which also maps `Mr` to `mister`.

**Try it yourself.**

1. With the defaults the WER is 1.0000, 17 substitutions out of 17. This is the "as written" failure of block 3.
2. Tick the box. The WER falls to 0.0588: one substitution, `mr` against `mister`, out of 17. Whisper's normaliser removes that one too, which is how block 3 reaches 0.0898 on all clips.
3. Choose "One word, long guess". WER is 5.0000: five insertions against a one-word reference. A WER above 1 means more words were invented than there were to hear.

## Designing with it

- **Choose by failure, not by headline score.** CTC for streaming and low latency, with a language model attached. An attention model for fluent offline transcripts, with a check for hallucination. Both can be wrapped around the same encoder.
- **Normalise before you compare, and say how.** Publish the normaliser with the number. A WER without its normaliser is not comparable with anything.
- **Evaluate on your own audio.** Read-aloud audiobooks are the easy case. Phone calls, accents, overlapping speakers and domain words all cost accuracy, and a 73-clip sample as here is a smoke test, not a benchmark.
- **Detect invention.** Compare the transcript length with the audio length, watch for repeated phrases, and flag segments with very low audio energy before you transcribe them. Block 4 got ` you` from silence: words from nothing are a failure mode, not a result.
- **Measure speed in real-time factor.** The processing time divided by the audio time. Below 1 means faster than real time; a streaming product needs headroom well below that.

## Where this stands in 2026

Whisper remains a common open baseline: its model card lists Apache 2.0 weights in sizes from 39 million to 1,550 million parameters, and the transformers library loads them in a few lines, as above. CTC and attention are still the two basic designs in current recognisers, often combined. I did not survey newer systems or leaderboards for this chapter, so I make no claim about which model is best today. The method here, scoring your own audio with a stated normaliser and an interval, applies to any of them.

## Common mistakes

1. **Comparing WER numbers with different normalisation.** Scores of 0.9983 and 0.0898 came from the same output. It feels fair to compare "WER on LibriSpeech" across papers. State the normaliser, apply it to both sides and re-score the systems yourself.
2. **Declaring a winner from 20 clips.** On the first 20 clips base looked worse than tiny; on 73 it was ahead; the interval included zero. Use a bootstrap or more audio, and report the interval.
3. **Trusting fluent output.** The 0 dB transcript read like English, and silence produced the word ` you`. Recognisers with a built-in language model produce plausible text for audio that does not support it. Add a check on audio level and on repetition.
4. **Using greedy decoding and calling it the model's best.** In block 1 greedy read `b` while `ab` was more probable. Beam search or a language model recovers such cases, at a cost in speed.
5. **Forgetting the blank rule.** Dropping blanks before merging repeats turns "hello" into "helo". Merge first, drop second, and keep a blank between double letters.

## Practice questions

<details>
<summary><strong>Q1 (Easy).</strong> The CTC rule maps `a a _ b b _ _` to which text, and which path would you need for the text `aa`?</summary>

The repeated `a a` merges to `a`, the blank is dropped, the repeated `b b` merges to `b`, and trailing blanks are dropped, so the text is `ab`. To write `aa` you need a blank between the two letters, for example `a _ a`. Without the blank the merge rule would collapse them.

</details>

<details>
<summary><strong>Q2 (Easy).</strong> A reference has 10 words. The recogniser makes 2 substitutions, 1 deletion and 1 insertion. What is the WER?</summary>

(2 + 1 + 1) / 10 = 0.40. The denominator is the reference length, which is why WER can exceed 1.

</details>

<details>
<summary><strong>Q3 (Medium).</strong> Why does the CTC sum use addition over paths and multiplication along a path?</summary>

Along a path, each frame's symbol is a separate event, and CTC treats frames as independent given the audio, so the probability that all of them occur is the product. Different paths are different, mutually exclusive ways of getting the same text, so the probability that one of them occurs is the sum. In the worked example the five products 0.1080, 0.1800, 0.0900, 0.0720 and 0.0180 add to 0.4680.

</details>

<details>
<summary><strong>Q4 (Medium).</strong> In the chapter's second CTC table, greedy decoding reads `b` but the full sum favours `ab`. Which of the two would you ship, and what would you change to get the better one cheaply?</summary>

Prefer the one with the higher total probability, `ab` at 0.4480 against 0.3520. Greedy is a one-line decoder and loses information by choosing blank in the first two frames, where the margin was small. Beam search keeps several candidate paths and merges the ones that collapse to the same text, so it recovers `ab`. A language model on top would also help, because it knows which word sequences are plausible.

</details>

<details>
<summary><strong>Q5 (Stretch).</strong> Block 3 gives tiny.en 0.0898 and base.en 0.0830 on 73 clips, a gap of about 0.007. A colleague says this proves bigger is better. What would you reply, and what would you run?</summary>

The gap is small compared with the sampling noise. The first 20 clips gave the opposite sign (0.0530 against 0.0684), and a bootstrap over clips gives an interval of -0.0104 to +0.0241 for tiny minus base, which includes zero. I would resample clips, as in the block, and I would add more audio, ideally from the conditions of use. Only if the interval excludes zero on that data would I claim a difference.

</details>

<details>
<summary><strong>Q6 (Stretch).</strong> The last decoder layer's attention had a rank correlation of 0.642 with time, yet the transcript is correct. Does that show the attention is not used?</summary>

No. The correlation measures how closely one layer's strongest attended frame follows the clock, not whether the layer matters. The last layer may be doing something other than aligning: for example attending to a few salient frames or acting as a general summary. The transcript is produced by all layers together. To test whether a layer matters, remove or randomise it and measure WER, which is a different experiment from the one in block 5.

</details>

## Go deeper

All sources were opened on 8 October 2026.

- Graves A, Fernandez S, Gomez F and Schmidhuber J, "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks", ICML 2006 ([paper PDF](https://www.cs.toronto.edu/~graves/icml_2006.pdf)). The original CTC paper, with a TIMIT experiment against an HMM baseline.
- Chan W, Jaitly N, Le QV and Vinyals O, [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211) (arXiv 1508.01211, August 2015). The attention speech model whose abstract is summarised above.
- Radford A, Kim JW, Xu T, Brockman G, McLeavey C and Sutskever I, [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) (arXiv 2212.04356, December 2022). Section 2.2 for the model, the text normaliser discussion in the evaluation section, and the limitations on hallucination.
- Model cards: [openai/whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en) and [openai/whisper-base.en](https://huggingface.co/openai/whisper-base.en), both Apache 2.0. The tiny card self-reports a LibriSpeech test-clean WER of 8.437; this chapter's 0.0898 is on a 73-clip dev-clean sample with a different normaliser setup, so the two are not comparable.
- LibriSpeech: Panayotov V, Chen G, Povey D and Khudanpur S, ICASSP 2015, from [OpenSLR resource 12](https://www.openslr.org/12), CC BY 4.0. The sample used is the `hf-internal-testing/librispeech_asr_dummy` set of dev-clean clips.
- [jiwer](https://pypi.org/project/jiwer/) 4.0.0 for `wer`, `cer` and `process_words`; [torch.nn.functional.ctc_loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.ctc_loss.html) for the loss.

## Check yourself

- I can apply the CTC collapse rule to a path and say why a blank is needed between double letters.
- I can add up the paths for a short target by hand and turn the total into a loss.
- I can explain why greedy CTC decoding can disagree with the most probable text.
- I can compute word error rate by hand, say how it can exceed 1, and compare it with a library.
- I can explain why the same transcript scored 0.9983 and 0.0898, and what to publish alongside a WER.
- I can describe how a recogniser fails in noise and how to test for invented text.

## Where to go next

Next chapter: [speech synthesis and voice agents](/docs/theory/speech/speech-synthesis-and-voice-agents), where the arrow runs the other way, from text to audio, and where we measure the delay between the last word you say and the first word the agent says. A related chapter: [attention for sequence-to-sequence models](/docs/theory/dnn/attention-mechanism-for-seq2seq-models), the mechanism behind the decoder used here.
