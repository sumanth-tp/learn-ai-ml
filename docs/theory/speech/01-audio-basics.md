---
id: speech-audio-basics
title: "Audio Basics: Waveforms, Spectrograms and the Mel Scale"
sidebar_label: "1 · Audio basics"
sidebar_position: 1
slug: /theory/speech/audio-basics
description: "How sound becomes numbers a model can read: sampling and aliasing, the Fourier transform, the window trade-off between pitch and time, and the mel filterbank that Whisper uses, rebuilt in numpy and checked against the library."
tags: [speech, audio, spectrogram, mel-scale, fft, sampling, whisper, scipy]
---

import Infographic from '@site/src/components/Infographic';
import SpectrumResolutionLab from '@site/src/components/viz/SpectrumResolutionLab';

**In one line.** A recogniser never hears sound: it reads a table of numbers made by cutting the waveform into short frames, measuring how much energy each frame has at each pitch, and squeezing the pitches onto a scale that matches the ear.

:::tip Before you start
- **You should already know** what an array and a matrix product are (the dot product is explained in [vector-space term weighting](/docs/theory/ir/vector-space-and-term-weighting)) and what a sine wave looks like from school maths.
- **Reading time:** about 40 minutes, plus about two minutes to run the code and download one small model.
- **After this chapter you can** explain why a sampling rate must be at least twice the highest pitch, choose a window length for a job, build a mel filterbank by hand, reproduce Whisper's input features to within 1e-5 and say what a lower rate costs a recogniser.
:::

:::note Not from a lecture
This chapter was written for this site from the sources under Go deeper. Every number is printed by the code in the chapter. Environment: Python 3.14, NumPy 2.5.3, SciPy 1.18.1, transformers 5.18.0. The real clip comes from LibriSpeech, released under the Creative Commons Attribution 4.0 licence. Sources were opened on 8 October 2026.
:::

## In 30 seconds

Think of a guitar string being plucked. The air moves back and forth hundreds of times a second, and a microphone writes down the air pressure 16,000 times a second. That list of numbers is a waveform. It holds everything, but a model cannot easily see which pitch is in it, just as you cannot see a chord by looking at a wiggly line.

So we cut the line into overlapping pieces 25 thousandths of a second long, and for each piece we ask: how much of each pitch is in here? Stacked side by side, the answers form a picture called a spectrogram. A final squeeze, the mel scale, gives the low pitches more room than the high ones, as your ear does. That picture is what speech models read.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Sample | One reading of air pressure | 16,000 per second in speech work |
| Sampling rate | Readings per second, in hertz (Hz) | 16,000 Hz |
| Nyquist frequency | Half the sampling rate, the highest pitch that can be recorded truthfully | 8,000 Hz at 16,000 Hz |
| Aliasing | A pitch above Nyquist showing up as a false lower pitch | 700 Hz recorded at 1,000 Hz looks like 300 Hz |
| Fourier transform (FFT) | The recipe that says how much of each pitch a piece of signal contains | 8 samples in, 5 pitch measurements out |
| Window | A smooth bump that fades a frame to zero at both ends | Hann window |
| Spectrogram | Frames side by side, each described by its pitch content | 481 frames for a 4.8 s clip |
| Mel scale | A pitch scale spaced the way the ear hears it | 1,000 Hz is 15 mel, 8,000 Hz is 45.2 mel |
| Filterbank | A row of overlapping triangles that add up the energy in neighbouring pitches | 80 triangles |

## The idea in plain words

Start with a question. Someone plays the note A above middle C, 440 vibrations a second. How would you teach a computer to say which note it was, given only air-pressure readings?

You could count how often the wave crosses zero, but that fails the moment two notes play together. The better route is to compare the signal against a set of pure test waves, one per pitch, and note how strongly each one matches. That is the Fourier transform. A strong match at 440 Hz and nothing elsewhere means one pure note. Speech is a mixture of hundreds of such matches that change every few hundredths of a second, which is why we cannot take one transform of a whole sentence. We take many short ones.

Now a smaller example with numbers you can follow by hand. Take eight readings of a wave: 0, 1, 0, -1, 0, 1, 0, -1. The pattern repeats every four readings, so it fits exactly two cycles into the eight. Compare it against a wave that makes `k` cycles over the eight readings, multiplying reading by reading and adding up. For `k = 2` every product has the same sign and the total is 4. For every other `k` the positive and negative products cancel to 0. In words: a pitch bin is large only when the signal contains that many cycles.

Two things follow from this picture. First, a short piece of signal cannot tell close pitches apart, because with only a few readings there are few cycles to compare. Second, if you take readings too slowly, a fast wave can pass through the same readings as a slow wave. That is aliasing, and it is why recording rate matters.

<Infographic src="/img/speech/audio-to-features.svg" alt="Six cards in a row showing the path from 77,040 audio samples through 25 millisecond frames, a Hann window, a 201-bin FFT, an 80-filter mel bank, and a log and scale step, with a result card and a check against the Whisper feature extractor showing a largest difference of 8.0e-06." caption="Read the six cards left to right. The two cards underneath say what comes out and how close our rebuilt version is to the library's." />

## Worked example, step by step

Three small calculations, each reproduced by a code block below.

1. **Pitch bin by hand.** The eight readings above give a magnitude of 4 at `k = 2` and 0 elsewhere, so the list for `k = 0` to `4` is `[0, 0, 4, 0, 0]`. Block 1 computes it with a plain sum and with `np.fft.rfft`.
2. **Aliasing.** A 700 Hz tone recorded at 1,000 Hz readings per second has Nyquist 500 Hz, which is below 700. The tone folds back by the distance to the sampling rate: 1,000 - 700 = 300 Hz. At 800 readings per second it folds to 800 - 700 = 100 Hz. At 4,000 or 16,000 the tone is below Nyquist and stays at 700 Hz.
3. **Mel positions.** The mel scale used by Whisper is a straight line up to 1,000 Hz, `mel = Hz / 66.67`, so 1,000 Hz is 15 mel. Above that it is logarithmic, `mel = 15 + ln(Hz / 1000) / 0.0688`. That puts 4,000 Hz at 35.16 mel and 8,000 Hz at 45.25 mel. Below 1,000 Hz sits 15 / 45.25 = 33 per cent of the scale. With 80 filters spread evenly in mel, 80 x 0.33 = 26.5 of them fall under 1,000 Hz. Block 3 prints 26.

<Infographic src="/img/speech/audio-worked-example.svg" alt="Three panels. The first shows eight samples and a hand discrete Fourier transform with magnitude 4 at bin 2. The second shows a 700 hertz tone seen at 700, 700, 300 and 100 hertz when recorded at 16000, 4000, 1000 and 800 hertz. The third shows the mel scale split into three ranges with shares of 33, 45 and 22 per cent." caption="Start with the left panel, which is small enough to check with a pencil, then compare the middle table with block 1's printed lines." />

## How it works

### Why does the sampling rate matter?

A wave is only pinned down by its readings if there are at least two per cycle. With fewer, several different waves pass through the same readings, and the recorder has no way to tell which one was real. It reports the slowest candidate. The highest pitch that can be recorded truthfully, half the sampling rate, is the Nyquist frequency.

Speech lives mostly below 4,000 Hz, with the sharp "s" and "f" sounds reaching higher, so 16,000 Hz readings (Nyquist 8,000 Hz) are the standard for recognition. Telephone audio at 8,000 Hz cuts everything above 4,000 Hz, which is one reason "f" and "s" are hard to tell apart on a phone. Real recorders put a filter in front of the converter to remove pitches above Nyquist before sampling, because after sampling the damage cannot be undone.

### What does a window do?

A frame cut from a signal has sharp edges. The transform treats the frame as if it repeated forever, so the jump from the last reading back to the first looks like a click, and clicks contain every pitch. Multiplying the frame by a smooth bump that fades to zero at both ends, the Hann window, removes the click. The price is that each pitch spreads a little over its neighbours. That spread is why two close tones can merge into one peak.

The pitch spacing of the transform is the sampling rate divided by the number of readings in one frame. A 400-reading frame at 16,000 Hz gives bins 40 Hz apart. Longer frames give narrower bins, so pitch detail improves while time detail gets worse, because one frame now averages over more of the sound. That is the central trade-off of this chapter, and the second block measures it.

### Why mel?

Pitch perception is not linear. The step from 200 Hz to 300 Hz sounds large, while 6,000 Hz to 6,100 Hz is barely noticeable. The mel scale, built from listening experiments, spaces pitches by how different they sound. Speech models use it for two reasons: it throws away high-pitch detail that carries little meaning, and it cuts the number of features per frame from 201 to 80.

There are two common definitions of the scale. The older HTK formula is `2595 x log10(1 + Hz / 700)`. The one used by librosa and by Whisper (the Slaney version) is straight up to 1,000 Hz and logarithmic after. They place the filters differently, and mixing them produces features that run but give worse results.

<Infographic src="/img/speech/window-tradeoff.svg" alt="Two tables and three cards. The first table lists the smallest gap between two tones that still shows two peaks for window lengths of 5, 10, 25, 50 and 100 milliseconds. The second lists the ridge width of a fast frequency sweep for windows of 5, 25, 100 and 400 milliseconds. Three cards describe short, 25 millisecond and long windows." caption="Left table first: each doubling of the window halves the gap that can be resolved. The right table shows the cost, a sweep smeared once the window is long." />

## Code you can run

The first three blocks are CPU only and finish in seconds. The first needs only numpy, the second adds scipy, and the third downloads the Whisper feature settings (a few kilobytes) and reads a clip from the LibriSpeech sample in the Hugging Face cache. The fourth also downloads `whisper-tiny.en` (about 150 MB) and takes about half a minute.

### 1. Sampling, aliasing and the transform by hand

This block records a 700 Hz tone at four sampling rates and asks the transform which pitch it sees, then checks the eight-reading example against numpy.

```python
import numpy as np

def tone(freq, seconds, sr):
    t = np.arange(int(seconds * sr)) / sr
    return np.sin(2 * np.pi * freq * t)

def peak_hz(x, sr):
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / sr)[spectrum.argmax()]

print("a 700 Hz tone, recorded at different sampling rates")
for sr in (16000, 4000, 1000, 800):
    seen = peak_hz(tone(700, 1.0, sr), sr)
    print(f"  rate {sr:5d} Hz  Nyquist {sr // 2:5d} Hz  strongest frequency seen {seen:6.1f} Hz")

x = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0])
by_hand = [abs(sum(x[n] * np.exp(-2j * np.pi * k * n / 8) for n in range(8))) for k in range(5)]
print("DFT of 8 samples, by hand :", np.round(by_hand, 3))
print("DFT of 8 samples, np.fft  :", np.round(np.abs(np.fft.rfft(x)), 3))
```

**Reading the output.** At 16,000 and 4,000 readings per second the tone is below Nyquist and shows at 700 Hz. At 1,000 it is above Nyquist (500 Hz) and appears at 300 Hz; at 800 it appears at 100 Hz. Those are the 1,000 - 700 and 800 - 700 of the worked example. The two DFT lines match: 4 at bin 2, zeros elsewhere. A wrong sampling rate does not raise an error. It gives a perfectly clean tone at the wrong pitch.

**Line by line.**

- `np.hanning(len(x))` fades the frame so the edges do not add false pitches.
- `np.fft.rfftfreq(len(x), 1 / sr)` gives the pitch of each bin in hertz. `rfft` keeps only the half of the output that is not a mirror image for real signals.
- The `by_hand` list is the definition written as a sum: each reading times a wave of `k` cycles, added up, magnitude taken.

### 2. The window trade-off

We now ask how close two tones can be before one window length merges them, and how well each window follows a sweeping pitch. A tone pair "resolves" when two separate peaks reach at least half the height of the tallest.

```python
import numpy as np
from scipy.signal import chirp, find_peaks, spectrogram

sr = 16000

def peaks_found(f1, f2, ms):
    n = int(sr * ms / 1000)
    t = np.arange(n) / sr
    x = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(n), 16 * n))
    found, _ = find_peaks(spectrum, height=spectrum.max() * 0.5)
    return len(found)

print("a 1000 Hz tone plus a second tone: smallest gap that shows two separate peaks")
for ms in (5, 10, 25, 50, 100):
    n = int(sr * ms / 1000)
    smallest = next(gap for gap in range(5, 400, 5) if peaks_found(1000, 1000 + gap, ms) == 2)
    print(f"  window {ms:3d} ms = {n:4d} samples  bin width {sr / n:6.1f} Hz  smallest gap resolved {smallest:4d} Hz")

t = np.arange(sr) / sr
sweep = chirp(t, f0=500, f1=3500, t1=1.0, method="linear")
print("a sweep from 500 Hz to 3500 Hz in one second (3000 Hz per second)")
for ms in (5, 25, 100, 400):
    n = int(sr * ms / 1000)
    freqs, times, power = spectrogram(sweep, fs=sr, window="hann", nperseg=n, noverlap=n // 2, mode="magnitude")
    column = power[:, np.argmin(np.abs(times - 0.5))]
    middle = times[np.argmin(np.abs(times - 0.5))]
    width = (column >= column.max() / 2).sum() * (freqs[1] - freqs[0])
    print(f"  window {ms:3d} ms  frames {power.shape[1]:4d}  frame at {middle:.3f} s  true {500 + 3000 * middle:6.0f} Hz  ridge {freqs[column.argmax()]:6.1f} Hz  ridge width {width:6.1f} Hz")
```

**Reading the output.** Doubling the window halves the bin width and roughly halves the gap that can be separated: 150 Hz at 5 ms, 30 Hz at 25 ms, 10 Hz at 100 ms. Longer windows separate close pitches.

The sweep table shows the other side. The sweep climbs 3,000 Hz every second. A 5 ms window sees the right pitch but spreads it over 600 Hz, because 5 ms of signal cannot pin a pitch down. A 25 ms window gives the sharpest ridge, 120 Hz. At 100 ms the ridge is wider again, 150 Hz, because the pitch has moved 300 Hz while the window was open. At 400 ms there are only 4 frames in the whole second and the ridge is 602 Hz wide.

**What surprised me.** The smallest gap resolved is smaller than the bin width for windows up to 50 ms: 30 Hz against a 40 Hz bin at 25 ms. Zero-padding the transform draws the peak shape in between the bins, so "bin width equals resolution" is only roughly true. Peaks found this way are also not at the true frequencies: the lab shows a 5 ms window reporting 1,012.5 and 1,137.5 Hz for tones at 1,000 and 1,150 Hz. Close tones pull each other's peaks inward.

**Line by line.**

- `np.fft.rfft(x * np.hanning(n), 16 * n)` pads with zeros to 16 times the length. This does not add information; it only draws the existing spectrum on a finer grid so peaks can be found.
- `find_peaks(..., height=spectrum.max() * 0.5)` counts a bump only if it reaches half the tallest one, so tiny ripples do not count.
- `spectrogram(..., nperseg=n, noverlap=n // 2)` is SciPy's frame-and-transform in one call, with frames overlapping by half.

### 3. A mel filterbank by hand, against Whisper's own

Now the full pipeline on a real clip: frames, window, FFT, mel triangles, log and scale. We write every step in numpy and compare with the feature extractor Whisper ships. If the two agree, we understand the preprocessing.

```python
import glob, io
import numpy as np, pandas as pd, soundfile as sf
from transformers import WhisperFeatureExtractor

path = glob.glob("/Users/sumanth.tp/.cache/huggingface/hub/datasets--hf-internal-testing--librispeech_asr_dummy/snapshots/*/clean/*.parquet")[0]
row = pd.read_parquet(path).iloc[1]
audio, sr = sf.read(io.BytesIO(row.audio["bytes"]), dtype="float32")
print(row.id, f"{len(audio) / sr:.3f} s at {sr} Hz:", row.text)

def hz_to_mel(f):
    f = np.asarray(f, dtype=float)
    return np.where(f >= 1000.0, 15.0 + np.log(np.maximum(f, 1e-9) / 1000.0) / (np.log(6.4) / 27.0), f / (200.0 / 3))

def mel_to_hz(m):
    m = np.asarray(m, dtype=float)
    return np.where(m >= 15.0, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), (200.0 / 3) * m)

def mel_filterbank(n_mels, n_fft, sr):
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    edges = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2))
    bank = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        bank[i] = np.maximum(0, np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid))) * 2.0 / (hi - lo)
    return bank, edges

bank, edges = mel_filterbank(80, 400, sr)
centres = edges[1:-1]
print("filter centres below 1 kHz:", int((centres < 1000).sum()), "of 80;  between 4 kHz and 8 kHz:", int((centres >= 4000).sum()))

n_fft, hop = 400, 160
padded = np.pad(audio, n_fft // 2, mode="reflect")
frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
power = np.abs(np.fft.rfft(frames * np.hanning(n_fft + 1)[:-1], axis=1)) ** 2
log_mel = np.log10(np.maximum(bank @ power[:-1].T, 1e-10))
mine = (np.maximum(log_mel, log_mel.max() - 8.0) + 4.0) / 4.0

extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny.en")
library = extractor(audio, sampling_rate=sr, return_tensors="np").input_features[0]
gap = np.abs(mine - library[:, : mine.shape[1]])
print("my features", mine.shape, " library features", library.shape, "(padded to 30 s)")
print(f"largest difference {gap.max():.1e}, mean difference {gap.mean():.1e}")
print("value range", round(float(mine.min()), 3), "to", round(float(mine.max()), 3))
```

**Reading the output.** The clip lasts 4.815 s, so a hop of 10 ms gives 481 frames of 80 numbers each: shape `(80, 481)`. The library pads to 30 s and returns `(80, 3000)`; we compare the first 481 columns. The largest difference is 8.0e-06 and the mean 1.3e-07, which is float32 rounding. Values lie between -0.704 and 1.296, because Whisper scales its features to roughly -1 to 1. Of 80 filters, 26 are centred below 1,000 Hz and 18 sit between 4,000 and 8,000 Hz: the ear's detail is spent on the low end.

**Line by line.**

- `hz_to_mel` and `mel_to_hz` implement the Slaney scale: a line below 1,000 Hz and a logarithm above. The constant `np.log(6.4) / 27.0` is the step size of the log part.
- Each row of `bank` is a triangle that rises to 1 at its centre and falls to 0 at the neighbours' centres; `2.0 / (hi - lo)` scales it so wide triangles at high pitch do not add up to more energy than narrow ones at low pitch.
- `np.pad(audio, n_fft // 2, mode="reflect")` pads half a frame on each side so the first frame is centred on sample 0, as the library does.
- `np.hanning(n_fft + 1)[:-1]` gives the periodic Hann window the library uses, which differs very slightly from the symmetric one.
- `log_mel.max() - 8.0` clamps the quietest values to eight decades below the loudest, then `(x + 4.0) / 4.0` shifts and scales. Skip this step and every feature is off by a constant, which still runs but pushes inputs away from what the model saw in training.

### 4. What a lower sampling rate costs a real recogniser

Block 1 showed aliasing on a pure tone. This block asks what it does to speech. It takes 25 real clips and rebuilds each at lower rates, either with a proper filtered resampler or by simply dropping samples with no filter, and scores Whisper `tiny.en` on the results (chapter 2 explains the recogniser; here it is only a measuring stick).

```python
import glob, io
import jiwer, numpy as np, pandas as pd, soundfile as sf, torch
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.utils import logging

logging.set_verbosity_error()
torch.set_num_threads(4)
path = glob.glob("/Users/sumanth.tp/.cache/huggingface/hub/datasets--hf-internal-testing--librispeech_asr_dummy/snapshots/*/clean/*.parquet")[0]
frame = pd.read_parquet(path)
waves = [sf.read(io.BytesIO(a["bytes"]), dtype="float32")[0] for a in frame.audio]
keep = [i for i, w in enumerate(waves) if len(w) < 30 * 16000][:25]
waves, refs = [waves[i] for i in keep], [frame.text[i] for i in keep]

def share_above(wave, cutoff_hz, sr=16000):
    power = np.abs(np.fft.rfft(wave)) ** 2
    freqs = np.fft.rfftfreq(len(wave), 1 / sr)
    return power[freqs >= cutoff_hz].sum() / power.sum()

print("share of signal power above 4 kHz in the 25 clips:", f"{np.mean([share_above(w, 4000) for w in waves]):.4f}")
print("share of signal power above 2 kHz in the 25 clips:", f"{np.mean([share_above(w, 2000) for w in waves]):.4f}")

def filtered(wave, rate):
    low = resample_poly(wave, rate, 16000)
    return resample_poly(low, 16000, rate).astype("float32")[: len(wave)]

def naive(wave, factor):
    low = wave[::factor]
    return np.repeat(low, factor).astype("float32")[: len(wave)]

conditions = {
    "original 16 kHz": waves,
    "8 kHz, filtered": [filtered(w, 8000) for w in waves],
    "4 kHz, filtered": [filtered(w, 4000) for w in waves],
    "8 kHz, naive drop of every 2nd sample": [naive(w, 2) for w in waves],
    "4 kHz, naive drop of 3 in 4 samples": [naive(w, 4) for w in waves],
}
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").eval()
normalise = processor.tokenizer.normalize
print(f"{'condition':40s} WER on {len(waves)} clips")
for name, clips in conditions.items():
    hyps = []
    for clip in clips:
        feats = processor(clip, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            hyps.append(processor.batch_decode(model.generate(feats, num_beams=1, do_sample=False), skip_special_tokens=True)[0])
    print(f"{name:40s} {jiwer.wer([normalise(r) for r in refs], [normalise(h) for h in hyps]):.4f}")
```

**Reading the output.** On average 6.9 per cent of the signal power in these clips lies above 4 kHz, and 8.2 per cent above 2 kHz, so nearly all of the power that sits above 2 kHz is above 4 kHz: the hissing "s" and "f" sounds. The original clips score a WER of 0.0551. Filtered to 8 kHz (the telephone rate) the score is 0.0886, and filtered to 4 kHz it is 0.1949. Dropping samples without a filter is worse at the same rate: 0.1122 at 8 kHz and 0.4055 at 4 kHz, twice the damage of the filtered version.

The result has two parts. First, band-limiting costs accuracy in proportion to what it removes, even though the lost band holds a small share of the power: those few per cent carry the consonants. Second, aliasing is an extra loss on top, because the unfiltered version folds high pitches down onto the speech band instead of merely deleting them.

:::note A fair-comparison caveat
The naive branch repeats each kept sample to return to 16 kHz, which adds its own blockiness, while the filtered branch uses a polyphase resampler in both directions. So the gap between the two mixes aliasing with a cruder way back up. The ordering is what I would trust; the size of the gap is not a clean measurement of aliasing alone.
:::

**Line by line.**

- `share_above` sums the power of the FFT bins at or above a cutoff and divides by the total, per clip, then block 4 averages over clips.
- `resample_poly(wave, rate, 16000)` low-pass filters and changes the rate by the ratio of two integers. Calling it again in the other direction returns to 16 kHz with the band above the lower Nyquist removed.
- `wave[::factor]` keeps every `factor`-th sample with no filter, which is what an unfiltered decimation does, and `np.repeat` holds each kept value to refill the original length.

## Try it yourself

The lab uses the same Hann-windowed transform as block 2, written in TypeScript and checked against the Python numbers. The defaults reproduce the 25 ms row of block 2.

<SpectrumResolutionLab />

**What each control does.**

- **Window** is the frame length. It sets the number of readings and the bin width.
- **Gap to second tone** places a second tone above a fixed 1,000 Hz tone.
- **Tone played** and **Sampling rate** drive the aliasing readout: the pitch you would see after recording.

**Try it yourself.**

1. Keep 25 ms and set the gap to 30 Hz. Two peaks remain, at 997.5 and 1,032.5 Hz. Set 25 Hz and they merge into one at 1,012.5 Hz. Why: block 2's table says 25 ms resolves 30 Hz and not less.
2. Set the window to 5 ms and the gap to 150 Hz, then 145 Hz. Two peaks, then one. At 5 ms the bin is 200 Hz wide, so you need nearly one bin of separation.
3. With the tone at 700 Hz, set the sampling rate to 1,000 Hz. The readout says 300 Hz. Raise the rate to 4,000 Hz and it returns to 700 Hz. Why: 700 Hz is above Nyquist (500 Hz) only in the first case.

## Designing with it

- **Pick the rate from the content, not from habit.** 16,000 Hz covers speech, and block 4 shows what lower rates cost: WER 0.0551 at 16 kHz, 0.0886 at 8 kHz and 0.1949 at 4 kHz. Music wants 44,100 Hz or more. Resample once, at the start, with a filtering resampler, and keep a record of the original rate.
- **Match the front end to the model.** A model trained on 25 ms windows, 10 ms hops, 80 Slaney-mel bands and Whisper's log scaling expects exactly that. A different mel scale, a different window or a missing clamp will not crash, but accuracy drops quietly.
- **Choose the window for the job.** Speech recognition keeps 25 ms because speech changes character every few hundredths of a second. As rules of thumb, not laws, pitch tracking for music wants a longer window, 50 to 100 ms, and detecting a click or a clap wants a shorter one, 5 to 10 ms.
- **Treat the spectrogram as an image when it helps.** The table has a time axis and a pitch axis, so convolutional networks (see [convolutional networks](/docs/theory/dnn/what-a-convolutional-neural-network-is)) work on it directly. Whisper instead feeds the columns, one per time step, into a Transformer.

## Where this stands in 2026

The log-mel spectrogram is still the default input for speech recognition, including Whisper, whose paper specifies 16,000 Hz audio and an 80-channel log-magnitude mel spectrogram on 25 ms windows with a 10 ms stride. Research has also moved toward models that read the waveform or learned filters directly, but the mel front end remains the common baseline because it is cheap, stable and well understood. The next chapter uses the exact features built here to drive a real recogniser.

## Common mistakes

1. **Resampling without a filter.** Dropping samples to lower the rate folds the energy above the new Nyquist frequency into the band you keep. It feels harmless because the file plays. In block 4, unfiltered 4 kHz audio scored a WER of 0.4055 against 0.1949 for the filtered version of the same rate. Use a proper resampler, which filters first.
2. **Mixing mel scales.** A filterbank built with the HTK formula and a model trained on Slaney features both run. The mismatch only shows as a small accuracy loss that nobody traces back. Copy the front end that the model's card or code defines.
3. **Reading bin width as resolution.** A 40 Hz bin does not mean a 40 Hz limit: this chapter's 25 ms window split 30 Hz, and also pulled peaks away from the true pitch when tones were close. Test the claim you rely on with synthetic tones.
4. **Skipping the window.** A rectangular frame leaks energy from loud pitches into quiet ones, which fills the gaps between speech harmonics with false energy. Always window before the transform.
5. **Forgetting amplitude scale.** Features computed from audio in the range -1 to 1 and from 16-bit integers differ by 32,768 times. The log hides the difference as a constant shift that still ruins a model. Convert to float in -1 to 1 first.

## Practice questions

<details>
<summary><strong>Q1 (Easy).</strong> A 3,000 Hz tone is recorded at 4,000 samples per second. What pitch do you see, and why?</summary>

Nyquist is 2,000 Hz, so 3,000 Hz is too high. It folds back by the distance to the sampling rate: 4,000 - 3,000 = 1,000 Hz. A clean tone appears at 1,000 Hz and nothing warns you. The same calculation gave 300 Hz for the 700 Hz tone at 1,000 samples per second.

</details>

<details>
<summary><strong>Q2 (Easy).</strong> A frame has 400 readings at 16,000 Hz. How far apart are its pitch bins, and how many bins does `rfft` return?</summary>

The spacing is the rate divided by the readings: 16,000 / 400 = 40 Hz. A real signal of 400 readings gives 400 / 2 + 1 = 201 bins, from 0 to 8,000 Hz. Those 201 values are what the 80 mel triangles then add up.

</details>

<details>
<summary><strong>Q3 (Medium).</strong> Why does a window of 100 ms separate closer pitches than 25 ms, yet speech recognition still uses 25 ms?</summary>

A longer window contains more cycles of each pitch, so close pitches can be told apart. But one frame also averages over its whole length, so any change inside it is blurred. In speech the pitch and the sound shape change every few hundredths of a second. Block 2's sweep shows the cost: ridge width is 120 Hz at 25 ms and grows to 150 Hz at 100 ms because the pitch moved 300 Hz while the window was open.

</details>

<details>
<summary><strong>Q4 (Medium).</strong> Whisper pads every clip to 30 s, so a 4.815 s clip becomes 3,000 frames. What fraction of the table is real audio, and why is the padding harmless?</summary>

4.815 s gives 481 real frames, so 481 / 3,000 = 16 per cent. The rest is silence, which the model was trained to see after short clips. The cost is wasted compute: the encoder processes 1,500 frames after striding for a clip that needed about 240.

</details>

<details>
<summary><strong>Q5 (Stretch).</strong> Block 3 matches the library to 8.0e-06. Name two changes that would still produce a "working" pipeline but break that match.</summary>

Using the HTK mel formula instead of the Slaney one moves every filter centre. Removing the `max - 8` clamp, or the `(x + 4) / 4` scaling, shifts the output by a constant or changes its range. Both run without errors, and both put the features outside what the model saw in training. The way to detect it is the check in block 3: compare against the reference implementation on a real clip.

</details>

<details>
<summary><strong>Q6 (Medium).</strong> Only 8.2 per cent of the signal power in speech lies above 2 kHz, yet filtering to 4 kHz more than tripled the error rate in block 4 (0.0551 to 0.1949). Why does a small share of power matter so much?</summary>

Power is dominated by vowels, which are loud and low in pitch. Consonants such as "s", "f" and "t" are quiet and high, and they are what separates "sip" from "tip" or "fit" from "sit". The recogniser needs those few per cent of the power to tell words apart. Counting power is the wrong way to measure how much information a band carries.

</details>

<details>
<summary><strong>Q7 (Stretch).</strong> Two tones 30 Hz apart split into two peaks at 25 ms in block 2, even though the bin is 40 Hz wide. Does that mean the 40 Hz bin width is wrong?</summary>

No. The bin width is the spacing of the transform's output when it uses exactly as many points as readings. Zero-padding computes the same underlying curve at finer points, so a peak that falls between two bins can still be found. The curve itself has a width that depends on the window: a Hann window's main lobe is about four bins wide, yet the half-height rule found two peaks. The honest summary is that resolution depends on the window, on how you detect a peak and on the signal, and the number to trust is the one you measure.

</details>

## Go deeper

All sources were opened on 8 October 2026.

- Radford A, Kim JW, Xu T, Brockman G, McLeavey C and Sutskever I, [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) (arXiv 2212.04356, December 2022). Section 2.2 gives the 16,000 Hz, 80-channel, 25 ms and 10 ms front end that block 3 rebuilds. The paper is on arXiv; the model weights are released under Apache 2.0.
- Panayotov V, Chen G, Povey D and Khudanpur S, "LibriSpeech: an ASR corpus based on public domain audio books", ICASSP 2015. The corpus is about 1,000 hours of 16 kHz read English speech under CC BY 4.0, from [OpenSLR resource 12](https://www.openslr.org/12). The clip used here is from the dev-clean subset.
- [Hugging Face model card for openai/whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en): Apache 2.0 licence, the 39 million parameter tiny size and the training-data description.
- [SciPy signal documentation](https://docs.scipy.org/doc/scipy/reference/signal.html) for `spectrogram`, `chirp` and `find_peaks`, as run in block 2.

## Check yourself

- I can say what pitch a tone will appear at when it is recorded below twice its frequency, and compute it.
- I can state why a frame is windowed before the transform and what the window costs.
- I can compute the bin spacing from the rate and the frame length, and explain why it is not the same as resolution.
- I can explain why speech work uses the mel scale and name the two common definitions.
- I can rebuild Whisper's log-mel features in numpy and say which three steps would silently break the match.
- I can say what lowering the sampling rate with and without a filter did to a recogniser's error rate, using the numbers from block 4.

## Where to go next

Next chapter: [speech recognition with CTC, attention and Whisper](/docs/theory/speech/speech-recognition), where these features feed a real recogniser and we compute word error rate. A related chapter: [convolutional networks](/docs/theory/dnn/what-a-convolutional-neural-network-is), for treating the spectrogram as an image.
