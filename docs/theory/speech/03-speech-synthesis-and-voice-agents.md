---
id: speech-synthesis-voice-agents
title: "Speech Synthesis and Voice Agents: Vocoders, Latency Budgets and Turn-Taking"
sidebar_label: "3 · Synthesis and voice agents"
sidebar_position: 3
slug: /theory/speech/speech-synthesis-and-voice-agents
description: "How text becomes speech in two stages (an acoustic model and a neural vocoder), what each costs on a laptop CPU, and how the stage times, an end-of-turn rule and sentence streaming set the delay between your last word and the agent's first sound."
tags: [speech, tts, vocoder, hifi-gan, speecht5, griffin-lim, latency, turn-taking, voice-agents]
---

import Infographic from '@site/src/components/Infographic';
import ReplyLatencyLab from '@site/src/components/viz/ReplyLatencyLab';

**In one line.** A synthesiser first plans the sound as a mel spectrogram and then a vocoder fills in the waveform, and in a voice agent the delay you feel is the end-of-turn wait plus every stage that follows it, shortened by starting to speak before the reply is finished.

:::tip Before you start
- **You should already know** what a log-mel spectrogram is ([audio basics](/docs/theory/speech/audio-basics)) and how a recogniser is scored with word error rate ([speech recognition](/docs/theory/speech/speech-recognition)).
- **Reading time:** about 50 minutes, plus about six minutes to run all the code and download three small models.
- **After this chapter you can** explain what an acoustic model and a vocoder each do, measure a synthesiser's real-time factor, compare a classical and a neural vocoder, and compute when a voice agent's first sound arrives from its stage times.
:::

:::note Not from a lecture
This chapter was written for this site from the sources under Go deeper. Every number is printed by the code in the chapter, on a laptop CPU shared with other jobs, so timings moved by up to three times between my runs and the ranges are quoted where it matters. Environment: Python 3.14, torch 2.14.1, transformers 5.18.0, SciPy 1.18.1, jiwer 4.0.0. Models: `microsoft/speecht5_tts` and `microsoft/speecht5_hifigan` (MIT), `openai/whisper-tiny.en` and `openai/whisper-base.en` (Apache 2.0), `HuggingFaceTB/SmolLM2-135M-Instruct`, and speaker vectors from the `Matthijs/cmu-arctic-xvectors` dataset (MIT). Test audio is LibriSpeech dev-clean (CC BY 4.0). Sources were opened on 8 October 2026.
:::

## In 30 seconds

Think of an actor given a script. First they decide how the line should sound: where to pause, which words rise, how fast. Then they actually produce the sound. A speech synthesiser splits the work the same way. The first stage, the acoustic model, writes a rough picture of the sound, a mel spectrogram. The second stage, the vocoder, turns that picture into the actual waveform your speaker plays.

A voice agent adds three more waits around it: it must decide you have stopped talking, transcribe you, and think of a reply. Add the times up and you get the pause you feel before the agent speaks. Cutting that pause is mostly about doing stages at the same time instead of one after another.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Acoustic model | The stage that turns text into a mel spectrogram | 184 frames of 80 numbers for a 2.9 s sentence |
| Vocoder | The stage that turns a mel spectrogram into a waveform | HiFi-GAN, 12.7 million parameters |
| Speaker embedding | A vector that says whose voice to imitate | 512 numbers |
| Griffin-Lim | A classical vocoder that guesses the missing phase by repeating transform and inverse transform | 32 passes |
| Real-time factor (RTF) | Time to make the audio divided by the length of the audio | 0.24 means 1 second of speech takes 0.24 s |
| End-of-turn wait | How long the agent waits in silence before deciding you have finished | 700 ms |
| Time to first audio | From your last word to the first sound from the agent | 1,570 ms in the lab's default |
| Streaming by sentence | Synthesising and playing sentence 1 while sentence 2 is still being written | saves 598 ms in the lab's default |
| Barge-in | You interrupt while the agent is talking | covered in the voice-agent chapter |

## The idea in plain words

Start with the size of the problem. One second of speech at 16,000 samples a second is 16,000 numbers, and they must be exactly right or you hear a buzz. Writing 16,000 numbers directly from the words "hello there" is hard. A mel spectrogram is much smaller: with a hop of 256 samples, one second is 62.5 frames of 80 numbers, about 5,000 numbers, and it describes the sound in terms a model can plan.

Here is the smallest example with numbers. A 2-second reply at 16,000 samples a second is 32,000 samples. At 256 samples per frame that is 32,000 / 256 = 125 frames, each 16 ms long. The acoustic model writes 125 x 80 = 10,000 numbers. The vocoder then expands each frame back into 256 samples, 125 x 256 = 32,000. In words: the first stage works at 1 / 256 of the final time resolution, and the vocoder supplies the rest.

A spectrogram of magnitudes is not enough to make a waveform. It lacks the **phase**, which says where in its cycle each frequency component is. Two signals with the same magnitudes and different phases sound different. Griffin-Lim guesses the phase by a loop: pick random phases, build a signal, analyse it again, keep the new phases but restore the target magnitudes, repeat. A neural vocoder instead learns to produce the waveform directly from many recordings.

Now the delay. You say your last word. The agent waits 700 ms to be sure you have stopped, transcribes you in 141 ms, takes 142 ms to start its reply and then writes at 88 tokens a second, and the synthesiser needs 0.22 s of work for each second of speech. If the agent waits for its whole two-sentence reply, you hear nothing for 2,168 ms. If it speaks as soon as the first sentence is written and made, you hear the first sound at 1,570 ms and the second sentence is ready long before the first finishes. In words: streaming does not make any stage faster, it only stops later stages from delaying the first sound.

<Infographic src="/img/speech/tts-two-stages.svg" alt="A four-box pipeline: text, a SpeechT5 acoustic model of 144.4 million parameters writing 184 mel frames, a HiFi-GAN vocoder of 12.7 million parameters expanding each frame to 256 samples, and 2.94 seconds of audio. Below, a table of round-trip results through Whisper and two boxes: where it breaks (digits and abbreviations) and why there are two stages." caption="Read the top row left to right: the numbers between boxes are the shapes that pass between stages. Then the table: speaking a sentence and listening to it again is a cheap test of the synthesiser." />

## Worked example, step by step

Two hand calculations, matched by the code below.

**Frames to samples.**

1. A sentence of 184 mel frames. The hop is 256 samples at 16,000 Hz, so each frame is 256 / 16,000 = 0.016 s.
2. Duration: 184 x 0.016 = 2.944 s, which the block prints as 2.94 s.
3. Samples: 184 x 256 = 47,104. Block 1 prints "samples per mel frame: 256" for the last sentence.

**The first-sound delay.** Use the lab's defaults: end-of-turn wait 700 ms, speech to text 141 ms, model first token 142 ms at 88 tokens per second, sentences of 12 tokens, 2.1 s of audio per sentence at real-time factor 0.22.

1. Text ready: 700 + 141 = 841 ms.
2. First sentence written: the first token arrives 142 ms later, and the other 11 tokens take 11 / 88 = 0.125 s. 841 + 142 + 125 = 1,108 ms.
3. First sentence made: 0.22 x 2.1 s = 0.462 s of synthesis. 1,108 + 462 = 1,570 ms, when sound begins.
4. Whole reply instead: 24 tokens take 142 + 23 / 88 x 1,000 = 403 ms after 841, so the text is ready at 1,244 ms. Both sentences take 2 x 462 = 924 ms to make. 1,244 + 924 = 2,168 ms.
5. Saved by streaming: 2,168 - 1,570 = 598 ms.

<Infographic src="/img/speech/reply-latency.svg" alt="Two stacked bars. Streamed by sentence: a 700 millisecond end-of-turn wait, 141 milliseconds of speech to text, 267 milliseconds of language model and 462 milliseconds of synthesis, first audio at 1,570 milliseconds. Waiting for the whole reply: the same wait and speech to text, 403 milliseconds of language model and 924 milliseconds of synthesis, first audio at 2,168 milliseconds. Below, a table of measured stage times and a table of how many pauses inside read speech each end-of-turn wait would cut." caption="Compare the two bars first: the wait and the speech to text are identical, and the saving comes from the purple and green segments. The right-hand table is the reason the blue segment cannot simply be made small." />

## How it works

### What does the acoustic model do?

It reads text (as token ids, often letters or phonemes) and writes mel frames, one at a time, each depending on the ones before it. SpeechT5 is an encoder-decoder Transformer from the paper by Ao and colleagues (2021, ACL 2022) that handles both speech and text, with small input and output networks for each type of data. For synthesis it is conditioned on a **speaker embedding**: a 512-number x-vector computed from recordings of a real speaker. The model card for `microsoft/speecht5_tts` lists LibriTTS as its training data and shows the embedding being loaded from the CMU ARCTIC x-vector set.

The decoder is autoregressive, so the cost grows with the length of the sentence, and it was the larger share of the work in every run of the first block: 0.3 to 0.9 s against 0.1 to 0.3 s for the vocoder.

### What does a neural vocoder add?

HiFi-GAN (Kong, Kim and Bae, 2020) is a generative adversarial network: a generator that produces waveform from a mel spectrogram, and discriminators that try to tell its output from real recordings. The paper's abstract reports synthesis 167.9 times faster than real time on one V100 GPU for 22.05 kHz audio, and a small version at 13.4 times real time on a CPU. The version used here has 12.7 million parameters and runs in about 0.1 to 0.3 s for a 3-second sentence on this CPU. Because it was trained on real speech, it supplies phase and fine detail that the spectrogram does not contain.

### Why does the text need cleaning first?

The acoustic model sees characters. Numbers, abbreviations and symbols such as "3 PM" or "21" have no spelling that tells the model how to say them. Real systems put a text-normalisation step in front that expands them into words. Block 1 shows what happens without it.

### What sets the delay in a voice agent?

Four things in series: the end-of-turn wait, speech to text, the language model, and synthesis, plus network time if any stage is remote. Two of them are choices rather than costs. The end-of-turn wait is a rule you pick, and it trades speed against cutting people off mid-thought. Streaming by sentence is a design you choose, and it works only while synthesis stays ahead of playback, which is exactly what a real-time factor below 1 means.

People are fast. The Stivers and colleagues study of turn-taking in 10 languages found the average gap between turns within 250 ms of the cross-language mean, and the figure usually quoted for a typical gap is around 200 ms (I could not confirm that number in the full text, so treat it as the commonly cited value). An agent whose first sound arrives 1,570 ms after you finish will feel slow however good its voice is. The voice-agent chapter ([voice and realtime agents](/docs/agentic-frontier/voice-and-realtime-agents)) measures turn-ending with a trained detector and covers interruption; this chapter's blocks 4 to 6 give the budget and the synthesis side.

<Infographic src="/img/speech/vocoder-comparison.svg" alt="Two tables and two notes. The first table compares Griffin-Lim with 1, 8, 32 and 128 passes against HiFi-GAN by time and relative mel error, from 0.2541 down to 0.0929 for Griffin-Lim and 0.1214 for HiFi-GAN. The second shows Whisper base.en word error rates of 0.0699 on the original, 0.0591 after HiFi-GAN and 0.0699 after Griffin-Lim. The notes say mel error is not naturalness and that listening tests were not run." caption="Left table first: more Griffin-Lim passes lower the mel error, and 128 passes beat the neural vocoder on that number. The right table and the red note explain why that number is not the whole story." />

## Code you can run

Six blocks. Blocks 1, 3 and 5 download models on first use (SpeechT5 about 600 MB with its vocoder, Whisper base.en about 290 MB, SmolLM2 about 270 MB). Everything is CPU only. Speech synthesis with SpeechT5 is random unless seeded, because its decoder applies dropout at inference, so the blocks call `torch.manual_seed(0)` before each sentence. If the Hub is slow once the models are cached, set `HF_HUB_OFFLINE=1`.

### 1. Speak it, then listen to it

The block runs the two-stage synthesiser on four sentences, times each stage, and sends the audio to Whisper `base.en` to see whether a recogniser can read it back.

```python
import io, time, zipfile
import jiwer, numpy as np, torch
from huggingface_hub import hf_hub_download
from transformers import (SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor,
                          WhisperForConditionalGeneration, WhisperProcessor)
from transformers.utils import logging

logging.set_verbosity_error()
torch.set_num_threads(4)
tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").eval()
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").eval()
print(f"acoustic model {sum(p.numel() for p in tts.parameters()) / 1e6:.1f} M parameters, vocoder {sum(p.numel() for p in vocoder.parameters()) / 1e6:.1f} M")

archive = zipfile.ZipFile(hf_hub_download("Matthijs/cmu-arctic-xvectors", "spkrec-xvect.zip", repo_type="dataset"))
names = sorted(n for n in archive.namelist() if n.endswith(".npy"))
speaker = torch.tensor(np.load(io.BytesIO(archive.read(names[7306])))).unsqueeze(0)
print("speaker embedding", tuple(speaker.shape), "from", names[7306].split("/")[-1])

asr_processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
asr = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en").eval()
normalise = asr_processor.tokenizer.normalize

sentences = ["The quick brown fox jumps over the lazy dog.",
             "Please transfer two hundred and fifty pounds to the savings account.",
             "The meeting is at 3 PM on 21 June.",
             "The meeting is at three in the afternoon on the twenty first of June."]
with torch.no_grad():
    tts.generate_speech(tts_processor(text="warm up", return_tensors="pt")["input_ids"], speaker)

print(f"{'frames':>6} {'seconds':>7} {'acoustic':>9} {'vocoder':>8} {'RTF':>5}  WER after Whisper")
for text in sentences:
    ids = tts_processor(text=text, return_tensors="pt")["input_ids"]
    torch.manual_seed(0)
    start = time.perf_counter()
    with torch.no_grad():
        mel = tts.generate_speech(ids, speaker)
        middle = time.perf_counter()
        audio = vocoder(mel).numpy()
    end = time.perf_counter()
    seconds = len(audio) / 16000
    feats = asr_processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        heard = asr_processor.batch_decode(asr.generate(feats, num_beams=1, do_sample=False), skip_special_tokens=True)[0]
    score = jiwer.wer(normalise(text), normalise(heard))
    print(f"{mel.shape[0]:6d} {seconds:7.2f} {middle - start:8.2f}s {end - middle:7.2f}s {(end - start) / seconds:5.2f}  {score:.3f}")
    print(f"       typed : {text}")
    print(f"       heard : {heard.strip()}")
print("samples per mel frame:", len(audio) // mel.shape[0])
```

**Reading the output.** The first sentence is 184 mel frames, 2.94 s of audio, so 184 x 256 = 47,104 samples. The acoustic model took about 0.5 s and the vocoder about 0.2 s, so the real-time factor was 0.24 to 0.29 in my runs. The larger of the two costs is the acoustic model. The recogniser read the first two sentences back with a WER of 0.000. For the second, Whisper wrote `£250` for "two hundred and fifty pounds"; Whisper's normaliser maps number words to digits on both sides, so it still counts as a match.

**What surprised me.** The third sentence, `The meeting is at 3 PM on 21 June.`, came back as `The meeting is at come on June.`, a WER of 0.333: `3 PM` became `come` and the `21` vanished. When the same date was written out in words (`at three in the afternoon on the twenty first of June`) the WER was 0.000. I did not listen to the audio, so I cannot say what the synthesiser actually said; I can say a recogniser could not recover it. The fix to try first is a text-normalisation step that spells numbers and abbreviations out.

**Line by line.**

- `tts.generate_speech(ids, speaker)` runs the acoustic model and returns the mel frames as a tensor of shape frames x 80. `vocoder(mel)` then returns the waveform.
- `torch.manual_seed(0)` fixes the dropout noise so the frame count and the transcript are the same on every run. Without it the first sentence gave 186, 190 and 184 frames on different runs.
- The warm-up call before the loop avoids counting one-time start-up work in the first timing.
- `normalise = asr_processor.tokenizer.normalize` is the same normaliser as in the recognition chapter, applied to the typed text and to what was heard.

### 2. Griffin-Lim from scratch against HiFi-GAN

Now the vocoder alone. We take the mel spectrogram of a real 4.82 s clip, rebuild a linear-frequency magnitude from it with a pseudo-inverse of the mel filters, and run Griffin-Lim for 1, 8, 32 and 128 passes, then HiFi-GAN. Each output is analysed back to mel and compared with the target.

```python
import glob, io, time
import numpy as np, pandas as pd, soundfile as sf, torch
from scipy.signal import istft, stft
from transformers import SpeechT5HifiGan, SpeechT5Processor
from transformers.utils import logging

logging.set_verbosity_error()
path = glob.glob("/Users/sumanth.tp/.cache/huggingface/hub/datasets--hf-internal-testing--librispeech_asr_dummy/snapshots/*/clean/*.parquet")[0]
wave = sf.read(io.BytesIO(pd.read_parquet(path).audio[1]["bytes"]), dtype="float32")[0]

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").eval()
filters = np.asarray(processor.feature_extractor.mel_filters)
hop, win = 256, 1024

def log_mel(signal):
    return processor(audio_target=signal, sampling_rate=16000, return_tensors="pt").input_values[0].numpy()

target = log_mel(wave)
band = np.maximum(10.0 ** target, 1e-10)
magnitude = np.maximum(np.linalg.pinv(filters.T) @ band.T, 0.0) / (win / 2)
print("mel", target.shape, "-> linear magnitude", magnitude.shape, "(513 bins by frames)")

def griffin_lim(magnitude, iterations):
    phase = np.exp(2j * np.pi * np.random.default_rng(0).random(magnitude.shape))
    for _ in range(iterations):
        _, signal = istft(magnitude * phase, nperseg=win, noverlap=win - hop)
        _, _, spec = stft(signal, nperseg=win, noverlap=win - hop)
        phase = np.ones(magnitude.shape, dtype=complex)
        phase[:, : spec.shape[1]] = np.exp(1j * np.angle(spec[:, : magnitude.shape[1]]))
    _, signal = istft(magnitude * phase, nperseg=win, noverlap=win - hop)
    return signal.astype("float32")

def mel_error(signal):
    other = log_mel(signal)
    frames = min(len(other), len(target))
    got, want = 10.0 ** other[:frames], 10.0 ** target[:frames]
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))

print(f"{'method':18s} {'seconds':>8s} {'relative mel error':>20s}")
for iterations in (1, 8, 32, 128):
    start = time.perf_counter()
    signal = griffin_lim(magnitude, iterations)
    print(f"Griffin-Lim {iterations:4d}   {time.perf_counter() - start:8.2f} {mel_error(signal):20.4f}")
start = time.perf_counter()
with torch.no_grad():
    neural = vocoder(torch.tensor(target)).numpy()
print(f"{'HiFi-GAN':18s} {time.perf_counter() - start:8.2f} {mel_error(neural):20.4f}")
print("audio length", round(len(wave) / 16000, 2), "s, HiFi-GAN parameters", round(sum(p.numel() for p in vocoder.parameters()) / 1e6, 2), "M")
```

**Reading the output.** The mel input is 301 frames by 80 and the rebuilt magnitude is 513 bins by 301 frames. Griffin-Lim's error falls with every extra pass: 0.2541 after one, 0.1609 after 8, 0.1168 after 32 and 0.0929 after 128, while the time grows from 0.01 s to 1.07 s. HiFi-GAN took 0.26 s and its error is 0.1214. Each error is the norm of the difference between the output's mel and the target's mel, divided by the norm of the target (in linear amplitude, not log).

**What surprised me.** The classical method beat the neural one on this number. Griffin-Lim has no training: it is an optimisation that is designed to match the magnitudes, so it does. HiFi-GAN is trained to sound like real speech, not to reproduce the input spectrogram exactly. Griffin and Lim's 1984 paper shows that the error between the spectrogram magnitudes of the estimate and the target decreases with each pass, which is the trend in the table. A low mel error is evidence of consistency with the input, not of naturalness.

**Line by line.**

- `np.linalg.pinv(filters.T) @ band.T` maps 80 mel bands back to 513 linear bins. Mel bands were an averaging step, so the mapping is only approximate; negative values are clipped to zero.
- `/ (win / 2)` corrects a scale difference: SciPy's transform divides by half the window length and the feature extractor does not.
- `phase = np.exp(2j * np.pi * rng.random(...))` starts from random phase. Each pass keeps the angle of the re-analysed signal and puts the target magnitude back.
- `processor(audio_target=signal, ...)` computes the log-mel in exactly the form the vocoder expects, so the comparison is between like and like.

### 3. Does a recogniser care which vocoder was used?

A different check: turn 12 real clips into mel spectrograms, rebuild each with HiFi-GAN and with 32-pass Griffin-Lim, and score Whisper `base.en` on the original and both rebuilds.

```python
import glob, io
import jiwer, numpy as np, pandas as pd, soundfile as sf, torch
from scipy.signal import istft, stft
from transformers import SpeechT5HifiGan, SpeechT5Processor, WhisperForConditionalGeneration, WhisperProcessor
from transformers.utils import logging

logging.set_verbosity_error()
torch.set_num_threads(4)
path = glob.glob("/Users/sumanth.tp/.cache/huggingface/hub/datasets--hf-internal-testing--librispeech_asr_dummy/snapshots/*/clean/*.parquet")[0]
frame = pd.read_parquet(path)
waves = [sf.read(io.BytesIO(a["bytes"]), dtype="float32")[0] for a in frame.audio]
keep = [i for i, w in enumerate(waves) if len(w) < 12 * 16000][:12]
waves, refs = [waves[i] for i in keep], [frame.text[i] for i in keep]

tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").eval()
filters = np.asarray(tts_processor.feature_extractor.mel_filters)
hop, win = 256, 1024

def griffin_lim(log_mel, iterations=32):
    band = np.maximum(10.0 ** log_mel, 1e-10)
    magnitude = np.maximum(np.linalg.pinv(filters.T) @ band.T, 0.0) / (win / 2)
    phase = np.exp(2j * np.pi * np.random.default_rng(0).random(magnitude.shape))
    for _ in range(iterations):
        _, signal = istft(magnitude * phase, nperseg=win, noverlap=win - hop)
        _, _, spec = stft(signal, nperseg=win, noverlap=win - hop)
        phase = np.ones(magnitude.shape, dtype=complex)
        phase[:, : spec.shape[1]] = np.exp(1j * np.angle(spec[:, : magnitude.shape[1]]))
    _, signal = istft(magnitude * phase, nperseg=win, noverlap=win - hop)
    return signal.astype("float32")

variants = {"original": [], "HiFi-GAN": [], "Griffin-Lim": []}
for wave in waves:
    mel = tts_processor(audio_target=wave, sampling_rate=16000, return_tensors="pt").input_values[0]
    with torch.no_grad():
        variants["HiFi-GAN"].append(vocoder(mel).numpy())
    variants["original"].append(wave)
    variants["Griffin-Lim"].append(griffin_lim(mel.numpy()))

asr_processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
asr = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en").eval()
normalise = asr_processor.tokenizer.normalize
print(f"{len(waves)} clips, {sum(len(w) for w in waves) / 16000:.1f} s, {sum(len(normalise(r).split()) for r in refs)} words")
for name, clips in variants.items():
    hyps = []
    for clip in clips:
        feats = asr_processor(clip, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            hyps.append(asr_processor.batch_decode(asr.generate(feats, num_beams=1, do_sample=False), skip_special_tokens=True)[0])
    errors = jiwer.process_words([normalise(r) for r in refs], [normalise(h) for h in hyps])
    print(f"{name:12s} Whisper base.en WER {errors.wer:.4f}  (S {errors.substitutions}, D {errors.deletions}, I {errors.insertions})")
```

**Reading the output.** The 12 clips last 80.8 s and hold 186 reference words. Whisper scores 0.0699 on the original audio (9 substitutions, 4 deletions), 0.0591 after HiFi-GAN (7, 4) and 0.0699 after Griffin-Lim (9, 4). Two words out of 186 separate the neural rebuild from the original, which is within noise on a sample this size.

The honest reading is two-sided. A recogniser reads both rebuilds as well as the original, so what listeners may hear as a metallic or smeared sound does not hurt a recogniser. And since recognition cannot tell the three apart, word error rate is not a way to compare vocoders for human listening. That needs a listening test (a mean opinion score), which I did not run, so this chapter makes no claim about how any of these sounds.

**Line by line.**

- `variants` holds the three versions of each clip, so all three go through exactly the same recogniser code.
- `jiwer.process_words(...)` returns the counts of substitutions, deletions and insertions as well as the rate.

### 4. How often does a pause inside a sentence look like the end of a turn?

An agent that waits for silence has to choose how long. This block finds, in 73 read-aloud clips, every silent stretch between the first and last voiced frame and counts how many each wait would have wrongly treated as the end of the speech. It uses plain energy, not a trained detector.

```python
import glob, io
import numpy as np, pandas as pd, soundfile as sf

path = glob.glob("/Users/sumanth.tp/.cache/huggingface/hub/datasets--hf-internal-testing--librispeech_asr_dummy/snapshots/*/clean/*.parquet")[0]
frame = pd.read_parquet(path)
waves = [sf.read(io.BytesIO(a["bytes"]), dtype="float32")[0] for a in frame.audio]

def internal_pauses(wave, sr=16000, hop_ms=10, floor_db=-35.0):
    hop = sr * hop_ms // 1000
    count = len(wave) // hop
    frames = wave[: count * hop].reshape(count, hop)
    level = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
    voiced = level > np.percentile(level, 95) + floor_db
    idx = np.flatnonzero(voiced)
    first, last = idx[0], idx[-1]
    pauses, run = [], 0
    for flag in voiced[first : last + 1]:
        if flag:
            if run:
                pauses.append(run * hop_ms)
            run = 0
        else:
            run += 1
    return pauses, (last - first + 1) * hop_ms / 1000

all_pauses, speech_seconds = [], 0.0
for wave in waves:
    pauses, seconds = internal_pauses(wave)
    all_pauses += pauses
    speech_seconds += seconds
all_pauses = np.array(all_pauses)
print(f"{len(waves)} clips, {speech_seconds:.1f} s from first to last voiced frame, {len(all_pauses)} internal pauses")
print(f"median pause {np.median(all_pauses):.0f} ms, 90th percentile {np.percentile(all_pauses, 90):.0f} ms, longest {all_pauses.max():.0f} ms")
print("endpoint wait   pauses it would cut   cuts per minute of speech")
for wait in (100, 200, 300, 500, 700, 1000):
    cuts = int((all_pauses >= wait).sum())
    print(f"{wait:8d} ms   {cuts:12d}   {cuts / (speech_seconds / 60):18.2f}")
```

**Reading the output.** Between first and last voiced frame there are 437.6 s of speech and 898 internal pauses. Most are tiny: the median is 30 ms and the 90th percentile 163 ms, probably the closures of consonants and small gaps between words. The longest is 950 ms. A wait of 200 ms would cut speech 74 times, 10.15 per minute. At 300 ms it is 50 cuts (6.86 per minute), at 500 ms 14 (1.92), at 700 ms 6 (0.82) and at 1,000 ms none.

So 700 ms in the lab is not arbitrary: it is where the cut-offs in this audio fall below one per minute. Each extra 100 ms of wait is paid on every turn, whether or not that turn contained a pause. These clips are an audiobook reading a text; hesitant spontaneous speech very likely has more and longer pauses, which I did not measure. The voice-agent chapter measures the same trade-off with a trained detector on a recording of ten utterances and gets the same shape.

**Line by line.**

- `floor_db=-35.0` marks a 10 ms frame as voiced when its level is within 35 dB of the loud frames (the 95th percentile of frame levels). The threshold is relative so quiet and loud clips are treated alike.
- `voiced[first : last + 1]` ignores silence before the first word and after the last, because those are not pauses inside speech.

### 5. What the three stages cost on this machine

Now the stage times of a small cascade: Whisper `tiny.en` transcribes the 4.82 s clip, SmolLM2 135M writes a reply, and SpeechT5 speaks a sentence. Each is timed three times after a warm-up and the median kept.

```python
import glob, io, time, zipfile
import numpy as np, pandas as pd, soundfile as sf, torch
from huggingface_hub import hf_hub_download
from transformers import (AutoModelForCausalLM, AutoTokenizer, SpeechT5ForTextToSpeech, SpeechT5HifiGan,
                          SpeechT5Processor, WhisperForConditionalGeneration, WhisperProcessor)
from transformers.utils import logging

logging.set_verbosity_error()
torch.set_num_threads(4)
path = glob.glob("/Users/sumanth.tp/.cache/huggingface/hub/datasets--hf-internal-testing--librispeech_asr_dummy/snapshots/*/clean/*.parquet")[0]
wave = sf.read(io.BytesIO(pd.read_parquet(path).audio[1]["bytes"]), dtype="float32")[0]
audio_seconds = len(wave) / 16000

asr_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
asr = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").eval()
llm_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm = AutoModelForCausalLM.from_pretrained(llm_name).eval()
tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").eval()
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").eval()
archive = zipfile.ZipFile(hf_hub_download("Matthijs/cmu-arctic-xvectors", "spkrec-xvect.zip", repo_type="dataset"))
names = sorted(n for n in archive.namelist() if n.endswith(".npy"))
speaker = torch.tensor(np.load(io.BytesIO(archive.read(names[7306])))).unsqueeze(0)

def timed(fn, repeats=3):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return result, sorted(times)[len(times) // 2]

def listen():
    feats = asr_processor(wave, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        ids = asr.generate(feats, num_beams=1, do_sample=False)
    return asr_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

def think(text, new_tokens):
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False)
    return tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

def speak(text):
    ids = tts_processor(text=text, return_tensors="pt")["input_ids"]
    torch.manual_seed(0)
    with torch.no_grad():
        mel = tts.generate_speech(ids, speaker)
        audio = vocoder(mel)
    return audio

listen(); think("hi", 1); speak("Hello there.")
heard, t_asr = timed(listen)
_, t_first = timed(lambda: think(heard, 1))
reply, t_reply = timed(lambda: think(heard, 25))
rate = 24 / (t_reply - t_first)
sentence = "Mister Quilter is a painter."
audio, t_tts = timed(lambda: speak(sentence))
seconds = len(audio) / 16000
print(f"heard      : {heard!r}")
print(f"reply      : {reply!r}")
print(f"speech to text for {audio_seconds:.2f} s of speech: median {t_asr * 1000:.0f} ms")
print(f"language model: first token median {t_first * 1000:.0f} ms, then {rate:.0f} tokens per second")
print(f"speech synthesis: {seconds:.2f} s of audio in median {t_tts * 1000:.0f} ms, real-time factor {t_tts / seconds:.2f}")
```

**Reading the output.** In the run I quote, Whisper took a median 141 ms for 4.82 s of speech, SmolLM2 took 142 ms for the first token and then wrote at 88 tokens per second, and the synthesiser needed 465 ms for 2.08 s of audio, a real-time factor of 0.22. The text the model replied with is not meaningful, because a 135-million-parameter model answering a transcribed sentence has nothing to answer; only its speed matters here.

These timings are not stable. Across my runs on this shared machine the speech to text step took 136 to 425 ms, the first token 106 to 425 ms, the writing speed 39 to 101 tokens per second and the synthesiser's real-time factor 0.22 to 0.59. My very first, unwarmed single measurement was slowest on every count, which is why the block uses a warm-up and a median. A latency budget built from one measurement is a guess. Budget from a percentile measured on the machine and the load that will run it.

**Line by line.**

- `listen(); think("hi", 1); speak("Hello there.")` runs each stage once before timing so model loading and first-call set-up are not counted.
- `rate = 24 / (t_reply - t_first)` takes the writing speed from the difference between a 25-token and a 1-token reply, which removes the first-token cost.
- `min_new_tokens=new_tokens` forces the model to write the stated number of tokens, so the two timings are for known lengths.

### 6. The latency model, in plain Python

The last block is the arithmetic of the worked example as a function, with the measured numbers, and it is the same model the lab runs. It prints the first-sound time for four end-of-turn waits, the effect of streaming, and what happens when synthesis is slower than real time.

```python
def reply_timeline(endpoint_ms, asr_ms, first_token_ms, tokens_per_s, tokens_per_sentence, sentences,
                   audio_s, tts_rtf, network_ms):
    text_ready = endpoint_ms + asr_ms
    synth_ms = tts_rtf * audio_s * 1000
    synth_end = play_end = stalls = 0.0
    first_audio = 0.0
    rows = []
    for i in range(sentences):
        ready = text_ready + first_token_ms + ((i + 1) * tokens_per_sentence - 1) / tokens_per_s * 1000
        synth_end = max(ready, synth_end) + synth_ms
        arrives = synth_end + network_ms
        start = arrives if i == 0 else max(arrives, play_end)
        gap = 0.0 if i == 0 else start - play_end
        if i == 0:
            first_audio = start
        stalls += gap
        play_end = start + audio_s * 1000
        rows.append((i + 1, ready, synth_end, start, gap))
    whole_ready = text_ready + first_token_ms + (sentences * tokens_per_sentence - 1) / tokens_per_s * 1000
    wait_for_all = whole_ready + synth_ms * sentences + network_ms
    return first_audio, stalls, play_end, wait_for_all, rows

measured = dict(asr_ms=141, first_token_ms=142, tokens_per_s=88, tokens_per_sentence=12, sentences=2, audio_s=2.1, tts_rtf=0.22, network_ms=0)
print("end-of-turn wait   first audio   whole reply first   saved   silent gaps")
for endpoint in (300, 500, 700, 1000):
    first, stalls, finish, whole, _ = reply_timeline(endpoint, **measured)
    print(f"{endpoint:12d} ms {first:10.0f} ms {whole:14.0f} ms {whole - first:7.0f} {stalls:10.0f} ms")
first, stalls, finish, whole, rows = reply_timeline(700, **measured)
for number, ready, synth_end, start, gap in rows:
    print(f"sentence {number}: text ready {ready:6.0f} ms, audio made {synth_end:6.0f} ms, plays from {start:6.0f} ms, gap before it {gap:5.0f} ms")
slow = dict(measured, tts_rtf=1.4)
first, stalls, finish, whole, _ = reply_timeline(700, **slow)
print(f"with a synthesiser at 1.4 x real time: first audio {first:.0f} ms, silent gaps {stalls:.0f} ms")
```

**Reading the output.** At the default 700 ms wait the first sound arrives at 1,570 ms, against 2,168 ms for a whole-reply agent, so streaming saves 598 ms. That saving does not depend on the wait: it is 598 ms at 300, 500, 700 and 1,000 ms. The wait changes the first-sound time one for one, from 1,170 ms at 300 ms to 1,870 ms at 1,000 ms. The two sentence rows show that sentence 2 is made at 2,032 ms but is not needed until 3,670 ms, so the gap before it is 0.

The last line is the failure. If the synthesiser ran at 1.4 times real time, the first sentence would be made later, at 4,048 ms, and each sentence after it would be made later than the previous one finishes playing: 840 ms of silence for a two-sentence reply. Streaming only works when the synthesiser stays ahead of playback.

**Line by line.**

- `synth_end = max(ready, synth_end) + synth_ms` models one synthesis worker: sentence `i` starts when its text is ready and the worker is free.
- `start = arrives if i == 0 else max(arrives, play_end)` says playback of a sentence starts when its audio has arrived and the previous sentence has finished.
- `gap` is the silence between the end of one sentence and the start of the next.

## Try it yourself

The lab runs the model of block 6 in your browser. Its defaults are the numbers of the worked example.

<ReplyLatencyLab />

**What each control does.**

- **End-of-turn wait** is the silence rule. **Speech to text**, **Model first token** and **Model speed** describe the first two stages.
- **Tokens per sentence**, **Sentences in reply** and **Audio per sentence** describe the reply.
- **Synthesis real-time factor** is the synthesiser's speed. **Network** adds a fixed delay before each sentence's audio arrives.

**Try it yourself.**

1. Keep the defaults and read: first audio 1,570 ms, whole reply 2,168 ms, 598 ms saved, 0 ms of gaps. These are block 6's numbers.
2. Set the end-of-turn wait to 300 ms. First audio becomes 1,170 ms, 400 ms sooner, and the saving from streaming stays 598 ms. Why: the wait comes before every other stage and moves everything by the same amount. The price is in block 4: 6.86 cut-offs per minute of speech instead of 0.82.
3. Set the synthesis real-time factor to 1.40. First audio becomes 4,048 ms and the replies pick up 840 ms of silent gaps. Now set sentences to 4: the gaps grow to 2,520 ms. Why: each sentence takes 2.94 s to make but only 2.1 s to play, so every sentence falls 0.84 s further behind.

## Designing with it

- **Budget the stages, then the rule.** Write each stage's median and 95th percentile, add them, and compare with a target. Choose the end-of-turn wait last, with the cut-off rate from your own audio.
- **Keep synthesis faster than playback.** The real-time factor must stay below 1 with margin, under load, or streaming creates gaps instead of removing delay. Test on the machine and the load that will run it.
- **Speak the first sentence early and keep it short.** A short first sentence reaches the loudspeaker sooner. A reply that starts "Sure." costs one short synthesis and buys time for the rest.
- **Normalise text before synthesis.** Expand numbers, dates, currencies and abbreviations. Check by speaking the text and reading it back with a recogniser, as block 1 does.
- **Do not use word error rate to choose a voice.** Block 3 shows a recogniser cannot tell a neural vocoder from a classical one. Use listening tests for naturalness and WER only for intelligibility.

## Where this stands in 2026

The two-stage design of this chapter, a model that plans a mel spectrogram and a vocoder that renders it, is the classical neural recipe, and it is what the openly licensed models here implement. Voice agents are also built the other way, with one model that takes speech in and gives speech out, and the voice-agent chapter compares the two. I did not survey current commercial or research synthesisers for this chapter, so I make no claim about which is best today. SpeechT5 (a 2021 paper) was chosen because it is MIT licensed and runs on a laptop CPU, not because it is the state of the art.

## Common mistakes

1. **Quoting one timing as the latency.** My own runs of the same code gave synthesis speeds from 0.22 to 0.59 of real time. A single run feels precise and is not. Use medians and a high percentile, after a warm-up, on the target machine.
2. **Choosing the shortest end-of-turn wait.** It makes the agent feel quick and cuts people off, 10 times a minute at 200 ms in read speech. Pick the wait from a table like block 4's, on audio like your users'.
3. **Streaming with a slow synthesiser.** Streaming assumes the next sentence is ready when the last one ends. At a real-time factor of 1.4 the agent stutters, with 840 ms of silence for two sentences.
4. **Feeding raw digits to the synthesiser.** `3 PM on 21 June` came back as `come on June`. Expand numbers and abbreviations to words first.
5. **Judging a vocoder by mel error or by WER.** Griffin-Lim at 128 passes had lower mel error than HiFi-GAN, and a recogniser could not tell the original from either. Neither number says how natural it sounds. Run a listening test.

## Practice questions

<details>
<summary><strong>Q1 (Easy).</strong> A reply is 3 seconds long at 16,000 Hz with a mel hop of 256 samples. How many mel frames does the acoustic model write, and how many numbers is that?</summary>

3 x 16,000 = 48,000 samples. 48,000 / 256 = 187.5, so about 188 frames (the model writes whole frames). Each frame has 80 numbers, so 188 x 80 = 15,040 numbers. The vocoder expands each frame to 256 samples, giving 188 x 256 = 48,128 samples, which is 3.008 s.

</details>

<details>
<summary><strong>Q2 (Easy).</strong> A synthesiser takes 0.6 s to make a 2-second sentence. What is its real-time factor, and can it keep up with playback?</summary>

0.6 / 2 = 0.30. It is below 1, so it makes audio faster than it is played: after the first sentence it stays ahead and the agent can stream sentence by sentence without gaps.

</details>

<details>
<summary><strong>Q3 (Medium).</strong> Using the lab's defaults, work out the first-audio time if the end-of-turn wait is 500 ms and the first sentence has 8 tokens instead of 12.</summary>

Text ready: 500 + 141 = 641 ms. First token 142 ms later, then 7 more tokens at 88 per second: 7 / 88 = 0.080 s, so 80 ms. The sentence is written at 641 + 142 + 80 = 863 ms. Synthesis of 2.1 s of audio at 0.22 is 462 ms, so the first sound is at 863 + 462 = 1,325 ms. (If the shorter sentence also has less audio, say 1.4 s, synthesis takes 308 ms and the first sound is at 1,171 ms.)

</details>

<details>
<summary><strong>Q4 (Medium).</strong> Griffin-Lim at 128 passes has a lower mel error than HiFi-GAN in block 2, but practitioners prefer HiFi-GAN. Reconcile the two facts.</summary>

The mel error measures how closely the output's spectrogram matches the input spectrogram. Griffin-Lim is designed to reduce exactly that, and its error never rises as passes are added. HiFi-GAN is trained against discriminators to produce waveforms that look like real recordings, which includes natural phase and fine detail that the mel spectrogram does not encode, so its mel error can stay higher. Naturalness has to be measured with listeners. This chapter did not run a listening test, so it gives no result on that.

</details>

<details>
<summary><strong>Q5 (Stretch).</strong> The pause table says a 700 ms wait cuts only 0.82 times per minute in read speech. Why might you still choose a shorter wait for a booking agent, and what would you do to make it safe?</summary>

A booking agent mostly hears short, task-focused answers ("Tuesday", "two people"), so long thinking pauses are rarer than in a lecture, and each extra 100 ms is paid on every turn. A shorter wait, say 400 to 500 ms, improves the feel of every turn. To make it safe, use a detector that also looks at what was said (a sentence that ends is likelier a finished turn than one that stops after "to"), allow the agent to take back a premature reply when the caller keeps talking, and measure the cut-off rate on real calls rather than on read speech. The voice-agent chapter covers these rules.

</details>

<details>
<summary><strong>Q6 (Stretch).</strong> In the latency model, streaming saved 598 ms at every end-of-turn wait. Predict what happens to the saving if the reply has four sentences instead of two, and check it with the model.</summary>

The whole-reply agent now waits for four sentences to be written and four to be made, while the streaming agent still waits for one of each. The text arrives later by 24 tokens more at 88 per second, about 273 ms, and the synthesis by two more sentences at 462 ms each, 924 ms. So the saving grows by about 1,197 ms, to about 1,795 ms, as long as the synthesiser stays ahead of playback. Run block 6 with `sentences=4` to confirm; the exact value is the difference between the two printed first-audio times.

</details>

## Go deeper

All sources were opened on 8 October 2026.

- Kong J, Kim J and Bae J, [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646) (arXiv 2010.05646, NeurIPS 2020). The abstract states 167.9 times real time on a V100 for 22.05 kHz audio and 13.4 times real time on a CPU for a small version.
- Ao J and colleagues, [SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing](https://arxiv.org/abs/2110.07205) (arXiv 2110.07205, ACL 2022). The model behind the acoustic stage.
- Model cards: [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts) (MIT, trained on LibriTTS; the card states no limitations and says more information is needed on risks and biases) and [microsoft/speecht5_hifigan](https://huggingface.co/microsoft/speecht5_hifigan) (MIT). Speaker vectors: [Matthijs/cmu-arctic-xvectors](https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors) (MIT, 512-number x-vectors from the CMU ARCTIC speakers).
- Griffin D and Lim J, "Signal estimation from modified short-time Fourier transform", IEEE Transactions on Acoustics, Speech and Signal Processing 32(2), 236 to 243, April 1984. The source of the algorithm in block 2; I read its abstract summary through a search result and did not open the full paper.
- Stivers T and colleagues, "Universals and cultural variation in turn taking in conversation", PNAS 106(26), 2009, in the [University of Groningen record](https://research.rug.nl/en/publications/universals-and-cultural-variation-in-turn-taking-in-conversation/). I read the abstract only (10 languages; gaps within 250 ms of the cross-language mean). PNAS itself refused automated access, so I could not check the 200 ms figure in the full text.
- [Voice and realtime agents](/docs/agentic-frontier/voice-and-realtime-agents) on this site: trained turn detection, interruption, echo and the speech-to-speech alternative.

## Check yourself

- I can explain what an acoustic model and a vocoder each produce, and give the shapes passed between them.
- I can compute frames, samples and duration for a given hop length.
- I can say why a mel spectrogram is not enough to build a waveform, and describe how Griffin-Lim and a neural vocoder each fill the gap.
- I can measure a synthesiser's real-time factor with a warm-up and a median, and say what it must stay below for streaming to work.
- I can add up a voice agent's stage times to a first-sound time, and say how much streaming by sentence saves.
- I can read a pause table and choose an end-of-turn wait from it, naming what it costs.

## Where to go next

Next chapter: this is the last chapter of the speech track. Continue with [voice and realtime agents](/docs/agentic-frontier/voice-and-realtime-agents) for trained turn detection and interruption. Back one chapter: [speech recognition](/docs/theory/speech/speech-recognition), the listening half of the same pipeline. A related chapter: [attention and Transformers](/docs/theory/nlp/attention-and-transformers), the architecture behind both SpeechT5 and Whisper.
