---
id: afr-voice
title: "Voice and Realtime Agents"
sidebar_label: "4 · Voice agents"
sidebar_position: 4
slug: /agentic-frontier/voice-and-realtime-agents
description: "Build the mental model of a voice agent: the cascade of voice activity detection, speech to text, model and text to speech against a single speech-to-speech model, the latency budget between the last word and the first audio, and how end-of-turn and interruption rules trade speed for mistakes, measured on real speech."
tags: [voice-agents, realtime, vad, turn-taking, barge-in, whisper, silero, latency]
---

import Infographic from '@site/src/components/Infographic';
import TurnTakingLab from '@site/src/components/viz/TurnTakingLab';

**In one line.** A voice agent is a conversation with a stopwatch: it must decide when you have finished speaking, answer within about a second, and stop talking the moment you cut in, and every one of those decisions trades speed against mistakes.

:::note Not from a lecture
Written for this site from the sources under Go deeper, opened on 8 October 2026. The speech is real: ten utterances from the public LibriSpeech sample on the Hugging Face hub (`hf-internal-testing/librispeech_asr_dummy`, validation split). The voice activity detector is Silero VAD 6.2.3, the speech recogniser is Whisper tiny.en and the language model is SmolLM2 135M, all on a CPU, with PyTorch 2.14.1, Transformers 5.18.0 and Python 3.14.6. I could not run a hosted speech-to-speech model, so that half of the comparison rests on vendor documentation and is marked as such.
:::

:::tip Before you start
You should already know:

- the agent loop of a model that reads input and replies ([What is agentic AI?](/docs/agentic-ai/what-is-agentic-ai));
- why a model's first token takes time and tokens then stream out ([why decoding is memory-bound](/docs/llm-engineering/why-decoding-is-memory-bound));
- what a token is and what a context window costs ([Context engineering](/docs/agentic-frontier/context-engineering)).

Reading time: about 40 minutes with the code. The first run downloads two small models and the audio sample.

After this chapter you can:

- list the stages of a cascade voice agent and add up its delay from the last word to the first audio;
- choose a silence rule and an interruption rule by reading a measured trade-off, not by guessing;
- decide when a speech-to-speech model is worth its trade-offs.
:::

## In 30 seconds

A voice agent has two ears and a mouth that must work together. The ears first find speech in a stream of sound (is anyone talking?), then turn it into words, and a language model writes a reply that is spoken aloud. The hard part is not any single stage. It is the timing: answering too early talks over a person who was only pausing, answering too late feels like a bad phone line, and talking on after the person interrupts feels rude.

Think of a good receptionist. They wait through your pause, answer quickly when you are done, and stop mid-sentence the moment you say "sorry, one thing". A voice agent must learn the same three habits from sound alone.

## Words you will meet

| Term | Plain meaning | Tiny example |
| --- | --- | --- |
| Voice activity detection (VAD) | A small model that says, for each slice of audio, whether someone is speaking | 32 ms of audio in, "speech" or "not speech" out |
| Endpointing | Deciding the person talking has finished a turn | 500 ms of silence after speech |
| Barge-in | The user speaking while the agent is talking | "wait, no" in the middle of a reply |
| Backchannel | A short sound that is not a turn | "mm-hm", "right" |
| Speech to text (ASR) | Turning audio into words (automatic speech recognition) | Whisper |
| Text to speech (TTS) | Turning words into audio | a spoken reply |
| Real-time factor | Processing time divided by audio length | 0.05 means 1 s of audio takes 0.05 s |
| Cascade | A chain of separate models: VAD, ASR, language model, TTS | this chapter's main example |
| Speech to speech | One model that reads audio and writes audio | a realtime API session |

## The idea in plain words

Picture a phone call to a shop. You say "I would like to book a flight", stop to think for a moment, then say "to Lisbon". The person on the line must not answer during your pause. After your last word they should answer fast, because a long silence reads as "did the line drop?". If you speak again while they talk, they must stop.

A computer hears none of this as words. It receives a stream of numbers, 16,000 a second. First a **voice activity detector** labels each 32 milliseconds as speech or not. A rule such as "silence for 500 ms means the turn is over" turns those labels into turns. The turn's audio goes to a speech recogniser, the text to a language model, and the reply to a speech synthesiser, whose first audio chunk starts playing while the rest is still being made.

Now the smallest example with numbers. A user says "I would like to book a flight" (1,900 ms), pauses 400 ms, and says "to Lisbon" (900 ms). If the rule is 300 ms, the 400 ms pause is long enough to end the turn, so the agent answers a sentence that stops at "flight". If the rule is 500 ms, the pause is shorter than the rule and the sentence stays whole, but at the end the agent must wait 500 ms before it even begins. In words: a shorter rule is faster and cuts people off, a longer rule is safe and slow, and no rule is both.

<Infographic src="/img/afr/voice-cascade-vs-speech-to-speech.svg" alt="Two pipelines: a cascade of microphone, voice activity detector, speech to text, language model, text to speech and speaker, and a single speech-to-speech model, above a bar showing 1,257 milliseconds from the last word to the first audio in five parts." caption="Read the top row left to right: that is the cascade. The bar at the bottom adds up its delays; two parts are measured here and two are assumed." />

## Worked example, step by step

Take the sentence above with an assumed pipeline: speech to text 400 ms, model first token 300 ms, text to speech first audio 150 ms, network 80 ms. The delay from the last word to the first audio is the silence rule plus all the stages.

1. **Rule 300 ms.** The 400 ms pause is longer than the rule, so the detector declares a turn after "flight". The turn is cut in two. The second turn ("to Lisbon") ends and waits 300 ms. Its delay is 300 + 400 + 300 + 150 + 80 = 1,230 ms, but the first turn has already triggered a wrong reply.
2. **Rule 500 ms.** The 400 ms pause is shorter than the rule, so the sentence stays whole. After "Lisbon" the agent waits 500 ms, then adds the stages: 500 + 400 + 300 + 150 + 80 = 1,430 ms.
3. **Rule 800 ms.** The same, with 800 ms of waiting: 800 + 400 + 300 + 150 + 80 = 1,730 ms.
4. **The cost of safety.** Going from 300 to 800 ms removes the cut-off and adds 500 ms to every turn of the conversation, not only the ones with a pause.

The first code block reproduces these three lines. The second block measures how often real speakers pause longer than each rule.

<Infographic src="/img/afr/voice-turn-taking-worked-example.svg" alt="A sentence split by a 400 millisecond pause, a table showing what three silence rules do to it and the delay each adds, and a table of segments and premature cut-offs on ten real utterances." caption="Look at the top bar first: the red gap is the pause. The table below shows which rules cut it. The last table is the same trade-off measured on real speech." />

## How it works

### The cascade, stage by stage

**Voice activity detection** runs on every 32 ms chunk of audio (512 samples at 16 kHz) and outputs a probability of speech. Silero VAD is a small neural detector (about 2 MB, MIT licence, per its package description) that handles 8 kHz and 16 kHz audio. Its probabilities are thresholded, and short gaps are smoothed away by the end-of-turn rule.

**Speech to text** turns a finished turn into words. Whisper is a family of models trained on 680,000 hours of audio, per its model card; `tiny.en` has about 39 million parameters and the card reports a LibriSpeech test-clean word error rate of 8.4 (self-reported). Run in batch, it needs the whole turn before it starts, so its delay after the last word grows with the length of the turn. Streaming recognisers start early and have a short delay at the end, which is why production systems usually stream.

**The language model** reads the text and starts writing. The delay that matters is the time to the first token, because the text-to-speech stage can start as soon as it has a clause.

**Text to speech** converts the reply in chunks. Its delay is the time to the first audio chunk, not the whole reply.

Everything after the silence rule can overlap. A streaming pipeline sends partial text to the model, and the first sentence to the voice, before the later stages have finished.

### Speech to speech

A **speech-to-speech** model takes audio and produces audio directly. OpenAI's Realtime API guide describes it as a way to build voice agents where the model works directly with audio, keeps conversation state and can call tools, reached over WebRTC from a browser or WebSocket from a trusted server. Google's Gemini Live API guide describes "native audio" output models that return 24 kHz audio from 16 kHz input. Two stages (ASR and TTS) disappear from the chain, and with them their errors: the model can hear tone, hesitation and non-words that a transcript drops. In return, you have less control over each stage, you cannot swap in a better recogniser or voice on their own, and the whole thing is harder to test, because the text in the middle no longer exists. I did not measure any hosted model, so this chapter makes no claim about their delay.

### Deciding that a turn has ended

There are three kinds of rule, from simplest to richest.

**Silence.** End the turn after a fixed gap. OpenAI's `server_vad` mode exposes `threshold`, `prefix_padding_ms` and `silence_duration_ms`, and Google's Live API exposes the same ideas with start and end sensitivity. Google's guide says values around 100 to 200 ms split utterances into fragments and recommends roughly 500 to 800 ms. Block 2 measures that on real speech.

**Meaning.** OpenAI's `semantic_vad` mode takes an `eagerness` setting (low, medium, high or auto) that waits longer when the words sound unfinished. LiveKit's agents documentation lists a turn-detector model as the recommended default, which uses meaning and acoustics on top of the VAD, with plain VAD-only and speech-recogniser endpointing as alternatives. The idea: "to Lisbon" after "I would like to book a flight" completes a thought, while "I would like to" does not, however long the silence.

**Manual.** Push-to-talk. The user decides. LiveKit lists it as a mode.

Even people do this: a well-known 2009 study of ten languages (Stivers and colleagues, PNAS) found that speakers in every language avoid overlapping talk and keep the silence between turns short, with average gaps differing across languages by only a few hundred milliseconds. A fixed rule of a second would feel slow to any of them.

### Interruption

When the user speaks over the agent, three things should happen quickly (this is design guidance, not a quotation from a specification): the agent stops speaking and discards queued audio, the model's partly spoken reply is truncated in the history to what the user actually heard, and any pending tool call is cancelled or set aside. Google's guide says that on an interruption the server cancels the ongoing generation, sends an `interrupted` message, and the client should stop playback and clear its audio queue; it also says pending function calls are discarded. A cascade built by hand must do all three itself.

The hard part is not stopping, it is deciding whether to stop. A cough, a keyboard click or a "mm-hm" are not turns. LiveKit's documentation describes `min_duration` and `min_words` settings for how much speech counts as an interruption, an adaptive mode that tries to tell a real interruption from backchannel, and a timeout for recovering from a false interruption. Block 4 measures the first of these.

### Echo

If the loudspeaker's sound reaches the microphone, the agent hears itself and interrupts itself. Browsers and phones apply echo cancellation; a server-side system that receives raw audio must supply it. The experiments below assume a clean microphone.

## Code you can run

Four blocks. The first is pure Python. The others use `silero-vad` 6.2.3, `transformers` 5.18.0, `datasets`, `soundfile` 0.14.0 and `torch` 2.14.1, download a few hundred megabytes on the first run, and print timings that vary with the machine's load. Everything that is not a timing (counts, error rates, detection decisions) is deterministic.

### 1. The worked example, replayed

First the toy sentence from the worked example. The "detector" is only a rule on pause lengths, so you can check the arithmetic.

```python
SPEECH_MS = [("I would like to book a flight", 1900), ("pause", 400), ("to Lisbon", 900)]
ASR_MS, LLM_MS, TTS_MS, NETWORK_MS = 400, 300, 150, 80

def replay(silence_rule_ms):
    heard, said = [], ""
    for text, length in SPEECH_MS:
        if text == "pause":
            if length >= silence_rule_ms:
                heard.append(said.strip())
                said = ""
            continue
        said += " " + text
    heard.append(said.strip())
    return heard

for rule in (300, 500, 800):
    heard = replay(rule)
    budget = rule + ASR_MS + LLM_MS + TTS_MS + NETWORK_MS
    print(f"rule {rule} ms: {len(heard)} turn(s) {heard}; first audio {budget} ms after the last word")
```

**Reading the output.** With a 300 ms rule the sentence becomes two turns, `I would like to book a flight` and `to Lisbon`, and the delay after the last word is 1,230 ms. With 500 and 800 ms the sentence is one turn, and the delays are 1,430 and 1,730 ms. These are the numbers worked by hand.

**Line by line.**

- `SPEECH_MS` is the sentence as pieces with lengths. The pause is a piece too.
- `replay` walks through the pieces. When it meets a pause at least as long as the rule, it closes the current turn.
- The budget is the rule plus the four assumed stages. Nothing here is measured.

### 2. A real detector on real speech

Now the real thing. We join ten LibriSpeech utterances, each followed by 1.5 seconds of silence, and ask Silero VAD for speech segments under six silence rules. Each true utterance should come out as one segment, so any extra segment is a cut in the middle of someone talking.

```python
import io
import time

import numpy as np
import soundfile
import torch
from datasets import Audio, load_dataset
from silero_vad import get_speech_timestamps, load_silero_vad

SR = 16000
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(decode=False))
utterances = []
for row in dataset.select(range(10)):
    wave, rate = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    utterances.append(wave)

gap = np.zeros(int(1.5 * SR), dtype=np.float32)
stream = np.concatenate([part for wave in utterances for part in (wave, gap)])
print(f"{len(utterances)} utterances, {len(stream) / SR:.1f} s of audio, sample rate {SR}")

model = load_silero_vad()
audio = torch.from_numpy(stream)
print("threshold  segments  premature cut-offs  delay added at every turn end")
for silence_ms in (100, 200, 300, 500, 800, 1200):
    spans = get_speech_timestamps(audio, model, sampling_rate=SR, threshold=0.5,
                                  min_silence_duration_ms=silence_ms, speech_pad_ms=0, return_seconds=True)
    print(f"{silence_ms:6d} ms {len(spans):9d} {len(spans) - len(utterances):19d} {silence_ms:22d} ms")

start = time.perf_counter()
get_speech_timestamps(audio, model, sampling_rate=SR, min_silence_duration_ms=300, return_seconds=True)
elapsed = time.perf_counter() - start
print(f"VAD time {elapsed:.2f} s for {len(stream) / SR:.1f} s of audio: {elapsed / (len(stream) / SR):.4f} of real time")
```

**Reading the output.** The stream is 124.8 seconds. With a 100 ms rule the detector returns 26 segments for 10 utterances, 16 premature cut-offs. At 300 ms there are 19 segments and 9 cut-offs. At 800 ms, 12 and 2. At 1,200 ms, 11 and 1: even a 1.2 second rule still cuts one utterance, because that person paused longer than 1.2 seconds. The last column is the cost, the same delay at every turn end.

**What surprised me.** The cut-offs fall slowly. Doubling the rule from 100 to 200 ms removes only five cut-offs of sixteen. Going from 500 to 800 ms removes two more and adds 300 ms to every turn. Google's Live API guide says values around 100 to 200 ms split utterances into fragments and suggests 500 to 800 ms; the measured table agrees with the first part and shows why the second part is a compromise. These are read-aloud audiobooks; casual speech with hesitations will cut more.

The last line shows the detector's speed: under one percent of real time (0.006 in the run quoted here), so detection itself is never the bottleneck.

**Line by line.**

- `soundfile.read(..., dtype="float32")` decodes the stored FLAC bytes into numbers. We use `Audio(decode=False)` so the library does not need an extra decoder.
- `gap` is 1.5 seconds of zeros, longer than every rule we test, so the turn boundaries are always found.
- `get_speech_timestamps(... min_silence_duration_ms=..., speech_pad_ms=0)` returns one span per run of speech separated by at least that much silence. `speech_pad_ms=0` removes the padding so the spans are exactly the speech.
- `len(spans) - len(utterances)` is the number of premature cut-offs, because every utterance contributes at least one span.

### 3. What the other stages cost

Now the other stages, measured: Whisper tiny.en on four utterances, with its word error rate, then the first-token time of the small language model, then the sum with the assumed stages.

```python
import io
import re
import statistics
import time

import numpy as np
import soundfile
import torch
from datasets import Audio, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

SR = 16000
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(decode=False))
clips = []
for row in dataset.select(range(4)):
    wave, _ = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    clips.append((wave, row["text"]))

asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", dtype=torch.float32)

def words(text):
    return re.sub(r"[^a-z' ]", "", text.lower()).split()

def error_rate(reference, hypothesis):
    ref, hyp = words(reference), words(hypothesis)
    table = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        previous, table[0] = table[0], i
        for j, h in enumerate(hyp, 1):
            previous, table[j] = table[j], min(table[j] + 1, table[j - 1] + 1, previous + (r != h))
    return table[-1] / len(ref)

asr({"raw": clips[0][0][:SR], "sampling_rate": SR})
print("clip  seconds  asr seconds  real-time factor  word error rate")
for index, (wave, text) in enumerate(clips):
    start = time.perf_counter()
    heard = asr({"raw": wave, "sampling_rate": SR})["text"]
    spent = time.perf_counter() - start
    print(f"{index:4d} {len(wave) / SR:8.1f} {spent:12.2f} {spent / (len(wave) / SR):17.3f} {error_rate(text, heard):16.3f}")

name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer, lm = AutoTokenizer.from_pretrained(name), AutoModelForCausalLM.from_pretrained(name).eval()
prompt = tokenizer.apply_chat_template([{"role": "user", "content": heard}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
first, rate = [], []
for _ in range(5):
    start = time.perf_counter()
    with torch.no_grad():
        lm.generate(**prompt, max_new_tokens=1, do_sample=False)
    first.append(time.perf_counter() - start)
    start = time.perf_counter()
    with torch.no_grad():
        out = lm.generate(**prompt, max_new_tokens=30, min_new_tokens=30, do_sample=False)
    rate.append(30 / (time.perf_counter() - start))
print(f"LLM first token: median {statistics.median(first) * 1000:.0f} ms; decoding: median {statistics.median(rate):.1f} tokens/s")

TTS_FIRST_AUDIO_MS, NETWORK_MS = 150, 80
batch_asr_ms = spent * 1000
print("endpoint wait + batch ASR + LLM first token + assumed TTS and network (ms):")
for endpoint_ms in (300, 500, 800):
    total = endpoint_ms + batch_asr_ms + statistics.median(first) * 1000 + TTS_FIRST_AUDIO_MS + NETWORK_MS
    print(f"  endpoint {endpoint_ms:4d}: {total:6.0f} ms from end of speech to first audio")
```

**Reading the output.** Each row is one utterance. The real-time factor is about 0.04 to 0.05, so a 12.5 second utterance takes about half a second to transcribe in batch, which is the delay the user feels after the last word. The word error rates are 0.059, 0.100, 0.000 and 0.042, deterministic for this model and audio. The language model's first token takes roughly 0.3 seconds on this CPU, and it decodes tens of tokens a second. The final lines add the silence rule, the batch recogniser time, the first token and the assumed 150 ms of speech synthesis and 80 ms of network: about 1.26 seconds for a 300 ms rule, 1.46 seconds for 500 ms and 1.76 seconds for 800 ms. Rerun it and the timings move by tens of milliseconds, and a loaded machine moves them more.

**What this shows and does not.** The batch recogniser's delay is not a property of Whisper: a streaming set-up would have been transcribing during the speech. The 135-million-parameter language model is far smaller than what a production agent would use, so its delay is a floor for this machine, not a figure for any hosted model. The speech synthesis and network times are assumptions, and are written as constants so you can replace them.

**Line by line.**

- `error_rate` is word error rate by edit distance: substitutions, insertions and deletions divided by the number of reference words.
- The warm-up call before the loop avoids counting model start-up as a delay.
- `min_new_tokens=30` with `max_new_tokens=30` forces exactly 30 tokens so the decoding rate is comparable. The first-token time uses `max_new_tokens=1`.
- `statistics.median` over five runs reduces noise from other work on the machine.

### 4. Should the agent stop talking?

Last, interruption. The agent is speaking, the microphone is clean, and a sound arrives. We feed three sounds to the detector in 32 ms chunks, as a live system would, and ask when each would stop the agent under four "speech needed" settings.

```python
import io

import numpy as np
import soundfile
import torch
from datasets import Audio, load_dataset
from silero_vad import get_speech_timestamps, load_silero_vad

SR, CHUNK = 16000, 512
row = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").cast_column("audio", Audio(decode=False))[1]
voice, _ = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
model = load_silero_vad()
first_word = get_speech_timestamps(torch.from_numpy(voice), model, sampling_rate=SR, return_seconds=False)[0]["start"]
voice = voice[first_word:]
rng = np.random.default_rng(0)
clips = {
    "real interruption (1.5 s of speech)": voice[: int(1.5 * SR)],
    "backchannel (0.25 s of speech)": voice[int(0.8 * SR): int(1.05 * SR)],
    "noise burst (0.3 s)": (0.2 * rng.standard_normal(int(0.3 * SR))).astype(np.float32),
}

def probabilities(wave):
    model.reset_states()
    padded = np.pad(wave, (0, (-len(wave)) % CHUNK))
    return [model(torch.from_numpy(padded[i:i + CHUNK]), SR).item() for i in range(0, len(padded), CHUNK)]

chunk_ms = CHUNK / SR * 1000
print(f"each decision covers {chunk_ms:.0f} ms of audio")
print("clip                                  longest run  first detection    stops agent after 0 / 100 / 250 / 400 ms of speech")
for name, wave in clips.items():
    probs = [p >= 0.5 for p in probabilities(wave)]
    run, best = 0, 0
    for flag in probs:
        run = run + 1 if flag else 0
        best = max(best, run)
    first = probs.index(True) * chunk_ms + chunk_ms if True in probs else None
    verdict = []
    for need_ms in (0, 100, 250, 400):
        fires = best * chunk_ms >= max(need_ms, 1)
        verdict.append(f"{(first + need_ms):5.0f} ms" if fires else "   never")
    shown = f"{first:6.0f} ms" if first is not None else "     none"
    print(f"{name:37s} {best:8d} ch {shown:>14s}    {'  '.join(verdict)}")
```

**Reading the output.** Each decision covers 32 ms. A real 1.5-second interruption gives a run of 46 speech chunks, is first detected at 64 ms and stops the agent after 64, 164, 314 or 464 ms depending on how much speech the rule needs first (0, 100, 250 or 400 ms). A 0.25-second backchannel, a short real syllable cut from the speech, gives a run of 7 chunks (224 ms): it stops the agent at 0 or 100 ms of required speech, and is ignored at 250 and 400 ms. White noise of the same loudness scale gives no speech chunks at all and never stops the agent.

**What surprised me.** The detector never mistook the noise burst for speech, so a noise gate is not what you need to tune. The risk is real speech that is not a turn. The rule "250 ms of speech" ignores the backchannel and costs 250 ms of reaction time on a real interruption. A rule of 0 ms reacts instantly and will stop the agent for every "mm-hm". The right setting depends on whether your callers interrupt more than they agree aloud.

**Line by line.**

- `probabilities` resets the detector's memory and feeds 512-sample chunks in order. The detector is recurrent, so the order matters.
- `best` is the longest unbroken run of speech chunks, which is what a "needs this much speech" rule checks.
- The first-detection time is the end of the first chunk judged as speech, so every time is a multiple of 32 ms.

## The lab

<TurnTakingLab />

The defaults, a 300 ms silence rule and the example latencies of the third block, reproduce the printed row of block 2 (19 segments, 9 premature cut-offs) and a budget of 1,257 ms, within the run-to-run noise of the timings in block 3. The interruption controls reproduce the table of block 4.

**What each control does.**

- **end of turn after silence** picks the silence rule. The grey boxes show what was said, the blue boxes what the detector returned, and red marks where it cut an utterance.
- **speech to text, model first token, text to speech, network** are the delays added to the rule. The sum is shown under the picture.
- **interruption sound** picks a sound, and **speech needed before the agent stops** is the interruption rule. The sentence below the controls says when, or whether, the agent stops.
- **show data** lists segments, cut-offs and the delay at each rule.

**Try it yourself.**

1. Go through the silence rules from 100 to 1,200 ms and watch the red cuts in the first 28 seconds disappear: 16, 11, 9, 4, 2, 1 over the whole stream. At 800 ms the total delay is 1,757 ms, 500 ms more than at 300. The cuts you remove are the ones you pay for on every turn.
2. Set text to speech to 50 ms, network to 0 and the model's first token to 100 ms. At a 300 ms rule the delay is 870 ms, and most of what is left is the recogniser and the rule. Speeding up the model gives less than shortening the wait.
3. Choose the backchannel and move speech needed from 0 to 250 ms: at 100 ms the agent stops after 164 ms, at 250 ms it never stops. Choose the real interruption: at 250 ms it stops after 314 ms. Then choose the noise burst: it never stops at any setting.

<Infographic src="/img/afr/voice-barge-in-and-budget.svg" alt="A table of how long it takes the agent to stop for three sounds under four speech-needed rules, a table of speech recogniser timings and word error rates on four clips, and three cards summarising end of turn, barge-in and the latency budget." caption="Left table first: when the agent stops. Then the recogniser measurements. The cards at the bottom are the lessons." />

## Designing with it

| Decision | Start here | Why |
| --- | --- | --- |
| Silence rule | 500 to 800 ms for general speech, shorter for short commands | Google's guide recommends 500 to 800 ms, and the measured table agrees |
| Smarter end of turn | A meaning-aware detector on top of VAD | A long silence is safe but slow; meaning lets a short silence end a finished sentence |
| Recogniser | Streaming, with partial results | Batch delay grows with the turn |
| Model | Smallest that does the job; stream the first sentence to the voice | The first token and first clause set the delay |
| Voice | Start synthesis on the first clause | Time to first audio, not full reply |
| Interruption | Require 200 to 300 ms of speech, and ignore known backchannel words | Measured: it ignores "mm-hm" and costs a quarter of a second |
| On interruption | Stop audio, truncate the history to what was heard, cancel pending tools | Otherwise the model believes the user heard words they did not |
| Speech to speech | When turn-taking feel and tone matter more than stage control | One model hears hesitation; you lose stage-by-stage testing |

Three habits for building one. Measure the budget stage by stage on your own audio before choosing models, because the stage you assume is slow rarely is. Log every turn decision with the audio and the verdict so a cut-off can be replayed. And test with messy audio: accents, background noise, a second speaker and a speaker on a speakerphone. Read-aloud audiobooks, as in this chapter, are the easy case. For evaluation methods see [operational evals](/docs/llm-evals/operational-evals) and for the model-side cost see [semantic caching, routing and cost](/docs/llm-engineering/semantic-caching-routing-and-cost).

## Where this stands in 2026

:::info Industry view
Two architectures are in production use. Cascades built with frameworks such as LiveKit Agents give control over every stage, and its documentation (checked 8 October 2026) lists a turn-detector model as the default alongside VAD-only, speech-recogniser endpointing and manual modes, with adaptive handling of interruptions. Speech-to-speech services from OpenAI and Google expose server-side voice activity detection with silence and sensitivity settings, and a semantic mode, and document interruption events. Both OpenAI's and Google's guides describe the model keeping conversation state and handling tool calls.

What is not settled: how much a speech-to-speech model gains over a well-tuned cascade in naturalness against what it costs in testability, and how to evaluate turn-taking quality objectively. I did not find a vendor-neutral benchmark that I could open and quote, so I give none. Voice also carries legal and privacy duties (recording consent, retention of audio) that differ by country and are outside this chapter.
:::

## Common mistakes

- **Fixing latency by shrinking the silence rule.** It is the easiest dial. But at 100 to 200 ms the measured stream was cut 16 and 11 times in 10 utterances. Fix the stages after the rule first, then decide how much cut-off you can bear.
- **Measuring delay from the end of the audio file.** It feels like the user's experience. The user feels the time from their last word, which includes the silence rule. Always add it.
- **Tuning on read-aloud speech.** Clean audiobooks are easy. Real callers pause, restart and talk over each other, so the cut-off rate will be higher. Test on recordings of your own users.
- **Treating any sound as an interruption.** It feels responsive. Backchannel and short words then stop the agent mid-sentence. Require a short run of speech, as measured above, and handle known acknowledgements as non-turns.
- **Leaving the history untruncated after an interruption.** The model wrote a long answer, the user heard half, and the model believes all of it was said. Truncate the stored reply to what was played.

## Practice questions

<details>
<summary><strong>Easy.</strong> In the worked example, why does the 300 ms rule split the sentence and the 500 ms rule not?</summary>

The pause inside the sentence is 400 ms. A rule ends the turn when silence lasts at least as long as the rule. 400 is at least 300, so the 300 ms rule ends the turn at the pause. 400 is less than 500, so the 500 ms rule keeps listening and the sentence stays whole.

</details>

<details>
<summary><strong>Easy.</strong> The real-time factor of Whisper tiny.en is about 0.05. How long does batch transcription of a 20-second turn take, and when does the user feel that delay?</summary>

About 20 × 0.05 = 1 second, and the user feels all of it after their last word, because a batch recogniser starts only when the turn is complete. A streaming recogniser would be transcribing during the speech and would have a much smaller delay at the end.

</details>

<details>
<summary><strong>Medium.</strong> From the measured table, how many cut-offs do you remove by moving the rule from 300 to 800 ms, and what does it cost per turn?</summary>

At 300 ms there are 9 premature cut-offs in 10 utterances and at 800 ms there are 2, so 7 are removed. The cost is 500 ms of extra waiting at the end of every turn, whether or not the person paused. A meaning-aware end-of-turn detector aims to keep the safety of a long rule on unfinished sentences and the speed of a short one on finished ones.

</details>

<details>
<summary><strong>Medium.</strong> An agent should ignore "mm-hm" but stop for "wait, no". Using block 4, which setting works for the 0.25-second backchannel and the 1.5-second interruption, and what does it cost?</summary>

A rule needing 250 ms of speech ignores the backchannel (longest run 224 ms) and stops for the interruption after 314 ms, which is 250 ms later than a rule of 0 would. A 100 ms rule stops for the backchannel as well. The cost is a quarter of a second of reaction time on real interruptions. A meaning-aware or word-level rule (a minimum of words, as LiveKit offers) can do better, because a single short word can still be an interruption such as "stop".

</details>

<details>
<summary><strong>Stretch.</strong> A speech-to-speech model removes the recogniser and the voice stage. Which delays in the budget disappear, which stay, and what can you not measure any more?</summary>

The batch recogniser (420 ms here) and text-to-speech first chunk (150 ms assumed) disappear as separate stages. The end-of-turn rule (300 ms here) stays, though it may be applied inside the service, and so does the network and the model's own time to first audio. What you lose is the text between stages: you cannot compute word error rate of the recogniser alone, inspect a transcript of what the model heard, or swap the voice. You also cannot reproduce this chapter's stage timings for a hosted model without measuring it, so measure end-to-end from the last word to the first audio.

</details>

<details>
<summary><strong>Stretch.</strong> After a barge-in at 2 seconds, the agent had planned a 10-second answer and a tool call. List what the system must do in the next 200 ms and what it must write to memory.</summary>

Stop audio playback and clear queued audio at once; cancel generation and any text-to-speech work in flight; cancel or set aside the pending tool call (and make sure a half-run tool is safe to abandon, for instance with an idempotency key); start capturing the user's turn. For memory, store the agent's reply truncated to the words actually played (here about the first 2 seconds), mark it as interrupted, and keep the unspoken part out of the history, so later turns do not assume the user heard it.

</details>

## Go deeper

All opened on 8 October 2026 unless stated.

- OpenAI, Realtime API guide and voice activity detection guide (speech-to-speech, WebRTC and WebSocket, `server_vad` with `threshold`, `prefix_padding_ms`, `silence_duration_ms`, `semantic_vad` with `eagerness`, `create_response` and `interrupt_response`). The guide's example values are not documented as defaults.
- Google, Gemini Live API guide (automatic activity detection, `start_of_speech_sensitivity`, `end_of_speech_sensitivity`, `prefix_padding_ms`, `silence_duration_ms` with 500 to 800 ms recommended, interruption handling, 16 kHz input and 24 kHz output).
- LiveKit Agents documentation, turn detection and interruption handling (turn-detector model, VAD, speech-recogniser endpointing and manual modes; adaptive and VAD interruption modes, `min_duration`, `min_words`).
- Tanya Stivers and colleagues, "Universals and cultural variation in turn-taking in conversation", Proceedings of the National Academy of Sciences 106(26), 2009 (abstract read; the full paper was not accessible to me).
- Alec Radford and colleagues, "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper), arXiv 2212.04356, December 2022; the `openai/whisper-tiny.en` model card (39 million parameters, Apache 2.0).
- The `silero-vad` package, version 6.2.3, described as a pre-trained voice activity detector for 8 kHz and 16 kHz audio, about 2 MB, MIT licence. LibriSpeech: Vassil Panayotov and colleagues, ICASSP 2015.
- On this site: [Context engineering](/docs/agentic-frontier/context-engineering), [Agent interoperability](/docs/agentic-frontier/agent-interoperability-mcp-and-a2a), [human in the loop](/docs/genai/langchain-advanced/human-in-the-loop), [streaming in LangChain](/docs/genai/langchain-advanced/streaming), [operational evals](/docs/llm-evals/operational-evals), [why decoding is memory-bound](/docs/llm-engineering/why-decoding-is-memory-bound).

**Not verified here.** Any hosted speech-to-speech model: no delay, quality or price is claimed. Streaming recognisers, real text-to-speech models and real network paths (the 150 ms and 80 ms are assumptions). Noisy, accented or overlapping speech. A semantic turn detector: I read the documentation and did not run one. Echo cancellation. The 200 ms human gap figure that is often quoted was not in the abstract I could open, so I do not state it.

## Check yourself

- I can list the stages of a cascade voice agent and add up the delay from the last word to the first audio.
- I can explain why a short silence rule cuts people off and a long one slows every turn, and read the measured trade-off.
- I can say what a voice activity detector does and does not decide.
- I can describe what must happen when a user interrupts, including what to do with the conversation history.
- I can say what a speech-to-speech model removes from the chain and what it takes away from testing.

## Where to go next

Next: [Automatic prompt optimisation and DSPy](/docs/agentic-frontier/automatic-prompt-optimisation-and-dspy), where the thing being tuned is the prompt itself. Related: [Computer use and browser agents](/docs/agentic-frontier/computer-use-and-browser-agents), another loop that has to decide what it sees and when to act.
