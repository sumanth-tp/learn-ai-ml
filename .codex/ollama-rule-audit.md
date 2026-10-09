# Ollama chapter rule audit, 7 October 2026

Audited `docs/genai/21-ollama-local-llms.md` against the current `.codex/AGENTS.md`, `.codex/write-like-claude.md`, `.codex/beginner-friendly-standard.md` and quality gate. This is a new check of the actual chapter, not a reuse of the previous pass report.

## Findings corrected

The current gate initially failed on 29 narration lines. Some still referred to “the video's”, “the source's”, or a speaker's actions. Rewrote them as explanations of the operation. Kept notebook-version provenance in the private source audit instead of a distracting comparison in the lesson.

Manual checks found that the quick preview exceeded five sentences, the plain-words and how-it-works labels were absent, and the closing design/current-version sections followed practice. Corrected those. Added first-use definitions for token, CPU, GPU, JSON, SDK, REST, HTTP, PDF, quantisation, context length, billion parameters and gigabytes. Moved two line-by-line explanations next to their executable blocks, before separate output fences.

Source examples, model tags, settings, functions and tool schemas remain unchanged. The video's progression is retained; the worked shop arithmetic precedes its tool code rather than moving it ahead of the earlier CLI material.

## Final checks

| Rule or check | Result |
| --- | --- |
| Video timestamps in headings, prose, links or index | None |
| Video pointers | Exactly one source link at the top |
| Speaker/source narration detector | Zero matches |
| Quick preview | Five sentences |
| Glossary | Ten terms, with other technical terms defined in prose |
| Required learning sections | Present; closing design/version sections precede mistakes/practice |
| Explanation after executable blocks | Reading the output and Line by line present for every block |
| Executable block length | Largest is 31 lines |
| Long prose paragraphs | No ordinary prose paragraph above 90 words in manual scan |
| Python | 27 blocks parse; 327 inline lines; real-library imports |
| Runtime evidence | Existing live Llama 3.1 experiment retained with its failed discount, not claimed as success |
| Original tool flow | Real SDK + mock HTTP checks pass for both products and history |
| Modelfile | Actual CLI + mock endpoint check passes |
| REST parser | Buffered JSON-line reassembly passes |
| Boards | Two, inspected in rendered browser |
| Lab | Default 900, cap 840, stopped execution and missing product checked |
| Python/TypeScript agreement | All 33 price cases agree |
| Browser | Desktop/mobile, infographic expand/zoom/Escape, one source link, no timestamp links, zero page errors, no 390px overflow |
| Typecheck | Pass |
| Final MDX compilation | Pass |
| Whitespace check | Pass |
| Production build | Client/server compile; fails on the two existing GraphRAG links |

Readability: Flesch 59.2; mean sentence 11.3 words; mean paragraph 29.7 words. Metrics and detailed scan: `.lecture-import/ollama-rule-audit/manual-report.json`.

Limits remain explicit on the chapter: the matching packages for every source model were not installed, and cloud inference was not replayed. No new model downloads, deletion, creation or cloud calls were performed during this audit. The build failure is on the GraphRAG page's links to contextual-retrieval-and-reranking and long-context-vs-rag; the Ollama chapter is absent from the broken-link list.

GATE: 1 of 1 chapters pass
