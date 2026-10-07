# Ollama source rebuild, 2026-10-07

User contract: rebuild `ollama-local-llms` from video YcAYmIFtA0o; translate the Hindi transcript first, preserve teaching order, use video frames and displayed code, and identify corrections instead of inventing examples.

Published chapter: `docs/genai/21-ollama-local-llms.md`.
Preserved pre-edit working copy: `docs/genai/_old/21-ollama-before-source-rebuild.md`.

## Primary evidence

- Video: https://www.youtube.com/watch?v=YcAYmIFtA0o (2:49:40).
- Matching cached metadata, original Hindi auto captions and frames: `.lecture-import/ollama-video-review/`.
- Original captions: 3,778 segments; no translated track available in the cached source review.
- Complete English working translation: `.lecture-import/ollama-video-rebuild/source-en-draft.md` and `.json`, covering all 255 consecutive 40-second blocks from 00:00 through 2:49:20. Generated from the original Hindi captions before using each range in the chapter. Machine translation is a working draft, not a publication: proper nouns and technical expressions require correction. Reviewed English meaning is recorded below and in the chapter.
- Translation uses Google's translation endpoint on this public video text. Some original caption fragments contain Hindi transliterations of English and translate badly; the draft was not copied directly into the notes.
- Video-description repository: https://github.com/campusx-official/Ollama-Youtube.
- Retrieved commit: `06244ad032b3ef982e1d4d8b9f514d3c9be60dba`; pinned source links in the chapter.
- All five notebooks read, including code and relevant saved outputs. Extracted the supplied DOCX outline for cross-checking, without treating its additional material as automatically part of the video.
- Forty cached frames reviewed in contact sheets; thirty additional frames extracted from the matching cached video and reviewed. Individual code/terminal frames enlarged where needed. Frames are analysis-only and not embedded in the course page.

## Reviewed English interpretation by source section

These are original summaries of the translated source meaning, not a substitute verbatim transcript.

| Start | English interpretation and evidence retained |
| --- | --- |
| 00:00 | Earlier GenAI demos used GPT APIs; students reported payment-method obstacles. |
| 01:02 | Paid proprietary access is the practical motivation, rather than an assertion that GPT quality is poor. |
| 01:37 | Mature downloadable DeepSeek, Qwen and GLM models motivate an alternative. |
| 02:42 | Video introduces the team's longer course and lets non-buyers start learning. |
| 03:16 | Detailed introduction; five interfaces promised. Nitish introduces, Ajay teaches. |
| 04:57 | Neural network, layers/connections, learned weights and biases. |
| 06:23 | Accessibility/control split: provider-owned model access versus downloadable files. |
| 09:54 | Downloading components, running locally and cloth/customisation analogy. Openness/licence overclaims visibly corrected. |
| 11:54 | Raw-weight friction: storage, working memory, compatibility. The cloth explanation continues into this region before the question about hosted subscriptions. |
| 15:17 | Ollama manages download/run/model management; WhatsApp analogy. |
| 18:51 | Privacy, offline access/latency, cost, ease, library, customisation, reduced dependence, management. Local/cloud and tuning distinctions made explicit. |
| 22:43 | Library families and capability filters; initially no thinking results because Vision remained selected; Qwen3-VL sizes. |
| 28:37 | Hardware guidance, OS, RAM, processor recommendation, disk-size examples, first download needs internet, CLI knowledge and optional GPU. |
| 32:22 | Windows setup downloaded and installed; model-versus-runtime distinction precedes terminal demonstration. |
| 34:31 | Version, cancelled Ministral pull, local list, corrected mistyped run, greeting, photosynthesis, Indian rights, suggested offline exercise. Image fails on Llama; succeeds after switching to Gemma. |
| 44:26 | show/help inspection; Llama lacks explicit defaults/system text; switch to Gemma; set menus; on-screen top_p 0.99; system instruction; CLI for experiments. |
| 51:41 | Install/import Python SDK; moon prompt; response metadata/text; streaming chunks. Incorrect generated moon explanation is labelled. |
| 56:38 | API field tour, binary image/base64 encoding, one-image caption, two-image story, funny system instruction, ocean prompt with temperature 0.3/top_p 0.5/top_k 45. |
| 1:04:14 | Context/history explanation via chat API docs, no name-recall demo; return to terminal for deletion; Python list/name/size, cancelled deepseek-r1 pull, show details. |
| 1:10:21 | Database, Chandigarh temperature and today's-news limitations; tool functions supply missing capabilities; model Tools support matters. |
| 1:17:54 | Functions, schemas, model choice of tool/arguments, application execution, resend user/assistant/tool history. Dehradun argument example. |
| 1:26:55 | Electronic-shop mock dictionary, 30% capped discount, mapping, schemas, singular message list, chat, dispatch, second call without tools. iPhone → laptop → five-year prompt, in that order. |
| 1:44:02 | General model plus instructions; student/shortcut analogy; Modelfile directives; sentiment file; create/list/run; JSON shape versus wrong NEUTRAL label; mention API creation without fabricating a program. |
| 1:59:04 | Remote-wrapper diagram then local endpoint; generate comparison, raw JSON-lines inspection then assembly; SDK listing and REST listing. Server-per-request claim corrected. |
| 2:14:11 | Company-policy PDF RAG diagram; LangChain components/interfaces; three adapter examples; embeddinggemma, not nomic-embed-text. No full RAG project demonstrated. |
| 2:28:14 | Memory constraints motivate remote hardware; eligible Cloud models; sign in/connect; DeepSeek cloud; Python stars prompt; signout/unauthorised; free/paid usage discussion and remote privacy trade-off. |
| 2:43:16 | New chat, Settings, local Llama greeting, Gemma image, cancelled DeepSeek download, search picker, GPT-OSS cloud. |
| 2:47:51 | Nitish returns, recaps, shows longer course with projects; historical promotion not current advice. |

## Code provenance and visible adjustments

- `Ollama.ipynb`: moon generate/stream cells 4–6; one/two images cells 7–8; system cell 9; options cell 11; list/pull/show cells 12–15. Imports repeated to make individual examples understandable. `.dict()` changed to `.model_dump()` with its source deprecation warning explained.
- `Tool Calling.ipynb`: inventory/functions/map/schemas cells 1–4; question and first call cell 6, using the iPhone variation witnessed in the recording; dispatch cell 8; final call cell 9. No invented standalone dispatch snippet or unlabelled multi-turn repair loop.
- Important: the duplicate shop example inside `Ollama.ipynb` has a 25% cap and three-year prompt; it is not the recorded separate notebook's 30% cap/five-year example.
- Recorded five-year final output claims 1140. Displayed first call requests inventory; later call does not pass tools or execute another request. Correct rule returns 900. Preserve failure, do not describe model-written text as execution.
- `Modelfile.txt`: same base, four parameters, instruction, scores and labels. Two assistant MESSAGE lines adjusted to raw JSON because installed CLI preserves the source's escape characters. This adjustment is explicitly explained; current CLI verified that plain JSON reaches the create payload.
- `Ollama using Rest API.ipynb`: cells 1, 4, 6–7, 9–10, 12–13. Preserve buffered `requests.post` and the two passes over lines; explain distinction from client HTTP streaming. Same black-holes prompt in both routes.
- `Ollama Using LangChain.ipynb`: cells 1–5. Preserve temperature=0, quantum-entanglement prompt, France prefix, embeddinggemma, document strings and vector inspection. Added prerequisite pull labelled as setup.
- `Ollama Cloud.ipynb`: cell 1, same cloud model and stars prompt.
- On-screen CLI top_p is 0.99; speech includes 0.90. Frame wins for displayed code.
- Repository image names have `(1)` suffixes; explain renaming/path changes instead of substituting an unrelated image.

## Verification

Working checks/scripts/logs remain under ignored `.lecture-import/ollama-video-rebuild/`.

- All 24 published Python blocks parse with `ast`.
- Actual Ollama Python SDK exercised with `httpx.MockTransport`: both stock prompts, explicit schemas, dictionary dispatch, assistant tool request and tool-result history, final call carrying no tool definitions.
- Tool schemas compared structurally with the instructor notebook.
- Business rules checked: laptop lookup, unknown product, five-year price 900, 30% cap price 840.
- Requests buffered-response JSON-lines parser checked over its second iteration.
- Actual installed Ollama CLI parsed the corrected published Modelfile against a mocked create endpoint; base model, four parameters, system and all four MESSAGE turns verified. Assistant contents parse as plain JSON.
- No new models downloaded or created; no live inference or deterministic-output claims.
- Browser verification on the existing localhost:3000 preview: source headings, 23 timestamp links, five Mermaid diagrams, expand/zoom/Escape, 390px mobile width without overflow and zero page errors. Screenshots retained privately.
- `git diff --check` passed for the chapter.
- `npm run build`: client and server compilation succeeded; whole-site link checking failed on the existing GraphRAG page's links to `/docs/genai/rag-advanced/contextual-retrieval-and-reranking` and `/docs/genai/rag-advanced/long-context-vs-rag`. Neither link is in the rewritten Ollama chapter. An existing earlier audit already reported the first broken link. No unrelated source files repaired.
- Final Modelfile syntax clarification was made after production compilation and checked through the current development preview and actual CLI parser. Whole-site build is not claimed as passing.

## Outstanding

No missing video sections identified in this review. Whole-site build remains blocked by the two unrelated GraphRAG links. No commit, push or deployment requested or performed.
