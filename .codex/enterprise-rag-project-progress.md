# Enterprise RAG project authoring state

## User contract

Create new Projects section, Enterprise RAG project, exactly two session Markdown chapters corresponding to https://www.youtube.com/watch?v=bjkjaqUZl4E and https://www.youtube.com/watch?v=jOgqWdck7BU. Preserve actual transcript order/content; no invented session material. Begin with problem statement, requirements/libraries, repository structure, then continuous detailed teaching and complete code walkthrough in demonstrated order. Include student interactions, session links/commands, visuals inspected from frames. Small corrections/production improvements only, explicitly `NOT from session`. User clarified: use transcripts, use browser to obtain complete transcripts and inspect comments for corrections; write self-contained polished teaching like existing docs/genai rather than constant 'instructor says' narration. External Claude Opus review score 90+ is a quality target, not a guarantee. No subagents authorised. No deployment or external paid calls requested.

## Sources and access

Google Drive skill read and announced; canonical fetch succeeded for resource Doc https://docs.google.com/document/d/1wMPQL2NJTzT70GLBVYr3hKrObCmYrwhwvTgoEb0PLWk/edit?tab=t.0. Cached `.lecture-import/enterprise-rag/materials/resource-links.txt`. It links teaching repo d-hackmt/8hr-MARATHON, deployment fork sourangshupal/8hr-MARATHON, tldraw, Notion commands, guardthisrag/guardrailz/letsgateway/ragasz Streamlit demos.

CLI caption downloads HTTP429 and youtube-transcript-api IpBlocked. **Browser UI succeeds!** Use cached playwright-core at /Users/sumanth.tp/.npm/_npx/e41f203b7505f1fb/node_modules/playwright-core/index.mjs, Chromium headless shell1228. Open YouTube, click `tp-yt-paper-button#expand`, then button `Show transcript`. Modern segments are `transcript-segment-view-model`; timestamp `[aria-hidden=true]`, text `[role=text]`. Session1 3688 segments 2:40–8:29:51; session2 4032 segments 4:52–9:11:39. Do not ask user for transcripts again; async question was answered with browser instruction and browser worked.

Session1 metadata title Build Enterprise-Grade RAG Applications | LIVE 8-Hour Marathon, uploaded2026-07-03, duration30599s. Session2 title Live 8 hours Marathon Building Multi-Modal Intelligence Systems: Document Processing Production RAG, uploaded2026-07-10, duration33108s. Detailed chapter timestamps in respective `.info.json` files under `.lecture-import/enterprise-rag/transcripts`.

Browser scripts /tmp/read-enterprise-rag-materials.mjs (done/finishing) and /tmp/read-enterprise-sessions.mjs (currently running exec session29919) store full HTML/text, transcript JSON, comments JSON and frame PNGs in materials/transcripts. Log /tmp/enterprise-rag-sessions-browser.log. Session2 transcript already saved as jOgqWdck7BU.browser.json; session1 initial extraction bjkjaqUZl4E.transcript.json has numeric start. Browser frame seeking yielded readyState1 at 1.5s: inspect images, do not claim verified frames until truly loaded. Need improve wait `readyState>=2` and seek settled if images blank/stale.

Tldraw browser read succeeded, all text saved whiteboard.txt and screenshotwhiteboard.png. Notion read succeeded ingestion-commands.txt (also has eval diagram text). Streamlit outer pages show only Hosted with Streamlit; may need iframe inspection to read actual apps. Browser metadata/HTML saved for each.

## Repositories pinned

Ignored local clones (do not execute deployment scripts or contact paid APIs):
- `.lecture-import/enterprise-rag/session-repo`: d-hackmt/8hr-MARATHON main commit52b771cbdea2e2215c823cc1ae522183b77a85b7 (2026-07-01). Branches teaching/00-scaffold, stage-1-ingestion, stage-2-basic-rag, stage-3-rerank-memory, stage-4-guardrails, stage-5-llm-gateway, stage-6-evals, Clean-main. This uses Gemini3072/local mpnet768 embeddings, Qdrant, FlashRank, Groq via Portkey, NeMo, Logfire, LangSmith, Ragas.
- `.lecture-import/enterprise-rag/deployment-repo`: sourangshupal/8hr-MARATHON deployment commitf97dc63318b11db3e2d806db4d6528fef7baebf8 (2026-07-10). Changed to Jina embeddings/reranking, OpenAI/Anthropic gateway, NeonPostgres, UpstashRedis, AWS ECS/ECR/ALB/Secrets/GitHubActions. Session2 covers these transitions; do not silently mix with session1.

## Known code checks already read

Teaching splitter splits blank-line paragraphs, target1500 characters, NO overlap and NO embeddings used for chunk boundaries. Oversized paragraph exceeds limit; docstring incorrectly claims max size guarantee. Must demonstrate small correction under NOT from session.
Teaching embedding lazy `_init`: probe Gemini then choose fallback once per process; Gemini3072 vs mpnet768; mid-batch failure retries and raises, not automatic dynamic fallback. Retries sleeps1,2,4 (4 attempts), comment mentions8 but no finalsleep. Same embedding family required for query/index, not merely matching dimension.
Teaching qdrant search query_points limit8 default, payloadtext/source/score; catchesallerrors→[] hides outages.
Teaching FlashRank `_get_ranker` Ranker(cache_dir='/tmp/flashrank') defaultmodel not pinned; docstring saysMiniLM, log saysTinyBERT. Verify actual default against pinnedFlashRank version. Reranking returns only text, dropping source IDs; fallback originalorder top_n onerror. Need preserveidentity smallNOTaddition ifdiscussed.
DOCS/06_KNOWN_GOTCHAS claims Logfire is permanently 'poisoned' if any call beforeconfigure; likely overstated, verifyofficialdocs rather than repeat. Correct initialiseearly; lazyinit does notguarantee millisecondstartup.
Notion clean command `python -m app.ingestion.processor DATA/true_data true` accompaniedtextclaimswipecollection; must checkactual CLIparser, donot endorseunsupportedargument. Docalsosaysnoisy_sample_10/15 paths maynotexist.

## Remaining work

1. Finish browser extraction both transcripts/comments, inspectframes/whiteboard, parse timedsegments into manageable chapter files. **Need read transcripts in depth, not only metadata.** No chapter contentwrittenyet.
2. Read full relevant code across ingestion, config, agents, gateway, guardrails, evals,UI and deployment branch, notebooks, docker/CI/AWS. Review comments for concrete issues and verify.
3. Follow session2's final ~1h40 multimodal section too: ColPali/ColQwen2.5, Nemotron/UnlimitedOCR, PP-DocLayout/GLM-OCR, L4GPU demonstrations. Need actualtranscript links/notebookrepos from transcript rather thaninvent.
4. Create docs/projects/_category_.json and docs/projects/enterprise-rag/_category_.json plus01-session-1.md,02-session-2.md (only2MDfiles; generatedcategorylandings). Possibly Projects navbar entry consistentnewsection request. FollowrepoAGENTS sources/diagrams/summary; user lateststyleoverrides video-narration.
5. Source-based fluentteaching originalprose, timestamps onlylightsectionlinks; studentQ&A paraphrasedactualtranscript. Setupproblemrequirementsrepofirst; fullorderedwalkthrough next; smallNOTfromsession correctnessnotes. Stateexecutionvalidation limits; nofabricatedAWSdeployment.
6. Validate Markdownlinks, Mermaidrendering/browser, build. Local offline codechecks meaningful, no paidcalls. Updateprojectmemory andfinalaccurate.

Workspace initial gitstatusclean for this newtask; interviewchangesfromprioruserturn apparentlycommitted. No projectfilescreatedyet;onlyignoredsourcecache andthismemory.

## Latest steering and transcript progress
User explicitly wants **student AND host interactions/questions**. Put substantive exchanges inline with actual timestamps, including follow-ups/corrections, not only student FAQ. User says **`NOT from session` should be only a tag placed where relevant in session flow; no grouped additions section**. Use inline bold/code tag, not separate chapter/additions appendix.

S1 transcript chapters05–27 have now been read in depth (05 first tool slightly truncated logistics; initial01–04 previous context). All substantive S1 source now understood. S2 still unread. Video images previously captured are only poster frame; MUST NOT treat as viewed. New `/tmp/enterprise-frames.mjs` running execsession6722 plays muted normally/skips available ad button, awaits real frames readyState>=2, writes verified-sN-T.png; log /tmp/enterprise-frames.log currently empty when last checked.

S1 substantive interaction evidence:
-1:49 MTEB leaderboard,768dimensionexample;1:58:13 Bhavesh asks Docling parser→yes,extension-routingusedhere.
-2:00 installfailsPythonversion;recreateuvvenvPython3.11;2:05Groqconsolekey,2:07GeminiAIStudiokey,2:09Qdrantcloud.Suppliedfree-tierquotasnotcurrentfacts.
-2:09:36 hostaskswhyQdrant→selfhost,filters,hybrid;Superlinkedcomparison;incorrectBM25comparisoncorrectedbyPaul3:03:38QdrantsupportsBM25.
-2:42student asksparser/LLMorchestrator→extensiondispatchdeterministic.
-2:51–3:13importantQA:ShivaKumar2:53:26dimensionmismatch;Rakesh2:56:24mixingembeddings;Paul2:57:06saysavoidmix/reindex.2:54MRLvariableoutputdimensions.2:55imagesdeferredS2.3:00websiteFirecrawl/Crawl4AI;3:03PPDocLayout.3:08updateddocsrequiredeleteobsoletechunks/versiontracking.3:10chunkstrategyexperiments;3:11longcontextvsretrieval.
-3:14resumesembeddingimplementation(probe→loadfallback→init→dimension→batch→embedtexts→query),3:25chunker explicitlyNOoverlap;3:29:55hostadmitsnointentionbehindomission.3:30processorfullflow.3:52commands;3:57:52liveimporttypoembedding/embeddingsstudenthelps;4:01missing__main__;4:03clean_args;4:06:30Geminiquota→initialfallbackdownload438MB;4:08Qdrantpoints.4:10noisy_sample_10 used (notnecessarilyrepodir).
-4:18studentasksprocessed_data→JSONparsedchunks/metadata;4:20Omarwhy-noise→retrievalstresscorpus.
-4:23retrieval;4:26rerankgreen/blackrelevanceblocks;bi- vs cross-encoder;4:33RRFdifferenthybridconcept.
-4:35–4:57PaulQA:codeawarechunkingbyfile;rate-limitreducecorpus;multimodalneedslayout/OCR;latencyfilters/quantisation;graph/sparse/densealternatives;multilinguale5/translation;legalMLEB/LegalBenchRAG;Crawl4AI;observabilityLangfuse/Phoenix. Severalovergeneralisationstonotcarryasfact:GraphRAGneverchunks,SPLADE=multivector,noOllamainproduction,neversearch>100kdocs.
-4:58recap;5:03LogfireappvsLangSmithLLM;5:11statebrain/handanalogy;5:18:05Surajaskswhyfinal_answerandmessages→historyvsresponsefield.5:24planner,5:29responder,5:32FlashRank,5:35retriever15→5,5:38graphMemorySaver. HostincorrectlyclaimsMemorySaverforgetsafter10–15questions,modelcache_dirstoresretrieveddocs. CorrectinlineNOTtags.
-5:45API→graph/query,5:52threadIDChatGPTURLanalogy;5:59uvicorn,6:03Swaggerhi/whoareyou,6:06lastquestionmemory,6:07autoscalepodsquerytraces.6:09Logfireauth/projectsuse.6:12span/trace/waterfall;hostcallsnestedspanstraces,correctterminologywithNOTtag.6:17UIcode,6:20greeting,6:21overrideidentityYashPatilYouTuber,6:23memorykeepsbadidentity,Kubernetesquery.
-6:37HRdemoAcmeCorp:remoteeligibility90dayprobation,requestvacation10businessdays,FAISSnotQdrant,linearLangChainnotLangGraph;coffeepancakes/Netflixrefusal,PII.6:52guardedappalsobypassidentity;6:54input/output/customrails;hostKananasksproductionoptionsBedrock/GuardrailsAI.Noabsoluteframeworksecurityclaims.
-6:57topic;6:59jailbreak;7:01livebypass;7:02sensitiveon-topicclusterhacking;7:03dialoguegreetings;7:05PII/urgency;7:06outputsanitise;7:08Harshkeyrotation(restartenv-loadedprocessnote);7:09Colangdefineuser/bot/flow;7:13examplesintentmatching;hostcallsFastEmbedvectorstoreincorrect(embeddinglibrary),andoversimplifiesNeMointentprocess;correctwithofficialsources.
-7:19hostoffersguardintegrationvsnewgatewaytopic→audiencechoosesgateway;so **S1doesnotfinishprojectNeMointegration**,S2does.7:29gatewaypurpose,7:31quotaerrortrace;7:35providerfallbackvs7:37taskrouting;7:38Balkrishnaaskswhatifgatewaydown→stilldependency;Portkey/LiteLLM/Bifrost/Cloudflarementioned.
-7:39cacheanalogyrepeatstudentresearch;7:41Amanretentionquestion(hostforeverwrong);7:42exactvssemantic;7:47Ganesh15yearsvs20yearscachequery →hostcallsdifferent;addfalsehitthresholdnumbersfilterinlineNOT.7:44virtualkeys/slugs;7:54PortkeyintegrationGroqrag1/rag2;7:58studentaskskeytrust(hosttrustedsothereisnoriskincorrect,briefvendorsecurityresponsibilityNOT).8:01routing/logs;8:04metadataYashcustomersupport,DHcodeassistant;8:06loadbalance120b/20b70/30then50/50;8:07invalidrag6fallback;8:09bothinvalidrag6rag7error10failuresbutdashboardnotupdated;donotclaimobservabilityproved.8:11cachehitrepeatNLP;8:15hostsplanS2evalsdeploymentmultimodal;8:17GoogleDocreplacesNotion;8:20Redisnormalvssemanticcachequestion.

Code additional verified:
S1processor office uses **unstructured.partition.auto.partition**, notdirectpython-docx/PPTX;requirementsbothlibs. PDFpypdffallbackpdfplumberdoesNOTOCR,andappendsblankpagetextafterallotherpages(reorders). HTMLnormalisestonewlinesnotdoubleblank→giantchunkpotential.
CLI --wipeonlydeletescollection; positionaltrue/noisyislabelNOTwipe. SourceNotionincorrect. process_filecatcheserrorsandlogs,jobstillprintscompleted;uuid4rerunsduplicates;zipcantruncate;subdirwalkonlyonelevel/rootfilesignoredifsubdirs. processed_dataJSONnochunkvectors. Noquality guarantee fromnoisyratio.
S1AgentState messagesAnnotated[List[dict],operator.add],documentsList[str],current_query,status,plan,final_answer. PlannerexactCONVERSATIONALelseoutputsearchquery;notliteralTECHNICAL. HistoryflattenedallnonuserasAssistant. Respondercontextcap25000chars,breakonfirstoversizeddoc,notokensbudget;nativePortkeycacheheader;retrieverdropssource metadataformatsCONTENTonly. API defaultthreaddefault_userunsafe sharedmemory;errorsreturnHTTP200statuserror;guardearlyreturnskipsgraphandmemory;sourcesarechunkstringsnotrealcitations. /graphdraw_mermaid_pngmayexternalrender.
S1requirementsunversionedexceptpydantic>=2;nopinnedreproducibilityclaim. mainrepoalreadyfinalNeMo+gatewaybutliveS1onlybaseline+independentdemos;usegitstage2/3forbaselinecodewhenneeded,thenfinalintegrationS2.
CommentsS1 all101top-levelloaded;readmost,38parserclarification;67true-dataoverwrite/modelnotfallback;69dimensionmismatch;86inline_config_blocked;89semanticspacenotjustdimension;78productionwithoutloadtest. Needrelevantrepliesread. S218loadedtop-levelalreadyreviewed.

## 2026-09-26 continuation, supersedes stale Remaining work above

User switched explicitly to a new task: add supplied ten-row chain-type table to `docs/genai/09-chains.md`. Table added, including naming corrections HyDE → HypotheticalDocumentEmbedder, agent executor → AgentExecutor, and legacy API note. Enterprise RAG is still unfinished; do not report it complete.

Enterprise RAG current deliverables:
- Both chapters exist under docs/projects/enterprise-rag; Projects navbar and category files exist. Everything delivered stays inside the two MD files (no external code/image assets).
- S1 has complete 18 stage-3 implementation files inline, existing explanations/doubts/summaries. Latest improvement adds security boundaries and detailed introductory walkthroughs before core code, two-turn state trace, retrieval example, and removes duplicate state class.
- S2 sections1–10 contain full guardrails/gateway, six eval files+goldens, deployment runtime files, Docker/compose. Section11 AWS now appended from source commands with complete taskdefs/CI/CD, correction tags, managed-service setup, image push, secret creation helper, task renderer, TLS completion, validation and cleanup order. AWS not run.
- 12 original clean whiteboard figures embedded as PNG data URIs inside Markdown: S1 simple LLM/RAG, original ingestion, two detailed ingestion flows, reranking; S2 five metric posters and evaluation implementation. Source images preserved, not falsely claimed redraws. Additional source infographics still need recreating (security, detailed architecture, AWS original, multimodal paradigms, etc.).
- Full 10-second storyboard visual sweep now inspected for BOTH sessions: 35 S1 contact sheets and37 S2. Sampling is not every literal frame. Full transcripts cached/read substantially; S2 small earlier truncation gaps still worth checking.
- S2 missing multimodal sections12+ (ColPali, Nemotron/Mistral, UnlimitedOCR, dual GLM-OCR/layout), full demos source code and final acceptance/summary. Most transcripts reviewed, OCR source files partially read.
- Need comprehensive content quality pass, especially move code behind motivation and avoid duplicate illustrative implementations. Need complete all source diagrams, code syntax checks, build and real browser verification.
- Active goal exists for completing both RAG chapters; no subagents authorised. Do not mark goal complete based on latest unrelated chain-table edit.

Latest helper scripts in /tmp (do not rerun append scripts):
- improve-rag-teaching.py: S1 teaching introductions, already applied.
- embed-rag-figures.py: 12 original inline figures, already applied.
- rag-aws-chapter.py: appended full section11, already applied.
- Earlier scripts enterprise-inline-code.py, enterprise-s2-integration.py, enterprise-evals-prose.md, enterprise-deploy-prose.md already applied.

AWS section11 review concerns for next pass:
- Uses source variable/network/SG/IAM blocks; new helper create_aws_secrets.py reads .env, creates11SecretsManager entries, writes ARNs. Renderer reads dotenv+env, replaces placeholders, resolves role ARNs; taskdefs corrected Jina collection and UI /ui path.
- Secret permission attaches to execution role only (corrected source). NAT single-AZ limitation noted. InitialHTTP then ACMHTTPS with redirect and updatedUIBACKEND_URL.
- CD uses workflow_run.head_sha and same-repo push predicate, fixes missingPortkeyconfig substitution, removes UIcontinue-on-error. Need ensure all substitutions and YAML parse good.
- Python snippets generated through Python triple strings may have literal newline escaping mistakes: inspect `write_text('\n'.join(...))` in generated Markdown; must fix if it became split source lines.
- Check AWS bashcommands and TLSrulelogic; no external cloud deployment validated.
- Build started for chains edit, log /tmp/learn-ai-ml-build.log; resolve status in active turn.

Verification after chain-table edit: `npm run build` passed (existing dependency-analysis warning from vscode-languageserver-types). Playwright served build and confirmed GenAI chains table is visible with exactly10bodyrows and Chain name/Description headers. RAG pages compiled but their visuals/runtime snippets have NOT yet undergone the required detailed browser/code validation.
