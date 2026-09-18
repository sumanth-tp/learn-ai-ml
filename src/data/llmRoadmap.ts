/**
 * Curriculum + resource shelf for the 7-month Production LLM Engineering track.
 *
 * The spine (phases, modules, topics, hands-on tasks, project mapping, priority)
 * follows the course outline; every resource attached to a unit is an external
 * link that was checked to exist at the time of writing. Paid-platform links
 * point at course landing pages, so availability and price can change.
 *
 * Everything here is plain data — the page in `src/pages/llm-roadmap.tsx` does
 * all rendering, and `src/lib/roadmapProgress.ts` owns the tick boxes.
 */

export type ResourceKind =
  | 'docs'
  | 'course'
  | 'udemy'
  | 'youtube'
  | 'linkedin'
  | 'paper'
  | 'article'
  | 'tool'
  | 'book'
  | 'dataset';

export type Priority = 'Must learn' | 'High' | 'Advanced';

export type Track = 'foundation' | 'module' | 'project';

export type Resource = {
  /** Stable id — the progress store keys off this, so never renumber it. */
  id: string;
  label: string;
  href: string;
  kind: ResourceKind;
  /** Why this one is on the list, in a few words. */
  note?: string;
};

export type Unit = {
  id: string;
  track: Track;
  phase: string;
  /** "Module 01", "Project 03", "Foundation" — the label from the outline. */
  code: string;
  topic: string;
  subTopics: string[];
  outcome: string;
  handsOn: string;
  project: string;
  priority: Priority;
  /** Rough study budget at 8 hrs/week. */
  hours: number;
  resources: Resource[];
};

export const KIND_LABEL: Record<ResourceKind, string> = {
  docs: 'Docs',
  course: 'Course',
  udemy: 'Udemy',
  youtube: 'YouTube',
  linkedin: 'LinkedIn Learning',
  paper: 'Paper',
  article: 'Article',
  tool: 'Tool',
  book: 'Book',
  dataset: 'Dataset',
};

/** Order used for the resource-library section and the legend. */
export const KIND_ORDER: ResourceKind[] = [
  'course',
  'udemy',
  'youtube',
  'linkedin',
  'docs',
  'paper',
  'article',
  'tool',
  'book',
  'dataset',
];

export const PHASES: {key: string; label: string; blurb: string}[] = [
  {
    key: 'Foundation',
    label: 'Foundation',
    blurb:
      'Python, PyTorch, Git, Docker and API basics — the floor everything else stands on.',
  },
  {
    key: 'Month 1',
    label: 'Month 1 · Transformers',
    blurb:
      'Attention and tokenization from first principles, then fine-tune the three architecture families and optimise inference.',
  },
  {
    key: 'Month 2',
    label: 'Month 2 · Fine-tuning lifecycle',
    blurb:
      'Pre-training vs post-training, dataset construction, PEFT, preference alignment, quantization and serving.',
  },
  {
    key: 'Month 2-3',
    label: 'Month 2–3 · Scaling & compression',
    blurb:
      'Mixture of Experts, reasoning models, small language models and knowledge distillation.',
  },
  {
    key: 'Month 3',
    label: 'Month 3 · RAG engineering',
    blurb:
      'LangChain orchestration, retrieval foundations, advanced and adaptive RAG, multimodal and graph retrieval.',
  },
  {
    key: 'Month 4',
    label: 'Month 4 · Agentic AI',
    blurb:
      'Structured output, function calling, MCP, LangGraph state machines, A2A and managed agent runtimes.',
  },
  {
    key: 'Month 5',
    label: 'Month 5 · Production LLM engineering',
    blurb:
      'Prompt engineering, context engineering, evaluation harnesses and eval-gated CI/CD.',
  },
  {
    key: 'Month 6-7',
    label: 'Month 6–7 · Multimodal AI',
    blurb:
      'Vision transformers, VLMs, speech-to-text and the embedding-model taxonomy.',
  },
  {
    key: 'Capstones',
    label: 'Capstone projects',
    blurb:
      'Five end-to-end builds that turn the modules into a portfolio — plus the QA/SDET-flavoured variants.',
  },
];

export const UNITS: Unit[] = [
  /* ------------------------------------------------------------------ *
   * Foundation
   * ------------------------------------------------------------------ */
  {
    id: 'f01',
    track: 'foundation',
    phase: 'Foundation',
    code: 'Foundation',
    topic: 'Python, PyTorch & engineering setup',
    subTopics: [
      'Python functions, classes, decorators, typing',
      'PyTorch tensors, autograd, training loops',
      'Git branching and pull-request flow',
      'Docker images, layers and multi-stage builds',
      'FastAPI request/response models and async',
    ],
    outcome: 'Ready for LLM engineering — can build and ship a small ML service.',
    handsOn: 'Build a set of ML utilities and wrap them in an API service.',
    project: 'Underpins every project',
    priority: 'Must learn',
    hours: 30,
    resources: [
      {
        id: 'f01-r1',
        label: 'The Python Tutorial (official)',
        href: 'https://docs.python.org/3/tutorial/',
        kind: 'docs',
        note: 'Skim chapters 9–12 if classes and modules feel shaky.',
      },
      {
        id: 'f01-r2',
        label: 'PyTorch — Learn the Basics',
        href: 'https://pytorch.org/tutorials/beginner/basics/intro.html',
        kind: 'docs',
        note: 'Tensors → autograd → optimisation loop in one sitting.',
      },
      {
        id: 'f01-r3',
        label: 'Deep Learning with PyTorch: a 60 Minute Blitz',
        href: 'https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html',
        kind: 'docs',
      },
      {
        id: 'f01-r4',
        label: 'Pro Git (free book)',
        href: 'https://git-scm.com/book/en/v2',
        kind: 'book',
        note: 'Chapters 2–3 cover everything the course needs.',
      },
      {
        id: 'f01-r5',
        label: 'Docker — Get started',
        href: 'https://docs.docker.com/get-started/',
        kind: 'docs',
      },
      {
        id: 'f01-r6',
        label: 'FastAPI Tutorial — User Guide',
        href: 'https://fastapi.tiangolo.com/tutorial/',
        kind: 'docs',
        note: 'Every serving layer in this track is FastAPI.',
      },
      {
        id: 'f01-r7',
        label: 'Andrej Karpathy — Neural Networks: Zero to Hero',
        href: 'https://www.youtube.com/@AndrejKarpathy',
        kind: 'youtube',
        note: 'Work the playlist end to end; it is the best intuition builder there is.',
      },
      {
        id: 'f01-r8',
        label: 'Applied AI: Getting Started with Hugging Face Transformers',
        href: 'https://www.linkedin.com/learning/applied-ai-getting-started-with-hugging-face-transformers',
        kind: 'linkedin',
      },
      {
        id: 'f01-r9',
        label: 'Hugging Face — LLM Course',
        href: 'https://huggingface.co/learn/llm-course',
        kind: 'course',
        note: 'The spine for Months 1–2; free and hands-on.',
      },
    ],
  },

  /* ------------------------------------------------------------------ *
   * Month 1
   * ------------------------------------------------------------------ */
  {
    id: 'm01',
    track: 'module',
    phase: 'Month 1',
    code: 'Module 01',
    topic: 'Transformer architecture & tokenization',
    subTopics: [
      'Embeddings: discrete → continuous space',
      'Self-attention, multi-head, masked multi-head, cross-attention',
      'Positional encoding',
      'Encoder-only, decoder-only, encoder–decoder',
      'Tokenizer taxonomy: word, subword, character, byte',
      'BPE, WordPiece, SentencePiece',
    ],
    outcome: 'Understand LLM fundamentals well enough to read any model card.',
    handsOn: 'Implement attention and tokenizer experiments from scratch.',
    project: 'Mini GPT',
    priority: 'Must learn',
    hours: 22,
    resources: [
      {
        id: 'm01-r1',
        label: 'Attention Is All You Need',
        href: 'https://arxiv.org/abs/1706.03762',
        kind: 'paper',
        note: 'Read it after the illustrated guide, not before.',
      },
      {
        id: 'm01-r2',
        label: 'The Illustrated Transformer',
        href: 'https://jalammar.github.io/illustrated-transformer/',
        kind: 'article',
      },
      {
        id: 'm01-r3',
        label: 'The Annotated Transformer (Harvard NLP)',
        href: 'https://nlp.seas.harvard.edu/annotated-transformer/',
        kind: 'article',
        note: 'Paper and code side by side — type it out yourself.',
      },
      {
        id: 'm01-r4',
        label: 'Karpathy — Let’s build GPT: from scratch, in code',
        href: 'https://www.youtube.com/watch?v=kCc8FmEb1nY',
        kind: 'youtube',
        note: 'The single highest-value video in this roadmap.',
      },
      {
        id: 'm01-r5',
        label: 'Karpathy — Let’s build the GPT Tokenizer',
        href: 'https://www.youtube.com/watch?v=zduSFxRajkE',
        kind: 'youtube',
        note: 'Covers BPE end to end, including the ugly edge cases.',
      },
      {
        id: 'm01-r6',
        label: '3Blue1Brown — Attention in transformers, visually explained',
        href: 'https://www.youtube.com/watch?v=eMlx5fFNoYc',
        kind: 'youtube',
      },
      {
        id: 'm01-r7',
        label: 'Stanford CS336 — Language Modeling from Scratch (Spring 2025)',
        href: 'https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_',
        kind: 'youtube',
        note: 'Graduate-level; lectures 1–3 map onto this module.',
      },
      {
        id: 'm01-r8',
        label: 'CS336 course site (assignments + slides)',
        href: 'https://cs336.stanford.edu/spring2025',
        kind: 'course',
      },
      {
        id: 'm01-r9',
        label: 'Hugging Face Tokenizers — building a tokenizer',
        href: 'https://huggingface.co/learn/llm-course/chapter6/1',
        kind: 'course',
      },
      {
        id: 'm01-r10',
        label: 'LLMs from Scratch: Build, Align & Fine-Tune with PyTorch',
        href: 'https://www.udemy.com/course/llm-mastery-hands-on-code-align-and-master-llms/',
        kind: 'udemy',
      },
      {
        id: 'm01-r11',
        label: 'Hugging Face Transformers: Introduction to Pretrained Models',
        href: 'https://www.linkedin.com/learning/hugging-face-transformers-introduction-to-pretrained-models',
        kind: 'linkedin',
      },
      {
        id: 'm01-r12',
        label: 'Build a Large Language Model (From Scratch) — Raschka',
        href: 'https://www.manning.com/books/build-a-large-language-model-from-scratch',
        kind: 'book',
        note: 'The best companion book for Months 1–2.',
      },
    ],
  },
  {
    id: 'm02',
    track: 'module',
    phase: 'Month 1',
    code: 'Module 02',
    topic: 'Fine-tuning transformer architectures in practice',
    subTopics: [
      'Coding attention mechanisms',
      'Fine-tuning DistilBERT on custom data',
      'Fine-tuning DistilGPT on custom data',
      'Fine-tuning T5 on custom data',
    ],
    outcome: 'Train transformer models on your own data with confidence.',
    handsOn: 'Fine-tune an NLP model end to end and measure the lift.',
    project: 'Domain assistant',
    priority: 'Must learn',
    hours: 20,
    resources: [
      {
        id: 'm02-r1',
        label: 'Transformers — Fine-tune a pretrained model',
        href: 'https://huggingface.co/docs/transformers/training',
        kind: 'docs',
      },
      {
        id: 'm02-r2',
        label: 'HF LLM Course — Fine-tuning a pretrained model',
        href: 'https://huggingface.co/learn/llm-course/chapter3/1',
        kind: 'course',
      },
      {
        id: 'm02-r3',
        label: 'Datasets library documentation',
        href: 'https://huggingface.co/docs/datasets/index',
        kind: 'docs',
      },
      {
        id: 'm02-r4',
        label: 'LLM Fine-Tuning with Hugging Face: LoRA, QLoRA, PEFT',
        href: 'https://www.udemy.com/course/fine-tuning-llm-with-hugging-face-transformers/',
        kind: 'udemy',
        note: 'Covers BERT, T5 and ViT fine-tuning as well as PEFT.',
      },
      {
        id: 'm02-r5',
        label: 'Applied AI: Building NLP Apps with Hugging Face Transformers',
        href: 'https://www.linkedin.com/learning/applied-ai-building-nlp-apps-with-hugging-face-transformers',
        kind: 'linkedin',
      },
      {
        id: 'm02-r6',
        label: 'T5: Exploring the Limits of Transfer Learning',
        href: 'https://arxiv.org/abs/1910.10683',
        kind: 'paper',
      },
    ],
  },
  {
    id: 'm03',
    track: 'module',
    phase: 'Month 1',
    code: 'Module 03',
    topic: 'Inference optimization, attention variants & scaling laws',
    subTopics: [
      'The naive decoding problem, KV cache and its memory math',
      'Flash Attention, PyTorch SDPA',
      'MHA, MQA, GQA, MLA',
      'PagedAttention & vLLM',
      'RoPE',
      'Kaplan and Chinchilla scaling laws',
    ],
    outcome: 'Optimise LLM inference and size models compute-optimally.',
    handsOn: 'Benchmark local model serving: tokens/sec and first-token latency.',
    project: 'Production LLM API',
    priority: 'High',
    hours: 20,
    resources: [
      {
        id: 'm03-r1',
        label: 'vLLM documentation',
        href: 'https://docs.vllm.ai/en/latest/',
        kind: 'docs',
        note: 'Start with the quickstart, then the paged-attention design notes.',
      },
      {
        id: 'm03-r2',
        label: 'FlashAttention: Fast and Memory-Efficient Exact Attention',
        href: 'https://arxiv.org/abs/2205.14135',
        kind: 'paper',
      },
      {
        id: 'm03-r3',
        label: 'Efficient Memory Management for LLM Serving (PagedAttention)',
        href: 'https://arxiv.org/abs/2309.06180',
        kind: 'paper',
      },
      {
        id: 'm03-r4',
        label: 'GQA: Training Generalized Multi-Query Transformer Models',
        href: 'https://arxiv.org/abs/2305.13245',
        kind: 'paper',
      },
      {
        id: 'm03-r5',
        label: 'RoFormer: Rotary Position Embedding (RoPE)',
        href: 'https://arxiv.org/abs/2104.09864',
        kind: 'paper',
      },
      {
        id: 'm03-r6',
        label: 'Scaling Laws for Neural Language Models (Kaplan)',
        href: 'https://arxiv.org/abs/2001.08361',
        kind: 'paper',
      },
      {
        id: 'm03-r7',
        label: 'Training Compute-Optimal LLMs (Chinchilla)',
        href: 'https://arxiv.org/abs/2203.15556',
        kind: 'paper',
      },
      {
        id: 'm03-r8',
        label: 'Transformers — LLM inference optimization',
        href: 'https://huggingface.co/docs/transformers/llm_optims',
        kind: 'docs',
      },
      {
        id: 'm03-r9',
        label: 'torch.nn.functional.scaled_dot_product_attention',
        href: 'https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html',
        kind: 'docs',
        note: 'The unified attention API the module refers to.',
      },
      {
        id: 'm03-r10',
        label: 'Umar Jamil — architecture and inference deep dives',
        href: 'https://www.youtube.com/@umarjamilai',
        kind: 'youtube',
        note: 'Line-by-line walkthroughs of KV cache, RoPE, GQA and Flash Attention.',
      },
    ],
  },

  /* ------------------------------------------------------------------ *
   * Month 2
   * ------------------------------------------------------------------ */
  {
    id: 'm04',
    track: 'module',
    phase: 'Month 2',
    code: 'Module 04',
    topic: 'LLM fine-tuning lifecycle & pre-training foundations',
    subTopics: [
      'Pre-training vs post-training',
      'What pre-training produces, and why base models are not useful raw',
      'Training objectives: CLM, MLM, Prefix-LM',
      'Data curation and filtering at scale',
      'Continued pre-training (CPT) for domain adaptation',
      'Multi-Token Prediction (MTP)',
    ],
    outcome: 'Know when to reach for CPT and when to go straight to SFT.',
    handsOn: 'Plan a post-training pipeline for one domain and cost it out.',
    project: 'AML/KYC Assistant',
    priority: 'Must learn',
    hours: 14,
    resources: [
      {
        id: 'm04-r1',
        label: 'HF LLM Course — Transformer models',
        href: 'https://huggingface.co/learn/llm-course/chapter1/1',
        kind: 'course',
      },
      {
        id: 'm04-r2',
        label: 'Ahead of AI — Sebastian Raschka',
        href: 'https://magazine.sebastianraschka.com/',
        kind: 'article',
        note: 'The clearest running commentary on post-training practice.',
      },
      {
        id: 'm04-r3',
        label: 'DeepSeek-V3 Technical Report (MTP in production)',
        href: 'https://arxiv.org/abs/2412.19437',
        kind: 'paper',
      },
      {
        id: 'm04-r4',
        label: 'The Pile / data curation at scale',
        href: 'https://arxiv.org/abs/2101.00027',
        kind: 'paper',
      },
      {
        id: 'm04-r5',
        label: 'CS336 lecture — Data',
        href: 'https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_',
        kind: 'youtube',
      },
    ],
  },
  {
    id: 'm05',
    track: 'module',
    phase: 'Month 2',
    code: 'Module 05',
    topic: 'Data preparation & synthetic dataset generation',
    subTopics: [
      'Dataset formats: instruction pairs, chat format',
      'Chat templates: ChatML, Llama-3, Mistral',
      'Loss masking — and what breaks without it',
      'Deduplication and filtering pipelines',
      'Self-Instruct, Alpaca, preference dataset generation',
      'LLM-as-judge scoring; distilabel, DataDreamer, Argilla',
      'Risks: model collapse and data poisoning',
    ],
    outcome: 'Build a domain SFT dataset that does not quietly poison the model.',
    handsOn: 'Generate, dedupe and score a synthetic instruction set.',
    project: 'AML/KYC Assistant',
    priority: 'Must learn',
    hours: 18,
    resources: [
      {
        id: 'm05-r1',
        label: 'Transformers — Chat templates',
        href: 'https://huggingface.co/docs/transformers/chat_templating',
        kind: 'docs',
        note: 'Read this before your first SFT run, not after it fails.',
      },
      {
        id: 'm05-r2',
        label: 'distilabel documentation',
        href: 'https://distilabel.argilla.io/latest/',
        kind: 'tool',
      },
      {
        id: 'm05-r3',
        label: 'Argilla documentation',
        href: 'https://docs.argilla.io/latest/',
        kind: 'tool',
      },
      {
        id: 'm05-r4',
        label: 'Self-Instruct: Aligning LMs with Self-Generated Instructions',
        href: 'https://arxiv.org/abs/2212.10560',
        kind: 'paper',
      },
      {
        id: 'm05-r5',
        label: 'Stanford Alpaca',
        href: 'https://crfm.stanford.edu/2023/03/13/alpaca.html',
        kind: 'article',
      },
      {
        id: 'm05-r6',
        label: 'The Curse of Recursion: model collapse',
        href: 'https://arxiv.org/abs/2305.17493',
        kind: 'paper',
      },
      {
        id: 'm05-r7',
        label: 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena',
        href: 'https://arxiv.org/abs/2306.05685',
        kind: 'paper',
      },
    ],
  },
  {
    id: 'm06',
    track: 'module',
    phase: 'Month 2',
    code: 'Module 06',
    topic: 'SFT, parameter-efficient methods & preference alignment',
    subTopics: [
      'Intrinsic dimensionality; LoRA math, rank, alpha, target modules',
      'QLoRA — 4-bit NF4, double quantization, paged optimizers',
      'DoRA, AdaLoRA, LoRA+',
      'Instruction tuning, chat tuning, chain-of-thought tuning',
      'RLHF with PPO — reward model, critic, KL penalty',
      'DPO — direct optimization without a reward model',
    ],
    outcome: 'Customise and align open models on a single mid-tier GPU.',
    handsOn: 'Create a full fine-tuning pipeline: SFT → DPO → eval.',
    project: 'AML/KYC Assistant',
    priority: 'Must learn',
    hours: 26,
    resources: [
      {
        id: 'm06-r1',
        label: 'LoRA: Low-Rank Adaptation of Large Language Models',
        href: 'https://arxiv.org/abs/2106.09685',
        kind: 'paper',
      },
      {
        id: 'm06-r2',
        label: 'QLoRA: Efficient Finetuning of Quantized LLMs',
        href: 'https://arxiv.org/abs/2305.14314',
        kind: 'paper',
      },
      {
        id: 'm06-r3',
        label: 'DoRA: Weight-Decomposed Low-Rank Adaptation',
        href: 'https://arxiv.org/abs/2402.09353',
        kind: 'paper',
      },
      {
        id: 'm06-r4',
        label: 'Direct Preference Optimization (DPO)',
        href: 'https://arxiv.org/abs/2305.18290',
        kind: 'paper',
      },
      {
        id: 'm06-r5',
        label: 'InstructGPT — training LMs to follow instructions with human feedback',
        href: 'https://arxiv.org/abs/2203.02155',
        kind: 'paper',
      },
      {
        id: 'm06-r6',
        label: 'PEFT documentation',
        href: 'https://huggingface.co/docs/peft/index',
        kind: 'docs',
      },
      {
        id: 'm06-r7',
        label: 'TRL — SFTTrainer, DPOTrainer, GRPO, ORPO',
        href: 'https://huggingface.co/docs/trl/index',
        kind: 'docs',
        note: 'The library the capstone fine-tuning project is built on.',
      },
      {
        id: 'm06-r8',
        label: 'Unsloth documentation',
        href: 'https://docs.unsloth.ai/',
        kind: 'tool',
        note: 'Halves the VRAM bill on consumer GPUs.',
      },
      {
        id: 'm06-r9',
        label: 'LLM Fine-Tuning for Beginners: HuggingFace & Unsloth',
        href: 'https://www.udemy.com/course/ai-with-huggingface/',
        kind: 'udemy',
        note: 'SFT, LoRA, QLoRA, DPO and GRPO in one pass.',
      },
      {
        id: 'm06-r10',
        label: 'DeepLearning.AI — Finetuning Large Language Models',
        href: 'https://www.deeplearning.ai/short-courses/finetuning-large-language-models/',
        kind: 'course',
      },
      {
        id: 'm06-r11',
        label: 'Fine-Tune & Deploy LLMs with QLoRA on SageMaker + Streamlit',
        href: 'https://www.udemy.com/course/fine-tune-deploy-llms-with-qlora-on-sagemaker-streamlit/',
        kind: 'udemy',
      },
    ],
  },
  {
    id: 'm07',
    track: 'module',
    phase: 'Month 2',
    code: 'Module 07',
    topic: 'Evaluation, quantization, deployment & tooling',
    subTopics: [
      'Benchmark types: knowledge, reasoning, instruction-following',
      'LLM-as-judge: MT-Bench, Chatbot Arena, domain evals',
      'GPTQ, AWQ, BNB NF4, FP8',
      'Merging LoRA adapters; serving many adapters from one base',
      'vLLM, SGLang, GGUF, llama.cpp, speculative decoding',
      'TRL, Unsloth, Axolotl, LLaMA-Factory, managed fine-tuning',
    ],
    outcome: 'Ship a quantized, evaluated model behind a real endpoint.',
    handsOn: 'Quantize a fine-tuned model and serve multiple adapters.',
    project: 'Production LLM API',
    priority: 'Must learn',
    hours: 22,
    resources: [
      {
        id: 'm07-r1',
        label: 'Transformers — Quantization overview',
        href: 'https://huggingface.co/docs/transformers/quantization/overview',
        kind: 'docs',
      },
      {
        id: 'm07-r2',
        label: 'GPTQ: Accurate Post-Training Quantization',
        href: 'https://arxiv.org/abs/2210.17323',
        kind: 'paper',
      },
      {
        id: 'm07-r3',
        label: 'AWQ: Activation-aware Weight Quantization',
        href: 'https://arxiv.org/abs/2306.00978',
        kind: 'paper',
      },
      {
        id: 'm07-r4',
        label: 'GGUF format on the Hugging Face Hub',
        href: 'https://huggingface.co/docs/hub/gguf',
        kind: 'docs',
        note: 'The practical entry point to llama.cpp conversion.',
      },
      {
        id: 'm07-r5',
        label: 'Fast Inference via Speculative Decoding',
        href: 'https://arxiv.org/abs/2211.17192',
        kind: 'paper',
      },
      {
        id: 'm07-r6',
        label: 'Axolotl documentation',
        href: 'https://docs.axolotl.ai/',
        kind: 'tool',
      },
      {
        id: 'm07-r7',
        label: 'LMArena (Chatbot Arena) leaderboard',
        href: 'https://lmarena.ai/',
        kind: 'tool',
      },
      {
        id: 'm07-r8',
        label: 'MLOps Tools: MLflow and Hugging Face',
        href: 'https://www.linkedin.com/learning/mlops-tools-mlflow-and-hugging-face',
        kind: 'linkedin',
      },
    ],
  },

  /* ------------------------------------------------------------------ *
   * Month 2–3 — scaling & compression
   * ------------------------------------------------------------------ */
  {
    id: 'm08',
    track: 'module',
    phase: 'Month 2-3',
    code: 'Module 08',
    topic: 'Mixture of Experts (MoE) for scaling LLMs',
    subTopics: [
      'The dense model scaling problem',
      'Routing and the MoE idea; architecture deep dive',
      'Load balancing and expert collapse',
      'Training vs inference trade-offs; sparse and soft MoE',
      'MoE vs dense: when to use each',
    ],
    outcome: 'Read and reason about frontier MoE model cards.',
    handsOn: 'Trace routing decisions through an open MoE checkpoint.',
    project: 'Architecture literacy',
    priority: 'Advanced',
    hours: 10,
    resources: [
      {
        id: 'm08-r1',
        label: 'Mixture of Experts Explained (HF blog)',
        href: 'https://huggingface.co/blog/moe',
        kind: 'article',
        note: 'Start here — it is the best single explainer.',
      },
      {
        id: 'm08-r2',
        label: 'Switch Transformers',
        href: 'https://arxiv.org/abs/2101.03961',
        kind: 'paper',
      },
      {
        id: 'm08-r3',
        label: 'Mixtral of Experts',
        href: 'https://arxiv.org/abs/2401.04088',
        kind: 'paper',
      },
      {
        id: 'm08-r4',
        label: 'CS336 lecture — Mixture of Experts',
        href: 'https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_',
        kind: 'youtube',
      },
    ],
  },
  {
    id: 'm09',
    track: 'module',
    phase: 'Month 2-3',
    code: 'Module 09',
    topic: 'Reasoning models, chain-of-thought & RL-only training',
    subTopics: [
      'What makes a reasoning model different',
      'Chain-of-thought as prompting and as training signal',
      'The training recipe',
      'DeepSeek-R1-Zero: skipping SFT entirely',
      'Distilling reasoning without RL',
    ],
    outcome: 'Decide when a reasoning model earns its latency and cost.',
    handsOn: 'Compare CoT prompting against a reasoning model on one task suite.',
    project: 'LLM Quality Platform',
    priority: 'High',
    hours: 12,
    resources: [
      {
        id: 'm09-r1',
        label: 'Chain-of-Thought Prompting Elicits Reasoning',
        href: 'https://arxiv.org/abs/2201.11903',
        kind: 'paper',
      },
      {
        id: 'm09-r2',
        label: 'Self-Consistency Improves Chain of Thought',
        href: 'https://arxiv.org/abs/2203.11171',
        kind: 'paper',
      },
      {
        id: 'm09-r3',
        label: 'DeepSeek-R1: Incentivizing Reasoning via RL',
        href: 'https://arxiv.org/abs/2501.12948',
        kind: 'paper',
        note: 'The R1-Zero section is the one the module is built around.',
      },
      {
        id: 'm09-r4',
        label: 'Karpathy — Deep Dive into LLMs like ChatGPT',
        href: 'https://www.youtube.com/watch?v=7xTGNNLPyMI',
        kind: 'youtube',
        note: 'Covers pre-training → SFT → RLHF → reasoning in one long sitting.',
      },
    ],
  },
  {
    id: 'm10',
    track: 'module',
    phase: 'Month 2-3',
    code: 'Module 10',
    topic: 'Small language models & knowledge distillation',
    subTopics: [
      'Why SLMs matter: cost, latency, privacy',
      'Distillation, pruning, quantization',
      'Student–teacher paradigm; hard vs soft labels',
      'Temperature scaling and the KL divergence loss',
      'Attention transfer; building a distillation pipeline',
    ],
    outcome: 'Compress a capable model to edge-deployable size.',
    handsOn: 'Run a distillation pipeline and an ablation over loss terms.',
    project: 'EdgeReason capstone',
    priority: 'High',
    hours: 16,
    resources: [
      {
        id: 'm10-r1',
        label: 'Distilling the Knowledge in a Neural Network (Hinton)',
        href: 'https://arxiv.org/abs/1503.02531',
        kind: 'paper',
      },
      {
        id: 'm10-r2',
        label: 'DistilBERT',
        href: 'https://arxiv.org/abs/1910.01108',
        kind: 'paper',
      },
      {
        id: 'm10-r3',
        label: 'Phi-3 Technical Report',
        href: 'https://arxiv.org/abs/2404.14219',
        kind: 'paper',
        note: 'The clearest case study in data-quality-driven SLM design.',
      },
      {
        id: 'm10-r4',
        label: 'Fundamentals of SLM Fine-Tuning: LoRA, Quantization & Edge',
        href: 'https://www.udemy.com/course/edge-ai-with-slms-fine-tuning-local-deployment/',
        kind: 'udemy',
      },
      {
        id: 'm10-r5',
        label: 'Ollama documentation',
        href: 'https://docs.ollama.com/',
        kind: 'tool',
        note: 'Quickest way to run the distilled result locally.',
      },
    ],
  },

  /* ------------------------------------------------------------------ *
   * Month 3 — RAG
   * ------------------------------------------------------------------ */
  {
    id: 'm15',
    track: 'module',
    phase: 'Month 3',
    code: 'Module 15',
    topic: 'LangChain for RAG: ecosystem, patterns & production',
    subTopics: [
      'Architecture, model abstraction, LCEL deep dive',
      'Prompting, structured output, output parsers, tool calling',
      'Memory, document loaders, text splitters, vector stores',
      'Agentic RAG pipelines; multimodal and text-to-SQL',
      'Callbacks, tracing and LangSmith; security and responsible AI',
    ],
    outcome: 'Orchestrate production RAG rather than gluing scripts together.',
    handsOn: 'Build and deploy a secured LangChain RAG API.',
    project: 'Enterprise RAG',
    priority: 'Must learn',
    hours: 24,
    resources: [
      {
        id: 'm15-r1',
        label: 'LangChain documentation (Python)',
        href: 'https://python.langchain.com/docs/introduction/',
        kind: 'docs',
      },
      {
        id: 'm15-r2',
        label: 'LangChain Expression Language (LCEL)',
        href: 'https://python.langchain.com/docs/concepts/lcel/',
        kind: 'docs',
      },
      {
        id: 'm15-r3',
        label: 'LangSmith documentation',
        href: 'https://docs.smith.langchain.com/',
        kind: 'docs',
        note: 'Tracing and evals — used again in Month 5.',
      },
      {
        id: 'm15-r4',
        label: 'Ultimate RAG Bootcamp Using LangChain, LangGraph & LangSmith',
        href: 'https://www.udemy.com/course/ultimate-rag-bootcamp-using-langchainlanggraph-langsmith/',
        kind: 'udemy',
        note: 'Closest match to this module and Module 17.',
      },
      {
        id: 'm15-r5',
        label: 'LangChain & LangGraph Mastery: RAG, Agents & AI Workflows',
        href: 'https://www.udemy.com/course/langchain-langgraph-mastery-rag-agents-ai-workflows/',
        kind: 'udemy',
      },
      {
        id: 'm15-r6',
        label: 'LinkedIn Learning — Generative AI topic hub',
        href: 'https://www.linkedin.com/learning/topics/generative-ai',
        kind: 'linkedin',
        note: 'Filter to RAG and vector-database courses.',
      },
    ],
  },
  {
    id: 'm16',
    track: 'module',
    phase: 'Month 3',
    code: 'Module 16',
    topic: 'Foundations of RAG: embeddings, chunking & retrieval',
    subTopics: [
      'Vanilla RAG end to end',
      'Choosing embedding models',
      'Chunking strategies and their failure modes',
      'BM25 sparse retrieval',
      'SPLADE and ColBERT multi-vector retrieval',
    ],
    outcome: 'Build a retrieval baseline you can actually measure against.',
    handsOn: 'Index a real corpus; compare dense, sparse and hybrid recall.',
    project: 'Enterprise RAG',
    priority: 'Must learn',
    hours: 18,
    resources: [
      {
        id: 'm16-r1',
        label: 'Retrieval-Augmented Generation (original paper)',
        href: 'https://arxiv.org/abs/2005.11401',
        kind: 'paper',
      },
      {
        id: 'm16-r2',
        label: 'Chunking strategies for LLM applications',
        href: 'https://www.pinecone.io/learn/chunking-strategies/',
        kind: 'article',
      },
      {
        id: 'm16-r3',
        label: 'ColBERT: Efficient and Effective Passage Search',
        href: 'https://arxiv.org/abs/2004.12832',
        kind: 'paper',
      },
      {
        id: 'm16-r4',
        label: 'SPLADE: Sparse Lexical and Expansion Model',
        href: 'https://arxiv.org/abs/2107.05720',
        kind: 'paper',
      },
      {
        id: 'm16-r5',
        label: 'Sentence Transformers documentation',
        href: 'https://sbert.net/',
        kind: 'docs',
      },
      {
        id: 'm16-r6',
        label: 'Qdrant documentation',
        href: 'https://qdrant.tech/documentation/',
        kind: 'docs',
        note: 'The vector store used by the legal-RAG capstone.',
      },
      {
        id: 'm16-r7',
        label: 'Basic RAG with LangChain and LangGraph — Ollama',
        href: 'https://www.udemy.com/course/agentic-rag-with-langchain-and-langgraph/',
        kind: 'udemy',
        note: 'Runs fully local, which is handy for practising at zero API cost.',
      },
    ],
  },
  {
    id: 'm17',
    track: 'module',
    phase: 'Month 3',
    code: 'Module 17',
    topic: 'Advanced RAG: query transforms, rerankers & adaptive retrieval',
    subTopics: [
      'Hybrid and meta-hybrid RAG',
      'Query transformations and rerankers',
      'Self-RAG, Corrective RAG, Adaptive RAG',
      'Contextual retrieval',
      'Agentic RAG and RAG evaluation',
    ],
    outcome: 'Raise retrieval precision with measurements, not vibes.',
    handsOn: 'Add reranking + routing and prove the gain with RAGAS.',
    project: 'Enterprise RAG',
    priority: 'Must learn',
    hours: 22,
    resources: [
      {
        id: 'm17-r1',
        label: 'Self-RAG: Learning to Retrieve, Generate and Critique',
        href: 'https://arxiv.org/abs/2310.11511',
        kind: 'paper',
      },
      {
        id: 'm17-r2',
        label: 'Corrective Retrieval Augmented Generation',
        href: 'https://arxiv.org/abs/2401.15884',
        kind: 'paper',
      },
      {
        id: 'm17-r3',
        label: 'Adaptive-RAG',
        href: 'https://arxiv.org/abs/2403.14403',
        kind: 'paper',
      },
      {
        id: 'm17-r4',
        label: 'HyDE: Precise Zero-Shot Dense Retrieval',
        href: 'https://arxiv.org/abs/2212.10496',
        kind: 'paper',
      },
      {
        id: 'm17-r5',
        label: 'Anthropic — Introducing Contextual Retrieval',
        href: 'https://www.anthropic.com/news/contextual-retrieval',
        kind: 'article',
        note: 'Cheap trick, large recall gain — implement it.',
      },
      {
        id: 'm17-r6',
        label: 'RAGAS documentation',
        href: 'https://docs.ragas.io/',
        kind: 'docs',
        note: 'Faithfulness, answer relevancy, context precision/recall.',
      },
      {
        id: 'm17-r7',
        label: 'LangGraph tutorials — adaptive and corrective RAG',
        href: 'https://langchain-ai.github.io/langgraph/examples/',
        kind: 'docs',
      },
    ],
  },
  {
    id: 'm18',
    track: 'module',
    phase: 'Month 3',
    code: 'Module 18',
    topic: 'Vector quantization, multimodal RAG & emerging patterns',
    subTopics: [
      'Scalar, binary and product quantization',
      'ColPali paradigm; layout detection and OCR-free parsing',
      'VL embeddings and VL rerankers',
      'Graph RAG with Neo4j',
      'Vectorless RAG (PageIndex); caching and semantic caching',
      'PII masking, guardrails, prompt-injection defence',
    ],
    outcome: 'Scale retrieval and secure it against injection and leakage.',
    handsOn: 'Index scanned PDFs with ColPali; add a graph retriever and guardrails.',
    project: 'Enterprise RAG / Playwright Knowledge Graph Assistant',
    priority: 'Must learn',
    hours: 24,
    resources: [
      {
        id: 'm18-r1',
        label: 'ColPali: Efficient Document Retrieval with VLMs',
        href: 'https://arxiv.org/abs/2407.01449',
        kind: 'paper',
      },
      {
        id: 'm18-r2',
        label: 'From Local to Global: a Graph RAG approach',
        href: 'https://arxiv.org/abs/2404.16130',
        kind: 'paper',
      },
      {
        id: 'm18-r3',
        label: 'Neo4j GraphRAG for Python',
        href: 'https://neo4j.com/docs/neo4j-graphrag-python/current/',
        kind: 'docs',
      },
      {
        id: 'm18-r4',
        label: 'Qdrant — quantization guide',
        href: 'https://qdrant.tech/documentation/guides/quantization/',
        kind: 'docs',
      },
      {
        id: 'm18-r5',
        label: 'NVIDIA NeMo Guardrails',
        href: 'https://developer.nvidia.com/nemo-guardrails',
        kind: 'tool',
      },
      {
        id: 'm18-r6',
        label: 'Microsoft Presidio — PII detection and anonymization',
        href: 'https://microsoft.github.io/presidio/',
        kind: 'tool',
      },
      {
        id: 'm18-r7',
        label: 'OWASP Top 10 for LLM Applications',
        href: 'https://genai.owasp.org/llm-top-10/',
        kind: 'article',
        note: 'Prompt injection, insecure output handling, data leakage.',
      },
      {
        id: 'm18-r8',
        label: 'RAG Agents: Build Apps & GPTs with APIs/MCP, LangChain & n8n',
        href: 'https://www.udemy.com/course/rag-agents-build-apps-gpts-with-apismcp-langchain-n8n/',
        kind: 'udemy',
      },
    ],
  },

  /* ------------------------------------------------------------------ *
   * Month 4 — agents
   * ------------------------------------------------------------------ */
  {
    id: 'm19',
    track: 'module',
    phase: 'Month 4',
    code: 'Module 19',
    topic: 'Agent foundations: structured output, function calling & MCP',
    subTopics: [
      'Pydantic models, validators, settings management',
      'Function schema design and the request/response lifecycle',
      'Parallel calls, forced tool choice, structured output vs tools',
      'Building a tool executor loop from scratch',
      'MCP architecture: hosts, clients, servers, stdio/SSE',
      'MCP security: authentication, authorization, trust boundaries',
    ],
    outcome: 'Make an LLM drive real tools safely and predictably.',
    handsOn: 'Write an MCP server and client, then wire them to an agent.',
    project: 'AI QA Engineer Agent',
    priority: 'Must learn',
    hours: 22,
    resources: [
      {
        id: 'm19-r1',
        label: 'Model Context Protocol documentation',
        href: 'https://modelcontextprotocol.io/',
        kind: 'docs',
        note: 'Read the spec section, not just the quickstart.',
      },
      {
        id: 'm19-r2',
        label: 'FastMCP — the fast Python MCP framework',
        href: 'https://gofastmcp.com/',
        kind: 'tool',
      },
      {
        id: 'm19-r3',
        label: 'Pydantic documentation',
        href: 'https://docs.pydantic.dev/latest/',
        kind: 'docs',
      },
      {
        id: 'm19-r4',
        label: 'Claude — tool use overview',
        href: 'https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview',
        kind: 'docs',
      },
      {
        id: 'm19-r5',
        label: 'OpenAI — function calling guide',
        href: 'https://platform.openai.com/docs/guides/function-calling',
        kind: 'docs',
        note: 'Pair with the Claude doc to see the universal contract.',
      },
      {
        id: 'm19-r6',
        label: 'Anthropic — Building effective agents',
        href: 'https://www.anthropic.com/engineering/building-effective-agents',
        kind: 'article',
        note: 'The single best piece on when *not* to build an agent.',
      },
      {
        id: 'm19-r7',
        label: 'Hugging Face — AI Agents Course',
        href: 'https://huggingface.co/learn/agents-course',
        kind: 'course',
      },
    ],
  },
  {
    id: 'm20',
    track: 'module',
    phase: 'Month 4',
    code: 'Module 20',
    topic: 'LangGraph: stateful, multi-agent & human-in-the-loop workflows',
    subTopics: [
      'Core graph concepts, state management, nodes, edges, routing',
      'Tool calling and the ReAct pattern',
      'Human-in-the-loop interrupts',
      'Memory and persistence across sessions',
      'Multi-agent supervisor systems; streaming and observability',
    ],
    outcome: 'Build autonomous systems that can be paused, resumed and audited.',
    handsOn: 'Create a tool-using agent with HITL gates on every write.',
    project: 'Autonomous QA Engineer Agent',
    priority: 'Must learn',
    hours: 26,
    resources: [
      {
        id: 'm20-r1',
        label: 'LangGraph documentation',
        href: 'https://langchain-ai.github.io/langgraph/',
        kind: 'docs',
      },
      {
        id: 'm20-r2',
        label: 'ReAct: Synergizing Reasoning and Acting',
        href: 'https://arxiv.org/abs/2210.03629',
        kind: 'paper',
      },
      {
        id: 'm20-r3',
        label: 'Complete Agentic AI Bootcamp With LangGraph and LangChain',
        href: 'https://www.udemy.com/course/complete-agentic-ai-bootcamp-with-langgraph-and-langchain/',
        kind: 'udemy',
      },
      {
        id: 'm20-r4',
        label: 'LangGraph — Develop LLM powered AI agents',
        href: 'https://www.udemy.com/course/langgraph/',
        kind: 'udemy',
      },
      {
        id: 'm20-r5',
        label: 'Production AI Agents with LangChain + LangGraph [2026]',
        href: 'https://www.udemy.com/course/production-ai-agents/',
        kind: 'udemy',
        note: 'Adds testing, security, FastAPI and Docker — closest to this track.',
      },
      {
        id: 'm20-r6',
        label: 'LangChain Academy',
        href: 'https://academy.langchain.com/',
        kind: 'course',
        note: 'Free first-party LangGraph course.',
      },
    ],
  },
  {
    id: 'm21',
    track: 'module',
    phase: 'Month 4',
    code: 'Module 21',
    topic: 'Production agents: observability, A2A & Bedrock AgentCore',
    subTopics: [
      'Agent observability with LangSmith and Logfire',
      'A2A: agent cards, task lifecycle, HTTP/SSE/JSON-RPC',
      'A2A vs MCP — when to use which',
      'AgentCore runtime: microVM isolation, sessions',
      'AgentCore memory, gateway, identity, browser, code interpreter',
      'Cedar policies and real-time tool-call interception',
    ],
    outcome: 'Run agents in production with identity, policy and traces.',
    handsOn: 'Deploy a multi-agent system to a managed agent runtime.',
    project: 'AI QA Engineer Agent',
    priority: 'High',
    hours: 22,
    resources: [
      {
        id: 'm21-r1',
        label: 'A2A Protocol documentation',
        href: 'https://a2a-protocol.org/',
        kind: 'docs',
      },
      {
        id: 'm21-r2',
        label: 'Amazon Bedrock AgentCore documentation',
        href: 'https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html',
        kind: 'docs',
      },
      {
        id: 'm21-r3',
        label: 'Strands Agents SDK',
        href: 'https://strandsagents.com/',
        kind: 'tool',
      },
      {
        id: 'm21-r4',
        label: 'Pydantic Logfire documentation',
        href: 'https://logfire.pydantic.dev/docs/',
        kind: 'tool',
      },
      {
        id: 'm21-r5',
        label: 'OpenTelemetry — GenAI semantic conventions',
        href: 'https://opentelemetry.io/docs/specs/semconv/gen-ai/',
        kind: 'docs',
        note: 'What LangSmith, Langfuse and CloudWatch all normalise to.',
      },
      {
        id: 'm21-r6',
        label: 'Cedar policy language',
        href: 'https://www.cedarpolicy.com/',
        kind: 'docs',
      },
    ],
  },

  /* ------------------------------------------------------------------ *
   * Month 5 — production engineering
   * ------------------------------------------------------------------ */
  {
    id: 'm22',
    track: 'module',
    phase: 'Month 5',
    code: 'Module 22',
    topic: 'Prompt engineering: structure, techniques & refinement',
    subTopics: [
      'Prompt anatomy: instruction, context, input, output format',
      'Zero-, one- and few-shot prompting; system prompts and roles',
      'Chain-of-thought and step-back prompting',
      'Structured generation: JSON mode, XML tags',
      'Sensitivity and robustness testing',
      'Prompt chaining, meta-prompting, self-refinement loops',
    ],
    outcome: 'Operate reliable LLM applications instead of fragile ones.',
    handsOn: 'Build a prompt suite with robustness tests and versioning.',
    project: 'LLM Quality Platform',
    priority: 'Must learn',
    hours: 14,
    resources: [
      {
        id: 'm22-r1',
        label: 'Claude — prompt engineering overview',
        href: 'https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview',
        kind: 'docs',
      },
      {
        id: 'm22-r2',
        label: 'OpenAI — prompt engineering guide',
        href: 'https://platform.openai.com/docs/guides/prompt-engineering',
        kind: 'docs',
      },
      {
        id: 'm22-r3',
        label: 'Prompt Engineering Guide (DAIR.AI)',
        href: 'https://www.promptingguide.ai/',
        kind: 'article',
      },
      {
        id: 'm22-r4',
        label: 'DeepLearning.AI — ChatGPT Prompt Engineering for Developers',
        href: 'https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/',
        kind: 'course',
      },
      {
        id: 'm22-r5',
        label: 'Introduction to Prompt Engineering for Generative AI',
        href: 'https://www.linkedin.com/learning/introduction-to-prompt-engineering-for-generative-ai-24636124',
        kind: 'linkedin',
      },
      {
        id: 'm22-r6',
        label: 'Prompt Engineering: How to Talk to the AIs',
        href: 'https://www.linkedin.com/learning/prompt-engineering-how-to-talk-to-the-ais',
        kind: 'linkedin',
      },
    ],
  },
  {
    id: 'm23',
    track: 'module',
    phase: 'Month 5',
    code: 'Module 23',
    topic: 'Context engineering: memory, pruning & retrieved context',
    subTopics: [
      'Context window anatomy: tokens, ordering, recency bias',
      'RAG as context construction',
      'Memory architectures: short-term, episodic, semantic, procedural',
      'Tool results and structured data as context',
      'Compression and pruning (LLMLingua, RECOMP)',
      'Multi-turn state and context accumulation',
    ],
    outcome: 'Keep long-running agents coherent and affordable.',
    handsOn: 'Add compression + memory tiers to an existing agent.',
    project: 'LLM Quality Platform',
    priority: 'Must learn',
    hours: 14,
    resources: [
      {
        id: 'm23-r1',
        label: 'Anthropic — Effective context engineering for AI agents',
        href: 'https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents',
        kind: 'article',
      },
      {
        id: 'm23-r2',
        label: 'Lost in the Middle: how LMs use long contexts',
        href: 'https://arxiv.org/abs/2307.03172',
        kind: 'paper',
        note: 'Explains why ordering matters more than window size.',
      },
      {
        id: 'm23-r3',
        label: 'LLMLingua: compressing prompts',
        href: 'https://arxiv.org/abs/2310.05736',
        kind: 'paper',
      },
      {
        id: 'm23-r4',
        label: 'RECOMP: compression and selective augmentation',
        href: 'https://arxiv.org/abs/2310.04408',
        kind: 'paper',
      },
      {
        id: 'm23-r5',
        label: 'LangGraph — memory and persistence',
        href: 'https://langchain-ai.github.io/langgraph/concepts/memory/',
        kind: 'docs',
      },
    ],
  },
  {
    id: 'm24',
    track: 'module',
    phase: 'Month 5',
    code: 'Module 24',
    topic: 'Harness engineering: evaluation, benchmarking & agent CI/CD',
    subTopics: [
      'Evaluation harness fundamentals; why ad-hoc testing fails',
      'Benchmarking (MMLU, GSM8K, TruthfulQA)',
      'Inspect AI: Task, Solver, Scorer',
      'LLM-as-judge pipelines: model-graded scoring, calibration, bias',
      'Tool execution sandboxes (E2B, Modal, Docker)',
      'ReAct harness patterns: iteration guards, fallbacks, token budgets',
      'Prompt regression and snapshot testing with PromptFoo',
      'Agent CI/CD: eval gating on PRs, cost regression guards',
    ],
    outcome: 'Gate every prompt and code change behind measurable evals.',
    handsOn: 'Create an evaluation pipeline that blocks a regressing PR.',
    project: 'LLM Evaluation Platform',
    priority: 'Must learn',
    hours: 24,
    resources: [
      {
        id: 'm24-r1',
        label: 'Inspect AI documentation (UK AISI)',
        href: 'https://inspect.aisi.org.uk/',
        kind: 'docs',
        note: 'Task/Solver/Scorer is the mental model the module teaches.',
      },
      {
        id: 'm24-r2',
        label: 'Inspect Evals — 200+ prebuilt evaluations',
        href: 'https://ukgovernmentbeis.github.io/inspect_evals/',
        kind: 'tool',
      },
      {
        id: 'm24-r3',
        label: 'PromptFoo documentation',
        href: 'https://www.promptfoo.dev/docs/intro/',
        kind: 'docs',
      },
      {
        id: 'm24-r4',
        label: 'RAGAS — metrics reference',
        href: 'https://docs.ragas.io/en/stable/concepts/metrics/',
        kind: 'docs',
      },
      {
        id: 'm24-r5',
        label: 'E2B — code interpreter sandboxes',
        href: 'https://e2b.dev/docs',
        kind: 'tool',
      },
      {
        id: 'm24-r6',
        label: 'Modal documentation',
        href: 'https://modal.com/docs',
        kind: 'tool',
      },
      {
        id: 'm24-r7',
        label: 'Weights & Biases documentation',
        href: 'https://docs.wandb.ai/',
        kind: 'tool',
        note: 'Loss curves for the distillation and fine-tuning projects.',
      },
      {
        id: 'm24-r8',
        label: 'MLOps Essentials on LinkedIn Learning',
        href: 'https://www.linkedin.com/learning/topics/machine-learning',
        kind: 'linkedin',
        note: 'Filter to MLOps; the CI/CD fundamentals transfer directly.',
      },
    ],
  },

  /* ------------------------------------------------------------------ *
   * Month 6–7 — multimodal
   * ------------------------------------------------------------------ */
  {
    id: 'm11',
    track: 'module',
    phase: 'Month 6-7',
    code: 'Module 11',
    topic: 'Vision foundations: from CNNs to Vision Transformers',
    subTopics: [
      'Image as a sequence of patches; patch embedding, CLS token',
      '2D positional encoding; transformer encoder on visual tokens',
      'Attention maps — what a ViT actually sees',
      'CNN vs ViT inductive bias',
      'Pre-trained encoders: CLIP, SigLIP, DINO/DINOv2',
    ],
    outcome: 'Use pre-trained vision encoders for downstream tasks.',
    handsOn: 'Visualise ViT attention; embed a document corpus with SigLIP.',
    project: 'Multimodal Agent',
    priority: 'Advanced',
    hours: 16,
    resources: [
      {
        id: 'm11-r1',
        label: 'An Image is Worth 16x16 Words (ViT)',
        href: 'https://arxiv.org/abs/2010.11929',
        kind: 'paper',
      },
      {
        id: 'm11-r2',
        label: 'CLIP: Learning Transferable Visual Models',
        href: 'https://arxiv.org/abs/2103.00020',
        kind: 'paper',
      },
      {
        id: 'm11-r3',
        label: 'SigLIP: Sigmoid Loss for Language-Image Pre-training',
        href: 'https://arxiv.org/abs/2303.15343',
        kind: 'paper',
      },
      {
        id: 'm11-r4',
        label: 'DINOv2: Learning Robust Visual Features',
        href: 'https://arxiv.org/abs/2304.07193',
        kind: 'paper',
      },
      {
        id: 'm11-r5',
        label: 'Hugging Face — Computer Vision Course',
        href: 'https://huggingface.co/learn/computer-vision-course',
        kind: 'course',
      },
    ],
  },
  {
    id: 'm12',
    track: 'module',
    phase: 'Month 6-7',
    code: 'Module 12',
    topic: 'Visual Language Models: connecting vision and text',
    subTopics: [
      'The three-component VLM architecture',
      'Visual encoder, aligner/projector, LLM backbone',
      'How visual tokens enter the LLM embedding space',
      'Training strategies for vision-language alignment',
    ],
    outcome: 'Build multimodal applications on open VLMs.',
    handsOn: 'Run document QA over screenshots with an open VLM.',
    project: 'Document intelligence system',
    priority: 'Advanced',
    hours: 14,
    resources: [
      {
        id: 'm12-r1',
        label: 'Visual Instruction Tuning (LLaVA)',
        href: 'https://arxiv.org/abs/2304.08485',
        kind: 'paper',
      },
      {
        id: 'm12-r2',
        label: 'Flamingo: a VLM for Few-Shot Learning',
        href: 'https://arxiv.org/abs/2204.14198',
        kind: 'paper',
      },
      {
        id: 'm12-r3',
        label: 'Vision Language Models Explained (HF blog)',
        href: 'https://huggingface.co/blog/vlms',
        kind: 'article',
      },
      {
        id: 'm12-r4',
        label: 'Transformers — vision-language model docs',
        href: 'https://huggingface.co/docs/transformers/tasks/idefics',
        kind: 'docs',
      },
    ],
  },
  {
    id: 'm13',
    track: 'module',
    phase: 'Month 6-7',
    code: 'Module 13',
    topic: 'Speech-to-text models & fine-tuning Whisper',
    subTopics: [
      'Speech AI and STT foundations',
      'Whisper architecture and API',
      'Building STT pipelines',
      'Dataset preparation and fine-tuning Whisper on custom audio',
    ],
    outcome: 'Transcribe domain audio accurately enough to build on.',
    handsOn: 'Fine-tune Whisper on a domain accent/vocabulary set.',
    project: 'Multimodal Agent',
    priority: 'Advanced',
    hours: 14,
    resources: [
      {
        id: 'm13-r1',
        label: 'Robust Speech Recognition via Weak Supervision (Whisper)',
        href: 'https://arxiv.org/abs/2212.04356',
        kind: 'paper',
      },
      {
        id: 'm13-r2',
        label: 'Hugging Face — Audio Course',
        href: 'https://huggingface.co/learn/audio-course',
        kind: 'course',
      },
      {
        id: 'm13-r3',
        label: 'Fine-tune Whisper for multilingual ASR (HF blog)',
        href: 'https://huggingface.co/blog/fine-tune-whisper',
        kind: 'article',
        note: 'Follow it verbatim once, then swap in your own data.',
      },
    ],
  },
  {
    id: 'm14',
    track: 'module',
    phase: 'Month 6-7',
    code: 'Module 14',
    topic: 'Embedding models: taxonomy, Matryoshka & fine-tuning',
    subTopics: [
      'Dense, sparse, quantized, binary and multi-vector embeddings',
      'Matryoshka Representation Learning — flexible dims at query time',
      'Using MRL embeddings in production',
      'Embedding fine-tuning strategies',
    ],
    outcome: 'Trade retrieval quality against cost deliberately.',
    handsOn: 'Fine-tune an embedding model on domain pairs; measure nDCG.',
    project: 'Enterprise RAG',
    priority: 'High',
    hours: 14,
    resources: [
      {
        id: 'm14-r1',
        label: 'Matryoshka Representation Learning',
        href: 'https://arxiv.org/abs/2205.13147',
        kind: 'paper',
      },
      {
        id: 'm14-r2',
        label: 'Binary and scalar embedding quantization (HF blog)',
        href: 'https://huggingface.co/blog/embedding-quantization',
        kind: 'article',
      },
      {
        id: 'm14-r3',
        label: 'MTEB leaderboard',
        href: 'https://huggingface.co/spaces/mteb/leaderboard',
        kind: 'tool',
        note: 'Pick embedding models here, not from blog posts.',
      },
      {
        id: 'm14-r4',
        label: 'Sentence Transformers — training and fine-tuning',
        href: 'https://sbert.net/docs/sentence_transformer/training_overview.html',
        kind: 'docs',
      },
    ],
  },

  /* ------------------------------------------------------------------ *
   * Capstones
   * ------------------------------------------------------------------ */
  {
    id: 'p01',
    track: 'project',
    phase: 'Capstones',
    code: 'Project 01',
    topic: 'MedScript AI — domain-specific fine-tuning for healthcare',
    subTopics: [
      'Synthetic instruction data with distilabel',
      'Stage 1: QLoRA SFT on Llama-3.1-8B-Instruct',
      'Stage 2: DPO preference alignment',
      'ROUGE-L, BERTScore and LLM-as-judge evaluation',
      'Multi-adapter vLLM server with per-request hot-swapping',
      'FastAPI + Docker + SageMaker endpoint',
    ],
    outcome: 'A domain LLM portfolio piece with a live endpoint.',
    handsOn: 'Ship base vs SFT vs SFT+DPO with an evaluation report.',
    project: 'Domain LLM portfolio',
    priority: 'Must learn',
    hours: 40,
    resources: [
      {
        id: 'p01-r1',
        label: 'TRL — SFTTrainer',
        href: 'https://huggingface.co/docs/trl/sft_trainer',
        kind: 'docs',
      },
      {
        id: 'p01-r2',
        label: 'TRL — DPOTrainer',
        href: 'https://huggingface.co/docs/trl/dpo_trainer',
        kind: 'docs',
      },
      {
        id: 'p01-r3',
        label: 'Clinical Q&A datasets on the Hub',
        href: 'https://huggingface.co/datasets?search=chatdoctor',
        kind: 'dataset',
      },
      {
        id: 'p01-r4',
        label: 'PubMedQA on the Hub',
        href: 'https://huggingface.co/datasets?search=pubmedqa',
        kind: 'dataset',
      },
      {
        id: 'p01-r5',
        label: 'vLLM — LoRA adapters',
        href: 'https://docs.vllm.ai/en/latest/features/lora.html',
        kind: 'docs',
      },
      {
        id: 'p01-r6',
        label: 'Amazon SageMaker — deploy models for inference',
        href: 'https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html',
        kind: 'docs',
      },
    ],
  },
  {
    id: 'p02',
    track: 'project',
    phase: 'Capstones',
    code: 'Project 02',
    topic: 'EdgeReason — distillation to a CPU-servable small model',
    subTopics: [
      'Teacher logit extraction and soft labels at temperature',
      'KL divergence + attention transfer implemented from scratch',
      'Ablation study across loss configurations',
      'GGUF conversion and multiple quantization levels',
      'llama-server: OpenAI-compatible CPU inference',
      'Tokens/sec and first-token latency benchmarks',
    ],
    outcome: 'Close most of the quality gap at a fraction of serving cost.',
    handsOn: 'Distil on GSM8K, quantize, and benchmark against the teacher.',
    project: 'Edge inference portfolio',
    priority: 'High',
    hours: 34,
    resources: [
      {
        id: 'p02-r1',
        label: 'GSM8K dataset',
        href: 'https://huggingface.co/datasets/openai/gsm8k',
        kind: 'dataset',
      },
      {
        id: 'p02-r2',
        label: 'torch.nn.KLDivLoss',
        href: 'https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html',
        kind: 'docs',
        note: 'Watch the log-probs argument order — it is the classic bug here.',
      },
      {
        id: 'p02-r3',
        label: 'GGUF quantization on the Hub',
        href: 'https://huggingface.co/docs/hub/gguf',
        kind: 'docs',
      },
      {
        id: 'p02-r4',
        label: 'Weights & Biases — experiment tracking',
        href: 'https://docs.wandb.ai/guides/track/',
        kind: 'tool',
      },
    ],
  },
  {
    id: 'p03',
    track: 'project',
    phase: 'Capstones',
    code: 'Project 03',
    topic: 'LexisGraph — enterprise legal RAG with ColPali, Neo4j & hybrid retrieval',
    subTopics: [
      'ColPali multi-vector indexing with no OCR',
      'Qdrant MaxSim late-interaction scoring',
      'Elasticsearch BM25 + Neo4j entity graph',
      'Reciprocal Rank Fusion and cross-encoder reranking',
      'Adaptive query router across retrievers',
      'Presidio PII masking and guardrails',
      'RAGAS faithfulness gate on golden QA pairs',
    ],
    outcome: 'Production retrieval skill on messy, scanned, real documents.',
    handsOn: 'Index CUAD contracts; route, fuse, rerank and gate on RAGAS.',
    project: 'Enterprise RAG portfolio',
    priority: 'Must learn',
    hours: 44,
    resources: [
      {
        id: 'p03-r1',
        label: 'CUAD — Contract Understanding Atticus Dataset',
        href: 'https://www.atticusprojectai.org/cuad',
        kind: 'dataset',
      },
      {
        id: 'p03-r2',
        label: 'Qdrant — multivector / late interaction',
        href: 'https://qdrant.tech/documentation/concepts/vectors/',
        kind: 'docs',
      },
      {
        id: 'p03-r3',
        label: 'Elasticsearch — BM25 and relevance tuning',
        href: 'https://www.elastic.co/docs/solutions/search/full-text',
        kind: 'docs',
      },
      {
        id: 'p03-r4',
        label: 'Neo4j Cypher manual',
        href: 'https://neo4j.com/docs/cypher-manual/current/',
        kind: 'docs',
      },
      {
        id: 'p03-r5',
        label: 'Reciprocal Rank Fusion (original paper)',
        href: 'https://dl.acm.org/doi/10.1145/1571941.1572114',
        kind: 'paper',
      },
      {
        id: 'p03-r6',
        label: 'RAGAS — evaluating a RAG pipeline',
        href: 'https://docs.ragas.io/en/stable/getstarted/rag_eval/',
        kind: 'docs',
      },
    ],
  },
  {
    id: 'p04',
    track: 'project',
    phase: 'Capstones',
    code: 'Project 04',
    topic: 'AutoOps — DevOps multi-agent system with HITL & A2A',
    subTopics: [
      'LangGraph supervisor-worker graph with conditional routing',
      'FastMCP servers per agent over stdio and SSE',
      'A2A peer delegation and discovery',
      'HITL interrupts with SQLite checkpoint resume',
      'MicroVM isolation, gateway, memory and identity',
      'Policy-based access control and cross-agent observability',
    ],
    outcome: 'The agentic AI portfolio piece — and the closest to SDET work.',
    handsOn: 'Triage an incident → issue → review → approval, end to end.',
    project: 'Autonomous QA Engineer Agent',
    priority: 'Must learn',
    hours: 44,
    resources: [
      {
        id: 'p04-r1',
        label: 'LangGraph — human-in-the-loop',
        href: 'https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/',
        kind: 'docs',
      },
      {
        id: 'p04-r2',
        label: 'LangGraph — multi-agent systems',
        href: 'https://langchain-ai.github.io/langgraph/concepts/multi_agent/',
        kind: 'docs',
      },
      {
        id: 'p04-r3',
        label: 'FastMCP — server and client patterns',
        href: 'https://gofastmcp.com/getting-started/welcome',
        kind: 'tool',
      },
      {
        id: 'p04-r4',
        label: 'A2A — agent cards and task lifecycle',
        href: 'https://a2a-protocol.org/latest/specification/',
        kind: 'docs',
      },
      {
        id: 'p04-r5',
        label: 'Playwright MCP-driven browser automation',
        href: 'https://playwright.dev/docs/intro',
        kind: 'docs',
        note: 'The hook into your SDET differentiation.',
      },
    ],
  },
  {
    id: 'p05',
    track: 'project',
    phase: 'Capstones',
    code: 'Project 05',
    topic: 'ShipLLM — LLMOps CI/CD with eval-gated deployment',
    subTopics: [
      'Stage gates: lint/unit → prompt regression → eval score → cost/latency',
      'PR blocked on eval failure with score delta as a comment',
      'Container build, rolling deploy, staging smoke tests',
      'Blue/green with gradual traffic shifting and auto-rollback',
      'Golden test cases committed as the regression baseline',
      'LLM metrics dashboard and tracing',
    ],
    outcome: 'The LLMOps skill that makes the other four projects shippable.',
    handsOn: 'Make a prompt change trip the gate, then roll it back.',
    project: 'LLM Evaluation Platform',
    priority: 'Must learn',
    hours: 40,
    resources: [
      {
        id: 'p05-r1',
        label: 'PromptFoo — CI/CD integration',
        href: 'https://www.promptfoo.dev/docs/integrations/ci-cd/',
        kind: 'docs',
      },
      {
        id: 'p05-r2',
        label: 'Inspect AI — scorers and eval logs',
        href: 'https://inspect.aisi.org.uk/scorers.html',
        kind: 'docs',
      },
      {
        id: 'p05-r3',
        label: 'Docker — multi-stage builds',
        href: 'https://docs.docker.com/build/building/multi-stage/',
        kind: 'docs',
      },
      {
        id: 'p05-r4',
        label: 'AWS — blue/green deployments',
        href: 'https://docs.aws.amazon.com/whitepapers/latest/overview-deployment-options/bluegreen-deployments.html',
        kind: 'article',
      },
      {
        id: 'p05-r5',
        label: 'Langfuse — LLM observability',
        href: 'https://langfuse.com/docs',
        kind: 'tool',
      },
      {
        id: 'p05-r6',
        label: 'AI Engineering — Chip Huyen',
        href: 'https://huyenchip.com/books/',
        kind: 'book',
        note: 'The best book-length treatment of evaluation and LLMOps.',
      },
    ],
  },
];

export const TOTAL_RESOURCES = UNITS.reduce(
  (sum, unit) => sum + unit.resources.length,
  0,
);

export const TOTAL_HOURS = UNITS.reduce((sum, unit) => sum + unit.hours, 0);

/* ------------------------------------------------------------------ *
 * Build-along projects
 *
 * One or more hands-on walkthroughs per unit — a video you code along with,
 * or an article/notebook you work through — kept separate from the reference
 * shelf above so the "read about it" and "build it" halves stay distinct.
 * A handful are YouTube search links rather than a fixed video, because the
 * good walkthrough for that topic changes every few months; those are noted.
 * ------------------------------------------------------------------ */

export const BUILD_PROJECTS: Record<string, Resource[]> = {
  f01: [
    {
      id: 'f01-b1',
      label: 'Build micrograd — a tiny autograd engine, from scratch',
      href: 'https://www.youtube.com/watch?v=VMj-3S1tku0',
      kind: 'youtube',
      note: 'Backprop stops being magic after this one. ~2.5 hrs, code along.',
    },
    {
      id: 'f01-b2',
      label: 'Train an image classifier in PyTorch (CIFAR-10)',
      href: 'https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html',
      kind: 'article',
      note: 'Your first complete train/eval loop.',
    },
  ],
  m01: [
    {
      id: 'm01-b1',
      label: 'Build a GPT from scratch, line by line (Karpathy)',
      href: 'https://www.youtube.com/watch?v=kCc8FmEb1nY',
      kind: 'youtube',
      note: 'The canonical build. Do not just watch it — type it.',
    },
    {
      id: 'm01-b2',
      label: 'Build a tokenizer, block by block (HF course)',
      href: 'https://huggingface.co/learn/llm-course/chapter6/8',
      kind: 'article',
      note: 'Normalizer → pre-tokenizer → model → decoder, assembled by hand.',
    },
  ],
  m02: [
    {
      id: 'm02-b1',
      label: 'Fine-tune a model with the Trainer API (HF course)',
      href: 'https://huggingface.co/learn/llm-course/chapter3/3',
      kind: 'article',
      note: 'The smallest complete fine-tune you can run today.',
    },
    {
      id: 'm02-b2',
      label: 'Fine-tune Llama with QLoRA using the TRL SFT Trainer',
      href: 'https://www.youtube.com/watch?v=XOlyOWE4YaM',
      kind: 'youtube',
    },
  ],
  m03: [
    {
      id: 'm03-b1',
      label: 'vLLM quickstart — serve a model and benchmark it',
      href: 'https://docs.vllm.ai/en/latest/getting_started/quickstart.html',
      kind: 'article',
      note: 'Then re-run with and without prefix caching and compare.',
    },
    {
      id: 'm03-b2',
      label: 'Optimizing LLMs for speed and memory (HF guide)',
      href: 'https://huggingface.co/docs/transformers/llm_tutorial_optimization',
      kind: 'article',
      note: 'Quantization, Flash Attention and KV-cache variants, measured.',
    },
  ],
  m04: [
    {
      id: 'm04-b1',
      label: 'End-to-end LLM fine-tuning: LoRA, QLoRA and full fine-tune',
      href: 'https://www.youtube.com/watch?v=jrf5vyOEMr8',
      kind: 'youtube',
      note: 'Good map of where CPT, SFT and alignment sit relative to each other.',
    },
  ],
  m05: [
    {
      id: 'm05-b1',
      label: 'Open-Source AI Cookbook — synthetic data recipes',
      href: 'https://huggingface.co/learn/cookbook/index',
      kind: 'article',
      note: 'Runnable notebooks for generation, judging and filtering.',
    },
    {
      id: 'm05-b2',
      label: 'Build a domain SFT dataset with distilabel pipelines',
      href: 'https://distilabel.argilla.io/latest/sections/pipeline_samples/',
      kind: 'article',
    },
  ],
  m06: [
    {
      id: 'm06-b1',
      label: 'Fine-tune Llama 3.2 with QLoRA — step by step',
      href: 'https://www.youtube.com/watch?v=xBgSivyCwi8',
      kind: 'youtube',
    },
    {
      id: 'm06-b2',
      label: 'A fistful of dollars: fine-tune LLaMA 2 7B with QLoRA',
      href: 'https://www.youtube.com/watch?v=5L4s9mi9eUc',
      kind: 'youtube',
      note: 'Strong on the cost/VRAM arithmetic, not just the code.',
    },
    {
      id: 'm06-b3',
      label: 'TRL — run your first DPO training',
      href: 'https://huggingface.co/docs/trl/dpo_trainer',
      kind: 'article',
    },
  ],
  m07: [
    {
      id: 'm07-b1',
      label: 'Fine-tune any LLM with LLaMA-Factory (WebUI + CLI, LoRA + QLoRA)',
      href: 'https://www.youtube.com/watch?v=RL38OsL5ycY',
      kind: 'youtube',
    },
    {
      id: 'm07-b2',
      label: 'Quantize a model and push GGUF to the Hub',
      href: 'https://huggingface.co/docs/hub/gguf',
      kind: 'article',
    },
  ],
  m08: [
    {
      id: 'm08-b1',
      label: 'Implement a Mixture-of-Experts layer from scratch',
      href: 'https://www.youtube.com/results?search_query=mixture+of+experts+from+scratch+pytorch+implementation',
      kind: 'youtube',
      note: 'Search link — this topic dates fast; pick the most recent walkthrough.',
    },
  ],
  m09: [
    {
      id: 'm09-b1',
      label: 'Train a reasoning model with GRPO',
      href: 'https://www.youtube.com/results?search_query=train+reasoning+model+GRPO+tutorial+unsloth',
      kind: 'youtube',
      note: 'Search link — GRPO tooling changed twice in 2025; take the newest.',
    },
    {
      id: 'm09-b2',
      label: 'Reasoning recipes in the Open-Source AI Cookbook',
      href: 'https://huggingface.co/learn/cookbook/index',
      kind: 'article',
    },
  ],
  m10: [
    {
      id: 'm10-b1',
      label: 'Crash course: knowledge distillation, built end to end',
      href: 'https://www.youtube.com/watch?v=p-Q2lAx3YAk',
      kind: 'youtube',
      note: 'Vision models, but the loss plumbing is identical for LLMs.',
    },
    {
      id: 'm10-b2',
      label: 'Distil-Whisper: distillation via large-scale pseudo labelling',
      href: 'https://arxiv.org/abs/2311.00430',
      kind: 'paper',
      note: 'A real distillation recipe you can copy the structure of.',
    },
  ],
  m15: [
    {
      id: 'm15-b1',
      label: 'Complete RAG crash course with LangChain (2 hrs)',
      href: 'https://www.youtube.com/watch?v=o126p1QN_RI',
      kind: 'youtube',
    },
    {
      id: 'm15-b2',
      label: 'LangChain — build a RAG application (official tutorial)',
      href: 'https://python.langchain.com/docs/tutorials/rag/',
      kind: 'article',
    },
  ],
  m16: [
    {
      id: 'm16-b1',
      label: 'RAG From Scratch — LangChain video series',
      href: 'https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x',
      kind: 'youtube',
      note: 'Indexing → retrieval → generation → query translation → routing.',
    },
    {
      id: 'm16-b2',
      label: 'Learn RAG from scratch — full tutorial by a LangChain engineer',
      href: 'https://www.youtube.com/watch?v=sVcwVQRHIc8',
      kind: 'youtube',
    },
  ],
  m17: [
    {
      id: 'm17-b1',
      label: 'Build Self-RAG / Corrective RAG graphs (LangGraph tutorials)',
      href: 'https://langchain-ai.github.io/langgraph/examples/',
      kind: 'article',
      note: 'The adaptive-RAG notebooks are the module in runnable form.',
    },
    {
      id: 'm17-b2',
      label: 'RAG with LangChain — a complete build tutorial',
      href: 'https://www.youtube.com/watch?v=YLPNA1j7kmQ',
      kind: 'youtube',
    },
  ],
  m18: [
    {
      id: 'm18-b1',
      label: 'Build multimodal RAG with ColPali',
      href: 'https://www.youtube.com/results?search_query=ColPali+multimodal+RAG+tutorial+qdrant',
      kind: 'youtube',
      note: 'Search link — ColPali tooling is young; take the newest walkthrough.',
    },
    {
      id: 'm18-b2',
      label: 'Neo4j — build a GraphRAG pipeline in Python',
      href: 'https://neo4j.com/docs/neo4j-graphrag-python/current/',
      kind: 'article',
      note: 'Entity extraction → graph build → retriever, with working code.',
    },
  ],
  m19: [
    {
      id: 'm19-b1',
      label: 'MCP quickstart — build your first MCP server',
      href: 'https://modelcontextprotocol.io/quickstart/server',
      kind: 'article',
      note: 'Then point a real client at it and watch the handshake.',
    },
    {
      id: 'm19-b2',
      label: 'Build an MCP client and server wired into a LangGraph workflow',
      href: 'https://www.youtube.com/watch?v=pHTYLcWFp6w',
      kind: 'youtube',
    },
  ],
  m20: [
    {
      id: 'm20-b1',
      label: 'End-to-end multi-agent system: LangGraph + MCP + supervisor + HITL',
      href: 'https://www.youtube.com/watch?v=BM39OouLNsM',
      kind: 'youtube',
      note: 'The closest single video to capstone Project 04.',
    },
    {
      id: 'm20-b2',
      label: 'Full multi-agent app with MCP, LangGraph, a database and FastAPI',
      href: 'https://www.youtube.com/watch?v=LZAGlCqmhZQ',
      kind: 'youtube',
    },
  ],
  m21: [
    {
      id: 'm21-b1',
      label: 'Multi-agent system with MCP on LangGraph (supervisor architecture)',
      href: 'https://www.youtube.com/watch?v=OnG5E9WHbbs',
      kind: 'youtube',
    },
    {
      id: 'm21-b2',
      label: 'Deploy an agent to Bedrock AgentCore Runtime',
      href: 'https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-getting-started.html',
      kind: 'article',
    },
  ],
  m22: [
    {
      id: 'm22-b1',
      label: 'PromptFoo — getting started with prompt testing',
      href: 'https://www.promptfoo.dev/docs/getting-started/',
      kind: 'article',
      note: 'Turn "the prompt feels better" into a number in about 20 minutes.',
    },
    {
      id: 'm22-b2',
      label: 'Claude prompt library — patterns to reverse-engineer',
      href: 'https://docs.claude.com/en/resources/prompt-library/library',
      kind: 'article',
    },
  ],
  m23: [
    {
      id: 'm23-b1',
      label: 'LangGraph — persistence and checkpointing, built up step by step',
      href: 'https://langchain-ai.github.io/langgraph/concepts/persistence/',
      kind: 'article',
    },
    {
      id: 'm23-b2',
      label: 'Add long-term memory to an agent (LangGraph memory guide)',
      href: 'https://langchain-ai.github.io/langgraph/concepts/memory/',
      kind: 'article',
    },
  ],
  m24: [
    {
      id: 'm24-b1',
      label: 'Inspect AI tutorial — write your first Task, Solver and Scorer',
      href: 'https://inspect.aisi.org.uk/tutorial.html',
      kind: 'article',
    },
    {
      id: 'm24-b2',
      label: 'PromptFoo in CI — gate a pull request on eval scores',
      href: 'https://www.promptfoo.dev/docs/integrations/ci-cd/',
      kind: 'article',
      note: 'This is Project 05 in miniature — do it early.',
    },
  ],
  m11: [
    {
      id: 'm11-b1',
      label: 'Vision Transformers and knowledge distillation (lecture + build)',
      href: 'https://www.youtube.com/watch?v=J_q-PEYikEo',
      kind: 'youtube',
    },
    {
      id: 'm11-b2',
      label: 'HF Computer Vision Course — ViT hands-on chapters',
      href: 'https://huggingface.co/learn/computer-vision-course',
      kind: 'article',
    },
  ],
  m12: [
    {
      id: 'm12-b1',
      label: 'Fine-tune a vision-language model (Cookbook notebooks)',
      href: 'https://huggingface.co/learn/cookbook/index',
      kind: 'article',
      note: 'Filter the index for "VLM" and "multimodal".',
    },
    {
      id: 'm12-b2',
      label: 'Build a document-QA app on an open VLM',
      href: 'https://www.youtube.com/results?search_query=fine+tune+vision+language+model+tutorial+qwen+vl',
      kind: 'youtube',
      note: 'Search link — open VLMs turn over fast; take the newest.',
    },
  ],
  m13: [
    {
      id: 'm13-b1',
      label: 'Fine-tune Whisper for speech transcription',
      href: 'https://www.youtube.com/watch?v=anplUNnkM68',
      kind: 'youtube',
    },
    {
      id: 'm13-b2',
      label: 'Whisper data preparation and fine-tuning with Unsloth',
      href: 'https://www.youtube.com/watch?v=OfQNgPfv97s',
      kind: 'youtube',
    },
  ],
  m14: [
    {
      id: 'm14-b1',
      label: 'Train and fine-tune sentence transformers (HF blog)',
      href: 'https://huggingface.co/blog/how-to-train-sentence-transformers',
      kind: 'article',
    },
    {
      id: 'm14-b2',
      label: 'Sentence Transformers — training overview with runnable configs',
      href: 'https://sbert.net/docs/sentence_transformer/training_overview.html',
      kind: 'article',
    },
  ],
  p01: [
    {
      id: 'p01-b1',
      label: 'Fine-tune a medical LLM: dataset → SFT → DPO',
      href: 'https://www.youtube.com/results?search_query=fine+tune+medical+LLM+QLoRA+DPO+project',
      kind: 'youtube',
      note: 'Search link — use it for shape, then follow your own dataset.',
    },
    {
      id: 'p01-b2',
      label: 'Serve many LoRA adapters from one base model (vLLM)',
      href: 'https://docs.vllm.ai/en/latest/features/lora.html',
      kind: 'article',
    },
  ],
  p02: [
    {
      id: 'p02-b1',
      label: 'Knowledge distillation, implemented from scratch',
      href: 'https://www.youtube.com/watch?v=p-Q2lAx3YAk',
      kind: 'youtube',
    },
    {
      id: 'p02-b2',
      label: 'Convert to GGUF and serve on CPU',
      href: 'https://huggingface.co/docs/hub/gguf',
      kind: 'article',
    },
  ],
  p03: [
    {
      id: 'p03-b1',
      label: 'Build a hybrid retriever with RRF and a cross-encoder',
      href: 'https://qdrant.tech/documentation/concepts/hybrid-queries/',
      kind: 'article',
    },
    {
      id: 'p03-b2',
      label: 'Evaluate the pipeline with RAGAS end to end',
      href: 'https://docs.ragas.io/en/stable/getstarted/rag_eval/',
      kind: 'article',
    },
  ],
  p04: [
    {
      id: 'p04-b1',
      label: 'End-to-end multi-agent system with supervisor, guardrails and HITL',
      href: 'https://www.youtube.com/watch?v=BM39OouLNsM',
      kind: 'youtube',
      note: 'Follow it, then swap the tools for your own MCP servers.',
    },
    {
      id: 'p04-b2',
      label: 'LangGraph — human-in-the-loop interrupts, built step by step',
      href: 'https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/',
      kind: 'article',
    },
  ],
  p05: [
    {
      id: 'p05-b1',
      label: 'Gate a pull request on eval scores with PromptFoo in CI',
      href: 'https://www.promptfoo.dev/docs/integrations/ci-cd/',
      kind: 'article',
    },
    {
      id: 'p05-b2',
      label: 'Multi-stage container builds for every service',
      href: 'https://docs.docker.com/build/building/multi-stage/',
      kind: 'article',
    },
  ],
};

export function buildsFor(unitId: string): Resource[] {
  return BUILD_PROJECTS[unitId] ?? [];
}

export const TOTAL_BUILDS = Object.values(BUILD_PROJECTS).reduce(
  (sum, list) => sum + list.length,
  0,
);

/** Everything with a tick box: reference resources + build-along projects. */
export const TOTAL_LINKS = TOTAL_RESOURCES + TOTAL_BUILDS;
