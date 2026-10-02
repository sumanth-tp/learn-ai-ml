export type PathLink = {
  label: string;
  docId?: string;
  to?: string;
  note?: string;
};

export type Stage = {
  id: string;
  number: number | null;
  title: string;
  sidebarLabel: string;
  kicker: string;
  summary: string;
  units: string[];
  pairWith?: PathLink[];
  milestone?: PathLink;
  outcomes: string[];
  optional?: boolean;
  tone: 'indigo' | 'teal' | 'violet' | 'sky' | 'amber' | 'rose' | 'slate';
};

export const UNIT_LABELS: Record<string, string> = {
  intro: 'AI/ML learning roadmap',
  'theory/statistics': 'Statistics',
  code: 'Coding',
  scaler: 'Scaler engineering notes',
  'theory/ml': 'Machine learning',
  'theory/dnn': 'Deep learning (DNN)',
  'theory/cv': 'Computer vision',
  'theory/nlp': 'Natural Language Processing',
  'theory/ir': 'Information retrieval',
  'research-papers': 'Research papers',
  genai: 'Generative AI with LangChain',
  'llm-engineering': 'LLM engineering',
  'mlops/distributed': 'Distributed machine learning',
  'agentic-ai': 'Agentic AI using LangGraph',
  mcp: 'Model Context Protocol',
  'agentic-frontier': 'Agent frontier',
  'llm-evals': 'LLM Evaluation',
  'theory/seml': 'Software Engineering for ML',
  'mlops/data': 'Data management for ML',
  'mlops/platform': 'ML platform operations',
  governance: 'Safety and governance',
  projects: 'AI engineering projects',
  senior: 'Senior engineering craft',
  interviews: 'Interview preparation',
  'theory/drl': 'Deep Reinforcement Learning',
  'theory/timeseries': 'Time series',
  'theory/recsys': 'Recommender systems',
  'theory/causal': 'Causal inference',
  'theory/gnn': 'Graph machine learning',
  'theory/speech': 'Speech and audio',
  'theory/udl': 'Unsupervised deep learning',
  'theory/va': 'Video analysis',
  daily: 'Daily notes',
  cheetsheet: 'Cheatsheets',
  'document-collection': 'Miscellaneous collection',
};

export const STAGES: Stage[] = [
  {
    id: 'start',
    number: null,
    title: 'Start here',
    sidebarLabel: 'Start here',
    kicker: 'Orientation',
    summary:
      'The one-year roadmap, and the interactive path that maps it onto the notes on this site.',
    units: ['intro'],
    outcomes: ['I know which stage I am in and what the next milestone is.'],
    tone: 'indigo',
  },
  {
    id: 'foundations',
    number: 1,
    title: 'Foundations',
    sidebarLabel: 'Foundations',
    kicker: 'Maths, Python and engineering basics',
    summary:
      'Statistics and probability, the Python data stack, and the programming, DSA and system-design fundamentals everything later leans on.',
    units: ['theory/statistics', 'code', 'scaler'],
    milestone: {
      label: 'Python capstone: order-flow service',
      docId: 'code/python/capstone/py-capstone',
      note: 'A tested, packaged Python service built end to end.',
    },
    outcomes: [
      'I can describe a dataset statistically and test a hypothesis about it.',
      'I can clean, reshape and plot data with NumPy, pandas and seaborn.',
      'I can write, test and package a small Python service.',
    ],
    tone: 'teal',
  },
  {
    id: 'machine-learning',
    number: 2,
    title: 'Machine learning',
    sidebarLabel: 'Machine learning',
    kicker: 'Classical models, evaluation and judgement',
    summary:
      'Regression, trees, nearest neighbours, support vector machines, Bayesian classifiers, ensembles and clustering, with the evaluation, calibration and explanation habits that decide whether a model can be trusted.',
    units: ['theory/ml'],
    pairWith: [
      {
        label: 'scikit-learn cheatsheet',
        docId: 'cheetsheet/scikit-learn-master-cheatsheet',
        note: 'Keep it open while you work through the chapters.',
      },
    ],
    milestone: {
      label: 'Capstone: an end-to-end tabular pipeline',
      docId: 'theory/ml/evaluation-and-practice/ml-capstone',
      note: 'Split, baseline, tune, calibrate, explain and document one model.',
    },
    outcomes: [
      'I can pick between linear models, trees, SVMs and boosting, and say why.',
      'I can evaluate a model honestly: leakage-free splits, the right metric, a calibrated threshold.',
      'I can explain a prediction to a non-specialist.',
    ],
    tone: 'sky',
  },
  {
    id: 'deep-learning',
    number: 3,
    title: 'Deep learning and vision',
    sidebarLabel: 'Deep learning & vision',
    kicker: 'From perceptrons to transformers, and from pixels to detections',
    summary:
      'Neural networks from first principles: backpropagation, optimisation, CNNs, RNNs, attention and the transformer, then classical and modern computer vision.',
    units: ['theory/dnn', 'theory/cv'],
    pairWith: [
      {
        label: 'PyTorch, from tensors to training loops',
        to: '/docs/category/pytorch',
        note: 'Lives under Coding. Work through it alongside the theory.',
      },
    ],
    milestone: {
      label: 'Cat vs dog image classifier',
      docId: 'theory/dnn/cat-vs-dog-project',
      note: 'Train, regularise and evaluate a CNN on real images.',
    },
    outcomes: [
      'I can explain backpropagation and why a network is or is not learning.',
      'I can train a CNN and a sequence model in PyTorch.',
      'I can explain self-attention and the transformer block.',
    ],
    tone: 'indigo',
  },
  {
    id: 'language',
    number: 4,
    title: 'Language and retrieval',
    sidebarLabel: 'Language & retrieval',
    kicker: 'NLP, search and the papers behind modern LLMs',
    summary:
      'Vector semantics, language models and transformers, how search and retrieval work from inverted indexes to neural ranking, then the landmark papers from the Transformer and BERT to LoRA, ReAct and DeepSeek-R1.',
    units: ['theory/nlp', 'theory/ir', 'research-papers'],
    milestone: {
      label: 'NLP capstone: document assistant',
      docId: 'theory/nlp/capstone/nlp-capstone',
      note: 'Retrieval plus generation over your own documents.',
    },
    outcomes: [
      'I can explain how embeddings, tokenisation and language models fit together.',
      'I can explain why search uses BM25, dense vectors and reranking together.',
      'I can read a landmark ML paper and summarise its method and results.',
    ],
    tone: 'sky',
  },
  {
    id: 'llm-apps',
    number: 5,
    title: 'Building with LLMs',
    sidebarLabel: 'Building with LLMs',
    kicker: 'LangChain, RAG, tools and agents',
    summary:
      'Models, prompts, structured output, chains and runnables, then retrieval-augmented generation, tool calling and LangChain v1 agents.',
    units: ['genai'],
    pairWith: [
      {
        label: 'Production LLM Engineering track',
        to: '/llm-roadmap',
        note: 'A tick-box curriculum for this stage onwards.',
      },
    ],
    milestone: {
      label: 'GenAI capstone',
      docId: 'genai/capstone',
      note: 'A complete LangChain application with retrieval and tools.',
    },
    outcomes: [
      'I can build a RAG pipeline and explain each of its components.',
      'I can get structured output from a model and call tools from it.',
    ],
    tone: 'violet',
  },
  {
    id: 'llm-engineering',
    number: 6,
    title: 'LLM engineering',
    sidebarLabel: 'LLM engineering',
    kicker: 'Adapt, serve and scale models',
    summary:
      'Fine-tune and distil models, choose between tuning, retrieval and prompting, serve them with the right engine and quantisation, size GPUs and cost, and train at scale.',
    units: ['llm-engineering', 'mlops/distributed'],
    milestone: {
      label: 'A real LoRA fine-tune on a small model',
      docId: 'llm-engineering/adapting-models/llme-sft-lora',
      note: 'Prepare data, train with TRL and PEFT, merge and measure before and after.',
    },
    outcomes: [
      'I can decide between prompting, retrieval and fine-tuning with evidence.',
      'I can serve a model and explain its latency, throughput and cost.',
      'I can explain how a model too big for one GPU is trained.',
    ],
    tone: 'violet',
  },
  {
    id: 'agents',
    number: 7,
    title: 'Agents and tools',
    sidebarLabel: 'Agents & MCP',
    kicker: 'LangGraph, the Model Context Protocol and the agent frontier',
    summary:
      'Stateful workflows, memory, human-in-the-loop and multi-agent systems in LangGraph, exposing and consuming tools through MCP, then context engineering and agent interoperability.',
    units: ['agentic-ai', 'mcp', 'agentic-frontier'],
    milestone: {
      label: 'Customer-support agent',
      docId: 'agentic-ai/agentic-ai-project-1-customer-support-agent',
      note: 'A LangGraph agent with tools, memory and a human in the loop.',
    },
    outcomes: [
      'I can model an agent as a LangGraph state machine with persistence.',
      'I can build an MCP server and connect it to an agent host.',
    ],
    tone: 'rose',
  },
  {
    id: 'production',
    number: 8,
    title: 'Evaluate, secure and ship',
    sidebarLabel: 'Evaluate, secure & ship',
    kicker: 'Evals, data, operations and governance',
    summary:
      'Offline and online evaluation, data pipelines and observability, the software engineering that turns a model into a dependable system, and the safety and governance that keep it trustworthy.',
    units: ['llm-evals', 'theory/seml', 'mlops/data', 'mlops/platform', 'governance'],
    pairWith: [
      {
        label: 'FastAPI in depth',
        to: '/docs/category/fast-api-in-depth',
        note: 'Lives under Coding. Serving models and LLM gateways behind an API.',
      },
    ],
    milestone: {
      label: 'RAG evaluation with a CI gate',
      docId: 'llm-evals/llm-evals-project-1-rag-evaluation-and-ci-gate',
      note: 'Block a merge when answer quality regresses.',
    },
    outcomes: [
      'I can design an evaluation set and gate releases on it.',
      'I can take a model from notebook to a monitored service.',
      'I can name the data, drift and governance risks of a deployed model.',
    ],
    tone: 'amber',
  },
  {
    id: 'projects',
    number: 9,
    title: 'Real systems and senior craft',
    sidebarLabel: 'Real systems & craft',
    kicker: 'End-to-end projects, system design and technical leadership',
    summary:
      'Enterprise RAG, a secure clinical assistant, AI security with guardrails and AgentOps, a complete ten-hour agentic course, then the design, estimation and leadership skills that make an engineer senior.',
    units: ['projects', 'senior'],
    milestone: {
      label: 'Enterprise RAG, session 1',
      docId: 'projects/enterprise-rag/enterprise-rag-session-1',
      note: 'Start with the project that pulls stages 5 to 8 together.',
    },
    outcomes: [
      'I can build, secure and deploy a production-style LLM system.',
      'I can explain the design decisions behind it in an interview.',
    ],
    tone: 'teal',
  },
  {
    id: 'interviews',
    number: 10,
    title: 'Get interview-ready',
    sidebarLabel: 'Interview-ready',
    kicker: '400 practical questions, labs and mocks',
    summary:
      'Programming, statistics and modelling, AI systems, evaluation and production, then coding labs and full mock interviews.',
    units: ['interviews'],
    milestone: {
      label: 'Mock interviews',
      docId: 'interviews/mock-interviews',
      note: 'Rehearse full loops before the real thing.',
    },
    outcomes: ['I can answer applied ML and LLM-systems questions under time pressure.'],
    tone: 'rose',
  },
  {
    id: 'elective-rl',
    number: null,
    title: 'Elective: reinforcement learning',
    sidebarLabel: 'Elective · RL',
    kicker: 'Optional lane, any time after stage 3',
    summary:
      'The agent-environment loop through DQN, PPO and GRPO to deploying a policy. GRPO connects straight back to how reasoning LLMs are trained.',
    units: ['theory/drl'],
    milestone: {
      label: 'Dynamic-pricing agent',
      docId: 'theory/drl/capstone/drl-capstone',
      note: 'Train and evaluate an RL agent on a business problem.',
    },
    outcomes: ['I can frame a problem as an MDP and train a policy for it.'],
    optional: true,
    tone: 'violet',
  },
  {
    id: 'elective-specialised',
    number: null,
    title: 'Elective: specialised ML',
    sidebarLabel: 'Elective · Specialised',
    kicker: 'Optional lane, chosen by the problem in front of you',
    summary:
      'Time series, recommender systems, causal inference, graph learning, speech, unsupervised deep learning and video.',
    units: [
      'theory/timeseries',
      'theory/recsys',
      'theory/causal',
      'theory/gnn',
      'theory/speech',
      'theory/udl',
      'theory/va',
    ],
    outcomes: [],
    optional: true,
    tone: 'teal',
  },
  {
    id: 'reference',
    number: null,
    title: 'Reference shelf',
    sidebarLabel: 'Reference shelf',
    kicker: 'Keep open while you work',
    summary:
      'Daily study notes, library cheatsheets and the miscellaneous collection. Dip in from any stage.',
    units: ['daily', 'cheetsheet', 'document-collection'],
    outcomes: [],
    optional: true,
    tone: 'slate',
  },
];

export const LEFTOVER_STAGE_ID = 'more';

export const MAIN_STAGES = STAGES.filter((stage) => stage.number !== null);

export function stageById(id: string | undefined): Stage | undefined {
  return STAGES.find((stage) => stage.id === id);
}

export const PATH_ROUTE = '/path';
