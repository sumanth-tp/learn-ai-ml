import {useEffect, useMemo, useState} from 'react';

import styles from './CodeWalkthrough.module.css';

type Paper =
  | 'transformer' | 'bert' | 'gpt1' | 'gpt2' | 'gpt3' | 'rag' | 'lora'
  | 'instructgpt' | 'react' | 'clip' | 'llama' | 'deepseek-r1' | 'resnet' | 'ddpm';

type Step = {title: string; start: number; end: number; explanation: string};
type Guide = {file: string; steps: Step[]};

const GUIDES: Record<Paper, Guide> = {
  transformer: {file: 'attention.py', steps: [
    {title: '1 · Attention', start: 1, end: 35, explanation: 'Start with imports and multi-head attention. Follow the reshape into heads, scaled QKᵀ scores, causal mask, softmax, value mixture and output projection.'},
    {title: '2 · Encoder/decoder', start: 36, end: 85, explanation: 'Add sinusoidal positions, then compare encoder self-attention with decoder causal self-attention and encoder–decoder cross-attention.'},
    {title: '3 · Training batch', start: 86, end: 92, explanation: 'Construct source and shifted target sequences. The start token enters the decoder; every label is the next target token.'},
    {title: '4 · Train & generate', start: 93, end: 111, explanation: 'Train with teacher forcing, then switch to autoregressive generation and verify exact sequence reversal on held-out examples.'},
  ]},
  bert: {file: 'bert.py', steps: [
    {title: '1 · Encoder blocks', start: 1, end: 50, explanation: 'Build bidirectional self-attention and the residual feed-forward block. There is no triangular causal mask.'},
    {title: '2 · MLM/NSP data', start: 51, end: 74, explanation: 'Create sentence pairs, select MLM positions and apply the 80/10/10 corruption rule while retaining the original token targets.'},
    {title: '3 · BERT heads', start: 75, end: 94, explanation: 'Combine token, position and segment embeddings, then attach separate masked-token and next-sentence heads.'},
    {title: '4 · Pretrain/fine-tune', start: 95, end: 116, explanation: 'Pre-train both objectives, replace the output task with a classifier and fine-tune the encoder on labelled examples.'},
  ]},
  gpt1: {file: 'gpt1.py', steps: [
    {title: '1 · Causal stack', start: 1, end: 49, explanation: 'Build masked self-attention and the Transformer block. The mask ensures a position never reads its future label.'},
    {title: '2 · Language model', start: 50, end: 82, explanation: 'Map token IDs through embeddings and causal blocks to next-token logits, with a small task head available for transfer.'},
    {title: '3 · LM objective', start: 83, end: 98, explanation: 'Shift ordinary text by one position and pre-train the shared backbone with cross-entropy.'},
    {title: '4 · Fine-tune', start: 99, end: 112, explanation: 'Attach a classifier, update it together with the pretrained backbone and compare task predictions with labels.'},
  ]},
  gpt2: {file: 'gpt2.py', steps: [
    {title: '1 · Decoder', start: 1, end: 49, explanation: 'Implement the reusable causal Transformer components that make every prefix a next-token prediction problem.'},
    {title: '2 · LM + sampling', start: 50, end: 82, explanation: 'The model includes both forward logits and autoregressive generation, where temperature and top-k alter decoding rather than training.'},
    {title: '3 · Text objective', start: 83, end: 93, explanation: 'Turn character sequences into shifted inputs and labels and initialize the finite-context language model.'},
    {title: '4 · Train & prompt', start: 94, end: 105, explanation: 'Train on task-shaped text, then supply a prefix and sample its continuation without a task-specific gradient update.'},
  ]},
  gpt3: {file: 'gpt3.py', steps: [
    {title: '1 · Decoder', start: 1, end: 49, explanation: 'Use the same causal architecture; in-context learning changes the sequence presented to the model, not the layer type.'},
    {title: '2 · LM interface', start: 50, end: 86, explanation: 'Produce next-token distributions and define the shifted-token loss shared by demonstrations and the query.'},
    {title: '3 · Changing tasks', start: 87, end: 98, explanation: 'Generate a different key-to-value mapping in every row so the answer must be inferred from demonstrations in that row.'},
    {title: '4 · Train & test', start: 99, end: 121, explanation: 'Train across many temporary mappings, then measure whether the final query is answered from its in-context examples.'},
  ]},
  rag: {file: 'rag.py', steps: [
    {title: '1 · Generator', start: 1, end: 86, explanation: 'Build the small causal generator used after evidence is selected. This is parametric memory.'},
    {title: '2 · Corpus/index', start: 87, end: 95, explanation: 'Encode a fixed external corpus. The document rows form the non-parametric memory searched by the retriever.'},
    {title: '3 · Joint RAG model', start: 96, end: 132, explanation: 'Score documents, combine retrieval and generation log-probabilities, and locate where sequence- and token-level marginalization differ.'},
    {title: '4 · Train & inspect', start: 133, end: 158, explanation: 'Optimize the joint likelihood, inspect retrieved documents and verify that the answer changes with relevant memory.'},
  ]},
  lora: {file: 'lora.py', steps: [
    {title: '1 · Low-rank layer', start: 1, end: 30, explanation: 'Freeze the base matrix and define trainable A and B factors. Their product is the only learned weight update.'},
    {title: '2 · Controlled task', start: 31, end: 39, explanation: 'Create a known rank-two target shift so the experiment can test exactly what the adapter should recover.'},
    {title: '3 · Optimize adapter', start: 40, end: 44, explanation: 'Train only A and B and verify that the frozen base receives no gradient update.'},
    {title: '4 · Save & merge', start: 45, end: 59, explanation: 'Store the adapter separately, reload it with the matching base, merge BA into the weight and check output equivalence.'},
  ]},
  instructgpt: {file: 'instructgpt.py', steps: [
    {title: '1 · Policy/reward', start: 1, end: 32, explanation: 'Define two different models: a policy that chooses responses and a reward model that scores prompt–response pairs.'},
    {title: '2 · SFT', start: 33, end: 38, explanation: 'Teach an initial policy from demonstrations. This provides the policy and the frozen reference used later.'},
    {title: '3 · Preferences', start: 39, end: 49, explanation: 'Fit the reward model from chosen/rejected pairs with a pairwise ranking loss.'},
    {title: '4 · PPO', start: 50, end: 83, explanation: 'Roll out an old policy, calculate advantages and clipped ratios, apply reference KL control, then compare final choices.'},
  ]},
  react: {file: 'react.py', steps: [
    {title: '1 · Tools/parsing', start: 1, end: 34, explanation: 'Define the action grammar and parsing boundary. Only parsed, allowlisted actions should reach an environment.'},
    {title: '2 · Environment', start: 35, end: 46, explanation: 'Keep authoritative tool state outside the model and return observations as data.'},
    {title: '3 · Policies', start: 47, end: 63, explanation: 'Separate the model integration point from a deterministic fixture so the runner can be tested without an API.'},
    {title: '4 · Agent loop', start: 64, end: 89, explanation: 'Alternate policy output, validated action and real observation under a hard step limit, then persist the trace.'},
  ]},
  clip: {file: 'clip.py', steps: [
    {title: '1 · Paired data', start: 1, end: 19, explanation: 'Generate paired image features and text labels. Each row supplies one positive pair and batch negatives.'},
    {title: '2 · Dual encoders', start: 20, end: 29, explanation: 'Project both modalities, normalize their vectors and form a temperature-scaled all-pairs similarity matrix.'},
    {title: '3 · Contrastive loss', start: 30, end: 45, explanation: 'Train in both image-to-text and text-to-image directions so the diagonal pair wins in rows and columns.'},
    {title: '4 · Classify', start: 46, end: 52, explanation: 'Encode class text once and classify held-out images by their nearest text vector without fitting a new output head.'},
  ]},
  llama: {file: 'llama.py', steps: [
    {title: '1 · RMSNorm/RoPE', start: 1, end: 24, explanation: 'Implement magnitude normalization and rotate query/key coordinate pairs according to position.'},
    {title: '2 · Decoder layer', start: 25, end: 43, explanation: 'Combine pre-normalized causal attention with the gated SwiGLU feed-forward transformation.'},
    {title: '3 · Full model', start: 44, end: 58, explanation: 'Stack decoder layers, tie the token interface to logits and implement autoregressive generation.'},
    {title: '4 · Train & sample', start: 59, end: 72, explanation: 'Train the complete narrow model on shifted sequences and verify generation rather than isolated component outputs.'},
  ]},
  'deepseek-r1': {file: 'deepseek_r1.py', steps: [
    {title: '1 · Policy', start: 1, end: 28, explanation: 'Define a tiny response policy with a frozen-reference copy. Each response still consists of token decisions.'},
    {title: '2 · Rewards/SFT', start: 29, end: 47, explanation: 'Calculate token log-probabilities, verify final answers and provide a cold-start supervised training function.'},
    {title: '3 · GRPO', start: 48, end: 72, explanation: 'Sample response groups, normalize rewards within each prompt and optimize clipped ratios with a reference penalty.'},
    {title: '4 · Distil', start: 73, end: 86, explanation: 'Filter verified teacher generations and train a smaller student with ordinary supervised targets.'},
  ]},
  resnet: {file: 'resnet.py', steps: [
    {title: '1 · Basic block', start: 1, end: 18, explanation: 'Follow the two-convolution residual branch and the identity or projected shortcut before addition.'},
    {title: '2 · Bottleneck', start: 19, end: 28, explanation: 'Reduce, process and expand channels with 1×1, 3×3 and 1×1 convolutions while preserving the shortcut contract.'},
    {title: '3 · Network/data', start: 29, end: 45, explanation: 'Assemble stages and create a spatial classification task that requires convolutional features.'},
    {title: '4 · Train & evaluate', start: 46, end: 54, explanation: 'Optimize on training batches, switch BatchNorm to evaluation mode and check held-out predictions.'},
  ]},
  ddpm: {file: 'ddpm.py', steps: [
    {title: '1 · Schedule/noising', start: 1, end: 26, explanation: 'Create beta, alpha and cumulative alpha-bar values, then jump directly from a clean image to any sampled timestep.'},
    {title: '2 · Time-conditioned U-Net', start: 27, end: 51, explanation: 'Inject the timestep into residual convolutional blocks so one network can denoise every noise level.'},
    {title: '3 · Toy images/train', start: 52, end: 69, explanation: 'Generate simple clean images, add known Gaussian noise and train the network to predict that noise.'},
    {title: '4 · Reverse sample', start: 70, end: 82, explanation: 'Start from Gaussian noise, apply every reverse update and omit fresh noise at the final step before saving images.'},
  ]},
};

export default function CodeWalkthrough({paper}: {paper: Paper}) {
  const guide = GUIDES[paper];
  const [source, setSource] = useState('');
  const [active, setActive] = useState(0);

  useEffect(() => {
    let live = true;
    fetch(`/examples/research-papers/${guide.file}`)
      .then((response) => response.text())
      .then((text) => { if (live) setSource(text); })
      .catch(() => { if (live) setSource('# Open the complete script using the link below.'); });
    return () => { live = false; };
  }, [guide.file]);

  const step = guide.steps[active];
  const code = useMemo(() => {
    if (!source) return '# Loading code walkthrough…';
    return source.split('\n').slice(step.start - 1, step.end).join('\n');
  }, [source, step]);

  return (
    <section className={styles.wrap} aria-label="Progressive code walkthrough">
      <header className={styles.head}>
        <p className={styles.title}>Build the implementation in stages</p>
        <p className={styles.subtitle}>Read one responsibility at a time; the complete runnable file remains linked below.</p>
      </header>
      <nav className={styles.steps} aria-label="Code stages">
        {guide.steps.map((item, index) => (
          <button key={item.title} type="button"
                  className={`${styles.step} ${index === active ? styles.stepActive : ''}`}
                  onClick={() => setActive(index)} aria-pressed={index === active}>
            {item.title}
          </button>
        ))}
      </nav>
      <p className={styles.explanation}>{step.explanation}</p>
      <pre className={styles.code}><code>{code}</code></pre>
      <footer className={styles.footer}>
        <span>Lines {step.start}–{step.end}</span>
        <a href={`/examples/research-papers/${guide.file}`}>Open complete runnable script</a>
      </footer>
    </section>
  );
}

