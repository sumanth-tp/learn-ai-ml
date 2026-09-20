import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Heading from '@theme/Heading';
import Layout from '@theme/Layout';
import type {ComponentType, ReactNode} from 'react';

import {useDocsIndex, useSectionCounts} from '@site/src/lib/docsIndex';
import {
  ArrowIcon,
  BoltIcon,
  BoxIcon,
  BrainIcon,
  ChartIcon,
  ChatIcon,
  CodeIcon,
  LayersIcon,
  SigmaIcon,
  SparkIcon,
} from '@site/src/components/Icons';
import ContinueLearning from '@site/src/components/ContinueLearning';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import styles from './index.module.css';

type Track = {
  title: string;
  blurb: string;
  to: string;
  meta: string;
  Icon: ComponentType<{className?: string}>;
  tone: 'indigo' | 'teal' | 'amber' | 'rose' | 'sky' | 'violet';
};

const TRACKS: Track[] = [
  {
    title: 'Research Papers',
    blurb: 'Read the original papers, work through the ideas and equations, and run the complete teaching implementations.',
    to: '/docs/research-papers',
    meta: '14 papers',
    Icon: LayersIcon,
    tone: 'indigo',
  },
  {
    title: 'Deep Learning',
    blurb:
      'Perceptrons through backprop, optimizers, CNNs, RNNs, attention and transformers — derived, then coded.',
    to: '/docs/category/dnn',
    meta: '99 notes',
    Icon: BrainIcon,
    tone: 'indigo',
  },
  {
    title: 'Statistics & Probability',
    blurb:
      'Distributions, hypothesis testing, power, p-hacking and FDR — with the plots that make them click.',
    to: '/docs/category/statistics',
    meta: '50 notes',
    Icon: SigmaIcon,
    tone: 'teal',
  },
  {
    title: 'Code Tracks',
    blurb:
      'NumPy, pandas, Matplotlib, Seaborn, Plotly, PyTorch and FastAPI — runnable notebooks, not slideware.',
    to: '/docs/category/coding',
    meta: '9 tracks',
    Icon: CodeIcon,
    tone: 'violet',
  },
  {
    title: 'Cheatsheets',
    blurb:
      'One-page recalls for PyTorch, scikit-learn, SQL, Docker, K8s, LangChain, vector DBs, W&B and more.',
    to: '/docs/category/cheetsheet',
    meta: '26 sheets',
    Icon: BoltIcon,
    tone: 'amber',
  },
  {
    title: 'Interview Prep',
    blurb:
      'Question banks across ML, DL, statistics, SQL, Python, DSA and system design — with worked answers.',
    to: '/docs/category/interview',
    meta: '11 sets',
    Icon: ChatIcon,
    tone: 'rose',
  },
  {
    title: 'Engineering & Systems',
    blurb:
      'Scaler-track notes on programming constructs, DSA, system design and shipping real projects.',
    to: '/docs/category/scaler',
    meta: '48 notes',
    Icon: BoxIcon,
    tone: 'sky',
  },
];

function useStats(): {value: string; label: string}[] {
  const docs = useDocsIndex();
  const counts = useSectionCounts();
  const n = (key: string) => counts.get(key) ?? 0;
  return [
    {value: `${docs.length}`, label: 'written notes'},
    {value: `${n('dnn') + n('drl') + n('nlp')}`, label: 'deep learning, RL & NLP'},
    {value: `${n('cheatsheets')}`, label: 'master cheatsheets'},
    {value: `${n('interviews')}`, label: 'interview question banks'},
  ];
}

const PATH: {step: string; title: string; body: string; to: string}[] = [
  {
    step: '01',
    title: 'Foundations',
    body: 'Linear algebra, probability, calculus for ML and the Python stack you will actually type every day.',
    to: '/docs/category/statistics',
  },
  {
    step: '02',
    title: 'Core ML',
    body: 'Regression to ensembles with scikit-learn: features, validation, metrics and the failure modes behind each.',
    to: '/docs/category/cheetsheet',
  },
  {
    step: '03',
    title: 'Deep Learning',
    body: 'Backpropagation by hand, then CNNs, sequence models, attention and transformers in PyTorch.',
    to: '/docs/category/dnn',
  },
  {
    step: '04',
    title: 'Ship It',
    body: 'FastAPI services, Docker, experiment tracking and the system-design questions that follow in interviews.',
    to: '/docs/category/coding',
  },
];

const CODE_SAMPLE = `class TinyMLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)

loss = F.cross_entropy(model(xb), yb)
loss.backward()          # ∂L/∂w for every weight
opt.step(); opt.zero_grad()`;

function Hero() {
  const {siteConfig} = useDocusaurusContext();
  const stats = useStats();

  return (
    <header className={styles.hero}>
      <div className={styles.heroGlow} aria-hidden="true" />
      <div className={styles.heroGrid} aria-hidden="true" />

      <div className={`container ${styles.heroInner}`}>
        <div className={styles.heroCopy}>
          <span className={styles.badge}>
            <SparkIcon className={styles.badgeIcon} />
            Zero to production-ready
          </span>

          <Heading as="h1" className={styles.heroTitle}>
            Learn AI &amp; ML the way it{' '}
            <span className={styles.gradientText}>actually works</span>
          </Heading>

          <p className={styles.heroSubtitle}>
            {siteConfig.tagline}. An open, ever-growing notebook of
            mathematics, code and engineering practice — from the first
            derivative to a served model.
          </p>

          <div className={styles.heroActions}>
            <Link className={styles.primaryCta} to="/docs/intro">
              Start the roadmap
              <ArrowIcon className={styles.ctaIcon} />
            </Link>
            <Link className={styles.secondaryCta} to="/docs/category/cheetsheet">
              Browse cheatsheets
            </Link>
          </div>

          <dl className={styles.stats}>
            {stats.map((stat) => (
              <div key={stat.label} className={styles.stat}>
                <dt className={styles.statValue}>{stat.value}</dt>
                <dd className={styles.statLabel}>{stat.label}</dd>
              </div>
            ))}
          </dl>
        </div>

        <div className={styles.heroVisual} aria-hidden="true">
          <div className={styles.codeCard}>
            <div className={styles.codeChrome}>
              <span className={styles.dot} data-dot="red" />
              <span className={styles.dot} data-dot="amber" />
              <span className={styles.dot} data-dot="green" />
              <span className={styles.codeFile}>train.py</span>
            </div>
            <pre className={styles.codeBody}>
              <code>{CODE_SAMPLE}</code>
            </pre>
          </div>
          <div className={styles.floatCard} data-float="one">
            <ChartIcon className={styles.floatIcon} />
            <div>
              <strong>val_acc</strong>
              <span>0.9412 ▲</span>
            </div>
          </div>
          <div className={styles.floatCard} data-float="two">
            <LayersIcon className={styles.floatIcon} />
            <div>
              <strong>epoch 12/20</strong>
              <span>loss 0.211</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

function Tracks() {
  const counts = useSectionCounts();
  const liveMeta: Record<string, string> = {
    'Deep Learning': `${counts.get('dnn') ?? 0} notes`,
    'Statistics & Probability': `${counts.get('stats') ?? 0} notes`,
    Cheatsheets: `${counts.get('cheatsheets') ?? 0} sheets`,
    'Interview Prep': `${counts.get('interviews') ?? 0} sets`,
    'Engineering & Systems': `${counts.get('scaler') ?? 0} notes`,
  };
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.sectionHead}>
          <Heading as="h2" className={styles.sectionTitle}>
            Pick a track
          </Heading>
          <p className={styles.sectionSubtitle}>
            Six deep collections, each written to be read end to end — or raided
            the night before an interview.
          </p>
        </div>

        <div className={styles.trackGrid}>
          {TRACKS.map(({title, blurb, to, meta, Icon, tone}) => (
            <Link key={title} to={to} className={styles.track} data-tone={tone}>
              <span className={styles.trackIcon}>
                <Icon className={styles.trackIconGlyph} />
              </span>
              <div className={styles.trackBody}>
                <h3 className={styles.trackTitle}>{title}</h3>
                <p className={styles.trackBlurb}>{blurb}</p>
              </div>
              <div className={styles.trackFoot}>
                <span className={styles.trackMeta}>{liveMeta[title] ?? meta}</span>
                <ArrowIcon className={styles.trackArrow} />
              </div>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

function LearningPath() {
  return (
    <section className={`${styles.section} ${styles.sectionAlt}`}>
      <div className="container">
        <div className={styles.sectionHead}>
          <Heading as="h2" className={styles.sectionTitle}>
            A path, not a pile
          </Heading>
          <p className={styles.sectionSubtitle}>
            Four stages, in order. Each one assumes only what the stage before it
            taught you.
          </p>
        </div>

        <ol className={styles.path}>
          {PATH.map(({step, title, body, to}) => (
            <li key={step} className={styles.pathItem}>
              <Link to={to} className={styles.pathCard}>
                <span className={styles.pathStep}>{step}</span>
                <h3 className={styles.pathTitle}>{title}</h3>
                <p className={styles.pathBody}>{body}</p>
              </Link>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

function ClosingCta() {
  return (
    <section className={styles.ctaSection}>
      <div className="container">
        <div className={styles.ctaCard}>
          <div>
            <Heading as="h2" className={styles.ctaTitle}>
              Start where you are
            </Heading>
            <p className={styles.ctaText}>
              New to the field? Take the roadmap. Prepping for interviews? Jump
              straight to the question banks.
            </p>
          </div>
          <div className={styles.ctaActions}>
            <Link className={styles.primaryCta} to="/docs/intro">
              Open the roadmap
              <ArrowIcon className={styles.ctaIcon} />
            </Link>
            <Link className={styles.secondaryCta} to="/docs/category/interview">
              Interview prep
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();

  return (
    <Layout
      title={siteConfig.title}
      description="Structured, code-first notes on Python, statistics, machine learning, deep learning, PyTorch, FastAPI and MLOps — from fundamentals to production.">
      <Hero />
      <main>
        <Tracks />
        <ContinueLearning />
        <HomepageFeatures />
        <LearningPath />
        <ClosingCta />
      </main>
    </Layout>
  );
}
