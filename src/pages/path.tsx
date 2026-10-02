import Link from '@docusaurus/Link';
import useBrokenLinks from '@docusaurus/useBrokenLinks';
import Heading from '@theme/Heading';
import Layout from '@theme/Layout';
import {useMemo, type CSSProperties, type ReactNode} from 'react';

import {MAIN_STAGES} from '@site/src/data/learningPath';
import {outcomeKey, setOutcome, useOutcomes} from '@site/src/lib/pathProgress';
import {
  currentStageOf,
  pct,
  useResumeDoc,
  useStageProgress,
  type ResolvedLink,
  type StageProgress,
  type StageStatus,
} from '@site/src/lib/stageProgress';

import styles from './path.module.css';

const STATUS_LABEL: Record<StageStatus, string> = {
  done: 'Complete',
  active: 'In progress',
  todo: 'Not started',
};

function Bar({value, label}: {value: number; label: string}) {
  return (
    <div
      className={styles.track}
      role="progressbar"
      aria-valuenow={value}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label}>
      <div className={styles.fill} style={{width: `${value}%`}} />
    </div>
  );
}

function PathLinkRow({link, kind}: {link: ResolvedLink; kind: 'milestone' | 'pair'}) {
  const body = (
    <>
      <span className={styles.linkKind}>{kind === 'milestone' ? 'Milestone' : 'Pair with'}</span>
      <span className={styles.linkLabel}>
        {link.label}
        {link.read && <span className={styles.readTag}>read</span>}
      </span>
      {link.note && <span className={styles.linkNote}>{link.note}</span>}
    </>
  );
  return link.href ? (
    <Link to={link.href} className={styles.pathLink} data-kind={kind}>
      {body}
    </Link>
  ) : (
    <div className={styles.pathLink} data-kind={kind}>
      {body}
    </div>
  );
}

function Outcomes({item}: {item: StageProgress}) {
  const ticked = useOutcomes();
  if (item.stage.outcomes.length === 0) {
    return null;
  }
  return (
    <fieldset className={styles.outcomes}>
      <legend className={styles.outcomesTitle}>Check yourself</legend>
      {item.stage.outcomes.map((outcome, index) => {
        const key = outcomeKey(item.stage.id, index);
        return (
          <label key={key} className={styles.outcome}>
            <input
              type="checkbox"
              checked={Boolean(ticked[key])}
              onChange={(event) => setOutcome(key, event.target.checked)}
            />
            <span>{outcome}</span>
          </label>
        );
      })}
    </fieldset>
  );
}

function StageCard({item, current}: {item: StageProgress; current: boolean}) {
  const {stage} = item;
  useBrokenLinks().collectAnchor(stage.id);
  return (
    <li
      id={stage.id}
      className={styles.stage}
      data-tone={stage.tone}
      data-status={item.status}
      data-current={current || undefined}
      data-optional={stage.optional || undefined}>
      <div className={styles.marker} aria-hidden="true">
        {item.status === 'done' ? '✓' : (stage.number ?? '·')}
      </div>

      <article className={styles.card}>
        <header className={styles.cardHead}>
          <div>
            <p className={styles.kicker}>
              {stage.number !== null ? `Stage ${stage.number} · ` : ''}
              {stage.kicker}
            </p>
            <Heading as="h2" className={styles.stageTitle}>
              {stage.title}
            </Heading>
          </div>
          <div className={styles.badges}>
            {current && <span className={styles.hereBadge}>You are here</span>}
            <span className={styles.statusBadge} data-status={item.comingSoon ? 'soon' : item.status}>
              {item.comingSoon ? 'Coming soon' : STATUS_LABEL[item.status]}
            </span>
          </div>
        </header>

        <p className={styles.summary}>{stage.summary}</p>

        {item.comingSoon ? (
          <p className={styles.soon}>In preparation. Its chapters appear here as they are published.</p>
        ) : (
          <div className={styles.stageProgress}>
            <Bar value={item.pct} label={`${stage.title} progress`} />
            <span className={styles.count}>
              {item.read}/{item.total} notes · {item.pct}%
            </span>
          </div>
        )}

        <ul className={styles.units}>
          {item.units.map((unit) => (
            <li key={unit.key}>
              {unit.href ? (
                <Link to={unit.href} className={styles.unit}>
                  <span className={styles.unitLabel}>{unit.label}</span>
                  <span className={styles.unitCount}>
                    {unit.read}/{unit.total}
                  </span>
                  <span className={styles.unitTrack} aria-hidden="true">
                    <span style={{width: `${pct(unit.read, unit.total)}%`}} />
                  </span>
                </Link>
              ) : (
                <span className={styles.unit}>{unit.label}</span>
              )}
            </li>
          ))}
        </ul>

        {(item.pairWith.length > 0 || item.milestone) && (
          <div className={styles.links}>
            {item.pairWith.map((link) => (
              <PathLinkRow key={link.label} link={link} kind="pair" />
            ))}
            {item.milestone && <PathLinkRow link={item.milestone} kind="milestone" />}
          </div>
        )}

        <Outcomes item={item} />

        {item.nextUp && (
          <Link to={item.nextUp.permalink} className={styles.nextUp}>
            <span className={styles.nextUpLabel}>
              {item.read === 0 ? 'Start with' : 'Next up'}
            </span>
            <span className={styles.nextUpTitle}>{item.nextUp.title}</span>
            <span aria-hidden="true">→</span>
          </Link>
        )}
      </article>
    </li>
  );
}

export default function LearningPath(): ReactNode {
  const progress = useStageProgress();
  const resume = useResumeDoc(progress);
  const current = currentStageOf(progress);

  const main = progress.filter((item) => !item.stage.optional);
  const optional = progress.filter((item) => item.stage.optional);

  const totals = useMemo(() => {
    const steps = main.filter((item) => item.stage.number !== null && !item.comingSoon);
    const total = steps.reduce((sum, item) => sum + item.total, 0);
    const read = steps.reduce((sum, item) => sum + item.read, 0);
    return {
      total,
      read,
      pct: pct(read, total),
      stagesDone: steps.filter((item) => item.status === 'done').length,
      stages: steps.length,
    };
  }, [main]);

  return (
    <Layout
      title="Learning path"
      description="Every section of the site placed in learning order, with progress, milestones and a resume point.">
      <main className={styles.page}>
        <div className="container">
          <header className={styles.head}>
            <p className={styles.eyebrow}>Learning path</p>
            <Heading as="h1" className={styles.title}>
              From foundations to interview-ready
            </Heading>
            <p className={styles.subtitle}>
              Every section of the site, placed in the order to learn it. {MAIN_STAGES.length} stages,
              each ending in a milestone project. The sidebar follows the
              same order, so <kbd>Next</kbd> at the foot of any note keeps you on the path. For
              the week-by-week plan behind it, read the{' '}
              <Link to="/docs/intro">one-year roadmap</Link>.
            </p>
          </header>

          <section className={styles.overview} aria-label="Your progress">
            <div className={styles.overviewStats}>
              <div>
                <span className={styles.bigNumber}>{totals.pct}%</span>
                <span className={styles.bigLabel}>
                  {totals.read} of {totals.total} notes on the main path
                </span>
              </div>
              <div>
                <span className={styles.bigNumber}>
                  {totals.stagesDone}/{totals.stages}
                </span>
                <span className={styles.bigLabel}>stages complete</span>
              </div>
            </div>
            <Bar value={totals.pct} label="Main path progress" />
            {resume && (
              <Link to={resume.permalink} className={styles.resume}>
                <span className={styles.resumeLabel}>
                  {totals.read === 0 ? 'Begin' : 'Resume'}
                  {current?.stage.number ? ` · stage ${current.stage.number}` : ''}
                </span>
                <span className={styles.resumeTitle}>{resume.title}</span>
                <span aria-hidden="true">→</span>
              </Link>
            )}
            <p className={styles.privacy}>
              Progress comes from <em>Mark as read</em> on each note and is kept in this
              browser only.
            </p>
          </section>

          <nav
            className={styles.rail}
            aria-label="Stages"
            style={{'--steps': MAIN_STAGES.length} as CSSProperties}>
            {main
              .filter((item) => item.stage.number !== null)
              .map((item) => (
                <a
                  key={item.stage.id}
                  href={`#${item.stage.id}`}
                  className={styles.railStep}
                  data-status={item.status}
                  data-soon={item.comingSoon || undefined}
                  data-current={item === current || undefined}
                  data-tone={item.stage.tone}>
                  <span className={styles.railDot}>
                    {item.status === 'done' ? '✓' : item.stage.number}
                  </span>
                  <span className={styles.railLabel}>{item.stage.sidebarLabel}</span>
                </a>
              ))}
          </nav>

          <ol className={styles.timeline}>
            {main.map((item) => (
              <StageCard key={item.stage.id} item={item} current={item === current} />
            ))}
          </ol>

          <Heading as="h2" className={styles.sideTitle}>
            Off the main path
          </Heading>
          <ol className={styles.timeline} data-variant="optional">
            {optional.map((item) => (
              <StageCard key={item.stage.id} item={item} current={false} />
            ))}
          </ol>
        </div>
      </main>
    </Layout>
  );
}
