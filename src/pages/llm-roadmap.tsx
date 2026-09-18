import Heading from '@theme/Heading';
import Layout from '@theme/Layout';
import {useMemo, useState, type ReactNode} from 'react';

import {
  KIND_LABEL,
  KIND_ORDER,
  PHASES,
  TOTAL_HOURS,
  TOTAL_LINKS,
  UNITS,
  buildsFor,
  type Priority,
  type Resource,
  type ResourceKind,
  type Unit,
} from '@site/src/data/llmRoadmap';
import {
  clearRoadmap,
  setLinkDone,
  setLinksDone,
  setUnitDone,
  useDoneLinks,
  useDoneUnits,
} from '@site/src/lib/roadmapProgress';

import styles from './llm-roadmap.module.css';

const PRIORITIES: Priority[] = ['Must learn', 'High', 'Advanced'];

/** Every tick-boxable link on a unit, builds first — that is the study order. */
function linksOf(unit: Unit): Resource[] {
  return [...buildsFor(unit.id), ...unit.resources];
}

function normalise(value: string) {
  return value.toLowerCase().normalize('NFKD');
}

function haystackOf(unit: Unit) {
  return normalise(
    [
      unit.code,
      unit.topic,
      unit.outcome,
      unit.handsOn,
      unit.project,
      unit.phase,
      ...unit.subTopics,
      ...linksOf(unit).map((link) => `${link.label} ${link.note ?? ''}`),
    ].join(' '),
  );
}

function pct(done: number, total: number) {
  return total ? Math.round((done / total) * 100) : 0;
}

/* -------------------------------------------------------------------------- */

function ProgressBar({value, label}: {value: number; label?: string}) {
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

function KindTag({kind}: {kind: ResourceKind}) {
  return (
    <span className={styles.kind} data-kind={kind}>
      {KIND_LABEL[kind]}
    </span>
  );
}

function LinkRow({
  link,
  done,
  badge,
}: {
  link: Resource;
  done: boolean;
  /** Which unit this link belongs to — only shown in the library listing. */
  badge?: string;
}) {
  return (
    <li className={styles.linkRow} data-done={done || undefined}>
      <label className={styles.linkCheck}>
        <input
          type="checkbox"
          checked={done}
          onChange={() => setLinkDone(link.id, !done)}
          aria-label={`Mark "${link.label}" complete`}
        />
        <span className={styles.box} aria-hidden="true" />
      </label>
      <div className={styles.linkBody}>
        <a
          className={styles.linkTitle}
          href={link.href}
          target="_blank"
          rel="noopener noreferrer">
          {link.label}
        </a>
        <KindTag kind={link.kind} />
        {link.note && <p className={styles.linkNote}>{link.note}</p>}
      </div>
      {badge && <span className={styles.libUnit}>{badge}</span>}
    </li>
  );
}

/* -------------------------------------------------------------------------- */

function UnitRows({
  unit,
  unitDone,
  doneLinks,
  expanded,
  onToggleExpand,
}: {
  unit: Unit;
  unitDone: boolean;
  doneLinks: Record<string, number>;
  expanded: boolean;
  onToggleExpand: () => void;
}) {
  const builds = buildsFor(unit.id);
  const links = linksOf(unit);
  const ids = links.map((link) => link.id);
  const linksDone = ids.filter((id) => doneLinks[id]).length;
  const allLinksDone = links.length > 0 && linksDone === links.length;

  const grouped = KIND_ORDER.map((kind) => ({
    kind,
    items: unit.resources.filter((resource) => resource.kind === kind),
  })).filter((group) => group.items.length > 0);

  return (
    <>
      <tr className={styles.row} data-done={unitDone || undefined}>
        <td className={styles.cellCheck}>
          <label className={styles.unitCheck}>
            <input
              type="checkbox"
              checked={unitDone}
              onChange={() => setUnitDone(unit.id, !unitDone)}
              aria-label={`Mark ${unit.code} complete`}
            />
            <span className={styles.box} aria-hidden="true" />
          </label>
        </td>
        <td className={styles.cellCode}>
          <span className={styles.code}>{unit.code}</span>
        </td>
        <td className={styles.cellTopic}>
          <button type="button" className={styles.topicButton} onClick={onToggleExpand}>
            <span className={styles.topicText}>{unit.topic}</span>
            <span className={styles.chevron} data-open={expanded || undefined} aria-hidden="true">
              ▸
            </span>
          </button>
          <p className={styles.outcome}>{unit.outcome}</p>
        </td>
        <td className={styles.cellHands}>{unit.handsOn}</td>
        <td className={styles.cellProject}>{unit.project}</td>
        <td className={styles.cellPriority}>
          <span className={styles.priority} data-level={unit.priority}>
            {unit.priority}
          </span>
        </td>
        <td className={styles.cellHours}>{unit.hours}h</td>
        <td className={styles.cellLinks}>
          <button
            type="button"
            className={styles.linkCount}
            onClick={onToggleExpand}
            data-complete={allLinksDone || undefined}
            aria-expanded={expanded}
            aria-label={`${linksDone} of ${links.length} resources complete — show details`}>
            {linksDone}/{links.length}
          </button>
        </td>
      </tr>

      {expanded && (
        <tr className={styles.detailRow}>
          <td colSpan={8}>
            <div className={styles.detail}>
              <div className={styles.detailGrid}>
                <section>
                  <h4 className={styles.detailHead}>Sub-topics</h4>
                  <ul className={styles.subTopics}>
                    {unit.subTopics.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </section>
                <section>
                  <h4 className={styles.detailHead}>What you should be able to do</h4>
                  <p className={styles.detailText}>{unit.outcome}</p>
                  <h4 className={styles.detailHead}>Hands-on task</h4>
                  <p className={styles.detailText}>{unit.handsOn}</p>
                  <h4 className={styles.detailHead}>Feeds project</h4>
                  <p className={styles.detailText}>{unit.project}</p>
                </section>
              </div>

              <div className={styles.detailActions}>
                <span className={styles.detailProgress}>
                  {linksDone} of {links.length} links done
                </span>
                <button
                  type="button"
                  className={styles.ghostButton}
                  onClick={() => setLinksDone(ids, !allLinksDone)}>
                  {allLinksDone ? 'Clear all links' : 'Mark all links done'}
                </button>
              </div>

              {builds.length > 0 && (
                <section className={styles.buildBlock}>
                  <h4 className={styles.groupHead}>
                    Build along
                    <span className={styles.groupHint}>
                      code these — watching is not learning
                    </span>
                  </h4>
                  <ul className={styles.linkList}>
                    {builds.map((link) => (
                      <LinkRow key={link.id} link={link} done={Boolean(doneLinks[link.id])} />
                    ))}
                  </ul>
                </section>
              )}

              <section>
                <h4 className={styles.groupHead}>Study & reference</h4>
                <div className={styles.groups}>
                  {grouped.map((group) => (
                    <div key={group.kind} className={styles.group}>
                      <h5 className={styles.groupKind}>{KIND_LABEL[group.kind]}</h5>
                      <ul className={styles.linkList}>
                        {group.items.map((link) => (
                          <LinkRow
                            key={link.id}
                            link={link}
                            done={Boolean(doneLinks[link.id])}
                          />
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </section>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

/* -------------------------------------------------------------------------- */

export default function LlmRoadmap(): ReactNode {
  const doneUnits = useDoneUnits();
  const doneLinks = useDoneLinks();

  const [query, setQuery] = useState('');
  const [phase, setPhase] = useState('all');
  const [priority, setPriority] = useState('all');
  const [hideDone, setHideDone] = useState(false);
  const [open, setOpen] = useState<Record<string, boolean>>({});

  const decorated = useMemo(
    () => UNITS.map((unit) => ({unit, haystack: haystackOf(unit)})),
    [],
  );

  const visible = useMemo(() => {
    const needle = normalise(query.trim());
    return decorated
      .filter(({unit, haystack}) => {
        if (phase !== 'all' && unit.phase !== phase) return false;
        if (priority !== 'all' && unit.priority !== priority) return false;
        if (hideDone && doneUnits[unit.id]) return false;
        return needle === '' || haystack.includes(needle);
      })
      .map(({unit}) => unit);
  }, [decorated, query, phase, priority, hideDone, doneUnits]);

  const unitsDone = UNITS.filter((unit) => doneUnits[unit.id]).length;
  const linksDone = Object.keys(doneLinks).length;
  const hoursDone = UNITS.reduce(
    (sum, unit) => sum + (doneUnits[unit.id] ? unit.hours : 0),
    0,
  );
  const overall = pct(unitsDone, UNITS.length);

  const allOpen = visible.length > 0 && visible.every((unit) => open[unit.id]);

  const toggleAll = () => {
    const next: Record<string, boolean> = {...open};
    visible.forEach((unit) => {
      next[unit.id] = !allOpen;
    });
    setOpen(next);
  };

  const library = useMemo(() => {
    const byKind = new Map<ResourceKind, {unit: Unit; link: Resource}[]>();
    UNITS.forEach((unit) => {
      linksOf(unit).forEach((link) => {
        const list = byKind.get(link.kind) ?? [];
        list.push({unit, link});
        byKind.set(link.kind, list);
      });
    });
    return KIND_ORDER.filter((kind) => byKind.has(kind)).map((kind) => ({
      kind,
      items: byKind.get(kind)!,
    }));
  }, []);

  return (
    <Layout
      title="Production LLM Engineering roadmap"
      description="A 7-month roadmap from transformers to production LLM systems — every module mapped to courses, docs, papers, YouTube build-alongs and capstone projects, with a progress tracker.">
      <main className={styles.page}>
        <div className="container">
          <header className={styles.head}>
            <p className={styles.eyebrow}>Roadmap</p>
            <Heading as="h1" className={styles.title}>
              Production LLM Engineering
            </Heading>
            <p className={styles.subtitle}>
              Seven months, at roughly eight hours a week, from attention maths to
              eval-gated deployment. Every module carries a build-along project and a
              shelf of courses, docs and papers; tick things off as you go and the
              progress is kept in this browser.
            </p>
            <ul className={styles.metaRow}>
              <li>
                <strong>{UNITS.length}</strong> units
              </li>
              <li>
                <strong>{TOTAL_LINKS}</strong> curated links
              </li>
              <li>
                <strong>~{TOTAL_HOURS}</strong> study hours
              </li>
              <li>
                <strong>5</strong> capstone projects
              </li>
            </ul>
          </header>

          <section className={styles.summary} aria-label="Your progress">
            <div className={styles.summaryTop}>
              <div className={styles.summaryStats}>
                <div className={styles.stat}>
                  <span className={styles.statValue}>{unitsDone}</span>
                  <span className={styles.statLabel}>/ {UNITS.length} units</span>
                </div>
                <div className={styles.stat}>
                  <span className={styles.statValue}>{linksDone}</span>
                  <span className={styles.statLabel}>/ {TOTAL_LINKS} links</span>
                </div>
                <div className={styles.stat}>
                  <span className={styles.statValue}>{hoursDone}</span>
                  <span className={styles.statLabel}>/ {TOTAL_HOURS} hours</span>
                </div>
              </div>
              <div className={styles.summaryActions}>
                <span className={styles.summaryPct}>{overall}%</span>
                {(unitsDone > 0 || linksDone > 0) && (
                  <button
                    type="button"
                    className={styles.resetButton}
                    onClick={() => {
                      if (window.confirm('Clear all roadmap progress in this browser?')) {
                        clearRoadmap();
                      }
                    }}>
                    Reset
                  </button>
                )}
              </div>
            </div>
            <ProgressBar value={overall} label="Overall roadmap progress" />
            <p className={styles.summaryNote}>
              Progress is stored in this browser only — nothing is uploaded. Paid-course
              links point at landing pages, so check current pricing and syllabus before
              buying.
            </p>
          </section>

          <div className={styles.controls}>
            <input
              type="search"
              className={styles.search}
              placeholder="Search topics, tools, papers, courses…"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              aria-label="Search the roadmap"
            />
            <select
              className={styles.select}
              value={phase}
              onChange={(event) => setPhase(event.target.value)}
              aria-label="Filter by phase">
              <option value="all">All phases</option>
              {PHASES.map((item) => (
                <option key={item.key} value={item.key}>
                  {item.label}
                </option>
              ))}
            </select>
            <select
              className={styles.select}
              value={priority}
              onChange={(event) => setPriority(event.target.value)}
              aria-label="Filter by priority">
              <option value="all">All priorities</option>
              {PRIORITIES.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
            <label className={styles.toggle}>
              <input
                type="checkbox"
                checked={hideDone}
                onChange={(event) => setHideDone(event.target.checked)}
              />
              Hide completed
            </label>
            <button type="button" className={styles.ghostButton} onClick={toggleAll}>
              {allOpen ? 'Collapse all' : 'Expand all'}
            </button>
          </div>

          {visible.length === 0 && (
            <p className={styles.empty}>
              Nothing matches those filters. Clear the search or switch the phase back to
              “All phases”.
            </p>
          )}

          {PHASES.map((phaseDef) => {
            const units = visible.filter((unit) => unit.phase === phaseDef.key);
            if (units.length === 0) return null;

            const phaseUnits = UNITS.filter((unit) => unit.phase === phaseDef.key);
            const phaseDone = phaseUnits.filter((unit) => doneUnits[unit.id]).length;
            const phaseHours = phaseUnits.reduce((sum, unit) => sum + unit.hours, 0);

            return (
              <section key={phaseDef.key} className={styles.phase}>
                <header className={styles.phaseHead}>
                  <div>
                    <Heading as="h2" className={styles.phaseTitle}>
                      {phaseDef.label}
                    </Heading>
                    <p className={styles.phaseBlurb}>{phaseDef.blurb}</p>
                  </div>
                  <div className={styles.phaseMeta}>
                    <span className={styles.phaseCount}>
                      {phaseDone}/{phaseUnits.length} done
                    </span>
                    <span className={styles.phaseHours}>~{phaseHours}h</span>
                  </div>
                </header>
                <ProgressBar
                  value={pct(phaseDone, phaseUnits.length)}
                  label={`${phaseDef.label} progress`}
                />

                <div className={styles.tableWrap}>
                  <table className={styles.table}>
                    <thead>
                      <tr>
                        <th scope="col" className={styles.cellCheck}>
                          <span className={styles.srOnly}>Done</span>
                        </th>
                        <th scope="col">Module</th>
                        <th scope="col">Topic</th>
                        <th scope="col" className={styles.cellHands}>
                          Hands-on
                        </th>
                        <th scope="col" className={styles.cellProject}>
                          Project
                        </th>
                        <th scope="col">Priority</th>
                        <th scope="col" className={styles.cellHours}>
                          Time
                        </th>
                        <th scope="col">Links</th>
                      </tr>
                    </thead>
                    <tbody>
                      {units.map((unit) => (
                        <UnitRows
                          key={unit.id}
                          unit={unit}
                          unitDone={Boolean(doneUnits[unit.id])}
                          doneLinks={doneLinks}
                          expanded={Boolean(open[unit.id])}
                          onToggleExpand={() =>
                            setOpen((current) => ({
                              ...current,
                              [unit.id]: !current[unit.id],
                            }))
                          }
                        />
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            );
          })}

          <section className={styles.library}>
            <Heading as="h2" className={styles.phaseTitle}>
              Browse by resource type
            </Heading>
            <p className={styles.phaseBlurb}>
              The same {TOTAL_LINKS} links, grouped by where they live — handy when you
              want “every Udemy course on this track” or “just the papers”. Tick boxes are
              shared with the tables above.
            </p>
            {library.map((group) => {
              const groupDone = group.items.filter(
                ({link}) => doneLinks[link.id],
              ).length;
              return (
                <details key={group.kind} className={styles.libGroup}>
                  <summary className={styles.libSummary}>
                    <KindTag kind={group.kind} />
                    <span className={styles.libName}>{KIND_LABEL[group.kind]}</span>
                    <span className={styles.libCount}>
                      {groupDone}/{group.items.length}
                    </span>
                  </summary>
                  <ul className={styles.linkList}>
                    {group.items.map(({unit, link}) => (
                      <LinkRow
                        key={link.id}
                        link={link}
                        done={Boolean(doneLinks[link.id])}
                        badge={unit.code}
                      />
                    ))}
                  </ul>
                </details>
              );
            })}
          </section>

          <section className={styles.howto}>
            <Heading as="h2" className={styles.phaseTitle}>
              How to work through this
            </Heading>
            <ol className={styles.howtoList}>
              <li>
                <strong>Do not read linearly.</strong> Each unit is: skim the docs, watch
                or read the build-along, build it, then go back to the papers for the
                parts that confused you.
              </li>
              <li>
                <strong>One build per unit is the minimum bar.</strong> A ticked unit with
                no code written is a lie you tell yourself in month five.
              </li>
              <li>
                <strong>Start Project 05 in month one.</strong> Eval-gated CI is easier to
                grow alongside the other projects than to bolt on at the end.
              </li>
              <li>
                <strong>Courses are scaffolding, not the goal.</strong> Pick one paid
                course per phase at most; the free docs and papers here carry the weight.
              </li>
              <li>
                <strong>Months 6–7 are optional if you are short on time.</strong>{' '}
                Multimodal is marked Advanced for a reason — the RAG, agent and LLMOps
                months are what get read on a CV.
              </li>
            </ol>
          </section>
        </div>
      </main>
    </Layout>
  );
}
