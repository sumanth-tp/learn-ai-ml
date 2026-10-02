import type {ReactNode} from 'react';
import {useMemo, useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const PAD = {top: 16, right: 20, bottom: 18, left: 20};
const POPULATION = 100000;

export function posterior(prior: number, sensitivity: number, falsePositiveRate: number) {
  const hit = sensitivity * prior;
  const evidence = hit + falsePositiveRate * (1 - prior);
  return evidence === 0 ? 0 : hit / evidence;
}

export function naiveBayes(
  prior: number,
  likelihoods: {spam: number; ham: number; present: boolean}[],
) {
  let spam = prior;
  let ham = 1 - prior;
  likelihoods.forEach((l) => {
    spam *= l.present ? l.spam : 1 - l.spam;
    ham *= l.present ? l.ham : 1 - l.ham;
  });
  return {spam, ham, posterior: spam + ham === 0 ? 0 : spam / (spam + ham)};
}

const COUNTS = {
  spam: {total: 26, free: 5, money: 3, meeting: 0},
  ham: {total: 26, free: 0, money: 0, meeting: 2},
  vocabulary: 32,
};
const MESSAGE: ('free' | 'money' | 'meeting')[] = ['free', 'money', 'meeting'];

export function smoothed(alpha: number) {
  const score = (label: 'spam' | 'ham') => {
    const c = COUNTS[label];
    return MESSAGE.reduce(
      (acc, w) => acc * ((c[w] + alpha) / (c.total + alpha * COUNTS.vocabulary)),
      0.5,
    );
  };
  const spam = score('spam');
  const ham = score('ham');
  return {spam, ham, posterior: spam + ham === 0 ? null : spam / (spam + ham)};
}

const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;
const sci = (v: number) => (v === 0 ? '0' : v.toExponential(3));

export default function BayesRuleLab() {
  const dark = useDarkViz();
  const [view, setView] = useState<'disease' | 'spam' | 'smoothing'>('disease');
  const [logPrevalence, setLogPrevalence] = useState(-3);
  const [sensitivity, setSensitivity] = useState(99);
  const [fpr, setFpr] = useState(5);
  const [tests, setTests] = useState(1);
  const [priorSpam, setPriorSpam] = useState(40);
  const [freeSpam, setFreeSpam] = useState(80);
  const [moneySpam, setMoneySpam] = useState(60);
  const [freeHam, setFreeHam] = useState(10);
  const [moneyHam, setMoneyHam] = useState(20);
  const [freePresent, setFreePresent] = useState(true);
  const [moneyPresent, setMoneyPresent] = useState(true);
  const [alpha, setAlpha] = useState(1);

  const colA = seriesColor(0, dark);
  const colB = seriesColor(1, dark);
  const colC = seriesColor(2, dark);
  const strong = dark ? '#dfe3e9' : '#23262b';

  const prevalence = Math.pow(10, logPrevalence);
  const sens = sensitivity / 100;
  const fp = fpr / 100;

  const stages = useMemo(() => {
    const beliefs = [prevalence];
    for (let i = 0; i < tests; i += 1) beliefs.push(posterior(beliefs[i], sens, fp));
    return beliefs;
  }, [prevalence, sens, fp, tests]);
  const before = stages[stages.length - 2];
  const after = stages[stages.length - 1];
  const sick = Math.round(POPULATION * before);
  const healthy = POPULATION - sick;
  const truePositives = Math.round(sick * sens);
  const falsePositives = Math.round(healthy * fp);
  const positives = truePositives + falsePositives;

  const spamResult = naiveBayes(priorSpam / 100, [
    {spam: freeSpam / 100, ham: freeHam / 100, present: freePresent},
    {spam: moneySpam / 100, ham: moneyHam / 100, present: moneyPresent},
  ]);
  const smooth = smoothed(alpha);

  const innerW = W - PAD.left - PAD.right;
  const barX = PAD.left;

  const controlsFor = (extra: ReactNode) => (
    <>
      <label className={s.control}>
        view
        <select className={s.select} value={view} onChange={(e) => setView(e.target.value as typeof view)}>
          <option value="disease">disease-test surprise</option>
          <option value="spam">Naive Bayes spam filter</option>
          <option value="smoothing">Laplace smoothing</option>
        </select>
      </label>
      {extra}
    </>
  );

  const diseaseControls = controlsFor(
    <>
      <label className={s.control}>
        prevalence
        <input type="range" min={-4} max={-0.3} step={0.05} value={logPrevalence}
               onChange={(e) => setLogPrevalence(Number(e.target.value))} />
        <span className={s.value}>{pct(prevalence, prevalence < 0.01 ? 2 : 1)}</span>
      </label>
      <label className={s.control}>
        sensitivity
        <input type="range" min={50} max={100} step={0.5} value={sensitivity}
               onChange={(e) => setSensitivity(Number(e.target.value))} />
        <span className={s.value}>{sensitivity.toFixed(1)}%</span>
      </label>
      <label className={s.control}>
        false-positive rate
        <input type="range" min={0} max={50} step={0.5} value={fpr}
               onChange={(e) => setFpr(Number(e.target.value))} />
        <span className={s.value}>{fpr.toFixed(1)}%</span>
      </label>
      <label className={s.control}>
        positive tests in a row
        <input type="range" min={1} max={3} step={1} value={tests}
               onChange={(e) => setTests(Number(e.target.value))} />
        <span className={s.value}>{tests}</span>
      </label>
    </>,
  );

  const slider = (label: string, value: number, set: (v: number) => void, min = 1, max = 99) => (
    <label className={s.control}>
      {label}
      <input type="range" min={min} max={max} step={1} value={value} onChange={(e) => set(Number(e.target.value))} />
      <span className={s.value}>{(value / 100).toFixed(2)}</span>
    </label>
  );

  const spamControls = controlsFor(
    <>
      {slider('P(spam)', priorSpam, setPriorSpam, 5, 95)}
      {slider('P(free | spam)', freeSpam, setFreeSpam)}
      {slider('P(money | spam)', moneySpam, setMoneySpam)}
      {slider('P(free | ham)', freeHam, setFreeHam)}
      {slider('P(money | ham)', moneyHam, setMoneyHam)}
      <label className={s.control}>
        <input type="checkbox" checked={freePresent} onChange={(e) => setFreePresent(e.target.checked)} />
        email contains "free"
      </label>
      <label className={s.control}>
        <input type="checkbox" checked={moneyPresent} onChange={(e) => setMoneyPresent(e.target.checked)} />
        email contains "money"
      </label>
    </>,
  );

  const smoothingControls = controlsFor(
    <label className={s.control}>
      alpha
      <input type="range" min={0} max={2} step={0.1} value={alpha}
             onChange={(e) => setAlpha(Number(e.target.value))} />
      <span className={s.value}>{alpha.toFixed(1)}</span>
    </label>,
  );

  const table =
    view === 'disease'
      ? {
          columns: ['quantity', 'value'],
          rows: [
            ['belief before the last test', before.toFixed(4)],
            ['people tested', POPULATION.toLocaleString('en-GB')],
            ['truly sick', sick.toLocaleString('en-GB')],
            ['sick and test positive', truePositives.toLocaleString('en-GB')],
            ['healthy and test positive', falsePositives.toLocaleString('en-GB')],
            ['P(disease | positive), last test', after.toFixed(4)],
            ...(tests > 1 ? stages.slice(1).map((b, i) => [`after positive test ${i + 1}`, b.toFixed(4)]) : []),
          ],
        }
      : view === 'spam'
        ? {
            columns: ['factor', 'spam', 'ham'],
            rows: [
              ['prior', (priorSpam / 100).toFixed(3), (1 - priorSpam / 100).toFixed(3)],
              [
                freePresent ? 'free present' : 'free absent',
                (freePresent ? freeSpam / 100 : 1 - freeSpam / 100).toFixed(3),
                (freePresent ? freeHam / 100 : 1 - freeHam / 100).toFixed(3),
              ],
              [
                moneyPresent ? 'money present' : 'money absent',
                (moneyPresent ? moneySpam / 100 : 1 - moneySpam / 100).toFixed(3),
                (moneyPresent ? moneyHam / 100 : 1 - moneyHam / 100).toFixed(3),
              ],
              ['product', spamResult.spam.toFixed(4), spamResult.ham.toFixed(4)],
              ['P(spam | words)', spamResult.posterior.toFixed(4), ''],
            ],
          }
        : {
            columns: ['word', 'spam count', 'ham count', 'P(word | spam)', 'P(word | ham)'],
            rows: MESSAGE.map((w) => [
              w,
              COUNTS.spam[w],
              COUNTS.ham[w],
              ((COUNTS.spam[w] + alpha) / (COUNTS.spam.total + alpha * COUNTS.vocabulary)).toFixed(4),
              ((COUNTS.ham[w] + alpha) / (COUNTS.ham.total + alpha * COUNTS.vocabulary)).toFixed(4),
            ]),
          };

  const hints = {
    disease:
      'Default: 0.1% prevalence, 99% sensitivity, 5% false positives. Of 100,000 people 100 are sick and 99 of them test positive, but 4,995 healthy people also test positive, so a positive result means 1.94%. Raise the prevalence or the number of positive tests in a row and watch the belief climb: two positives give 28.2%, three give 88.6% (assuming the tests err independently).',
    spam:
      'Default: prior 0.4 and both words present gives spam 0.4 × 0.8 × 0.6 = 0.192 against ham 0.6 × 0.1 × 0.2 = 0.012, so P(spam) = 94.1%. Untick "free" and the missing word counts as evidence too (1 − 0.8 for spam, 1 − 0.1 for ham).',
    smoothing:
      'The message is "free money meeting" with counts from the six spam and six ham training messages in the chapter. "meeting" never occurs in spam and "free" never in ham, so at alpha = 0 both products are exactly zero and there is no answer. At alpha = 1 the lab shows 0.8889, the same as scikit-learn.',
  };

  const scoreMax = Math.max(spamResult.spam, spamResult.ham, 1e-12);

  return (
    <VizPanel
      title={
        view === 'disease'
          ? 'The disease-test surprise'
          : view === 'spam'
            ? 'Naive Bayes: multiply the likelihoods'
            : 'Why Naive Bayes needs smoothing'
      }
      hint={hints[view]}
      legend={
        view === 'disease'
          ? [
              {label: 'sick and test positive', color: colA},
              {label: 'healthy but test positive', color: colB},
              {label: 'belief (prior, then posterior)', color: colC},
            ]
          : view === 'spam'
            ? [
                {label: 'spam score', color: colB},
                {label: 'ham score', color: colA},
                {label: 'posterior P(spam)', color: colC},
              ]
            : [
                {label: 'spam score', color: colB},
                {label: 'ham score', color: colA},
              ]
      }
      table={table}
      controls={view === 'disease' ? diseaseControls : view === 'spam' ? spamControls : smoothingControls}>
      {view === 'disease' && (
        <>
          <svg className={s.svg} viewBox={`0 0 ${W} ${130 + stages.length * 24}`} role="img"
               aria-label="Among everyone who tests positive, how many are truly sick">
            <text className={s.axisLabel} x={barX} y={PAD.top + 8}>
              Of {POPULATION.toLocaleString('en-GB')} people, the {positives.toLocaleString('en-GB')} who test positive
            </text>
            <rect x={barX} y={PAD.top + 18} width={innerW} height={34} fill={colB} opacity={0.85} rx={3} />
            <rect x={barX} y={PAD.top + 18}
                  width={positives ? Math.max(2, (truePositives / positives) * innerW) : 0}
                  height={34} fill={colA} rx={3} />
            <text className={s.dataLabel} x={barX} y={PAD.top + 68}>
              {truePositives.toLocaleString('en-GB')} truly sick
            </text>
            <text className={s.dataLabel} x={barX + innerW} y={PAD.top + 68} textAnchor="end">
              {falsePositives.toLocaleString('en-GB')} healthy
            </text>
            <text className={s.axisLabel} x={barX} y={PAD.top + 108}>belief that you are sick</text>
            {stages.map((b, i) => (
              <g key={i}>
                <rect x={barX} y={PAD.top + 118 + i * 24} width={innerW} height={14} fill="var(--border-subtle)" rx={3} />
                <rect x={barX} y={PAD.top + 118 + i * 24} width={Math.max(2, b * innerW)} height={14}
                      fill={colC} opacity={i === stages.length - 1 ? 1 : 0.5} rx={3} />
                <text className={s.dataLabel} x={barX + innerW} y={PAD.top + 130 + i * 24} textAnchor="end">
                  {i === 0 ? `prior ${pct(b, 2)}` : `after positive test ${i}: ${pct(b, 2)}`}
                </text>
              </g>
            ))}
          </svg>
          <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0, marginTop: '0.6rem'}}>
            <span>
              last test: P(disease | positive) = <strong>{pct(after, 2)}</strong> ({truePositives.toLocaleString('en-GB')} of{' '}
              {positives.toLocaleString('en-GB')} positives are sick)
            </span>
          </div>
        </>
      )}
      {view === 'spam' && (
        <>
          <svg className={s.svg} viewBox={`0 0 ${W} 190`} role="img"
               aria-label="Unnormalised spam and ham scores and the resulting posterior">
            <text className={s.axisLabel} x={barX} y={PAD.top + 8}>unnormalised scores: prior × product of word likelihoods</text>
            <rect x={barX} y={PAD.top + 18} width={Math.max(2, (spamResult.spam / scoreMax) * (innerW - 90))} height={26} fill={colB} rx={3} />
            <text className={s.dataLabel} x={barX + Math.max(2, (spamResult.spam / scoreMax) * (innerW - 90)) + 8} y={PAD.top + 35}>
              spam {spamResult.spam.toFixed(3)}
            </text>
            <rect x={barX} y={PAD.top + 52} width={Math.max(2, (spamResult.ham / scoreMax) * (innerW - 90))} height={26} fill={colA} rx={3} />
            <text className={s.dataLabel} x={barX + Math.max(2, (spamResult.ham / scoreMax) * (innerW - 90)) + 8} y={PAD.top + 69}>
              ham {spamResult.ham.toFixed(3)}
            </text>
            <text className={s.axisLabel} x={barX} y={PAD.top + 108}>normalised: P(spam | the words)</text>
            <rect x={barX} y={PAD.top + 118} width={innerW} height={18} fill="var(--border-subtle)" rx={3} />
            <rect x={barX} y={PAD.top + 118} width={Math.max(2, spamResult.posterior * innerW)} height={18} fill={colC} rx={3} />
            <text className={s.dataLabel} x={barX + innerW} y={PAD.top + 154} textAnchor="end" fill={strong}>
              {pct(spamResult.posterior, 1)}
            </text>
          </svg>
          <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0, marginTop: '0.6rem'}}>
            <span>
              P(spam | words) = {spamResult.spam.toFixed(3)} / ({spamResult.spam.toFixed(3)} + {spamResult.ham.toFixed(3)}) ={' '}
              <strong>{spamResult.posterior.toFixed(3)}</strong>
            </span>
          </div>
        </>
      )}
      {view === 'smoothing' && (
        <>
          <svg className={s.svg} viewBox={`0 0 ${W} 150`} role="img"
               aria-label="Spam and ham scores for the message free money meeting">
            <text className={s.axisLabel} x={barX} y={PAD.top + 8}>scores for the message "free money meeting"</text>
            <rect x={barX} y={PAD.top + 18} width={smooth.spam === 0 ? 2 : Math.max(2, (smooth.spam / Math.max(smooth.spam, smooth.ham)) * (innerW - 150))} height={26} fill={colB} rx={3} />
            <text className={s.dataLabel} x={barX + (smooth.spam === 0 ? 2 : Math.max(2, (smooth.spam / Math.max(smooth.spam, smooth.ham)) * (innerW - 150))) + 8} y={PAD.top + 35}>
              spam {sci(smooth.spam)}
            </text>
            <rect x={barX} y={PAD.top + 52} width={smooth.ham === 0 ? 2 : Math.max(2, (smooth.ham / Math.max(smooth.spam, smooth.ham)) * (innerW - 150))} height={26} fill={colA} rx={3} />
            <text className={s.dataLabel} x={barX + (smooth.ham === 0 ? 2 : Math.max(2, (smooth.ham / Math.max(smooth.spam, smooth.ham)) * (innerW - 150))) + 8} y={PAD.top + 69}>
              ham {sci(smooth.ham)}
            </text>
          </svg>
          <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0, marginTop: '0.6rem'}}>
            <span>
              alpha = {alpha.toFixed(1)}:{' '}
              {smooth.posterior === null ? (
                <strong>0 / 0, no answer at all</strong>
              ) : (
                <>P(spam) = <strong>{smooth.posterior.toFixed(4)}</strong></>
              )}
            </span>
          </div>
        </>
      )}
    </VizPanel>
  );
}
