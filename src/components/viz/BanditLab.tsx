import {useCallback, useEffect, useRef, useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

type Strategy = 'greedy' | 'epsilon' | 'ucb';

const K = 8;
const W = 640;
const H = 210;
const PAD = {top: 14, right: 12, bottom: 26, left: 34};

function normal(rng: () => number, mean: number, sd: number) {
  const u = Math.max(rng(), 1e-9);
  const v = rng();
  return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type State = {
  q: number[];        // estimates
  n: number[];        // pull counts
  truth: number[];
  steps: number;
  reward: number;     // cumulative
  optimal: number;    // times the best arm was pulled
};

function freshState(seed: number): State {
  const rng = mulberry(seed);
  return {
    q: Array(K).fill(0),
    n: Array(K).fill(0),
    truth: Array.from({length: K}, () => normal(rng, 0, 1)),
    steps: 0,
    reward: 0,
    optimal: 0,
  };
}

export default function BanditLab() {
  const dark = useDarkViz();
  const [seed, setSeed] = useState(7);
  const [strategy, setStrategy] = useState<Strategy>('epsilon');
  const [epsilon, setEpsilon] = useState(0.1);
  const [state, setState] = useState<State>(() => freshState(7));
  const [running, setRunning] = useState(false);
  const rngRef = useRef(mulberry(seed * 977 + 13));

  const reset = useCallback(
    (nextSeed = seed) => {
      rngRef.current = mulberry(nextSeed * 977 + 13);
      setState(freshState(nextSeed));
      setRunning(false);
    },
    [seed],
  );

  const pull = useCallback(
    (count: number) => {
      setState((prev) => {
        const next: State = {
          ...prev,
          q: [...prev.q],
          n: [...prev.n],
        };
        const rng = rngRef.current;
        const best = next.truth.indexOf(Math.max(...next.truth));
        for (let i = 0; i < count; i += 1) {
          let arm: number;
          if (strategy === 'greedy') {
            arm = next.q.indexOf(Math.max(...next.q));
          } else if (strategy === 'epsilon') {
            arm =
              rng() < epsilon
                ? Math.floor(rng() * K)
                : next.q.indexOf(Math.max(...next.q));
          } else {
            const t = next.steps + 1;
            let bestScore = -Infinity;
            arm = 0;
            for (let a = 0; a < K; a += 1) {
              const bonus = next.n[a] === 0 ? 1e6 : 2 * Math.sqrt(Math.log(t) / next.n[a]);
              const score = next.q[a] + bonus;
              if (score > bestScore) {
                bestScore = score;
                arm = a;
              }
            }
          }
          const r = normal(rng, next.truth[arm], 1);
          next.n[arm] += 1;
          next.q[arm] += (r - next.q[arm]) / next.n[arm];
          next.steps += 1;
          next.reward += r;
          if (arm === best) next.optimal += 1;
        }
        return next;
      });
    },
    [strategy, epsilon],
  );

  useEffect(() => {
    if (!running) return undefined;
    const id = window.setInterval(() => pull(20), 60);
    return () => window.clearInterval(id);
  }, [running, pull]);

  useEffect(() => {
    if (state.steps >= 2000 && running) setRunning(false);
  }, [state.steps, running]);

  const maxAbs = Math.max(2, ...state.truth.map(Math.abs), ...state.q.map(Math.abs));
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const bandW = innerW / K;
  const y = (v: number) => PAD.top + innerH / 2 - (v / maxAbs) * (innerH / 2);
  const bestArm = state.truth.indexOf(Math.max(...state.truth));

  const estimateColor = seriesColor(0, dark);
  const truthColor = seriesColor(1, dark);

  return (
    <VizPanel
      title="Run a bandit: estimates climbing toward hidden values"
      hint="Greedy locks onto whichever arm looked good first. ε-greedy keeps sampling everything. UCB spends its exploration where it is still uncertain — watch how fast the best arm's estimate reaches its true value."
      legend={[
        {label: 'estimate Q(a)', color: estimateColor},
        {label: 'true value q*(a)', color: truthColor, note: 'hidden from the agent'},
      ]}
      table={{
        columns: ['arm', 'pulls', 'estimate Q(a)', 'true q*(a)'],
        rows: state.truth.map((t, i) => [
          i + 1 + (i === bestArm ? ' (best)' : ''),
          state.n[i],
          state.q[i].toFixed(3),
          t.toFixed(3),
        ]),
      }}
      controls={
        <>
          <label className={s.control}>
            strategy
            <select
              className={s.select}
              value={strategy}
              onChange={(e) => {
                setStrategy(e.target.value as Strategy);
                reset();
              }}>
              <option value="greedy">greedy</option>
              <option value="epsilon">ε-greedy</option>
              <option value="ucb">UCB</option>
            </select>
          </label>
          <label className={s.control}>
            ε
            <input
              type="range"
              min={0}
              max={0.5}
              step={0.01}
              value={epsilon}
              disabled={strategy !== 'epsilon'}
              onChange={(e) => setEpsilon(Number(e.target.value))}
            />
            <span className={s.value}>{epsilon.toFixed(2)}</span>
          </label>
          <button type="button" className={s.button} onClick={() => pull(50)}>
            pull ×50
          </button>
          <button type="button" className={s.button} onClick={() => setRunning((r) => !r)}>
            {running ? 'pause' : 'run'}
          </button>
          <button
            type="button"
            className={s.button}
            onClick={() => {
              const next = seed + 1;
              setSeed(next);
              reset(next);
            }}>
            new problem
          </button>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label="Bar chart of estimated action values against their hidden true values">
        <line className={s.axis} x1={PAD.left} y1={y(0)} x2={W - PAD.right} y2={y(0)} />
        {state.q.map((q, i) => {
          const x = PAD.left + i * bandW;
          const barW = bandW * 0.46;
          const top = Math.min(y(q), y(0));
          const height = Math.max(2, Math.abs(y(q) - y(0)));
          return (
            <g key={i}>
              <rect
                x={x + bandW * 0.5 - barW - 1}
                y={top}
                width={barW}
                height={height}
                rx={4}
                fill={estimateColor}
                opacity={state.n[i] ? 1 : 0.28}>
                <title>{`arm ${i + 1}: Q=${q.toFixed(2)} after ${state.n[i]} pulls`}</title>
              </rect>
              <line
                x1={x + bandW * 0.5 + 1}
                x2={x + bandW * 0.5 + barW + 1}
                y1={y(state.truth[i])}
                y2={y(state.truth[i])}
                stroke={truthColor}
                strokeWidth={2.5}
                strokeLinecap="round">
                <title>{`arm ${i + 1}: true value ${state.truth[i].toFixed(2)}`}</title>
              </line>
              <text
                className={s.tick}
                x={x + bandW / 2}
                y={H - 8}
                textAnchor="middle"
                style={{fontWeight: i === bestArm ? 700 : 400}}>
                {i + 1}
              </text>
              {/* the best arm is marked with a rule, not a glyph, and is also
                  named in the table view — never identified by colour alone */}
              {i === bestArm && (
                <line
                  x1={x + bandW / 2 - 7}
                  x2={x + bandW / 2 + 7}
                  y1={H - 4}
                  y2={H - 4}
                  stroke={truthColor}
                  strokeWidth={2}
                  strokeLinecap="round"
                />
              )}
            </g>
          );
        })}
      </svg>

      <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0}}>
        <span>
          pulls <strong>{state.steps}</strong>
        </span>
        <span>
          average reward{' '}
          <strong>{state.steps ? (state.reward / state.steps).toFixed(3) : '0.000'}</strong>
        </span>
        <span>
          % optimal action{' '}
          <strong>{state.steps ? ((100 * state.optimal) / state.steps).toFixed(1) : '0.0'}%</strong>
        </span>
      </div>
    </VizPanel>
  );
}
