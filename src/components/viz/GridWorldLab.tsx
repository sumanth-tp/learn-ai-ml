import {useCallback, useMemo, useState} from 'react';

import {sequentialColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const ROWS = 4;
const COLS = 5;
const GOAL = {r: 0, c: 4};
const TRAP = {r: 1, c: 4};
const WALLS = new Set(['1,1', '2,1', '2,3']);
const ACTIONS: {name: string; dr: number; dc: number; arrow: string}[] = [
  {name: 'up', dr: -1, dc: 0, arrow: '↑'},
  {name: 'down', dr: 1, dc: 0, arrow: '↓'},
  {name: 'left', dr: 0, dc: -1, arrow: '←'},
  {name: 'right', dr: 0, dc: 1, arrow: '→'},
];

const key = (r: number, c: number) => `${r},${c}`;
const isTerminal = (r: number, c: number) =>
  (r === GOAL.r && c === GOAL.c) || (r === TRAP.r && c === TRAP.c);

function reward(r: number, c: number, stepCost: number) {
  if (r === GOAL.r && c === GOAL.c) return 1;
  if (r === TRAP.r && c === TRAP.c) return -1;
  return stepCost;
}

function move(r: number, c: number, dr: number, dc: number) {
  const nr = Math.min(Math.max(r + dr, 0), ROWS - 1);
  const nc = Math.min(Math.max(c + dc, 0), COLS - 1);
  if (WALLS.has(key(nr, nc))) return {r, c};
  return {r: nr, c: nc};
}

function zeros() {
  return Array.from({length: ROWS}, () => Array(COLS).fill(0));
}

/** One synchronous sweep of the Bellman optimality update. */
function sweep(V: number[][], gamma: number, stepCost: number) {
  const next = V.map((row) => [...row]);
  let delta = 0;
  for (let r = 0; r < ROWS; r += 1) {
    for (let c = 0; c < COLS; c += 1) {
      if (WALLS.has(key(r, c))) continue;
      if (isTerminal(r, c)) {
        next[r][c] = reward(r, c, stepCost);
        continue;
      }
      let best = -Infinity;
      for (const a of ACTIONS) {
        const {r: nr, c: nc} = move(r, c, a.dr, a.dc);
        best = Math.max(best, reward(r, c, stepCost) + gamma * V[nr][nc]);
      }
      delta = Math.max(delta, Math.abs(best - V[r][c]));
      next[r][c] = best;
    }
  }
  return {V: next, delta};
}

export default function GridWorldLab() {
  const dark = useDarkViz();
  const [gamma, setGamma] = useState(0.9);
  const [stepCost, setStepCost] = useState(-0.04);
  const [V, setV] = useState(zeros);
  const [iterations, setIterations] = useState(0);
  const [delta, setDelta] = useState(Infinity);

  const step = useCallback(
    (times = 1) => {
      setV((current) => {
        let v = current;
        let d = 0;
        for (let i = 0; i < times; i += 1) {
          const out = sweep(v, gamma, stepCost);
          v = out.V;
          d = out.delta;
        }
        setDelta(d);
        setIterations((n) => n + times);
        return v;
      });
    },
    [gamma, stepCost],
  );

  const reset = useCallback(() => {
    setV(zeros());
    setIterations(0);
    setDelta(Infinity);
  }, []);

  const {min, max} = useMemo(() => {
    const flat = V.flat().filter((_, i) => !WALLS.has(key(Math.floor(i / COLS), i % COLS)));
    return {min: Math.min(...flat), max: Math.max(...flat)};
  }, [V]);

  const greedy = (r: number, c: number) => {
    let best = -Infinity;
    let arrow = '';
    for (const a of ACTIONS) {
      const {r: nr, c: nc} = move(r, c, a.dr, a.dc);
      const q = reward(r, c, stepCost) + gamma * V[nr][nc];
      if (q > best) {
        best = q;
        arrow = a.arrow;
      }
    }
    return arrow;
  };

  const CELL = 72;
  const W = COLS * CELL;
  const H = ROWS * CELL;

  return (
    <VizPanel
      title="Value iteration on a grid world"
      hint="Each sweep applies the Bellman optimality update once. Watch value spread outward from the goal, one ring per sweep, and the arrows settle into the optimal policy. Lower γ makes the agent short-sighted; a harsher step cost makes it hurry."
      legend={[
        {label: 'low value', color: sequentialColor(0.1, dark)},
        {label: 'high value', color: sequentialColor(0.95, dark)},
        {label: 'arrow = greedy action', color: 'transparent'},
      ]}
      table={{
        columns: ['cell', 'V(s)', 'greedy action'],
        rows: V.flatMap((row, r) =>
          row.map((v, c) => [
            WALLS.has(key(r, c)) ? `(${r},${c}) wall` : `(${r},${c})`,
            WALLS.has(key(r, c)) ? '—' : v.toFixed(3),
            WALLS.has(key(r, c)) || isTerminal(r, c) ? '—' : greedy(r, c),
          ]),
        ),
      }}
      controls={
        <>
          <label className={s.control}>
            γ
            <input
              type="range"
              min={0.5}
              max={0.99}
              step={0.01}
              value={gamma}
              onChange={(e) => {
                setGamma(Number(e.target.value));
                reset();
              }}
            />
            <span className={s.value}>{gamma.toFixed(2)}</span>
          </label>
          <label className={s.control}>
            step cost
            <input
              type="range"
              min={-0.5}
              max={0}
              step={0.01}
              value={stepCost}
              onChange={(e) => {
                setStepCost(Number(e.target.value));
                reset();
              }}
            />
            <span className={s.value}>{stepCost.toFixed(2)}</span>
          </label>
          <button type="button" className={s.button} onClick={() => step(1)}>
            sweep ×1
          </button>
          <button type="button" className={s.button} onClick={() => step(25)}>
            run to convergence
          </button>
          <button type="button" className={s.button} onClick={reset}>
            reset
          </button>
        </>
      }>
      <svg
        className={s.svg}
        viewBox={`0 0 ${W} ${H}`}
        style={{maxWidth: 420, margin: '0 auto'}}
        role="img"
        aria-label="Grid world heat map of state values with greedy policy arrows">
        {V.map((row, r) =>
          row.map((v, c) => {
            const wall = WALLS.has(key(r, c));
            const t = max > min ? (v - min) / (max - min) : 0;
            const x = c * CELL;
            const y = r * CELL;
            return (
              <g key={key(r, c)}>
                <rect
                  x={x + 1}
                  y={y + 1}
                  width={CELL - 2}
                  height={CELL - 2}
                  rx={6}
                  fill={wall ? 'var(--surface-2)' : sequentialColor(t, dark)}
                  stroke="var(--surface-raised)"
                  strokeWidth={2}>
                  <title>
                    {wall ? `wall at (${r},${c})` : `V(${r},${c}) = ${v.toFixed(3)}`}
                  </title>
                </rect>
                {!wall && (
                  <>
                    <text
                      x={x + CELL / 2}
                      y={y + CELL / 2 - 2}
                      textAnchor="middle"
                      style={{
                        fill: t > 0.55 ? '#ffffff' : 'var(--text-strong)',
                        fontSize: 13,
                        fontWeight: 600,
                      }}>
                      {v.toFixed(2)}
                    </text>
                    <text
                      x={x + CELL / 2}
                      y={y + CELL / 2 + 18}
                      textAnchor="middle"
                      style={{
                        fill: t > 0.55 ? 'rgba(255,255,255,0.85)' : 'var(--text-muted)',
                        fontSize: 15,
                      }}>
                      {r === GOAL.r && c === GOAL.c
                        ? 'goal'
                        : r === TRAP.r && c === TRAP.c
                          ? 'trap'
                          : greedy(r, c)}
                    </text>
                  </>
                )}
              </g>
            );
          }),
        )}
      </svg>

      <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0}}>
        <span>
          sweeps <strong>{iterations}</strong>
        </span>
        <span>
          max change this sweep{' '}
          <strong>{Number.isFinite(delta) ? delta.toFixed(5) : '—'}</strong>
        </span>
        <span>{delta < 1e-4 ? 'converged' : 'still propagating'}</span>
      </div>
    </VizPanel>
  );
}
