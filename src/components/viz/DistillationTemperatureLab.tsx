import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const W = 640;
const H = 300;
const PAD = {top: 26, right: 24, bottom: 40, left: 44};

export const CLASSES = ['car', 'truck', 'bus', 'cat', 'carrot'];
export const TEACHER = [9.0, 5.5, 4.6, 1.0, -2.0];
export const STUDENT_BASE = [6.0, 6.2, 3.0, 2.5, 0.0];

export function softmax(z: number[], T: number): number[] {
  const scaled = z.map((v) => v / T);
  const top = Math.max(...scaled);
  const e = scaled.map((v) => Math.exp(v - top));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

export function softLoss(teacher: number[], student: number[], T: number) {
  const p = softmax(teacher, T);
  const q = softmax(student, T);
  const kl = p.reduce((a, pi, i) => a + (pi > 0 ? pi * Math.log(pi / q[i]) : 0), 0);
  const grad = q.map((qi, i) => (qi - p[i]) / T);
  const gradNorm = Math.sqrt(grad.reduce((a, g) => a + g * g, 0));
  return {p, q, kl, gradNorm};
}

export function hardCrossEntropy(student: number[], label: number): number {
  return -Math.log(softmax(student, 1)[label]);
}

export default function DistillationTemperatureLab() {
  const dark = useDarkViz();
  const [T, setT] = useState(4);
  const [truck, setTruck] = useState(6.2);
  const [alpha, setAlpha] = useState(0);

  const student = STUDENT_BASE.map((v, i) => (i === 1 ? truck : v));
  const {p, q, kl, gradNorm} = softLoss(TEACHER, student, T);
  const ce = hardCrossEntropy(student, 0);
  const mixed = alpha * ce + (1 - alpha) * T * T * kl;

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const group = innerW / CLASSES.length;
  const barW = group * 0.3;
  const y = (v: number) => PAD.top + innerH - v * innerH;
  const teacherColor = seriesColor(0, dark);
  const studentColor = seriesColor(1, dark);

  const rows = CLASSES.map((c, i) => [c, TEACHER[i].toFixed(1), student[i].toFixed(1), p[i].toFixed(4), q[i].toFixed(4)]);

  return (
    <VizPanel
      title="Temperature softens the teacher"
      hint="Raise the temperature and the teacher's small probabilities (truck, bus) grow into a visible signal the student can copy. Defaults match the chapter's first code block: at T = 4 the KL is 0.10767 and the gradient norm times T squared is 0.9683. Drag the truck logit to see the student confuse truck with car."
      legend={[
        {label: 'teacher softmax(z / T)', color: teacherColor},
        {label: 'student softmax(z / T)', color: studentColor},
      ]}
      table={{columns: ['class', 'teacher logit', 'student logit', 'teacher p', 'student q'], rows}}
      controls={
        <>
          <label className={s.control}>
            temperature T
            <input type="range" min={1} max={20} step={0.5} value={T} onChange={(e) => setT(Number(e.target.value))} />
            <span className={s.value}>{T.toFixed(1)}</span>
          </label>
          <label className={s.control}>
            student truck logit
            <input type="range" min={0} max={10} step={0.1} value={truck} onChange={(e) => setTruck(Number(e.target.value))} />
            <span className={s.value}>{truck.toFixed(1)}</span>
          </label>
          <label className={s.control}>
            weight on the hard label
            <input type="range" min={0} max={1} step={0.05} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} />
            <span className={s.value}>{alpha.toFixed(2)}</span>
          </label>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label={`Teacher and student class probabilities at temperature ${T}`}>
        <line className={s.axis} x1={PAD.left} y1={y(0)} x2={W - PAD.right} y2={y(0)} />
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line className={s.grid} x1={PAD.left} y1={y(t)} x2={W - PAD.right} y2={y(t)} />
            <text className={s.tick} x={PAD.left - 6} y={y(t) + 3} textAnchor="end">{t}</text>
          </g>
        ))}
        {CLASSES.map((c, i) => {
          const x0 = PAD.left + i * group + group / 2;
          return (
            <g key={c}>
              <rect x={x0 - barW - 2} y={y(p[i])} width={barW} height={Math.max(0, y(0) - y(p[i]))} fill={teacherColor} />
              <rect x={x0 + 2} y={y(q[i])} width={barW} height={Math.max(0, y(0) - y(q[i]))} fill={studentColor} />
              <text className={s.dataLabel} x={x0 - barW / 2 - 2} y={y(p[i]) - 4} textAnchor="middle">{p[i].toFixed(3)}</text>
              <text className={s.dataLabel} x={x0 + barW / 2 + 2} y={y(q[i]) - 4} textAnchor="middle">{q[i].toFixed(3)}</text>
              <text className={s.axisLabel} x={x0} y={H - 18} textAnchor="middle">{c}</text>
            </g>
          );
        })}
      </svg>
      <div className={s.controls} style={{borderBottom: 0, paddingLeft: 0, marginTop: '0.4rem'}}>
        <span>
          KL(teacher || student) {kl.toFixed(5)} | gradient norm {gradNorm.toFixed(5)} | gradient norm x T^2{' '}
          {(gradNorm * T * T).toFixed(4)} | hard cross-entropy (T = 1) {ce.toFixed(4)} | mixed loss {mixed.toFixed(4)}
        </span>
      </div>
    </VizPanel>
  );
}
