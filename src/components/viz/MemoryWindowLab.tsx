import {useState} from 'react';
import VizPanel, {vizStyles as s} from './VizPanel';
import {LinePlot, Slider} from './CourseLabShared';
import {memoryAt, type MemoryStrategy} from './courseSimulations';
import c from './CourseLab.module.css';

const STRATEGIES = {buffer: 'Full conversation buffer', window: 'Sliding window', tokens: 'Token buffer', summary: 'Summary buffer'};
const FACTS = [{turn: 1, fact: 'Salary: ₹1,20,000 / month'}, {turn: 2, fact: 'Expenses: ₹60,000 / month'},
  {turn: 3, fact: 'Fixed deposit: ₹50,000'}, {turn: 3, fact: 'Risk preference: cautious'}];
const PROMPTS = ['Introduce salary', 'Describe expenses', 'Share savings and risk', 'Ask about SIPs', 'Discuss an emergency fund',
  'Am I saving enough for my salary?', 'Compare investment choices', 'Use my salary in the plan', 'Review the plan', 'Summarise using all my details'];
export default function MemoryWindowLab() {
  const [strategy, setStrategy] = useState<MemoryStrategy>('window');
  const [window, setWindow] = useState(3);
  const [budget, setBudget] = useState(450);
  const [turn, setTurn] = useState(6);
  const turns = Array.from({length: 10}, (_, i) => i + 1);
  const selected = turns.map(t => memoryAt(t, strategy, window, budget));
  const full = turns.map(t => memoryAt(t, 'buffer', window, budget).tokens);
  const current = selected[turn - 1];
  function factStatus(factTurn: number) {
    if (factTurn > turn) return 'Not yet given';
    const state = current.states[factTurn - 1];
    return state === 'verbatim' ? 'Visible in user message' : state === 'summarised' ? 'Available via summary' : 'Lost from prompt';
  }
  return <VizPanel title="Memory windows: cost versus retained facts"
    hint="Snapshot before the current reply: 132 system tokens, the supplied user-token counts, and 80 tokens per earlier reply. Summary size is an illustrative 40 + 12 tokens per summarised turn, capped at 150; this exercise assumes it preserves the four facts. Real summaries can lose details."
    controls={<>
      <label className={c.control}>Memory strategy<select className={s.select} value={strategy} onChange={e => setStrategy(e.target.value as MemoryStrategy)}>
        {Object.entries(STRATEGIES).map(([value, title]) => <option key={value} value={value}>{title}</option>)}
      </select></label>
      {strategy === 'window' && <Slider label="Window turns" value={window} min={1} max={10} onChange={setWindow} />}
      {(strategy === 'tokens' || strategy === 'summary') && <Slider label="Prompt token budget" value={budget} min={300} max={1000} step={10} onChange={setBudget} />}
      <Slider label="Conversation turn" value={turn} min={1} max={10} onChange={setTurn} />
    </>}
    table={{columns: ['Turn', 'Full buffer tokens', 'Chosen strategy tokens', 'User message retention'], rows: turns.map((t, i) => [t, full[i], selected[i].tokens, t <= turn ? current.states[i] : 'Future turn'])}}>
    <div className={c.score} aria-live="polite"><output aria-label="Prompt tokens">{current.tokens}</output> prompt tokens
      <span className={c.badge}>Full buffer: {full[turn - 1]}</span>
      {current.summaryTokens > 0 && <span className={c.badge}>{current.summaryTokens} summary tokens</span>}
    </div>
    <p className={c.note}>Turn {turn}: {PROMPTS[turn - 1]}. The sliding window includes the current user turn.</p>
    <div className={c.chips}>{current.states.map((state, i) => <span className={c.chip} key={i}>T{i + 1}: {state}</span>)}</div>
    <LinePlot xValues={turns} xLabel="Conversation turn" yLabel="Prompt tokens" yMax={1000} marker={turn}
      series={[{label: STRATEGIES[strategy], values: selected.map(row => row.tokens)}, {label: 'Full buffer', values: full, dashed: true}]} />
    <div className={c.stack}>{FACTS.map(fact => <div key={fact.fact} className={c.chip}>
      <strong>{fact.fact}</strong><br />{factStatus(fact.turn)}
    </div>)}</div>
    {strategy === 'tokens' && <p className={c.note}>The token buffer evicts individual messages. A reply can remain after its user message has gone; that does not guarantee the original fact is still present.</p>}
  </VizPanel>;
}
