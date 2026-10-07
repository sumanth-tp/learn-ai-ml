import {useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

type Step = {from: 'C' | 'S'; to: 'C' | 'S'; bytes: number; label: string; text: string};
const FLOWS = {"legacy":{"title":"MCP 2025-11-25: handshake, then one call","steps":[{"from":"C","to":"S","bytes":146,"label":"initialize (request)","text":"{\"jsonrpc\":\"2.0\",\"id\":0,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{},\"clientInfo\":{\"name\":\"c\",\"version\":\"1\"}}}"},{"from":"S","to":"C","bytes":247,"label":"initialize (result)","text":"{\"jsonrpc\":\"2.0\",\"id\":0,\"result\":{\"capabilities\":{\"prompts\":{\"listChanged\":false},\"resources\":{\"listChanged\":false,\"subscribe\":false},\"tools\":{\"listChanged\":false}},\"protocolVersion\":\"2025-11-25\",\"serverInfo\":{\"name\":\"pricing-demo\",\"version\":\"\"}}}"},{"from":"C","to":"S","bytes":54,"label":"notifications/initialized","text":"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}"},{"from":"C","to":"S","bytes":110,"label":"tools/call","text":"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"quote\",\"arguments\":{\"sku\":\"A1\",\"quantity\":4}}}"},{"from":"S","to":"C","bytes":151,"label":"result","text":"{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"content\":[{\"text\":\"4 x A1 = 50.00\",\"type\":\"text\"}],\"isError\":false,\"structuredContent\":{\"result\":\"4 x A1 = 50.00\"}}}"}]},"modern":{"title":"MCP 2026-07-28: one self-contained call","steps":[{"from":"C","to":"S","bytes":223,"label":"tools/call with _meta","text":"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"quote\",\"arguments\":{\"sku\":\"A1\",\"quantity\":4},\"_meta\":{\"io.modelcontextprotocol/protocolVersion\":\"2026-07-28\",\"io.modelcontextprotocol/clientCapabilities\":{}}}}"},{"from":"S","to":"C","bytes":259,"label":"result with resultType","text":"{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"content\":[{\"text\":\"4 x A1 = 50.00\",\"type\":\"text\"}],\"isError\":false,\"resultType\":\"complete\",\"structuredContent\":{\"result\":\"4 x A1 = 50.00\"},\"_meta\":{\"io.modelcontextprotocol/serverInfo\":{\"name\":\"pricing-demo\",\"version\":\"\"}}}}"}]},"mrtr":{"title":"MCP 2026-07-28: a call that needs the user (MRTR)","steps":[{"from":"C","to":"S","bytes":243,"label":"tools/call","text":"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"refund\",\"arguments\":{\"order_id\":\"A-17\"},\"_meta\":{\"io.modelcontextprotocol/protocolVersion\":\"2026-07-28\",\"io.modelcontextprotocol/clientCapabilities\":{\"elicitation\":{\"form\":{}}}}}}"},{"from":"S","to":"C","bytes":753,"label":"input_required, requestState","text":"{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"inputRequests\":{\"__main__:ask\":{\"method\":\"elicitation/create\",\"params\":{\"message\":\"Refund order A-17?\",\"mode\":\"form\",\"requestedSchema\":{\"properties\":{\"approve\":{\"title\":\"Approve\",\"type\":\"boolean\"}},\"required\":[\"approve\"],\"type\":\"object\"}}}},\"requestState\":\"<345 characters, opaque>\",\"resultType\":\"input_required\",\"_meta\":{\"io.modelcontextprotocol/serverInfo\":{\"name\":\"refund-demo\",\"version\":\"\"}}}}"},{"from":"C","to":"S","bytes":687,"label":"tools/call, inputResponses, requestState","text":"{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"refund\",\"arguments\":{\"order_id\":\"A-17\"},\"inputResponses\":{\"__main__:ask\":{\"action\":\"accept\",\"content\":{\"approve\":true}}},\"requestState\":\"<345 characters, opaque>\",\"_meta\":{\"io.modelcontextprotocol/protocolVersion\":\"2026-07-28\",\"io.modelcontextprotocol/clientCapabilities\":{\"elicitation\":{\"form\":{}}}}}}"},{"from":"S","to":"C","bytes":256,"label":"complete","text":"{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"content\":[{\"text\":\"refunded A-17\",\"type\":\"text\"}],\"isError\":false,\"resultType\":\"complete\",\"structuredContent\":{\"result\":\"refunded A-17\"},\"_meta\":{\"io.modelcontextprotocol/serverInfo\":{\"name\":\"refund-demo\",\"version\":\"\"}}}}"}]},"a2a":{"title":"A2A 1.0: two-turn task","steps":[{"from":"C","to":"S","bytes":32,"label":"GET agent card","text":"GET /.well-known/agent-card.json"},{"from":"S","to":"C","bytes":423,"label":"Agent Card","text":"{\"name\":\"Expense Approver\",\"description\":\"Approves small expenses.\",\"supportedInterfaces\":[{\"url\":\"http://127.0.0.1:62122/\",\"protocolBinding\":\"JSONRPC\",\"protocolVersion\":\"1.0\"}],\"version\":\"0.1.0\",\"capabilities\":{\"streaming\":true},\"defaultInputModes\":[\"text/plain\"],\"defaultOutputModes\":[\"text/plain\"],\"skills\":[{\"id\":\"approve\",\"name\":\"Approve expense\",\"description\":\"Checks an amount against a limit.\",\"tags\":[\"finance\"]}]}"},{"from":"C","to":"S","bytes":187,"label":"SendMessage","text":"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"SendMessage\",\"params\":{\"message\":{\"messageId\":\"1588af7c-96f0-4608-a9e5-836d751988c6\",\"role\":\"ROLE_USER\",\"parts\":[{\"text\":\"please approve my expense\"}]}}}"},{"from":"S","to":"C","bytes":582,"label":"Task: INPUT_REQUIRED","text":"{\"result\":{\"task\":{\"id\":\"e9855354-4068-4303-bef1-15973ceaae4a\",\"contextId\":\"55ee7d2b-ca84-43b2-94aa-396547b45026\",\"status\":{\"state\":\"TASK_STATE_INPUT_REQUIRED\",\"message\":{\"messageId\":\"30bf3769-8075-4338-a747-c6a75125701c\",\"role\":\"ROLE_AGENT\",\"parts\":[{\"text\":\"What is the amount?\"}]},\"timestamp\":\"2026-10-07T05:02:21.375784Z\"},\"history\":[{\"messageId\":\"1588af7c-96f0-4608-a9e5-836d751988c6\",\"contextId\":\"55ee7d2b-ca84-43b2-94aa-396547b45026\",\"taskId\":\"e9855354-4068-4303-bef1-15973ceaae4a\",\"role\":\"ROLE_USER\",\"parts\":[{\"text\":\"please approve my expense\"}]}]}},\"id\":1,\"jsonrpc\":\"2.0\"}"},{"from":"C","to":"S","bytes":271,"label":"SendMessage with taskId","text":"{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"SendMessage\",\"params\":{\"message\":{\"messageId\":\"c42a1676-9869-40a3-aab0-7fa3e7d071c0\",\"role\":\"ROLE_USER\",\"taskId\":\"e9855354-4068-4303-bef1-15973ceaae4a\",\"contextId\":\"55ee7d2b-ca84-43b2-94aa-396547b45026\",\"parts\":[{\"text\":\"amount 120\"}]}}}"},{"from":"S","to":"C","bytes":909,"label":"Task: COMPLETED","text":"{\"result\":{\"task\":{\"id\":\"e9855354-4068-4303-bef1-15973ceaae4a\",\"contextId\":\"55ee7d2b-ca84-43b2-94aa-396547b45026\",\"status\":{\"state\":\"TASK_STATE_COMPLETED\",\"timestamp\":\"2026-10-07T05:02:21.386555Z\"},\"artifacts\":[{\"artifactId\":\"5ea0f25a-7857-41f5-bb81-84e1b905129a\",\"parts\":[{\"text\":\"expense of 120: approved\",\"mediaType\":\"text/plain\"}]}],\"history\":[{\"messageId\":\"1588af7c-96f0-4608-a9e5-836d751988c6\",\"contextId\":\"55ee7d2b-ca84-43b2-94aa-396547b45026\",\"taskId\":\"e9855354-4068-4303-bef1-15973ceaae4a\",\"role\":\"ROLE_USER\",\"parts\":[{\"text\":\"please approve my expense\"}]},{\"messageId\":\"30bf3769-8075-4338-a747-c6a75125701c\",\"role\":\"ROLE_AGENT\",\"parts\":[{\"text\":\"What is the amount?\"}]},{\"messageId\":\"c42a1676-9869-40a3-aab0-7fa3e7d071c0\",\"contextId\":\"55ee7d2b-ca84-43b2-94aa-396547b45026\",\"taskId\":\"e9855354-4068-4303-bef1-15973ceaae4a\",\"role\":\"ROLE_USER\",\"parts\":[{\"text\":\"amount 120\"}]}]}},\"id\":2,\"jsonrpc\":\"2.0\"}"}]}} as Record<string, {title: string; steps: Step[]}>;

const ORDER = ['legacy', 'modern', 'mrtr', 'a2a'] as const;
const LEGACY_SETUP = 447;
const LEGACY_PER_CALL = 261;
const MODERN_PER_CALL = 482;

const W = 640;
const LANE_C = 120;
const LANE_S = 520;
const ROW = 34;

const pretty = (text: string) => {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
};

export default function ProtocolFlowLab() {
  const dark = useDarkViz();
  const [flow, setFlow] = useState<(typeof ORDER)[number]>('legacy');
  const [step, setStep] = useState(5);
  const [calls, setCalls] = useState(1);

  const steps = FLOWS[flow].steps;
  const shown = Math.min(step, steps.length);
  const current = steps[shown - 1];
  const height = 70 + steps.length * ROW;
  const requests = steps.filter((m) => m.from === 'C' && m.label !== 'notifications/initialized').length;
  const total = steps.slice(0, shown).reduce((a, m) => a + m.bytes, 0);

  const legacyBytes = LEGACY_SETUP + calls * LEGACY_PER_CALL;
  const modernBytes = calls * MODERN_PER_CALL;
  const colours = {c: seriesColor(0, dark), s: seriesColor(1, dark), dim: dark ? '#848c99' : '#9aa0a6', hot: dark ? DIVERGING.dark.positive : DIVERGING.light.positive};

  const rows = steps.map((m, i) => [i + 1, `${m.from === 'C' ? 'client to server' : 'server to client'}`, m.label, m.bytes]);

  return (
    <VizPanel
      title="Message flows, with the real bytes"
      hint="Pick a flow and step through it. Every message was captured from a real run (MCP Python SDK 2.3.0, A2A SDK 1.2.2). With the first flow at step 5 and 1 call in the session, the byte counter reproduces the printed 708 bytes for a legacy cold call against 482 for the modern one; raise the call count to see the handshake pay for itself."
      legend={[
        {label: 'client to server', color: colours.c},
        {label: 'server to client', color: colours.s},
      ]}
      table={{columns: ['step', 'direction', 'message', 'bytes'], rows}}
      controls={
        <>
          <label className={s.control}>
            flow
            <select
              className={s.select}
              value={flow}
              onChange={(e) => {
                const next = e.target.value as (typeof ORDER)[number];
                setFlow(next);
                setStep(FLOWS[next].steps.length);
              }}>
              {ORDER.map((key) => (
                <option key={key} value={key}>
                  {FLOWS[key].title}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            step
            <input type="range" min={1} max={steps.length} step={1} value={shown} onChange={(e) => setStep(Number(e.target.value))} />
            <span className={s.value}>
              {shown} of {steps.length}
            </span>
          </label>
          <label className={s.control}>
            calls in one session
            <input type="range" min={1} max={100} step={1} value={calls} onChange={(e) => setCalls(Number(e.target.value))} />
            <span className={s.value}>{calls}</span>
          </label>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${height}`} role="img" aria-label={`${FLOWS[flow].title}: ${steps.length} messages, showing the first ${shown}.`}>
        <text className={s.axisLabel} x={LANE_C} y={18} textAnchor="middle">
          client
        </text>
        <text className={s.axisLabel} x={LANE_S} y={18} textAnchor="middle">
          server
        </text>
        <line className={s.axis} x1={LANE_C} x2={LANE_C} y1={26} y2={height - 10} />
        <line className={s.axis} x1={LANE_S} x2={LANE_S} y1={26} y2={height - 10} />
        {steps.map((m, i) => {
          const y = 52 + i * ROW;
          const fromX = m.from === 'C' ? LANE_C : LANE_S;
          const toX = m.to === 'C' ? LANE_C : LANE_S;
          const on = i < shown;
          const colour = !on ? colours.dim : m.from === 'C' ? colours.c : colours.s;
          const dir = toX > fromX ? 1 : -1;
          return (
            <g key={i} opacity={on ? 1 : 0.3}>
              <line x1={fromX} x2={toX - dir * 8} y1={y} y2={y} stroke={colour} strokeWidth={i === shown - 1 ? 3 : 1.8} />
              <path d={`M${toX},${y} l${-dir * 9},-5 l0,10 z`} fill={colour} />
              <text className={s.tick} x={(LANE_C + LANE_S) / 2} y={y - 6} textAnchor="middle">
                {i + 1}. {m.label} ({m.bytes} B)
              </text>
            </g>
          );
        })}
      </svg>
      <p className={s.hint} style={{padding: '0.4rem 0 0'}} aria-live="polite">
        {requests} requests in this flow, {total} bytes exchanged so far. Cold call, MCP 2025-11-25: {LEGACY_SETUP + LEGACY_PER_CALL} bytes. MCP 2026-07-28: {MODERN_PER_CALL} bytes. With {calls}{' '}
        {calls === 1 ? 'call' : 'calls'} in one session: legacy {legacyBytes.toLocaleString('en-GB')} bytes, modern {modernBytes.toLocaleString('en-GB')} bytes
        ({legacyBytes < modernBytes ? 'the handshake has paid for itself' : 'stateless is still cheaper'}).
      </p>
      <pre style={{margin: '0.4rem 0 0', padding: '0.6rem', fontSize: '0.72rem', lineHeight: 1.45, overflow: 'auto', maxHeight: '15rem', background: 'var(--surface-1)', border: '1px solid var(--border-subtle)', borderRadius: 6, whiteSpace: 'pre-wrap', wordBreak: 'break-word'}}>
        {current ? pretty(current.text) : ''}
      </pre>
    </VizPanel>
  );
}
