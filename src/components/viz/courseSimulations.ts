/** Small deterministic teaching models; no external API calls or tokeniser. */
export const USER_TOKENS = [17, 9, 16, 9, 10, 12, 15, 11, 11, 14];
export const SYSTEM_TOKENS = 132;
export type MemoryStrategy = 'buffer' | 'window' | 'tokens' | 'summary';
type Message = {turn: number; kind: 'user' | 'assistant'; tokens: number};

export function memoryAt(turn: number, strategy: MemoryStrategy, window: number, budget: number) {
  let messages: Message[] = [];
  for (let t = 1; t <= turn; t++) {
    messages.push({turn: t, kind: 'user', tokens: USER_TOKENS[t - 1]});
    // Snapshot immediately before answering the current user message.
    if (t < turn) messages.push({turn: t, kind: 'assistant', tokens: 80});
  }
  const sum = () => messages.reduce((total, message) => total + message.tokens, SYSTEM_TOKENS);
  let summarised = 0;
  const summarySize = () => summarised ? Math.min(150, 40 + 12 * summarised) : 0;
  if (strategy === 'window') messages = messages.filter(message => message.turn > turn - window);
  if (strategy === 'tokens') {
    while (sum() > budget && messages.length > 1) messages.shift();
  }
  if (strategy === 'summary') {
    while (sum() + summarySize() > budget && messages[0]?.turn < turn) {
      const oldest = messages[0].turn;
      messages = messages.filter(message => message.turn !== oldest);
      summarised = oldest;
    }
  }
  const states = Array.from({length: turn}, (_, i) => {
    const t = i + 1;
    if (messages.some(m => m.turn === t && m.kind === 'user')) return 'verbatim';
    if (t <= summarised) return 'summarised';
    if (messages.some(m => m.turn === t)) return 'reply only';
    return 'evicted';
  });
  return {tokens: sum() + summarySize(), summaryTokens: summarySize(), states, messages};
}

export function forgettingSeries(halfLife: number, boost: number, threshold: number, reinforced: boolean) {
  let strength = 1;
  let prunedAt: number | null = null;
  const points = [{hour: 0, strength: 1, access: false}];
  for (let hour = 1; hour <= 120; hour++) {
    const access = reinforced && (hour === 24 || hour === 60);
    if (prunedAt === null) {
      strength *= 0.5 ** (1 / halfLife);
      if (access) strength = Math.min(1, strength + boost);
      // Repeated decay can land a few ulps below an exactly equal threshold.
      if (strength < threshold - 1e-12) {prunedAt = hour; strength = 0;}
    }
    points.push({hour, strength, access});
  }
  return {points, prunedAt};
}

export function hpaSimulation(users: number, duration: number) {
  let desired = 2;
  let lowMinutes = 0;
  const rows: {minute: number; users: number; running: number; desired: number; pending: number; cpu: number; failures: number}[] = [];
  for (let minute = 0; minute <= 15; minute++) {
    // Pods requested in the prior minute become available if capacity permits.
    let running = Math.min(4, desired);
    const activeUsers = minute < duration ? users : 0;
    const cpu = activeUsers * 8.3 / running;
    const target = Math.max(2, Math.min(6, Math.ceil(desired * cpu / 70)));
    if (target > desired) {
      desired = Math.min(target, desired + 2);
      lowMinutes = 0;
    } else if (cpu < 70) {
      lowMinutes++;
      // Samples are one minute apart: six low samples span five whole minutes.
      if (lowMinutes >= 6 && target < desired) desired = Math.max(target, desired - 1);
    } else {
      lowMinutes = 0;
    }
    running = Math.min(running, desired);
    rows.push({minute, users: activeUsers, running, desired, pending: desired - running, cpu,
      failures: cpu > 100 ? Math.max(0, 1 - 100 / cpu) : 0});
  }
  return rows;
}
