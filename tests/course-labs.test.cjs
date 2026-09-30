const {test, after} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const ts = require('typescript');

const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'course-lab-tests-'));
const source = fs.readFileSync(path.join(__dirname, '../src/components/viz/courseSimulations.ts'), 'utf8');
fs.writeFileSync(path.join(temp, 'models.js'), ts.transpileModule(source, {
  compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022},
}).outputText);
after(() => fs.rmSync(temp, {recursive: true, force: true}));
const {memoryAt, forgettingSeries, hpaSimulation} = require(path.join(temp, 'models.js'));

test('full buffer reproduces the notebook counts before the current reply', () => {
  assert.deepEqual([1, 2, 3, 10].map(t => memoryAt(t, 'buffer', 3, 450).tokens), [149, 238, 334, 976]);
  assert.equal(memoryAt(10, 'window', 1, 450).tokens, 146);
  assert.equal(memoryAt(10, 'window', 10, 450).tokens, 976);
});

test('window evicts salary by turn six; summary retains older facts within budget', () => {
  const window = memoryAt(6, 'window', 3, 450);
  assert.deepEqual(window.states, ['evicted', 'evicted', 'evicted', 'verbatim', 'verbatim', 'verbatim']);
  assert.equal(memoryAt(6, 'summary', 3, 450).states[0], 'summarised');
  for (const strategy of ['tokens', 'summary']) {
    for (const budget of [300, 450, 1000]) {
      for (let t = 1; t <= 10; t++) {
        const result = memoryAt(t, strategy, 3, budget);
        assert(result.tokens <= budget);
        assert.equal(result.messages.at(-1).kind, 'user');
        assert.equal(result.messages.at(-1).turn, t);
      }
    }
  }
});

test('token eviction can leave a reply after the user fact is evicted', () => {
  const result = memoryAt(10, 'tokens', 3, 320);
  assert(result.states.includes('reply only'));
  assert(result.tokens <= 320);
});

test('decay halves by the half-life and reinforcement improves retention', () => {
  const unused = forgettingSeries(24, 0.3, 0.1, false);
  const profile = forgettingSeries(24, 0.3, 0.1, true);
  assert(Math.abs(unused.points[24].strength - 0.5) < 1e-9);
  assert(Math.abs(profile.points[24].strength - 0.8) < 1e-9);
  assert.equal(unused.prunedAt, 80);
  assert.equal(profile.prunedAt, null);
  const early = forgettingSeries(6, 0.5, 0.3, true);
  assert(early.prunedAt < 24);
  assert.equal(early.points[24].strength, 0, 'an access must not revive deleted content');
});

test('HPA separates desired and running pods; scale down waits five minutes', () => {
  const low = hpaSimulation(10, 8);
  assert(low.every(row => row.desired === 2 && row.pending === 0));
  assert(Math.abs(low[0].cpu - 41.5) < 1e-9);
  const high = hpaSimulation(50, 8);
  assert.equal(high[0].desired, 4);
  assert.equal(high[0].running, 2);
  assert.equal(high[1].running, 4);
  assert.equal(high[1].desired, 6);
  assert.equal(high[1].pending, 2);
  assert(high.slice(8, 13).every(row => row.desired === 6));
  assert.equal(high[13].desired, 5);
  assert.equal(high[14].desired, 4);
  assert(high.every(row => row.running <= 4 && row.desired <= 6 && row.desired >= 2));
  assert(high.every(row => row.failures >= 0 && row.failures <= 1));
});

test('zero traffic and each supported load duration keep the model finite and bounded', () => {
  for (const users of [0, 10, 20, 50, 60]) for (const duration of [1, 8, 10]) {
    for (const row of hpaSimulation(users, duration)) {
      assert(Number.isFinite(row.cpu));
      assert(Number.isFinite(row.failures));
      assert.equal(row.desired, row.running + row.pending);
      assert(row.pending >= 0);
    }
  }
});
