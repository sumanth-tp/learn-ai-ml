const {test, after} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const ts = require('typescript');

// Run the browser-independent source and storage logic without a bundler.
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'ai-updates-tests-'));
for (const name of ['fetchers', 'history', 'sourceRequest']) {
  const source = fs.readFileSync(path.join(__dirname, '../src/components/AIInnovationHub', `${name}.ts`), 'utf8');
  fs.writeFileSync(path.join(temp, `${name}.js`), ts.transpileModule(source, {
    compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022},
  }).outputText);
}
after(() => fs.rmSync(temp, {recursive: true, force: true}));
const {fetchAllDiscoveries, fetchVideos} = require(path.join(temp, 'fetchers.js'));
const {sourceJSON} = require(path.join(temp, 'sourceRequest.js'));
const {discoveryKeys} = require(path.join(temp, 'history.js'));
const settings = {youtubeApiKey: '', githubToken: ''};
const day = 86400000;
const indices = Array.from({length: 15}, (_, i) => i);
const video = (i, days = 45) => ({type: 'stream', url: `/watch?v=video${String(i).padStart(6, '0')}`, title: `AI video ${i}`, uploaded: Date.now() - days * day});
const json = (body, status = 200) => new Response(JSON.stringify(body), {status, headers: {'Content-Type': 'application/json'}});

function mockSources(t, override = () => undefined) {
  const requests = [];
  const original = global.fetch;
  global.fetch = async (input, options) => {
    const url = new URL(input);
    requests.push({url, options});
    const custom = await override(url, options, requests);
    if (custom) return custom;
    if (url.hostname === 'api.openalex.org') return json({results: indices.map(i => ({id: `https://openalex.org/${i}`, title: `Paper ${i}`, publication_date: '2026-09-01'}))});
    if (url.pathname === '/api/daily_papers') return json(indices.map(i => ({paper: {id: `2609.${i}`, title: `Backup paper ${i}`, summary: 'A research abstract.', publishedAt: '2026-09-01', authors: [{name: 'Researcher'}]}})));
    if (url.pathname === '/api/models') return json(indices.map(i => ({id: `model-${i}`, createdAt: '2026-09-01'})));
    if (url.hostname === 'api.github.com') return json({items: indices.map(i => ({id: i, full_name: `tool-${i}`, html_url: `https://github.com/test/${i}`}))});
    if (url.hostname === 'api.piped.private.coffee') return json({items: indices.map(i => video(i)), nextpage: null});
    if (url.hostname === 'hn.algolia.com') return json({hits: indices.map(i => ({title: `AI tutorial ${i}`, url: `https://www.youtube.com/watch?v=backup${String(i).padStart(5, '0')}`, created_at: new Date().toISOString()})), nbPages: 1});
    throw new Error(`Unexpected test request: ${url.origin}${url.pathname}`);
  };
  t.after(() => { global.fetch = original; });
  return requests;
}

test('OpenAlex 503 retries once, then returns ten papers from the backup', async t => {
  const requests = mockSources(t, url => url.hostname === 'api.openalex.org' ? json({}, 503) : undefined);
  const result = await fetchAllDiscoveries(settings);
  assert.equal(result.items.length, 40);
  assert.equal(requests.filter(r => r.url.hostname === 'api.openalex.org').length, 2);
  assert(result.items.filter(i => i.category === 'papers').every(i => i.source.includes('Hugging Face Papers')));
  assert.equal(result.reports[0].status, 'ok');
  assert.match(result.reports[0].message, /Using Hugging Face Papers/);
  assert(requests.every(r => r.options.cache === 'no-store'));
});

test('90-day window accepts older videos, excludes expired/future dates, and loads another page', async t => {
  const requests = mockSources(t, url => {
    if (url.hostname !== 'api.piped.private.coffee') return;
    if (url.pathname === '/search') return json({items: [video(1, 89), video(2, 91), video(3, -1), video(4, 0.001)], nextpage: 'next-token'});
    assert.equal(url.searchParams.get('nextpage'), 'next-token');
    return json({items: indices.slice(5, 13).map(i => video(i, 60)), nextpage: null});
  });
  const result = await fetchAllDiscoveries(settings);
  const videos = result.items.filter(i => i.category === 'videos');
  assert.equal(videos.length, 10);
  assert(videos.some(i => i.id === 'video:video000001'));
  assert(videos.every(i => Date.parse(i.publishedAt) >= Date.now() - 90 * day));
  assert(!videos.some(i => ['video:video000002', 'video:video000003'].includes(i.id)));
  assert.equal(requests.filter(r => r.url.hostname === 'api.piped.private.coffee').length, 2);
});

test('video outage falls back to real video URLs with shared dates, not invented upload dates', async t => {
  const requests = mockSources(t, url => url.hostname === 'api.piped.private.coffee' ? json({}, 503) : undefined);
  const result = await fetchAllDiscoveries(settings);
  const videos = result.items.filter(i => i.category === 'videos');
  assert.equal(videos.length, 10);
  assert(videos.every(i => i.source === 'YouTube via Hacker News' && i.sharedAt && !i.publishedAt));
  const fallback = requests.find(r => r.url.hostname === 'hn.algolia.com');
  assert(Math.abs(Number(fallback.url.searchParams.get('numericFilters').split('>=')[1]) - (Date.now() / 1000 - 90 * 86400)) < 2);
});

test('repeats fill all four categories and saved records survive a complete outage', async t => {
  let outage = false;
  mockSources(t, () => outage ? json({}, 503) : undefined);
  const first = await fetchAllDiscoveries(settings);
  const seen = first.items.flatMap(discoveryKeys);
  const repeated = await fetchAllDiscoveries(settings, undefined, seen, first.items);
  assert.equal(repeated.items.length, 40);
  assert.equal(repeated.newCount, 20); // Five unseen records per category come first.
  outage = true;
  const cached = await fetchAllDiscoveries(settings, undefined, seen, first.items);
  assert.equal(cached.items.length, 40);
  assert.equal(cached.newCount, 0);
  assert(cached.reports.every(r => r.status === 'error' && r.count === 10 && r.message.includes('restored')));
  assert.equal(new Set(cached.items.map(i => i.id)).size, 40);
});

test('empty successful sources are marked partial instead of healthy', async t => {
  mockSources(t, url => url.pathname === '/api/models' ? json([]) : undefined);
  const result = await fetchAllDiscoveries(settings);
  const report = result.reports.find(r => r.source === 'Hugging Face');
  assert.equal(report.status, 'partial');
  assert.match(report.message, /Only 0 of 10/);
});

test('official YouTube search uses the same 90-day window', async t => {
  const requests = mockSources(t, url => url.hostname === 'www.googleapis.com' ? json({items: []}) : undefined);
  await fetchVideos({...settings, youtubeApiKey: 'test-key'});
  const after = Date.parse(requests[0].url.searchParams.get('publishedAfter'));
  assert(Date.now() - after >= 90 * day && Date.now() - after < 91 * day);
});

test('cancellation propagates without a retry', async t => {
  let requests = 0;
  const original = global.fetch;
  global.fetch = (_, options) => new Promise((_, reject) => {
    requests++;
    options.signal.addEventListener('abort', () => reject(options.signal.reason), {once: true});
  });
  t.after(() => { global.fetch = original; });
  const controller = new AbortController();
  const promise = sourceJSON('https://example.com', 'Test', controller.signal);
  controller.abort();
  await assert.rejects(promise, {name: 'AbortError'});
  assert.equal(requests, 1);
});
