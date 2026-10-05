import {useState} from 'react';

import {DIVERGING, seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

const DATA = {"docs":["Kestrel Motors builds the Aster Scooter in Lisbon.","Voltro supplies battery cells to Kestrel Motors.","Helion Group owns Voltro.","Nordic Safety Board regulates Helion Group.","Mara Ellis is chief executive of Kestrel Motors.","Kestrel Motors sells the Aster Scooter to Cobalt Rentals.","Calder Labs develops the drug Zerafil.","Lumen Therapeutics licenses Zerafil from Calder Labs.","Pharma Standards Agency approved Zerafil.","Pharma Standards Agency regulates Lumen Therapeutics.","Tomas Brandt is chief executive of Calder Labs.","Lumen Therapeutics sells Zerafil to Harbour Pharmacies.","Delmar Freight ships goods for Harbour Pharmacies.","Delmar Freight leases trucks from Quarry Fleet.","Quarry Fleet is owned by Helion Group.","Customs Directorate regulates Delmar Freight.","Ines Valdo is chief executive of Delmar Freight.","Cobalt Rentals partners with Delmar Freight.","Paylink processes payments for Cobalt Rentals.","Paylink is owned by Tidewater Capital.","Financial Conduct Office regulates Paylink.","Tidewater Capital funds Calder Labs.","Rhea Okafor is chief executive of Paylink.","Financial Conduct Office regulates Tidewater Capital."],"nodes":[["Aster Scooter",160,168,1],["Calder Labs",520,86,0],["Cobalt Rentals",250,110,1],["Customs Directorate",452,296,4],["Delmar Freight",372,214,4],["Financial Conduct Office",408,34,2],["Harbour Pharmacies",500,170,0],["Helion Group",150,305,3],["Ines Valdo",348,302,4],["Kestrel Motors",120,100,1],["Lisbon",46,150,1],["Lumen Therapeutics",596,140,0],["Mara Ellis",46,48,1],["Nordic Safety Board",250,318,3],["Paylink",318,62,2],["Pharma Standards Agency",580,306,0],["Quarry Fleet",262,250,3],["Rhea Okafor",236,28,2],["Tidewater Capital",422,108,2],["Tomas Brandt",604,44,0],["Voltro",58,262,3],["Zerafil",570,222,0]],"edges":[["Kestrel Motors","Aster Scooter","builds",0],["Kestrel Motors","Lisbon","located_in",0],["Kestrel Motors","Cobalt Rentals","sells_to",5],["Voltro","Kestrel Motors","supplies",1],["Helion Group","Voltro","owns",2],["Helion Group","Quarry Fleet","owns",14],["Nordic Safety Board","Helion Group","regulates",3],["Mara Ellis","Kestrel Motors","leads",4],["Cobalt Rentals","Aster Scooter","buys",5],["Cobalt Rentals","Delmar Freight","partners_with",17],["Calder Labs","Zerafil","develops",6],["Lumen Therapeutics","Zerafil","licenses",7],["Lumen Therapeutics","Calder Labs","licenses_from",7],["Lumen Therapeutics","Harbour Pharmacies","sells_to",11],["Pharma Standards Agency","Zerafil","approved",8],["Pharma Standards Agency","Lumen Therapeutics","regulates",9],["Tomas Brandt","Calder Labs","leads",10],["Harbour Pharmacies","Zerafil","buys",11],["Delmar Freight","Harbour Pharmacies","serves",12],["Delmar Freight","Quarry Fleet","leases_from",13],["Customs Directorate","Delmar Freight","regulates",15],["Ines Valdo","Delmar Freight","leads",16],["Paylink","Cobalt Rentals","serves",18],["Tidewater Capital","Paylink","owns",19],["Tidewater Capital","Calder Labs","funds",21],["Financial Conduct Office","Paylink","regulates",20],["Financial Conduct Office","Tidewater Capital","regulates",23],["Rhea Okafor","Paylink","leads",22]],"questions":[["Which regulator oversees the owner of Voltro?",[2,3]],["Which regulator oversees the company that owns the truck lessor of Delmar Freight?",[13,14,3]],["Who funds the developer of Zerafil, and which regulator oversees that funder?",[6,21,23]],["Which regulator oversees the parent of the supplier of Kestrel Motors?",[1,2,3]],["Who leads the carrier that serves the pharmacy chain buying Zerafil?",[11,12,16]],["Which regulator oversees the payment processor used by the rental firm that buys the Aster Scooter?",[5,18,20]],["Which agency regulates the company that licenses Zerafil?",[7,9]],["Who leads the developer of the drug that Harbour Pharmacies buys?",[11,6,10]],["What are the main themes across these documents?",[]]],"ranks":[[2,1,3,15,16,17,23,10,5,20,4,9,0,19,18,14,21,22,7,8,13,6,11,12],[15,16,13,17,12,10,20,4,5,22,19,23,18,2,21,3,14,0,9,8,7,1,6,11],[8,7,6,10,11,21,20,23,9,18,22,3,16,15,17,4,19,2,14,5,0,13,12,1],[4,5,1,0,3,2,17,15,16,13,10,14,22,20,9,21,18,23,8,12,19,7,6,11],[11,8,6,7,12,15,17,10,16,9,21,13,18,22,2,4,20,3,19,14,5,23,0,1],[5,0,18,20,17,3,15,13,2,23,22,14,21,8,19,16,10,4,9,1,7,11,6,12],[8,7,6,11,9,10,15,17,21,3,18,23,20,13,16,4,5,12,2,19,14,0,22,1],[12,11,9,6,21,8,7,10,15,17,16,23,19,20,22,4,2,3,13,18,14,0,5,1],[21,12,20,23,16,22,15,2,4,1,13,9,17,18,3,8,0,10,19,7,14,6,11,5]],"docCommunity":[1,3,3,3,1,1,0,0,0,0,0,0,4,4,3,4,4,1,2,2,2,2,2,2]} as {
  docs: string[];
  nodes: [string, number, number, number][];
  edges: [string, string, string, number][];
  questions: [string, number[]][];
  ranks: number[][];
  docCommunity: number[];
};

const W = 640;
const H = 340;
const GLOBAL = DATA.questions.length - 1;
const COMMUNITIES = 5;

type Method = 'vector' | 'path' | 'community';

const complete = (k: number) =>
  DATA.questions.slice(0, GLOBAL).filter((q, i) => q[1].every((d) => DATA.ranks[i].slice(0, k).includes(d))).length;

export default function GraphRetrievalLab() {
  const dark = useDarkViz();
  const [question, setQuestion] = useState(1);
  const [method, setMethod] = useState<Method>('vector');
  const [k, setK] = useState(4);

  const isGlobal = question === GLOBAL;
  const needed = DATA.questions[question][1];
  const effective: Method = isGlobal && method === 'path' ? 'vector' : method;

  let retrieved: number[];
  if (effective === 'vector') retrieved = DATA.ranks[question].slice(0, k);
  else if (effective === 'path') retrieved = needed;
  else retrieved = DATA.docs.map((_, i) => i);

  const got = new Set(retrieved);
  const found = needed.filter((d) => got.has(d));
  const missing = needed.filter((d) => !got.has(d));
  const touched = new Set(retrieved.map((d) => DATA.docCommunity[d]));

  const pos = new Map(DATA.nodes.map((n) => [n[0], n]));
  const neutral = dark ? '#848c99' : '#9aa0a6';
  const miss = dark ? DIVERGING.dark.negative : DIVERGING.light.negative;

  const status = isGlobal
    ? `${retrieved.length} documents read, touching ${touched.size} of ${COMMUNITIES} communities (themes)`
    : `${retrieved.length} documents given, ${found.length} of ${needed.length} needed found${missing.length ? `, missing ${missing.join(', ')}` : ', all there'}`;

  const rows = DATA.docs.map((text, i) => {
    const rank = DATA.ranks[question].indexOf(i) + 1;
    return [i, text, rank, got.has(i) ? 'yes' : '-', needed.includes(i) ? 'yes' : '-'];
  });

  return (
    <VizPanel
      title="Vector search or graph, on 24 documents"
      hint="Each line is one document turned into a relation. Pick a multi-hop question and give a vector search 2 to 8 documents, then switch to path following, which walks the relations the question needs. At k = 4 vector search completes only 2 of the 8 questions, the row printed by code block 1. For the global question the measure is how many of the 5 communities the context touches."
      legend={[
        {label: 'given to the reader', color: dark ? '#e6e8ec' : '#1f2328'},
        {label: 'needed but missing', color: miss},
        {label: 'entity, coloured by community', color: seriesColor(0, dark)},
      ]}
      table={{columns: ['doc', 'text', 'vector rank', 'given', 'needed'], rows}}
      controls={
        <>
          <label className={s.control}>
            question
            <select className={s.select} value={question} onChange={(e) => setQuestion(Number(e.target.value))}>
              {DATA.questions.map((q, i) => (
                <option key={q[0]} value={i}>
                  {i === GLOBAL ? 'Global: ' : `Q${i + 1}: `}
                  {q[0].length > 62 ? `${q[0].slice(0, 60)}...` : q[0]}
                </option>
              ))}
            </select>
          </label>
          <label className={s.control}>
            method
            <select className={s.select} value={effective} onChange={(e) => setMethod(e.target.value as Method)}>
              <option value="vector">Vector search, top k</option>
              <option value="path" disabled={isGlobal}>
                Path following
              </option>
              <option value="community">Community reports (all)</option>
            </select>
          </label>
          <label className={s.control}>
            k
            <input
              type="range"
              min={2}
              max={8}
              step={1}
              value={k}
              disabled={effective !== 'vector'}
              onChange={(e) => setK(Number(e.target.value))}
            />
            <span className={s.value}>{k}</span>
          </label>
          <span className={s.value} aria-live="polite">
            {status}
          </span>
        </>
      }>
      <svg
        className={s.svg}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Entity graph of 24 documents. ${status}. Method ${effective}.`}>
        {DATA.edges.map(([a, b, relation, doc]) => {
          const p = pos.get(a)!;
          const q = pos.get(b)!;
          const on = got.has(doc);
          const lost = !on && needed.includes(doc) && !isGlobal;
          const colour = on ? (dark ? '#e6e8ec' : '#1f2328') : lost ? miss : neutral;
          return (
            <g key={`${a}-${b}-${relation}-${doc}`}>
              <line
                x1={p[1]}
                y1={p[2]}
                x2={q[1]}
                y2={q[2]}
                stroke={colour}
                strokeWidth={on ? 2.6 : lost ? 2.2 : 1}
                strokeDasharray={lost ? '5 4' : undefined}
                opacity={on || lost ? 1 : 0.45}
              />
              {(on || lost) && effective !== 'community' && (
                <text className={s.tick} x={(p[1] + q[1]) / 2} y={(p[2] + q[2]) / 2 - 3} textAnchor="middle" fill={colour}>
                  {relation}
                </text>
              )}
            </g>
          );
        })}
        {DATA.nodes.map(([name, x, y, c]) => (
          <g key={name}>
            <circle cx={x} cy={y} r={7} fill={seriesColor(c, dark)} stroke="var(--surface-raised)" strokeWidth={1.5} />
            <text className={s.tick} x={x} y={y + 19} textAnchor="middle">
              {name}
            </text>
          </g>
        ))}
      </svg>
      <p className={s.hint} style={{padding: '0.4rem 0 0'}}>
        At k = {k}, vector search completes {complete(k)} of {GLOBAL} multi-hop questions. For the global question it
        touches {new Set(DATA.ranks[GLOBAL].slice(0, k).map((d) => DATA.docCommunity[d])).size} of {COMMUNITIES} communities.
      </p>
    </VizPanel>
  );
}
