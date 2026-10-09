# Labs: interactive visualisations

A lab lets the reader move one thing and watch another change. Build one for every concept in a chapter that moves: a threshold, a size, a rate, a weight, a schedule. A chapter needs at least one (the gate fails without it). There are about 150 labs in `src/components/viz/` already. Read two before you write one: `ChunkSplitLab.tsx` (data computed in Python, shipped as a `.ts` module) and `AvailabilityBudgetLab.tsx` (maths computed in the browser).

## Decide what the lab shows

Write one sentence: "Move X and watch Y, because Z." If you cannot write it, the concept does not need a lab, or you have picked the wrong control. Good controls change the reader's mind about something: chunk size against retrieval precision, a threshold against false positives, a sampling temperature against repetition. A lab that only animates a definition is decoration.

The lab's default settings must reproduce a number the chapter prints. The reader should see that number in the prose and then find it in the lab.

## Files

| File | What goes in it |
| --- | --- |
| `src/components/viz/<Name>Lab.tsx` | The component, built on `VizPanel` |
| `src/components/viz/<topic>Math.ts` | Pure functions for the maths, no React, so they can be checked |
| `src/components/viz/<topic>Data.ts` | Precomputed data when the maths needs Python (model outputs, real splits, embeddings). Generate it with a script that writes the file; never type numbers in by hand. |

Never edit `VizPanel*`, `palette.ts`, `CourseLab*`, `CodeWalkthrough*` or another agent's lab. Create the lab file before any chapter imports it. A page that imports a missing file breaks the whole build.

## The component

```tsx
import {useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export default function ThresholdLab() {
  const dark = useDarkViz();
  const [threshold, setThreshold] = useState(0.5);
  return (
    <VizPanel
      title="What one threshold does to precision and recall"
      hint="Each dot is one email. Move the threshold and count what lands on each side."
      controls={...}
      legend={[{label: 'spam', color: seriesColor(1, dark)}]}
      table={{columns: ['threshold', 'precision', 'recall'], rows: [...]}}>
      <svg role="img" aria-label="..." viewBox="0 0 640 260">...</svg>
    </VizPanel>
  );
}
```

`VizPanel` props: `title`, `hint`, `controls`, `legend` (label, colour, note), `table` (columns and rows, the accessible data view), `children` (the drawing). Colours come only from `palette.ts` (`seriesColor`, `sequentialColor`, `SERIES`, `DIVERGING`), through `useDarkViz()`, so the lab works in both themes.

Rules: deterministic (a fixed seed for any randomness), keyboard operable (native `<input type="range">`, `<select>` and buttons), an `aria-label` on every control and on the SVG, and a `table` that holds the same numbers as the picture.

## Cross-check the maths against Python

1. Put the formula in `<topic>Math.ts` as a pure function.
2. Compute the same quantity in the chapter's Python for three settings, including the default.
3. Compare. A short Node check is enough:

   ```bash
   npx -y tsx -e "import {f} from './src/components/viz/<topic>Math.ts'; console.log(f(0.5), f(0.7), f(0.9));"
   ```

4. The numbers must agree to the printed precision. If they do not, one of them is wrong. Find out which before going on.

## Embedding in the chapter

```mdx
import ThresholdLab from '@site/src/components/viz/ThresholdLab';

<ThresholdLab />

**What each control does.**

- **threshold** sets the score above which an email is marked spam, from 0.05 to 0.95.

**Try it yourself.**

1. Leave the default, 0.50. Precision is 0.83 and recall 0.71, the numbers the experiment printed above.
2. Set 0.80. Precision rises to 0.95 and recall falls to 0.40: fewer emails are flagged, and almost all of them are spam.
3. Set 0.20. Recall reaches 0.96, but one in three flagged emails is genuine. This is the setting for a filter that only moves mail into a folder, where a mistake is cheap.
```

Each experiment says what to set, what to watch, and why it changes, with numbers you saw in the lab yourself. (The numbers above show the format and are not from a real run.)

## Verify

```bash
npx tsc --noEmit
```

Then open the page in a browser (`npx docusaurus start`, or the isolated build in `.codex/AGENTS.md` section 9). Move every control, use the keyboard only once, switch to dark mode, and check the page at 390 px wide. Watch the console for errors. If you could not do the browser pass, say so in the report.
