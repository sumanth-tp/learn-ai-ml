# Boards: redrawing slides and whiteboards as images

On this site, "infographics" means board-style SVG images: dashed colour-coded groups, pastel cards with a bold title and monospace lines, labelled arrows, tables. Mermaid does not count for a slide or whiteboard from a source. Video frames and screenshots are never published. Every board is drawn from scratch with the kit and checked by eye.

## The kit

- `scripts/infographics/board.py`: the primitives. `Board(w, h, title, subtitle)`, then `group`, `card`, `cylinder`, `diamond`, `pill`, `text`, `person`, `bar`, `table`, `arrow(p1, p2, via=[...], label=...)`. Every shape returns a `Box` with `.top() .bottom() .left() .right()` anchors for arrows. Colours: `blue`, `green`, `orange`, `purple`, `red`, `teal`, `yellow`, `pink`, `grey`, `dark`.
- One script per track: `scripts/infographics/<track>.py`, with a `@board("file-name")` decorator, a `BOARDS` dict and a `__main__` that renders all boards or one by name. Copy the top of `scripts/infographics/llme_1.py`.
- Output: `static/img/<track>/<name>.svg`.
- Placement in the chapter, after the frontmatter import `import Infographic from '@site/src/components/Infographic';`:

  ```mdx
  <Infographic
    src="/img/<track>/<name>.svg"
    alt="What the board shows, in one sentence"
    caption="Where to look first"
  />
  ```

  `scripts/infographics/place.py` (`Doc.insert_after`, `Doc.replace_mermaid`) places tags by anchor text when you are editing many.

Do not edit `board.py` or another track's script. If you need a primitive, write a helper inside your own script.

## From a frame to a board

1. **Read the complete frame.** For a slide that builds up, use its last state. Grab it at full size (`yt_pack.py grab`).
2. **List what it says** before drawing: the groups, the items in each, the arrows and their labels, any numbers. This list, not the picture, is what you redraw.
3. **Fix it while you redraw.** A typo, a wrong arrow or an outdated name is corrected on the board and explained in the prose with a `:::note Correction`.
4. **Keep its teaching structure.** If the slide groups three things in a row, keep three in a row. If the colours encode categories, keep the encoding with the kit's colours.
5. **Every number on a board equals a printed number** in the chapter or in the source.

Add a board for a mechanism explained only in words, at most one per major idea, and caption it as an explanatory board. The chapter shape asks for a big-picture board near the top and a step-by-step board by the worked example. These are usually explanatory boards.

## Layout rules that avoid the usual defects

- Width 1000 to 1200 px. Height whatever the content needs.
- Text 12 px or more. Monospace characters are about 0.6 em wide, so a 12 px line of 40 characters needs about 290 px plus padding. Size cards from their longest line.
- Leave 30 px between cards so arrows and their labels have room. Route arrows around cards with `via=[...]`, never through them.
- At most about 12 cards per board. Split a dense slide into two boards rather than shrinking text.
- Put labels on arrows that carry meaning ("embeds", "top 4", "on error").

## Look at every board

```bash
python3 scripts/infographics/<track>.py <name>
node scripts/source-import/render_svg.mjs static/img/<track>/<name>.svg /tmp/<name>.png 1
```

The renderer exits with code 3 and lists any text outside the viewBox. Open the PNG and check: no clipped text, no text overlapping a card edge, no arrow crossing a card or a label, readable at 100 per cent, and the same reading order as the source. Fix and re-render until it is clean. Then open the page in a browser at desktop width and at 390 px. Boards scale down, so text that is fine at 1100 px can be unreadable at phone width. If it is, split the board.

## Captions and alt text

- `alt`: what the board shows, for someone who cannot see it ("Four stages of a RAG pipeline: load, split, embed, retrieve, with the chunk count at each").
- `caption`: where to look first ("Start at the left: each stage's output is the next stage's input").
- Never "Redrawn from the instructor's slide" and never a timestamp. The source line at the top of the chapter is the only reference to the source.
