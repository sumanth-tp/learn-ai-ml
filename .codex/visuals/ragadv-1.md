# Track B, agent B2: labs for docs/genai/rag-advanced

All labs sit on `VizPanel` with `useDarkViz()`, colours from `palette.ts`, a `table` prop, no external dependencies and no
randomness at render time. Numbers that Python produced are embedded; the chapter code prints the same numbers.

## GraphRetrievalLab (chapter 01)

- Data: the 24 one-sentence documents of chapter block 1, the 22 entities and 28 relations the rule-based extractor builds from
  them, the five Louvain communities (networkx 3.6.1, seed 0, sizes 6, 5, 4, 4, 3), and for each of the 8 multi-hop questions plus
  the global question "What are the main themes across these documents?" the full ranking of the 24 documents by cosine
  similarity under all-MiniLM-L6-v2. Node positions are hand-placed, grouped by community.
- Controls: question select (8 multi-hop questions and the global one; default Q2, the Delmar Freight question); method select
  (vector search, path following, community reports; path following is disabled for the global question); k slider 2 to 8
  (default 4, active for vector search only).
- Drawn: the entity graph; edges whose document was given to the reader are thick with the relation name, needed documents that
  are missing are dashed red, the rest are faint. Nodes are coloured by community. Status line: documents given, needed found,
  missing ids. A line under the plot gives the number of multi-hop questions vector search completes at the chosen k and the
  communities the global question touches.
- Expected numbers (chapter block 1 and 2): questions with every needed document at k = 2 to 8: 0, 1, 2, 3, 4, 4, 6 of 8, so
  2 of 8 at the default k = 4. Path following returns exactly the needed documents for 8 of 8 (Q2: documents 3, 13, 14).
  Global question: vector search touches 2 of 5 communities at k = 3 and 5, 3 of 5 at k = 8; reading all community reports
  touches 5 of 5.
- Table: document id, text, vector rank for the chosen question, whether it was given, whether it is needed.
- Keyboard: native select and range inputs.
