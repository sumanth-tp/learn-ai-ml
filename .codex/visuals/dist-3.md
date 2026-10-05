# Codex B: federated learning labs (2026-10-05)

Both labs use VizPanel, useDarkViz, palette colours, labelled native controls, responsive SVGs and data tables. Arithmetic is deterministic and mirrored in Python. Practice pages use boards and existing mechanisms via links; no additional lab names are assigned.

## FedAvgLab

Controls: client one examples 10–400 by 10 (default 100), client two examples 10–400 by 10 (default 300), first model -1–1 by 0.05 (default 0.4), second model -1–1 by 0.05 (default 0.8). Draw two model values and the weighted average on a number line. Table shows counts, normalised weights, model and weighted contribution. Default (100*0.4+300*0.8)/400=0.700000, unweighted mean=0.600000. No client data, privacy, or training outcome is inferred from this arithmetic.

## NonIidLab

Controls: heterogeneity gap 0–3 by 0.25 (default 1), period 1–8 (default 4), learning rate 0.02–0.2 by 0.02 (default 0.1). Fixed budget 24 local steps; equal-weight quadratics h=[1,4], minima [-gap,gap]. Central optimum=0.6*gap. Draw global model vs communication round and central optimum; table shows both clients before averaging, global model, disagreement and excess loss. Default 6 rounds; excess loss 0.035284. Gap zero starts exactly at both optima; gap here is target heterogeneity, not a full simulation of class-distribution non-IID data.
