# Recommender lab specifications

## FeedbackConfidenceLab
- Controls: implicit interaction count 0–5; alpha 1–5. Defaults: count 2, alpha 2, preference 1, confidence 5, matching chapter code.
- Drawing: observation versus inferred preference and confidence; unobserved does not mean dislike.
- Table: count, binary preference, confidence and interpretation for counts 0–5.

## MatrixFactorLab
- Controls: user first factor 0–3 in 0.5 steps; second fixed 1. Item A factors (2,1), B (0,2). Default user (1,1), scores A=3, B=2.
- Drawing: bars for dot product scores; no claim these hand-set factors were fitted.
- Table: item factors, per-dimension products, total score.

## CandidateRecallLab
- Controls: candidate count 1–4; fixed scores A .9, B .8, C .6, D .4 with relevant set {B,D}. Default k=2 gives recall 1/2.
- Drawing: ordered retrieval list with relevant items marked and cutoff.
- Table: item, score, relevance, retrieved; labels available, no ANN performance claim.

## SlateDiversityLab
- Controls: diversity bonus 0–0.4 in 0.1 steps; three items A(topic X,.9), B(X,.8), C(Y,.7). Default bonus 0.2 changes slate A,B to A,C under greedy second-position rule; unique topics 2 versus 1.
- Drawing: score-first and reranked top-two slates, with deterministic tie break favouring C at equal final score.
- Table: base relevance, topic, second-position adjusted score and selected item. The bonus is an illustrative rule, not a learned utility or causal metric.
