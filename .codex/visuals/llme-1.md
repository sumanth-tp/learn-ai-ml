# Track B, agent E1: labs for docs/llm-engineering/01-adapting-models

All labs sit on `VizPanel` with `useDarkViz()`, colours from `palette.ts`, a `table` prop, no external dependencies and no
randomness at render time. Numbers that Python produced are embedded; the chapter code prints the same numbers.

## AdaptationDecisionLab (chapter 01)

- Controls: failure type select (knowledge gap, behaviour or format gap, reasoning gap, latency or cost; default behaviour or
  format); "facts change often" checkbox (default off); labelled examples select 0, 120, 500, 2000, 5000 (default 2000); requests
  per month slider as log10 3 to 7, step 0.05 (default 5, which is 100,000); long-prompt tokens slider 500 to 6000, step 100
  (default 3000); cached share of the long prompt slider 0 to 0.9, step 0.1 (default 0).
- Model (the chapter's block 3, synthetic prices): big model 3.0 per million input tokens and 15.0 per million output tokens; small
  model 0.3 and 1.5; cache reads cost 0.1 of the input price; 20 output tokens per request. Options: long prompt (prompt tokens from
  the slider, cached share from the slider, no fixed cost); RAG (1000 prompt tokens, fixed 600 per month); tuned small model (60
  prompt tokens, fixed 3000 / 12 + 8 * 120 = 1210 per month).
- Advice: the chapter's `advise` function, with the break-even set to the rounded long-prompt versus tuned-model break-even.
- Drawn: log-log lines of monthly cost against requests per month for the three options, a marker at the chosen volume, and the
  break-even volumes as a table row. Advice shown as an ordered list under the plot.
- Default result (100,000 requests, 3000 tokens, no caching): long prompt 930, RAG 930, tuned small model 1,215 units per month;
  break-even long prompt versus tuned 130,783 requests per month. Set the cached share to 0.9: long prompt 201 and break-even 616,718.
  At 1,000,000 requests: tuned 1,258, long prompt 9,300, RAG 3,900.
- Table: cost per month at 10,000, 100,000 and 1,000,000 requests, and the break-evens.
- Keyboard: native range, select and checkbox inputs.

## ChatTemplateMaskLab (chapter 02)

- Data: three tickets (account, billing, technical) from block 2 of the chapter, each rendered by the real SmolLM2 chat template
  in two variants (the template's default system prompt, and the system prompt "Route the ticket."). Per token: the decoded
  string and the negative log-likelihood the real SmolLM2-135M-Instruct assigns it (float32, CPU), embedded rounded to 5
  decimals. Regenerable from the chapter's block 2 tokenisation plus one forward pass of the model per conversation (log-softmax of the
  next-token logits at each position).
- Controls: ticket select; system prompt select; loss mask select with three modes: completion only (labels from the end of
  the generation prompt to the last token, which is what TRL builds for prompt-completion data), assistant markers (the same
  without the trailing newline, what a patched template with generation markers returns) and every token (position 0 has no
  prediction, so it never counts).
- Drawn: a wrapped strip of token chips; trained tokens outlined and tinted, masked tokens faded. A status line gives token
  count, trained count, mean loss over trained tokens and mean over every token.
- Table: position, token, status, loss on the token.
- Default result (account ticket, default system prompt, completion only): 52 tokens, 8 trained (15%), mean loss 4.7013 over
  the trained tokens, 4.5594 over every token. Markers mode gives 7 trained tokens. Custom system prompt: 41 tokens.
- Keyboard: native select inputs.

## LoraRankLab (chapter 03)

- Model: the SmolLM2-135M-Instruct dimensions from its config: hidden 576, intermediate 1536, 30 layers, key/value width 192
  (3 key/value heads of 64), base parameters 134,515,008. LoRA parameters per module are rank * (in + out); modules: q_proj 576
  to 576, k_proj and v_proj 576 to 192, o_proj 576 to 576, gate_proj and up_proj 576 to 1536, down_proj 1536 to 576.
- Controls: rank select 1, 2, 4, 8, 16, 32, 64, 128, 256 (default 8); target modules select "q,v", "q,k,v,o", "all linear"
  (default all linear); lora_alpha slider 1 to 256 (default 16); rsLoRA checkbox (scaling alpha / sqrt(r) instead of alpha / r);
  base precision select float32 or bfloat16 (default float32).
- Drawn: a bar of trainable parameters against the base, and two stacked memory bars: full fine-tuning with Adam in float32
  (16 bytes per parameter) and LoRA (frozen base at the chosen precision, plus adapter weights 4 bytes, gradients 4 bytes and
  Adam moments 8 bytes per trainable parameter). Estimates only: activations and overhead are not counted.
- Expected numbers (chapter block 3): rank 8, all linear: 2,442,240 trainable, 1.82% of the base; adapter weights 9.8 MB,
  gradients 9.8 MB, Adam moments 19.5 MB; frozen base 538 MB (float32) or 269 MB (bfloat16); full fine-tuning 2,152 MB; LoRA
  float32 total is 27% of full fine-tuning. Rank 64, all linear: 19,537,920 (14.52%). Rank 1, q and v: 57,600 (0.04%).
  Scaling at the defaults: 16 / 8 = 2.
- Table: trainable parameters for every rank at the chosen target modules.
- Keyboard: native select, range and checkbox inputs.

## DpoLossLab (chapter 04)

- Controls: beta slider 0.01 to 1, step 0.01 (default 0.1); log-ratio of the chosen answer, log pi_theta(y+) minus log pi_ref(y+),
  slider -10 to 10, step 0.1 (default 2.0); log-ratio of the rejected answer, same range (default -1.5).
- Model (chapter block 1): implicit reward r = beta * log-ratio; margin = r(chosen) - r(rejected); loss = -log sigmoid(margin);
  gradient weight = sigmoid(-margin); the loss gradient with respect to the chosen log-probability is -beta * weight and with
  respect to the rejected log-probability is +beta * weight.
- Drawn: the loss and the gradient weight against the log-ratio gap (chosen minus rejected, -25 to 25) at the current beta, with
  a marker at the current gap; two bars for the implicit rewards of the chosen and rejected answers.
- Table: for the current log-ratios, beta = 0.01, 0.05, 0.1, 0.5, 1.0 with margin, loss, gradient weight and the gap needed to
  reach loss 0.1.
- Default result: gap 3.5, rewards +0.200 and -0.150, margin 0.350, loss 0.5334, gradient weight 0.4134. Both log-ratios 0 gives
  loss 0.6931 (ln 2). beta 0.5 with the default log-ratios: margin 1.750, loss 0.1602, weight 0.1480.
- Keyboard: native range inputs.
