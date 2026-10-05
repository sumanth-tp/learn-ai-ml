# Codex B: distributed learning labs (2026-10-05)

All three labs use VizPanel, useDarkViz and palette colours. Native labelled range inputs work by keyboard. SVGs have a responsive viewBox and textual titles; data tables carry every plotted value. Deterministic arithmetic, no external dependencies or random draws. Defaults mirror independent Python blocks in their chapters.

## StaleGradientLab

Controls: delay 0–5 (default 2), learning rate 0.05–0.8 by 0.05 (default 0.4), steps 5–40 (default 20). Objective 0.5w², initial w=1, use history[max(0, step-delay)] for each update. Draw current and fresh-gradient weights against accepted updates, with zero reference; data table shows update, delayed weight/loss and fresh weight/loss. Defaults reproduce the delay=2 output in stale.py. Early unavailable history uses the initial weight, explicitly stated. Delay is logical updates, not seconds.

## GradientCompressionLab

Controls: signed bits 2–16 (default 8), transmissions 1–12 (default 6), error feedback on/off (default on). Fixed vector [0.12,-0.7,1.5,-2,0.01,0.8,-0.4,0.3]; scale max(abs(corrected))/(2^(bits-1)-1); nearest magnitude rounds halves upwards, then restores sign. With feedback carry corrected minus decoded payload. Draw desired cumulative vector vs sent cumulative vector; table includes residual. Default one-step max error 0.007087, six-step residual norm 0.013588, raw fp32 32 bytes, ideal packed 8-bit payload 8 bytes plus one fp32 scale =12 bytes. Packed byte count ceil(8*bits/8); this is a payload model, no real network compression.

## LocalSgdLab

Controls: period 1–8 (default 4), learning rate 0.02–0.2 by 0.02 (default 0.1), total local steps 8–48 (default 24). Two equal-weight quadratic objectives with curvatures [1,4], minima [-1,2], initial global w=0. Train independently then average; partial final round is included. Draw local weights through every step and post-round global values; table gives round end, local models, average, central optimum 1.4 and excess objective. Default 6 rounds; Python local.py supplies model and excess loss.
