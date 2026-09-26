"""The end-to-end model-selection run.

    benchmark -> candidates -> judge -> calibration -> statistics
              -> contamination -> decision -> report -> store

One function, ``run_selection``, drives every stage so the CLI, the API and the
tests exercise exactly the same path. Adding a model to ``models.toml`` and
re-running costs only the new model's calls: everything else is a cache hit.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import asdict
from typing import Any

import numpy as np

from modelsel.config import Settings
from modelsel.contamination import completion_probe
from modelsel.dataset import dataset_hash, load_human_labels, load_split, split_path, write_benchmark
from modelsel.decision import apply_gates, pareto_frontier, recommend, weighted_matrix
from modelsel.evaluate import macro_f1_ci, per_slice, score_items, summarise
from modelsel.harness.cache import ResponseCache
from modelsel.harness.client import LLMCallError, LLMClient
from modelsel.harness.runner import run_candidates
from modelsel.judge.calibration import calibrate
from modelsel.judge.judge import Judge
from modelsel.llm.registry import Catalogue, load_catalogue
from modelsel.logging_setup import log_event
from modelsel.report import render_html, render_markdown
from modelsel.schemas import BenchmarkItem, ItemScore, Prediction, Split
from modelsel.stats import (
    bootstrap_ci,
    holm,
    mcnemar,
    minimum_detectable_effect,
    n_paired_means,
    n_paired_proportions,
    paired_bootstrap,
    permutation_test,
)
from modelsel.store import RunStore
from modelsel.tasks import PROMPT_VERSION

logger = logging.getLogger(__name__)

GAP_THRESHOLD = 0.08
"""Flag a model whose public-minus-private quality gap exceeds the median model's by this much."""


def ensure_data(settings: Settings) -> None:
    if not split_path(settings.data_dir, "test").exists():
        counts = write_benchmark(settings.data_dir, seed=settings.seed)
        log_event(logger, "benchmark_built", **counts)


def _memorised(catalogue: Catalogue, settings: Settings, candidates: list[str]) -> dict[str, dict[str, BenchmarkItem]]:
    """Contaminated fakes 'trained on' the splits named in their profile."""
    out: dict[str, dict[str, BenchmarkItem]] = {}
    for mid in candidates:
        spec = catalogue.spec(mid)
        if spec.fake and spec.fake.contaminated_on:
            seen: dict[str, BenchmarkItem] = {}
            for split in spec.fake.contaminated_on:
                for it in load_split(settings.data_dir, split, allow_private=True):  # type: ignore[arg-type]
                    seen[it.ticket] = it
            out[mid] = seen
    return out


async def judge_replies(
    judge: Judge, items: list[BenchmarkItem], preds: list[Prediction]
) -> dict[tuple[str, str], float | None]:
    by_id = {it.id: it for it in items}
    replies = [p for p in preds if p.task == "reply"]

    async def one(p: Prediction) -> tuple[tuple[str, str], float | None]:
        if p.error or not p.output.strip():
            return (p.model_id, p.item_id), 1.0
        it = by_id[p.item_id]
        try:
            res = await judge.pointwise(it.ticket, p.output, it.reference_reply)
        except LLMCallError:
            return (p.model_id, p.item_id), None
        return (p.model_id, p.item_id), res.score

    return dict(await asyncio.gather(*(one(p) for p in replies)))


async def pairwise_vs_baseline(
    judge: Judge,
    items: list[BenchmarkItem],
    preds: list[Prediction],
    baseline: str,
    candidates: list[str],
    resamples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Win rate of each candidate's reply against the baseline's, swap-and-averaged."""
    reply = {(p.model_id, p.item_id): p.output for p in preds if p.task == "reply"}
    out: dict[str, dict[str, float]] = {}
    for cand in candidates:
        if cand == baseline:
            continue
        results = await asyncio.gather(
            *(
                judge.pairwise(
                    it.ticket, reply.get((cand, it.id), ""), reply.get((baseline, it.id), ""), it.reference_reply
                )
                for it in items
            )
        )
        ci = bootstrap_ci([r.score_a for r in results], resamples=resamples, seed=seed)
        out[cand] = {
            "win_rate": ci.point,
            "low": ci.low,
            "high": ci.high,
            "consistency": float(np.mean([r.consistent for r in results])),
        }
    return out


def paired_tests(scores: list[ItemScore], reference_model: str, settings: Settings) -> dict[str, dict[str, Any]]:
    """Each candidate vs the reference model on the same items; Holm across candidates."""
    by = {(s.model_id, s.item_id): s for s in scores}
    ref_items = sorted(s.item_id for s in scores if s.model_id == reference_model)
    out: dict[str, dict[str, Any]] = {}
    for m in sorted({s.model_id for s in scores} - {reference_model}):
        ids = [i for i in ref_items if (m, i) in by]
        a = [by[(m, i)].composite for i in ids]
        b = [by[(reference_model, i)].composite for i in ids]
        boot = paired_bootstrap(a, b, resamples=settings.bootstrap_resamples, seed=settings.seed)
        perm = permutation_test(a, b, resamples=settings.permutation_resamples, seed=settings.seed)
        mc = mcnemar([by[(m, i)].label_correct for i in ids], [by[(reference_model, i)].label_correct for i in ids])
        out[m] = {
            "vs": reference_model,
            "n": len(ids),
            "composite_diff": boot.mean_diff,
            "diff_low": boot.ci.low,
            "diff_high": boot.ci.high,
            "p_bootstrap": boot.p_value,
            "p_permutation": perm.p_value,
            "mcnemar_only_candidate": mc.only_a_correct,
            "mcnemar_only_reference": mc.only_b_correct,
            "p_mcnemar": mc.p_value,
        }
    if out:
        keep = holm({m: v["p_permutation"] for m, v in out.items()})
        for m, v in out.items():
            v["significant_holm"] = keep[m]
    return out


def sample_size(scores: list[ItemScore], a: str, b: str, settings: Settings, mde: float) -> dict[str, Any]:
    by = {(s.model_id, s.item_id): s for s in scores}
    ids = sorted(i for (m, i) in by if m == a and (b, i) in by)
    if len(ids) < 5:
        return {}
    diffs = np.array([by[(a, i)].composite - by[(b, i)].composite for i in ids])
    sd = float(diffs.std(ddof=1)) or 1e-6
    disc = float(np.mean([by[(a, i)].label_correct != by[(b, i)].label_correct for i in ids]))
    out: dict[str, Any] = {
        "pair": [a, b],
        "n_items": len(ids),
        "sd_of_differences": sd,
        "discordant_rate": disc,
        "target_effect": mde,
        "n_needed_composite": n_paired_means(mde, sd),
        "mde_at_current_n": minimum_detectable_effect(len(ids), sd),
    }
    if disc >= mde > 0:
        out["n_needed_accuracy"] = n_paired_proportions(mde, disc)
    return out


async def check_contamination(
    client: LLMClient, model_ids: list[str], items: list[BenchmarkItem], scores: list[ItemScore]
) -> dict[str, dict[str, Any]]:
    """Completion probe on public and private items, plus the public-private quality gap.
    The gap is judged relative to the median model: everyone may find private items a
    little harder, only a contaminated model finds them much harder."""
    test_items = [it for it in items if it.split == "test"]
    priv_items = [it for it in items if it.split == "private"]
    out: dict[str, dict[str, Any]] = {}
    for m in model_ids:
        probe_test = await completion_probe(client, m, test_items[:20], split="test")
        entry: dict[str, Any] = {"probe_test": probe_test.mean_overlap, "probe_flagged": probe_test.flagged}
        priv = [s for s in scores if s.model_id == m and s.split == "private"]
        if priv and priv_items:
            probe_priv = await completion_probe(client, m, priv_items[:20], split="private")
            pub_q = float(np.mean([s.composite for s in scores if s.model_id == m and s.split == "test"]))
            priv_q = float(np.mean([s.composite for s in priv]))
            entry |= {
                "probe_private": probe_priv.mean_overlap,
                "test_quality": pub_q,
                "private_quality": priv_q,
                "gap": pub_q - priv_q,
            }
        out[m] = entry
    gaps = [v["gap"] for v in out.values() if "gap" in v]
    if gaps:
        median_gap = float(np.median(gaps))
        for v in out.values():
            if "gap" in v:
                v["gap_flagged"] = v["gap"] - median_gap > GAP_THRESHOLD
    return out


async def run_selection(
    settings: Settings,
    store: RunStore,
    *,
    include_private: bool = False,
    run_id: str | None = None,
    candidates: list[str] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    ensure_data(settings)
    catalogue = load_catalogue(settings.models_file)
    profile = catalogue.profiles[settings.profile]
    model_ids = candidates or profile.candidates
    splits: list[Split] = ["test", "private"] if include_private else ["test"]
    run_id = run_id or f"run-{uuid.uuid4().hex[:10]}"
    if store.get_run(run_id) is None:
        store.create_run(run_id, settings.profile, list(splits))
    store.set_status(run_id, "running")

    cache = ResponseCache(settings.cache_path)
    client = LLMClient(settings, catalogue, cache, memorised=_memorised(catalogue, settings, model_ids))
    try:
        items = [it for s in splits for it in load_split(settings.data_dir, s, allow_private=settings.allow_private)]
        dhash = dataset_hash(settings.data_dir, splits)
        log_event(logger, "run_started", run_id=run_id, models=model_ids, items=len(items), dataset_hash=dhash)

        preds = await run_candidates(client, run_id, model_ids, items)
        store.save_predictions(preds)

        judge = Judge(client, profile.judge)
        reply_scores = await judge_replies(judge, items, preds)
        scores = score_items(items, preds, reply_scores, catalogue.decision)
        store.save_scores(run_id, scores)

        test_items = [it for it in items if it.split == "test"]
        test_scores = [s for s in scores if s.split == "test"]
        test_preds = [p for p in preds if p.split == "test"]
        summaries = summarise(
            test_scores, test_items, test_preds, resamples=settings.bootstrap_resamples, seed=settings.seed
        )
        gold = {it.id: it.label.value for it in test_items}
        f1_ci = {
            m.model_id: macro_f1_ci(
                [s for s in test_scores if s.model_id == m.model_id],
                gold,
                resamples=settings.bootstrap_resamples,
                seed=settings.seed,
            )
            for m in summaries
        }
        pairwise = await pairwise_vs_baseline(
            judge, test_items, test_preds, profile.baseline, model_ids, settings.bootstrap_resamples, settings.seed
        )

        rows = load_human_labels(settings.data_dir)[: settings.judge_samples_for_calibration]
        calib = await calibrate(
            client,
            catalogue,
            profile.judge,
            profile.meta_judge,
            rows,
            spot_checks=settings.meta_judge_spot_checks,
            seed=settings.seed,
        )

        tests_vs_baseline = paired_tests(test_scores, profile.baseline, settings)
        contamination = await check_contamination(client, model_ids, items, scores)
        flagged = [m for m, v in contamination.items() if v.get("probe_flagged") or v.get("gap_flagged")]
        blocked = {m: "contamination suspected (see Contamination)" for m in flagged}

        apply_gates(summaries, catalogue.decision, blocked)
        ranked = sorted(summaries, key=lambda m: -m.quality)
        pvals: dict[tuple[str, str], float] = {}
        for i, a in enumerate(ranked):
            for b in ranked[i + 1 :]:
                pair = paired_tests(
                    [s for s in test_scores if s.model_id in {a.model_id, b.model_id}], b.model_id, settings
                )
                pvals[(a.model_id, b.model_id)] = pair[a.model_id]["p_permutation"]
        rec = recommend(summaries, catalogue.decision, p_values_vs=pvals, blocked=blocked)
        contenders = [m for m in ranked if m.passes][:2]
        size = (
            sample_size(
                test_scores,
                contenders[0].model_id,
                contenders[1].model_id,
                settings,
                catalogue.decision.min_detectable_effect,
            )
            if len(contenders) == 2
            else {}
        )

        caveats = list(rec.caveats) + calib.caveats
        if flagged:
            caveats.append(
                f"Possible benchmark contamination: {', '.join(flagged)}. Excluded from the recommendation; "
                "compare private-split numbers before trusting any public score for them."
            )
        if rec.model_id and catalogue.spec(rec.model_id).family == catalogue.spec(profile.judge).family:
            caveats.append(
                f"The recommended model shares the judge's family ({catalogue.spec(profile.judge).family}); "
                "re-score replies with a judge from another family before signing off."
            )
        if not include_private:
            caveats.append(
                "The private held-out split was not scored; run with --include-private before a release decision."
            )
        if size and size["mde_at_current_n"] > catalogue.decision.min_detectable_effect:
            caveats.append(
                f"{size['n_items']} test items can only detect composite differences of about "
                f"{size['mde_at_current_n']:.3f}; about {size['n_needed_composite']} are needed for "
                f"{catalogue.decision.min_detectable_effect:.2f}."
            )

        summary: dict[str, Any] = {
            "run_id": run_id,
            "profile": settings.profile,
            "splits": list(splits),
            "dataset_hash": dhash,
            "prompt_version": PROMPT_VERSION,
            "judge": profile.judge,
            "meta_judge": profile.meta_judge,
            "baseline": profile.baseline,
            "n_test_items": len(test_items),
            "models": [asdict(m) | {"macro_f1_ci": list(f1_ci[m.model_id]), "passes": m.passes} for m in summaries],
            "pareto": pareto_frontier([m for m in summaries if m.passes]),
            "matrix": weighted_matrix([m for m in summaries if m.passes], catalogue.decision),
            "weights": catalogue.decision.weights,
            "gates": {
                "min_json_validity": catalogue.decision.min_json_validity,
                "max_p95_latency_ms": catalogue.decision.max_p95_latency_ms,
                "max_cost_per_1k_tickets_usd": catalogue.decision.max_cost_per_1k_tickets_usd,
            },
            "pairwise_vs_baseline": pairwise,
            "tests_vs_baseline": tests_vs_baseline,
            "slices": {
                "ambiguous": per_slice(test_scores, "ambiguous"),
                "no_order_id": per_slice(test_scores, "no_order_id"),
            },
            "calibration": calib.to_dict(),
            "sample_size": size,
            "contamination": contamination,
            "recommendation": {
                "model_id": rec.model_id,
                "runner_up": rec.runner_up,
                "reason": rec.reason,
                "significant": rec.significant,
            },
            "caveats": caveats,
            "billed_usd": client.billed_usd,
            "cache_entries": cache.count(),
            "duration_s": round(time.perf_counter() - t0, 2),
        }
        md = render_markdown(summary)
        html = render_html(summary)
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        (settings.reports_dir / f"{run_id}.md").write_text(md, encoding="utf-8")
        (settings.reports_dir / f"{run_id}.html").write_text(html, encoding="utf-8")
        (settings.reports_dir / "latest.md").write_text(md, encoding="utf-8")
        (settings.reports_dir / "latest.html").write_text(html, encoding="utf-8")
        store.set_status(run_id, "succeeded", dataset_hash=dhash, summary=summary, report_md=md, report_html=html)
        log_event(
            logger,
            "run_finished",
            run_id=run_id,
            recommended=rec.model_id,
            billed_usd=round(client.billed_usd, 6),
            duration_s=summary["duration_s"],
        )
        return summary
    except Exception as exc:
        store.set_status(run_id, "failed", error=repr(exc))
        log_event(logger, "run_failed", logging.ERROR, run_id=run_id, error=repr(exc))
        raise
    finally:
        cache.close()
