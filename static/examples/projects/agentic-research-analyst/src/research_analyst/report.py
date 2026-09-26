"""Assemble the final report from verified claims: numbering, references, gaps, metrics."""

from __future__ import annotations

from research_analyst.graph.state import ResearchState
from research_analyst.models import (
    FinalReport,
    Gap,
    Reference,
    ReportMetrics,
    ReportSection,
    Usage,
)


def build_report(state: ResearchState) -> FinalReport:
    plan = state["plan"]
    sources = state.get("sources") or {}
    verified = state.get("verified") or []
    kept = [v for v in verified if v.status != "unsupported"]

    # references are numbered by first appearance, and only for sources actually cited
    numbers: dict[str, int] = {}
    for claim in kept:
        for sid in claim.citations:
            numbers.setdefault(sid, len(numbers) + 1)
    references = [
        Reference(
            number=n,
            source_id=sid,
            title=sources[sid].title,
            url=sources[sid].url,
            origin=sources[sid].origin,
            quality=sources[sid].quality,
        )
        for sid, n in numbers.items()
        if sid in sources
    ]

    sections = []
    headings = {k: v.heading for k, v in (state.get("sections") or {}).items()}
    for sq in plan.sub_questions:
        claims = [c for c in kept if c.sub_question_id == sq.id]
        if claims:
            sections.append(
                ReportSection(
                    sub_question_id=sq.id, heading=headings.get(sq.id, sq.question), claims=claims
                )
            )

    gaps = list(state.get("gaps") or [])
    gapped = {g.sub_question_id for g in gaps}
    covered = {s.sub_question_id for s in sections}
    for sq in plan.sub_questions:
        if sq.id not in covered and sq.id not in gapped:
            gaps.append(
                Gap(
                    sub_question_id=sq.id,
                    question=sq.question,
                    kind="no_evidence",
                    reason="no claim survived verification",
                )
            )

    all_checks = [ck for v in verified for ck in v.checks]
    precision = (
        sum(1 for ck in all_checks if ck.level != "no_support") / len(all_checks)
        if all_checks
        else 0.0
    )
    crit = state.get("critique")
    notes = list(dict.fromkeys(state.get("notes") or []))
    degraded = any(n.startswith("budget") for n in notes) or any(
        g.kind in ("timeout", "error", "budget") for g in gaps
    )
    metrics = ReportMetrics(
        claims_drafted=len(verified),
        claims_kept=len(kept),
        claims_removed=len(verified) - len(kept),
        claim_support_rate=round(
            sum(1 for v in verified if v.status == "supported") / len(verified), 3
        )
        if verified
        else 0.0,
        citation_precision=round(precision, 3),
        revisions=state.get("revisions", 0),
        critic_score=crit.overall if crit else None,
    )
    report = FinalReport(
        thread_id=state.get("thread_id", ""),
        question=state["question"],
        title=plan.title,
        sections=sections,
        references=references,
        gaps=gaps,
        degraded=degraded,
        degradation_notes=notes,
        metrics=metrics,
        usage=state.get("usage") or Usage(),
        markdown="",
    )
    return report.model_copy(update={"markdown": render_markdown(report, numbers)})


def render_markdown(report: FinalReport, numbers: dict[str, int]) -> str:
    lines = [f"# {report.title}", "", f"> {report.question}", ""]
    if report.degraded:
        lines += ["> **Degraded run.** Some work was skipped or failed; see Gaps and Notes.", ""]
    for sec in report.sections:
        lines += [f"## {sec.heading}", ""]
        for c in sec.claims:
            refs = "".join(f"[{numbers[sid]}]" for sid in c.citations if sid in numbers)
            flag = " *(partially supported)*" if c.status == "partial" else ""
            lines.append(f"- {c.text} {refs}{flag}")
        lines.append("")
    if report.gaps:
        lines += ["## Gaps", ""]
        lines += [f"- **{g.question}** ({g.kind}): {g.reason}" for g in report.gaps]
        lines.append("")
    if report.degradation_notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in report.degradation_notes] + [""]
    lines += ["## References", ""]
    lines += [
        f"{r.number}. {r.title}. {r.url} ({r.origin.value}, quality {r.quality:.2f})"
        for r in report.references
    ]
    m, u = report.metrics, report.usage
    lines += [
        "",
        "---",
        "",
        f"*Claims kept {m.claims_kept}/{m.claims_drafted}, citation precision "
        f"{m.citation_precision:.2f}, revisions {m.revisions}, "
        f"{u.total_tokens} tokens, ${u.cost_usd:.4f}.*",
        "",
    ]
    return "\n".join(lines)
