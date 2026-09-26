"""The custom benchmark: generation, splits, canary, loading and hashing.

The 150 items are generated from hand-written templates with a fixed seed so
the files are reproducible and reviewable. In a real team you would replace
the templates with anonymised production tickets that two people have
labelled; the file format and the loaders stay the same.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from modelsel.schemas import (
    BenchmarkItem,
    HumanLabel,
    Label,
    Priority,
    Sentiment,
    Split,
    TicketFields,
)

CANARY = "MODELSEL-CANARY-5f0c2b7e-8a41-4d7e-9c55-benchmark-do-not-train"
"""Embedded in every benchmark file. If a model can complete it, the files leaked."""

PRODUCTS = [
    "Aurora Desk Lamp",
    "Nimbus Headphones",
    "Cobalt Blender",
    "Terra Backpack",
    "Pulse Smartwatch",
    "Orbit Router",
    "Lumen E-Reader",
    "Vega Espresso Machine",
]

# fmt: off
NAMES = [
    "Priya", "Tom", "Amara", "Luis", "Mei", "Oliver", "Fatima", "Kenji", "Sofia", "Daniel",
    "Aisha", "Marco", "Hannah", "Ravi", "Chloe", "Ibrahim", "Elena", "Noah", "Zara", "Lukas",
    "Grace", "Omar", "Isla", "Arjun", "Nora", "Felix", "Leah", "Mateo", "Yuki", "Samuel",
]
# fmt: on

ACTIONS: dict[Label, str] = {
    Label.BILLING: "I have corrected the invoice and the adjusted amount will show on your next statement.",
    Label.REFUND: "I have issued a full refund to your original payment method within five working days.",
    Label.SHIPPING: "I have asked the courier to trace the parcel and will send you a tracking update within 24 hours.",
    Label.ACCOUNT_ACCESS: "I have sent a secure password reset link to the email address on your account.",
    Label.BUG_REPORT: "I have logged the fault with our engineering team and shared a workaround in the help centre article below.",
    Label.FEATURE_REQUEST: "I have passed your suggestion to the product team, who review requests every month.",
    Label.CANCELLATION: "I have cancelled your subscription and you will not be charged again.",
    Label.OTHER: "I have forwarded your message to the right team, who will reply within two working days.",
}

CLOSE = "Kind regards,\nSupport Team"


@dataclass(frozen=True)
class Template:
    label: Label
    text: str
    priority: Priority
    sentiment: Sentiment
    has_order: bool
    has_amount: bool
    tags: tuple[str, ...] = ()


T = Template
P, S, L = Priority, Sentiment, Label

# fmt: off
TEMPLATES: list[Template] = [
    T(L.BILLING, "Hi, my invoice for {order} shows {amount} but the {product} was on offer. Please fix the bill.", P.MEDIUM, S.NEUTRAL, True, True),
    T(L.BILLING, "Why was I charged {amount} for my {product}? The price on the website was lower. Order {order}.", P.MEDIUM, S.NEGATIVE, True, True),
    T(L.BILLING, "Could you send me a VAT invoice for order {order}? I bought a {product} for {amount}.", P.LOW, S.NEUTRAL, True, True),
    T(L.BILLING, "I was billed twice this month, {amount} each time, for the same {product}. This is not acceptable.", P.HIGH, S.NEGATIVE, False, True, ("ambiguous",)),
    T(L.REFUND, "The {product} from order {order} stopped working after two days. I want my {amount} back.", P.HIGH, S.NEGATIVE, True, True),
    T(L.REFUND, "I returned the {product} last week (order {order}). When will the refund of {amount} arrive?", P.MEDIUM, S.NEUTRAL, True, True),
    T(L.REFUND, "Charged twice for order {order}, please refund the duplicate {amount} payment for the {product}.", P.HIGH, S.NEGATIVE, True, True, ("ambiguous",)),
    T(L.REFUND, "Changed my mind about the {product}. It is unopened. Can I get a refund please?", P.LOW, S.NEUTRAL, False, False),
    T(L.SHIPPING, "My {product} (order {order}) was due on Monday and still has not arrived.", P.MEDIUM, S.NEGATIVE, True, False),
    T(L.SHIPPING, "Tracking for order {order} has said 'in transit' for nine days. Where is my {product}?", P.HIGH, S.NEGATIVE, True, False),
    T(L.SHIPPING, "Can you deliver the {product} to my office address instead? Order {order}.", P.LOW, S.NEUTRAL, True, False),
    T(L.SHIPPING, "The box arrived crushed and the {product} is damaged. I need this urgently for an event tomorrow.", P.URGENT, S.NEGATIVE, False, False, ("ambiguous",)),
    T(L.ACCOUNT_ACCESS, "I cannot log in. The password reset email never arrives.", P.HIGH, S.NEGATIVE, False, False, ("no_product",)),
    T(L.ACCOUNT_ACCESS, "My account is locked after too many attempts and I need to register my {product} warranty today.", P.URGENT, S.NEGATIVE, False, False),
    T(L.ACCOUNT_ACCESS, "How do I change the email address on my account? I no longer use the old one.", P.LOW, S.NEUTRAL, False, False, ("no_product",)),
    T(L.ACCOUNT_ACCESS, "Someone else seems to have logged into my account and ordered a {product}! Order {order}.", P.URGENT, S.NEGATIVE, True, False, ("security",)),
    T(L.BUG_REPORT, "The app crashes every time I pair my {product}. I have tried reinstalling.", P.MEDIUM, S.NEGATIVE, False, False),
    T(L.BUG_REPORT, "Firmware update 2.3 bricked my {product}. It will not turn on at all.", P.HIGH, S.NEGATIVE, False, False),
    T(L.BUG_REPORT, "Small thing: the {product} settings page shows the wrong time zone.", P.LOW, S.NEUTRAL, False, False),
    T(L.BUG_REPORT, "Checkout page throws an error when I try to buy a {product}. Tried two cards.", P.HIGH, S.NEGATIVE, False, False, ("ambiguous",)),
    T(L.FEATURE_REQUEST, "Love my {product}! Any chance of adding a dark mode to the companion app?", P.LOW, S.POSITIVE, False, False),
    T(L.FEATURE_REQUEST, "It would be great if the {product} could sync with my calendar.", P.LOW, S.POSITIVE, False, False),
    T(L.FEATURE_REQUEST, "Please offer the {product} in a left-handed version. Many of us would buy it.", P.LOW, S.NEUTRAL, False, False),
    T(L.CANCELLATION, "Please cancel my {product} care-plan subscription. I no longer need it.", P.MEDIUM, S.NEUTRAL, False, False),
    T(L.CANCELLATION, "Cancel order {order} immediately, I ordered the wrong {product}.", P.HIGH, S.NEUTRAL, True, False),
    T(L.CANCELLATION, "I am done with your service. Close my subscription and stop charging me {amount} a month.", P.HIGH, S.NEGATIVE, False, True, ("ambiguous", "no_product")),
    T(L.OTHER, "Do you have a physical shop in Manchester where I can try the {product}?", P.LOW, S.NEUTRAL, False, False),
    T(L.OTHER, "Just wanted to say the {product} is brilliant. Thanks to the team!", P.LOW, S.POSITIVE, False, False),
    T(L.OTHER, "Are you hiring for customer support roles?", P.LOW, S.NEUTRAL, False, False, ("no_product",)),
]
# fmt: on

SPLIT_SIZES: dict[Split, int] = {"dev": 40, "test": 80, "private": 30}
"""150 items. ``private`` is held out: scored only for release decisions."""


def action_for(label: Label) -> str:
    return ACTIONS[label]


def compose_reply(
    name: str,
    label: Label,
    product: str | None,
    order_id: str | None,
    *,
    parts: Iterable[str] = ("greeting", "ack", "action", "order", "close"),
    action_label: Label | None = None,
    filler_paragraphs: int = 0,
    close: str = CLOSE,
) -> str:
    """Build a reply from named parts. Candidate fakes and the calibration file share it."""
    chosen = set(parts)
    lines: list[str] = []
    if "greeting" in chosen:
        lines.append(f"Hi {name},")
    body: list[str] = []
    if "ack" in chosen:
        body.append(f"Thanks for getting in touch about your {product or 'account'}.")
    if "action" in chosen:
        body.append(action_for(action_label or label))
    if "order" in chosen and order_id:
        body.append(f"I have noted this against order {order_id}.")
    if body:
        lines.append(" ".join(body))
    for i in range(filler_paragraphs):
        lines.append(
            "We truly value you as a customer and we are always striving to improve every part "
            "of your experience with us, so please do not hesitate to reach out again at any time "
            f"if there is anything else at all that we can help you with{'' if i == 0 else ' today'}."
        )
    if "close" in chosen:
        lines.append(close)
    return "\n\n".join(lines)


def _render(template: Template, rng: random.Random, idx: int, split: Split) -> BenchmarkItem:
    name = NAMES[idx % len(NAMES)]
    product = None if "no_product" in template.tags else rng.choice(PRODUCTS)
    order_id = f"ORD-{rng.randint(10000, 99999)}" if template.has_order else None
    amount = round(rng.uniform(9, 480), 2) if template.has_amount else None
    ticket = template.text.format(
        order=order_id or "", amount=f"£{amount:.2f}" if amount is not None else "", product=product or ""
    )
    ticket = f"From: {name}\n\n{ticket}"
    tags = list(template.tags)
    if order_id is None:
        tags.append("no_order_id")
    fields = TicketFields(
        order_id=order_id,
        product=product,
        amount=amount,
        priority=template.priority,
        sentiment=template.sentiment,
    )
    return BenchmarkItem(
        id=f"{split}-{idx:03d}",
        split=split,
        customer_name=name,
        ticket=ticket,
        label=template.label,
        fields=fields,
        reference_reply=compose_reply(name, template.label, product, order_id),
        tags=tags,
    )


def build_items(seed: int = 7) -> list[BenchmarkItem]:
    """Stratified generation: every split sees every template family in proportion."""
    rng = random.Random(seed)
    items: list[BenchmarkItem] = []
    counter = 0
    for split, size in SPLIT_SIZES.items():
        order = list(range(len(TEMPLATES)))
        chosen: list[int] = []
        while len(chosen) < size:
            rng.shuffle(order)
            chosen.extend(order)
        for t_idx in chosen[:size]:
            items.append(_render(TEMPLATES[t_idx], rng, counter, split))
            counter += 1
    return items


def build_human_labels(items: list[BenchmarkItem], n: int = 40, seed: int = 11) -> list[HumanLabel]:
    """Simulated three-rater human labels over pairs of replies of known quality.

    Each reply is built from a random subset of parts; its hidden quality is the
    weighted share of parts present. Raters see the quality plus rater noise. Replace
    this file with your team's real ratings: the calibration code reads it unchanged.
    """
    rng = random.Random(seed)
    weights = {"greeting": 0.5, "ack": 1.0, "action": 2.0, "order": 0.5, "close": 0.5}
    authors = ["fake:frontier-large", "fake:balanced-mini", "fake:local-8b", "human-agent"]
    signatures = {"fake:frontier-large": "Warm regards,\nThe Support Team"}
    public = [it for it in items if it.split != "private"]
    rows: list[HumanLabel] = []

    def make(it: BenchmarkItem, author: str) -> tuple[str, float]:
        parts = [p for p in weights if rng.random() < 0.75]
        wrong_action = rng.random() < 0.15 and "action" in parts
        action_label = rng.choice([lb for lb in Label if lb != it.label]) if wrong_action else None
        filler = rng.choice([0, 0, 0, 1, 2])
        text = compose_reply(
            it.customer_name,
            it.label,
            it.fields.product,
            it.fields.order_id,
            parts=parts,
            action_label=action_label,
            filler_paragraphs=filler,
            close=signatures.get(author, CLOSE),
        )
        got = 0.0
        for part, w in weights.items():
            if part == "order" and it.fields.order_id is None:
                got += w  # nothing to cite, so nothing is missing
            elif part in parts and not (part == "action" and wrong_action):
                got += w
        return text, 1 + 4 * got / sum(weights.values())

    for k in range(n):
        it = public[rng.randrange(len(public))]
        a_author, b_author = rng.sample(authors, 2)
        reply_a, q_a = make(it, a_author)
        reply_b, q_b = make(it, b_author)
        ratings_a = [min(5, max(1, round(q_a + rng.gauss(0, 0.55)))) for _ in range(3)]
        ratings_b = [min(5, max(1, round(q_b + rng.gauss(0, 0.55)))) for _ in range(3)]
        diff = sum(ratings_a) - sum(ratings_b)
        preference = "A" if diff >= 2 else "B" if diff <= -2 else "tie"
        rows.append(
            HumanLabel(
                id=f"hl-{k:03d}",
                item_id=it.id,
                ticket=it.ticket,
                reference_reply=it.reference_reply,
                reply_a=reply_a,
                reply_b=reply_b,
                author_a=a_author,
                author_b=b_author,
                ratings_a=ratings_a,
                ratings_b=ratings_b,
                preference=preference,
            )
        )
    return rows


def split_path(data_dir: Path, split: Split) -> Path:
    if split == "private":
        return data_dir / "private" / "private_test.jsonl"
    return data_dir / "benchmark" / f"{split}.jsonl"


def write_benchmark(data_dir: Path, seed: int = 7) -> dict[str, int]:
    items = build_items(seed)
    counts: dict[str, int] = {}
    for split in SPLIT_SIZES:
        path = split_path(data_dir, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [it for it in items if it.split == split]
        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"_meta": {"canary": CANARY, "split": split, "seed": seed}}) + "\n")
            for it in rows:
                fh.write(it.model_dump_json() + "\n")
        counts[split] = len(rows)
    labels = build_human_labels(items)
    with (data_dir / "human_labels.jsonl").open("w", encoding="utf-8") as fh:
        for row in labels:
            fh.write(row.model_dump_json() + "\n")
    counts["human_labels"] = len(labels)
    return counts


class PrivateSplitLocked(PermissionError):
    """Raised when the private split is requested without MODELSEL_ALLOW_PRIVATE=true."""


def load_split(data_dir: Path, split: Split, *, allow_private: bool = False) -> list[BenchmarkItem]:
    if split == "private" and not allow_private:
        raise PrivateSplitLocked("the private split is locked; set MODELSEL_ALLOW_PRIVATE=true")
    path = split_path(data_dir, split)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `modelsel build-data` first")
    items: list[BenchmarkItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "_meta" in row:
            continue
        items.append(BenchmarkItem.model_validate(row))
    return items


def load_human_labels(data_dir: Path) -> list[HumanLabel]:
    path = data_dir / "human_labels.jsonl"
    return [HumanLabel.model_validate_json(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def dataset_hash(data_dir: Path, splits: Iterable[Split]) -> str:
    """Content hash of the split files: two runs are comparable only if this matches."""
    h = hashlib.sha256()
    for split in sorted(splits):
        h.update(split_path(data_dir, split).read_bytes())
    return h.hexdigest()[:16]
