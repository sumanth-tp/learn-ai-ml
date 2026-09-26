from ragate.pii import find_pii, leaked_pii, redact
from ragate.text import content_tokens, coverage, sentences, token_f1


def test_sentences_skip_headings_and_split_on_full_stops() -> None:
    text = "# Title\n\nFirst rule is 25 days. Second rule applies.\nThird one?"
    assert sentences(text) == ["First rule is 25 days.", "Second rule applies.", "Third one?"]


def test_content_tokens_drop_stopwords_and_stem() -> None:
    assert content_tokens("The days are approved") == ["day", "approv"]


def test_coverage_is_share_of_needle_words() -> None:
    assert coverage("carry over five days", "you can carry over days") == 0.75
    assert coverage("", "anything") == 0.0


def test_token_f1_symmetry_and_bounds() -> None:
    assert token_f1("25 days of leave", "25 days of leave") == 1.0
    assert token_f1("nothing shared", "completely different") == 0.0


def test_find_and_redact_pii() -> None:
    text = "Call +44 20 7946 0321 or mail priya.raman@fernhill-analytics.example, NI QQ 12 34 56 C"
    kinds = {k for k, _ in find_pii(text)}
    assert {"PHONE", "EMAIL", "NI_NUMBER"} <= kinds
    cleaned = redact(text)
    assert "7946" not in cleaned and "@" not in cleaned and "[NI_NUMBER]" in cleaned


def test_leak_ignores_pii_the_user_supplied() -> None:
    assert leaked_pii("We will email a@b.example", "my email is a@b.example") == []
    assert leaked_pii("Her email is c@d.example", "what is her email?") == [("EMAIL", "c@d.example")]
