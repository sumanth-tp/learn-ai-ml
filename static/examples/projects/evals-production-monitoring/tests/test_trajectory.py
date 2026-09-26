import pytest

from agentmon.evals.trajectory import (
    ExpectedCall,
    args_match,
    detect_loop,
    step_efficiency,
    tool_call_accuracy,
    trajectory_match,
)

BAL = {"name": "get_balance", "args": {"account_id": "ACC-1001"}}
TX = {"name": "list_transactions", "args": {"account_id": "ACC-1001", "limit": 5}}
FAQ = {"name": "search_help_center", "args": {"query": "fees"}}
E_BAL = ExpectedCall(name="get_balance", args={"account_id": "ACC-1001"})
E_TX = ExpectedCall(name="list_transactions", args={"account_id": "ACC-1001"})


def test_args_match_normalises_case_whitespace_and_numbers() -> None:
    assert args_match({"account_id": "acc-1001 "}, {"account_id": "ACC-1001"})
    assert args_match({"amount": 50}, {"amount": 50.0, "reference": "x"})
    assert not args_match({"amount": 50}, {"amount": 51})
    assert not args_match({"limit": 3}, {})


def test_tool_call_accuracy_counts_name_and_args_one_to_one() -> None:
    assert tool_call_accuracy([E_BAL, E_TX], [TX, BAL]) == 1.0
    assert tool_call_accuracy([E_BAL, E_BAL], [BAL]) == 0.5
    wrong = {"name": "get_balance", "args": {"account_id": "ACC-1002"}}
    assert tool_call_accuracy([E_BAL], [wrong]) == 0.0
    assert tool_call_accuracy([], []) == 1.0
    assert tool_call_accuracy([], [BAL]) == 0.0


@pytest.mark.parametrize(
    ("mode", "actual", "expected_result"),
    [
        ("exact", [BAL, TX], True),
        ("exact", [TX, BAL], False),
        ("exact", [BAL, FAQ, TX], False),
        ("in_order", [BAL, FAQ, TX], True),
        ("in_order", [TX, BAL], False),
        ("unordered", [TX, BAL], True),
        ("unordered", [TX, BAL, FAQ], False),
        ("superset", [FAQ, TX, BAL], True),
        ("superset", [BAL], False),
        ("subset", [BAL], True),
        ("subset", [BAL, FAQ], False),
    ],
)
def test_trajectory_modes(mode: str, actual: list, expected_result: bool) -> None:
    assert trajectory_match([E_BAL, E_TX], actual, mode) is expected_result  # type: ignore[arg-type]


def test_trajectory_can_ignore_args() -> None:
    other = {"name": "get_balance", "args": {"account_id": "ACC-9"}}
    assert not trajectory_match([E_BAL], [other], "exact")
    assert trajectory_match([E_BAL], [other], "exact", with_args=False)


def test_step_efficiency() -> None:
    assert step_efficiency([E_BAL], [BAL]) == 1.0
    assert step_efficiency([E_BAL], [BAL, BAL, BAL, BAL]) == 0.25
    assert step_efficiency([], []) == 1.0


def test_loop_detection() -> None:
    assert detect_loop([BAL, BAL, BAL])
    assert not detect_loop([BAL, BAL])
    assert detect_loop([BAL, TX, BAL, TX])
    assert not detect_loop([BAL, TX, FAQ])
