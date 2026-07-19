from examples.claudecode_ags.rewards.tool_loop_penalty import (
    adjust_episode_reward,
    episode_reward,
    longest_consecutive_tool_signature_run,
)


def _turn(tool: str, value: str) -> str:
    return (
        "<|im_start|>assistant\n"
        f"<tool_call><function={tool}><parameter=command>{value}</parameter></function></tool_call>"
        "<|im_end|>"
    )


def test_three_identical_turn_signatures_trigger_penalty() -> None:
    response = _turn("Bash", "pytest -q") * 3
    longest = longest_consecutive_tool_signature_run([response])
    assert longest == 3


def test_different_or_tool_free_turn_breaks_run() -> None:
    response = _turn("Bash", "pytest -q") * 2
    response += "<|im_start|>assistant\nI will inspect the failure.<|im_end|>"
    response += _turn("Bash", "pytest -q") * 2
    assert longest_consecutive_tool_signature_run([response]) == 2


def test_parallel_tool_order_is_ignored_but_parameters_are_not() -> None:
    one = (
        "<tool_call><function=Read><parameter=file_path>a.py</parameter></function></tool_call>"
        "<tool_call><function=Read><parameter=file_path>b.py</parameter></function></tool_call>"
    )
    two = (
        "<tool_call><function=Read><parameter=file_path>b.py</parameter></function></tool_call>"
        "<tool_call><function=Read><parameter=file_path>a.py</parameter></function></tool_call>"
    )
    three = (
        "<tool_call><function=Read><parameter=file_path>c.py</parameter></function></tool_call>"
        "<tool_call><function=Read><parameter=file_path>a.py</parameter></function></tool_call>"
    )
    response = "".join(
        f"<|im_start|>assistant\n{x}<|im_end|>" for x in (one, two, three)
    )
    assert longest_consecutive_tool_signature_run([response]) == 2


def test_timeout_outcome_reward_levels_and_tool_loop_priority() -> None:
    assert episode_reward(resolved=True, timed_out=False, tool_loop_penalized=False) == 1.0
    assert episode_reward(resolved=True, timed_out=True, tool_loop_penalized=False) == 0.5
    assert episode_reward(resolved=False, timed_out=False, tool_loop_penalized=False) == 0.0
    assert episode_reward(resolved=False, timed_out=True, tool_loop_penalized=False) == 0.0
    for resolved in (False, True):
        for timed_out in (False, True):
            assert episode_reward(
                resolved=resolved,
                timed_out=timed_out,
                tool_loop_penalized=True,
            ) == -1.0


def test_shared_adjustment_keeps_default_or_applies_full_policy() -> None:
    repeated = [_turn("Bash", "pytest -q") * 3]
    reward, details, audit = adjust_episode_reward(
        base_reward=1.0,
        reward_details={"mode": "binary"},
        resolved=True,
        exit_code=-1,
        responses=repeated,
        timeout_outcome_enabled=False,
        tool_loop_enabled=False,
    )
    assert reward == 1.0
    assert details == {"mode": "binary"}
    assert audit["tool_loop_penalized"] is False
    assert audit["tool_loop_detection_enabled"] is False

    reward, details, audit = adjust_episode_reward(
        base_reward=1.0,
        reward_details={"mode": "binary"},
        resolved=True,
        exit_code=-1,
        responses=repeated,
        timeout_outcome_enabled=True,
        tool_loop_enabled=True,
    )
    assert reward == -1.0
    assert details["base_reward_before_outcome_adjustment"] == 1.0
    assert audit["consecutive_tool_signature_max"] == 3
    assert audit["tool_loop_penalized"] is True
    assert audit["tool_loop_detection_enabled"] is True
    assert audit["agent_timed_out"] is True
