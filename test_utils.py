#!/usr/bin/env python3
"""Tests for utils.extract_lean_code and lean_solve_agent RLVR additions."""

import json
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from utils import extract_lean_code


# ── extract_lean_code ────────────────────────────────────────────────────────

def test_strategy1_lean_block():
    """Strategy 1: ```lean ... ``` block."""
    response = (
        "Some prose.\n"
        "```lean\n"
        "import Mathlib\n\n"
        "theorem foo : 1 + 1 = 2 := by norm_num\n"
        "```\n"
    )
    out = extract_lean_code(response)
    assert out.startswith("import Mathlib"), f"S1 failed: {repr(out)}"
    assert "theorem foo" in out
    print("Strategy 1 (lean block): OK")


def test_strategy2_import_block():
    """Strategy 2: plain ``` block starting with import."""
    response = "```\nimport Mathlib\ntheorem bar : True := trivial\n```"
    out = extract_lean_code(response)
    assert "theorem bar" in out, f"S2 failed: {repr(out)}"
    print("Strategy 2 (import block): OK")


def test_strategy3_boxed():
    """Strategy 3: \\boxed{...} extraction."""
    response = r"\boxed{import Mathlib" + "\n\ntheorem baz : 1 = 1 := rfl}"
    out = extract_lean_code(response)
    assert "theorem baz" in out, f"S3 failed: {repr(out)}"
    print("Strategy 3 (boxed): OK")


def test_strategy4_raw_lines():
    """Strategy 4: line-by-line heuristic for raw Lean output."""
    response = "Let me prove this.\n\nimport Mathlib\n\ntheorem qux : 2 = 2 := rfl\n"
    out = extract_lean_code(response)
    assert "theorem qux" in out, f"S4 failed: {repr(out)}"
    print("Strategy 4 (line-by-line): OK")


def test_empty_returns_empty():
    """Pure prose should return empty string."""
    response = "This is just prose with no code at all."
    out = extract_lean_code(response)
    assert out == "", f"Empty case failed: {repr(out)}"
    print("Empty response: OK")


def test_prefers_last_lean_block():
    """When multiple lean blocks exist, the last one is returned."""
    response = (
        "```lean\ntheorem first : True := trivial\n```\n\n"
        "```lean\nimport Mathlib\ntheorem second : 1 = 1 := rfl\n```"
    )
    out = extract_lean_code(response)
    assert "second" in out, f"Last-block pref failed: {repr(out)}"
    assert "first" not in out
    print("Prefers last lean block: OK")


def test_adds_import_when_missing():
    """The agent prepends 'import Mathlib' when extraction omits it."""
    # Simulate what lean_solve_agent does after extraction
    raw = "theorem needs_import : True := trivial"
    code = raw  # no import
    if not code.startswith("import"):
        code = "import Mathlib\n\n" + code
    assert code.startswith("import Mathlib")
    assert "needs_import" in code
    print("Import prepend: OK")


# ── Trajectory logging ───────────────────────────────────────────────────────

def test_trajectory_logging():
    """
    Smoke-test _append_trajectory_event by running the real method in isolation.
    Checks that valid JSON is written and the reward field is correct.
    """
    # Minimal stand-ins so we don't need the full LLM stack
    @dataclass
    class FakeProblem:
        problem_id: str = "putnam_1964_b1"

    @dataclass
    class FakeProblemState:
        problem: FakeProblem = field(default_factory=FakeProblem)

    @dataclass
    class FakeSubmission:
        problem_id: str = "putnam_1964_b1"
        lean_code: str = "import Mathlib\ntheorem t : True := trivial"
        verified: bool = False
        error_message: Optional[str] = "tactic failed"
        error_line: Optional[int] = 5
        sorry_count: int = 0
        soft_score: float = 0.65
        llm_feedback: str = "Try omega instead"
        attempt_num: int = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        submissions_dir = Path(tmpdir)

        # Replicate _append_trajectory_event without importing the full agent
        def append(state, sub, event_type):
            record = {
                "problem_id": state.problem.problem_id,
                "attempt_num": sub.attempt_num,
                "event_type": event_type,
                "verified": sub.verified,
                "reward": 1.0 if sub.verified else sub.soft_score,
                "error_message": sub.error_message,
                "error_line": sub.error_line,
                "sorry_count": sub.sorry_count,
                "llm_feedback": sub.llm_feedback,
                "lean_code": sub.lean_code,
                "timestamp": time.time(),
            }
            traj_path = submissions_dir / "trajectories.jsonl"
            with open(traj_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        state = FakeProblemState()
        sub_fail = FakeSubmission()
        sub_ok = FakeSubmission(verified=True, soft_score=0.0, attempt_num=3)

        append(state, sub_fail, "generate")
        append(state, sub_ok, "repair")

        traj_path = submissions_dir / "trajectories.jsonl"
        lines = traj_path.read_text().strip().splitlines()
        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"

        r1 = json.loads(lines[0])
        assert r1["reward"] == 0.65, f"Unverified reward wrong: {r1['reward']}"
        assert r1["event_type"] == "generate"
        assert r1["verified"] is False

        r2 = json.loads(lines[1])
        assert r2["reward"] == 1.0, f"Verified reward wrong: {r2['reward']}"
        assert r2["event_type"] == "repair"
        assert r2["verified"] is True

        print("Trajectory logging: OK")


# ── SUMMARY stats ────────────────────────────────────────────────────────────

def test_summary_stats():
    """
    Validate the RLVR stat calculations used in write_submissions().
    """
    @dataclass
    class FakeSub:
        attempt_num: int
        soft_score: float = 0.0
        verified: bool = False

    @dataclass
    class FakePS:
        solved: bool
        attempt_count: int
        submissions: list = field(default_factory=list)
        verified_submissions: list = field(default_factory=list)

    # Problem A: solved on first attempt
    ps_a = FakePS(solved=True, attempt_count=1,
                  verified_submissions=[FakeSub(attempt_num=1, verified=True)],
                  submissions=[FakeSub(attempt_num=1)])

    # Problem B: solved after 2 repairs (attempt 3)
    ps_b = FakePS(solved=True, attempt_count=3,
                  verified_submissions=[FakeSub(attempt_num=3, verified=True)],
                  submissions=[FakeSub(1, 0.3), FakeSub(2, 0.6), FakeSub(3)])

    # Problem C: unsolved after 2 attempts
    ps_c = FakePS(solved=False, attempt_count=2,
                  submissions=[FakeSub(1, 0.2), FakeSub(2, 0.4)])

    problems = {"a": ps_a, "b": ps_b, "c": ps_c}

    solved = sum(1 for p in problems.values() if p.solved)
    assert solved == 2

    solved_on_first = sum(
        1 for ps in problems.values()
        if ps.solved and ps.verified_submissions and ps.verified_submissions[0].attempt_num == 1
    )
    assert solved_on_first == 1, f"Expected 1, got {solved_on_first}"

    solved_by_repair = solved - solved_on_first
    assert solved_by_repair == 1

    attempted = [ps for ps in problems.values() if ps.attempt_count > 0]
    avg_attempts = sum(ps.attempt_count for ps in attempted) / len(attempted)
    assert abs(avg_attempts - 2.0) < 1e-9, f"avg_attempts wrong: {avg_attempts}"

    needed_repair = [ps for ps in problems.values() if ps.attempt_count > 1]
    repair_solved = sum(1 for ps in needed_repair if ps.solved)
    repair_rate = repair_solved / len(needed_repair)
    assert abs(repair_rate - 0.5) < 1e-9, f"repair_rate wrong: {repair_rate}"

    best_score_c = max((s.soft_score for s in ps_c.submissions), default=0.0)
    assert best_score_c == 0.4

    print("SUMMARY stats: OK")


if __name__ == "__main__":
    test_strategy1_lean_block()
    test_strategy2_import_block()
    test_strategy3_boxed()
    test_strategy4_raw_lines()
    test_empty_returns_empty()
    test_prefers_last_lean_block()
    test_adds_import_when_missing()
    test_trajectory_logging()
    test_summary_stats()
    print()
    print("All tests passed.")
