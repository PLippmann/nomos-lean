#!/usr/bin/env python3
"""
Lean 4 Formal Verification Agent for Olympiad Mathematics.

Refactored from nomos-1 to use Lean 4 type checking for deterministic proof verification
instead of LLM-based scoring.

Usage:
    python lean_solve_agent.py PutnamBench/ --time_limit_hours=1.0 --max_concurrent=8
"""

import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import fire
import httpx
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_delay,
    wait_exponential,
    retry_if_exception,
)

from lean_verifier import LeanVerifier, VerificationResult
from putnam_bench import PutnamBenchLoader, LeanProblem


# === Error Classes ===

class TokenLimitError(Exception):
    """Raised when request exceeds token limit."""
    pass


class IncompleteResponseError(Exception):
    """Raised when response has non-standard stop reason."""
    pass


def _is_token_limit_error(error: BaseException) -> bool:
    """Check if error is a token/context limit error."""
    error_str = str(error).lower()
    patterns = [
        "context_length_exceeded", "context length", "maximum context",
        "token limit", "too many tokens", "max_tokens", "reduce the length",
    ]
    return any(p in error_str for p in patterns)


def _should_retry(error: BaseException) -> bool:
    """Determine if we should retry this error."""
    if isinstance(error, (TokenLimitError, IncompleteResponseError)):
        return False
    if _is_token_limit_error(error):
        return False
    return True


# === Rich printing setup ===

def setup_rich():
    try:
        from rich.traceback import install
        install(show_locals=True, suppress=[asyncio])
        return True
    except Exception:
        return False


RICH_AVAILABLE = setup_rich()


def safeprint(*args, **kwargs):
    """Print with rich if available, fallback to regular print."""
    try:
        if RICH_AVAILABLE:
            from rich import print as rprint
            rprint(*args, **kwargs)
        else:
            print(*args, **kwargs)
    except Exception:
        try:
            print(*args, **kwargs)
        except Exception:
            pass


# === Data Classes ===

@dataclass
class LeanSubmission:
    """A single Lean proof submission."""
    problem_id: str
    lean_code: str
    verified: bool = False
    error_message: Optional[str] = None
    error_line: Optional[int] = None
    sorry_count: int = 0
    soft_score: float = 0.0
    llm_feedback: str = ""
    attempt_num: int = 0


@dataclass
class LeanProblemState:
    """State tracking for a single Lean problem."""
    problem: LeanProblem
    submissions: list[LeanSubmission] = field(default_factory=list)
    verified_submissions: list[LeanSubmission] = field(default_factory=list)
    failed_submissions: list[LeanSubmission] = field(default_factory=list)
    final_submission: Optional[LeanSubmission] = None
    solved: bool = False  # Flag for early termination
    attempt_count: int = 0


# === Main Agent ===

class LeanSolveAgent:
    """
    Async agent for solving Olympiad problems with Lean 4 formal verification.
    
    Workflow:
    1. Generate Lean proofs using LLM (DeepSeek V3)
    2. Verify proofs using `lake build` subprocess
    3. If verification fails, use LLM to score partial proof and generate feedback
    4. Repair loop: regenerate with error feedback
    5. First verified proof wins; parallel workers race to solve
    """

    def __init__(
        self,
        problems_dir: str,
        solve_prompt: str = "prompts/lean_solve.md",
        repair_prompt: str = "prompts/lean_repair.md",
        score_prompt: str = "prompts/lean_score.md",
        lean_project_dir: str = "lean_project",
        submissions_dir: Optional[str] = None,
        time_limit_hours: float = 3.0,
        max_concurrent: int = 8,
        max_repair_attempts: int = 3,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        problems_filter: Optional[str] = None,
        problems_limit: Optional[int] = None,
    ):
        """
        Initialize the Lean solve agent.
        
        Args:
            problems_dir: Path to PutnamBench repository
            solve_prompt: Path to Lean proof generation prompt
            repair_prompt: Path to proof repair prompt
            score_prompt: Path to partial proof scoring prompt
            lean_project_dir: Path to Lake project for verification
            submissions_dir: Output directory for successful proofs
            time_limit_hours: Total time limit
            max_concurrent: Max parallel workers
            max_repair_attempts: Max repair iterations per initial attempt
            model: LLM model name
            base_url: OpenAI-compatible API endpoint
            problems_filter: Regex filter for problem IDs
            problems_limit: Max problems to load
        """
        self.problems_dir = Path(problems_dir)
        self.lean_project_dir = Path(lean_project_dir)
        self.max_repair_attempts = max_repair_attempts
        self.problems_filter = problems_filter
        self.problems_limit = problems_limit

        # Timing
        self.time_limit_seconds = time_limit_hours * 3600
        self.early_stop_seconds = max(
            self.time_limit_seconds * 0.5,
            self.time_limit_seconds - (15 * 60)
        )
        self.max_concurrent = max_concurrent
        self.model = model

        # Output directory
        if submissions_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            submissions_dir = f"lean_submissions/{timestamp}"
        self.submissions_dir = Path(submissions_dir)

        # Load prompts
        base_dir = Path(__file__).parent
        self.solve_prompt_template = (base_dir / solve_prompt).read_text()
        self.repair_prompt_template = (base_dir / repair_prompt).read_text()
        self.score_prompt_template = (base_dir / score_prompt).read_text()

        # Initialize verifier
        self.verifier = LeanVerifier(lean_project_dir)

        # Initialize LLM client
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(600, connect=60.0),
        )

        # State
        self.problems: dict[str, LeanProblemState] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.start_time: float = 0
        self.stopping = False
        self.finalization_started = False
        self.solve_tasks: set[asyncio.Task] = set()
        self.round_robin_index: int = 0
        self.round_robin_lock = asyncio.Lock()

        # Stats
        self.stats = {
            "total_attempts": 0,
            "verified_proofs": 0,
            "api_calls": 0,
        }

    # === Timing ===

    def _time_remaining(self) -> float:
        return max(0, self.time_limit_seconds - (time.time() - self.start_time))

    def _should_stop_solving(self) -> bool:
        return self.stopping or (time.time() - self.start_time) >= self.early_stop_seconds

    # === LLM Calls ===

    @retry(
        stop=stop_after_delay(3 * 3600),
        wait=wait_exponential(multiplier=2, min=4, max=120),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    async def _call_llm(self, messages: list[dict]) -> str:
        """Call LLM with retries."""
        try:
            async with self.semaphore:
                self.stats["api_calls"] += 1
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=8192,
                )
                return response.choices[0].message.content or ""
        except Exception as e:
            if _is_token_limit_error(e):
                raise TokenLimitError(str(e)) from e
            raise

    def _extract_lean_code(self, response: str) -> str:
        """Extract Lean code from LLM response."""
        # Try to find code between ```lean ... ``` markers
        match = re.search(r'```lean\s*(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Try ``` ... ``` without language marker
        match = re.search(r'```\s*(import.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # If response starts with 'import', use as-is
        if response.strip().startswith('import'):
            return response.strip()
        
        # Fallback: return entire response
        return response.strip()

    def _extract_score_and_feedback(self, response: str) -> tuple[float, str]:
        """Extract score and feedback from scoring LLM response."""
        score = 0.0
        feedback = ""
        
        # Extract score: <score>X.X</score>
        score_match = re.search(r'<score>\s*([\d.]+)\s*</score>', response)
        if score_match:
            try:
                score = float(score_match.group(1))
                score = max(0.0, min(0.9, score))  # Clamp to valid range
            except ValueError:
                pass
        
        # Extract feedback: <feedback>...</feedback>
        feedback_match = re.search(r'<feedback>\s*(.*?)\s*</feedback>', response, re.DOTALL)
        if feedback_match:
            feedback = feedback_match.group(1).strip()
        
        return score, feedback

    # === Proof Generation ===

    async def generate_proof(self, problem_state: LeanProblemState) -> Optional[LeanSubmission]:
        """Generate a Lean proof for a problem."""
        if self._should_stop_solving() or problem_state.solved:
            return None

        problem = problem_state.problem
        
        try:
            prompt = self.solve_prompt_template.replace("{informal_statement}", problem.informal_statement)
            prompt = prompt.replace("{lean_statement}", problem.lean_statement)
            
            messages = [{"role": "user", "content": prompt}]
            response = await self._call_llm(messages)
            lean_code = self._extract_lean_code(response)
            
            # Ensure proper structure
            if not lean_code.startswith("import"):
                lean_code = f"{problem.imports}\n\n{lean_code}"
            
            problem_state.attempt_count += 1
            self.stats["total_attempts"] += 1
            
            return LeanSubmission(
                problem_id=problem.problem_id,
                lean_code=lean_code,
                attempt_num=problem_state.attempt_count,
            )
        except Exception as e:
            safeprint(f"[red]Error generating proof for {problem.problem_id}: {e}[/red]")
            return None

    # === Verification ===

    async def verify_submission(
        self, 
        problem_state: LeanProblemState, 
        submission: LeanSubmission
    ) -> bool:
        """Verify a Lean proof using lake build."""
        result = await self.verifier.verify_proof(submission.lean_code)
        
        submission.verified = result.success
        submission.error_message = result.error_message
        submission.error_line = result.error_line
        submission.sorry_count = result.sorry_count
        
        if result.success:
            safeprint(f"[bold green]✓ {problem_state.problem.problem_id}[/bold green] attempt {submission.attempt_num}: VERIFIED!")
            self.stats["verified_proofs"] += 1
        else:
            safeprint(f"[yellow]{problem_state.problem.problem_id}[/yellow] attempt {submission.attempt_num}: failed - {result.error_message[:50]}...")
        
        return result.success

    # === Scoring ===

    async def score_partial_proof(
        self, 
        problem_state: LeanProblemState, 
        submission: LeanSubmission
    ) -> tuple[float, str]:
        """Score a partial proof and generate feedback using LLM."""
        if submission.verified:
            return 1.0, "Proof verified successfully"
        
        problem = problem_state.problem
        
        try:
            prompt = self.score_prompt_template.replace("{informal_statement}", problem.informal_statement)
            prompt = prompt.replace("{lean_code}", submission.lean_code)
            prompt = prompt.replace("{error_message}", submission.error_message or "No error")
            prompt = prompt.replace("{sorry_count}", str(submission.sorry_count))
            
            messages = [{"role": "user", "content": prompt}]
            response = await self._call_llm(messages)
            score, feedback = self._extract_score_and_feedback(response)
            
            # Apply sorry penalty
            sorry_penalty = 0.05 * submission.sorry_count
            final_score = max(0.0, score - sorry_penalty)
            
            return final_score, feedback
        except Exception as e:
            safeprint(f"[yellow]Error scoring {problem.problem_id}: {e}[/yellow]")
            return 0.0, ""

    # === Repair ===

    async def repair_submission(
        self, 
        problem_state: LeanProblemState, 
        submission: LeanSubmission,
        feedback: str
    ) -> Optional[LeanSubmission]:
        """Generate a repaired proof using error feedback."""
        if self._should_stop_solving() or problem_state.solved:
            return None

        problem = problem_state.problem
        
        try:
            prompt = self.repair_prompt_template.replace("{informal_statement}", problem.informal_statement)
            prompt = prompt.replace("{lean_statement}", problem.lean_statement)
            prompt = prompt.replace("{previous_attempt}", submission.lean_code)
            prompt = prompt.replace("{error_message}", submission.error_message or "No error message")
            
            # Add LLM feedback if available
            llm_feedback_section = ""
            if feedback:
                llm_feedback_section = f"\n\n## LLM Analysis\n{feedback}"
            prompt = prompt.replace("{llm_feedback}", llm_feedback_section)
            
            messages = [{"role": "user", "content": prompt}]
            response = await self._call_llm(messages)
            lean_code = self._extract_lean_code(response)
            
            if not lean_code.startswith("import"):
                lean_code = f"{problem.imports}\n\n{lean_code}"
            
            problem_state.attempt_count += 1
            self.stats["total_attempts"] += 1
            
            return LeanSubmission(
                problem_id=problem.problem_id,
                lean_code=lean_code,
                attempt_num=problem_state.attempt_count,
            )
        except Exception as e:
            safeprint(f"[red]Error repairing {problem.problem_id}: {e}[/red]")
            return None

    # === Worker Logic ===

    async def solve_worker(self, problem_state: LeanProblemState) -> bool:
        """Main worker loop: generate, verify, repair."""
        if problem_state.solved:
            return True
        
        # Generate initial proof
        submission = await self.generate_proof(problem_state)
        if submission is None:
            return False
        
        problem_state.submissions.append(submission)
        
        # Verify
        success = await self.verify_submission(problem_state, submission)
        if success:
            problem_state.verified_submissions.append(submission)
            problem_state.solved = True
            return True
        
        # Score and get feedback for repair
        score, feedback = await self.score_partial_proof(problem_state, submission)
        submission.soft_score = score
        submission.llm_feedback = feedback
        
        # Repair loop
        for i in range(self.max_repair_attempts):
            if problem_state.solved or self._should_stop_solving():
                return problem_state.solved
            
            submission = await self.repair_submission(problem_state, submission, feedback)
            if submission is None:
                break
            
            problem_state.submissions.append(submission)
            
            if await self.verify_submission(problem_state, submission):
                problem_state.verified_submissions.append(submission)
                problem_state.solved = True
                return True
            
            # Get new feedback for next repair iteration
            score, feedback = await self.score_partial_proof(problem_state, submission)
            submission.soft_score = score
            submission.llm_feedback = feedback
        
        # Failed all repair attempts
        if submission:
            problem_state.failed_submissions.append(submission)
        return False

    # === Problem Selection ===

    async def _get_next_problem(self) -> Optional[LeanProblemState]:
        """Get the next unsolved problem to work on."""
        async with self.round_robin_lock:
            candidates = [p for p in self.problems.values() if not p.solved]
            if not candidates:
                return None
            
            # Prioritize problems with highest-scoring partial proofs
            candidates.sort(key=lambda p: (
                -max((s.soft_score for s in p.failed_submissions), default=0),
                p.problem.problem_id
            ))
            
            # Round-robin among top candidates
            self.round_robin_index = self.round_robin_index % len(candidates)
            selected = candidates[self.round_robin_index]
            self.round_robin_index += 1
            return selected

    # === Main Phases ===

    async def run_solving_phase(self):
        """Run the main solving phase with parallel workers."""
        safeprint(f"[bold blue]Starting solving phase with {self.max_concurrent} parallel workers...[/bold blue]")
        
        async def worker(worker_id: int):
            while not self._should_stop_solving():
                problem = await self._get_next_problem()
                if problem is None:
                    # All problems solved or no more work
                    await asyncio.sleep(1)
                    continue
                await self.solve_worker(problem)
                await asyncio.sleep(0.1)
        
        tasks = []
        for i in range(self.max_concurrent):
            task = asyncio.create_task(worker(i))
            self.solve_tasks.add(task)
            tasks.append(task)
        
        try:
            timeout = self.early_stop_seconds - (time.time() - self.start_time)
            if timeout > 0:
                done, pending = await asyncio.wait(tasks, timeout=timeout)
                for task in pending:
                    task.cancel()
        except Exception as e:
            safeprint(f"[yellow]Solving phase interrupted: {e}[/yellow]")
        
        self.stopping = True
        for task in self.solve_tasks:
            if not task.done():
                task.cancel()

    def _get_best_submission(self, problem_state: LeanProblemState) -> Optional[LeanSubmission]:
        """Get the best submission for a problem."""
        # Prefer verified
        if problem_state.verified_submissions:
            return problem_state.verified_submissions[0]
        
        # Then highest soft score
        all_subs = problem_state.submissions
        if all_subs:
            scored = [(s, s.soft_score) for s in all_subs]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[0][0]
        
        return None

    def write_submissions(self):
        """Write results to disk."""
        self.submissions_dir.mkdir(parents=True, exist_ok=True)
        
        for problem_state in self.problems.values():
            best = problem_state.final_submission or self._get_best_submission(problem_state)
            
            if best is None:
                continue
            
            # Write Lean file
            lean_file = self.submissions_dir / f"{problem_state.problem.problem_id}.lean"
            lean_file.write_text(best.lean_code)
            
            # Write summary
            summary_file = self.submissions_dir / f"{problem_state.problem.problem_id}.md"
            status = "✓ VERIFIED" if best.verified else f"✗ FAILED (score: {best.soft_score:.2f})"
            summary = f"# {problem_state.problem.problem_id}\n\n"
            summary += f"**Status**: {status}\n\n"
            summary += f"**Attempts**: {problem_state.attempt_count}\n\n"
            if best.error_message and not best.verified:
                summary += f"**Last Error**: {best.error_message}\n\n"
            summary += f"## Lean Code\n\n```lean\n{best.lean_code}\n```\n"
            summary_file.write_text(summary)
            
            safeprint(f"[green]Wrote {lean_file}[/green]")
        
        # Write overall summary
        summary_path = self.submissions_dir / "SUMMARY.md"
        solved = sum(1 for p in self.problems.values() if p.solved)
        total = len(self.problems)
        
        summary = f"# Lean Solve Agent Results\n\n"
        summary += f"- **Solved**: {solved}/{total} ({100*solved/total:.1f}%)\n"
        summary += f"- **Total Attempts**: {self.stats['total_attempts']}\n"
        summary += f"- **API Calls**: {self.stats['api_calls']}\n\n"
        summary += "## Problems\n\n"
        
        for pid, ps in sorted(self.problems.items()):
            status = "✓" if ps.solved else "✗"
            summary += f"- {status} `{pid}`: {ps.attempt_count} attempts\n"
        
        summary_path.write_text(summary)
        safeprint(f"[bold green]Summary written to {summary_path}[/bold green]")

    async def run(self):
        """Main entry point."""
        self.start_time = time.time()
        
        safeprint("[bold]Lean 4 Formal Verification Agent[/bold]")
        safeprint(f"Time limit: {self.time_limit_seconds/3600:.1f} hours")
        safeprint(f"Max concurrent: {self.max_concurrent}")
        safeprint(f"Repair attempts per proof: {self.max_repair_attempts}")
        safeprint("")
        
        # Load problems from PutnamBench
        try:
            loader = PutnamBenchLoader(self.problems_dir)
            problems = loader.load_problems(
                filter_pattern=self.problems_filter,
                limit=self.problems_limit
            )
        except Exception as e:
            safeprint(f"[red]Failed to load problems: {e}[/red]")
            return
        
        for p in problems:
            self.problems[p.problem_id] = LeanProblemState(problem=p)
        
        safeprint(f"[cyan]Loaded {len(self.problems)} problems[/cyan]")
        
        # Check Lean environment
        safeprint("[cyan]Checking Lean environment...[/cyan]")
        try:
            env_ok = await self.verifier.check_environment()
            if env_ok:
                safeprint("[green]✓ Lean environment ready[/green]")
            else:
                safeprint("[red]✗ Lean environment check failed. Run ./setup_lean.sh first.[/red]")
                return
        except Exception as e:
            safeprint(f"[red]✗ Lean environment error: {e}[/red]")
            safeprint("[yellow]Run ./setup_lean.sh to set up the Lean project[/yellow]")
            return
        
        safeprint("")
        
        try:
            await self.run_solving_phase()
            
            # Summary
            safeprint("")
            safeprint("[bold]Results:[/bold]")
            solved = sum(1 for p in self.problems.values() if p.solved)
            safeprint(f"  Solved: {solved}/{len(self.problems)}")
            safeprint(f"  Total attempts: {self.stats['total_attempts']}")
            safeprint(f"  API calls: {self.stats['api_calls']}")
            
        except Exception as e:
            safeprint(f"[red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()
        finally:
            self.write_submissions()
            elapsed = time.time() - self.start_time
            safeprint(f"\n[bold green]Done in {elapsed/60:.1f} minutes[/bold green]")


def main(
    problems_dir: str,
    solve_prompt: str = "prompts/lean_solve.md",
    repair_prompt: str = "prompts/lean_repair.md",
    score_prompt: str = "prompts/lean_score.md",
    lean_project_dir: str = "lean_project",
    submissions_dir: Optional[str] = None,
    time_limit_hours: float = 3.0,
    max_concurrent: int = 8,
    max_repair_attempts: int = 3,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com/v1",
    problems_filter: Optional[str] = None,
    problems_limit: Optional[int] = None,
):
    """
    Run the Lean 4 formal verification agent.
    
    Args:
        problems_dir: Path to PutnamBench directory (containing lean4/src/)
        solve_prompt: Path to proof generation prompt
        repair_prompt: Path to proof repair prompt
        score_prompt: Path to partial proof scoring prompt
        lean_project_dir: Path to Lake project for verification
        submissions_dir: Output directory for results
        time_limit_hours: Time limit in hours
        max_concurrent: Max parallel API requests
        max_repair_attempts: Max repair iterations per attempt
        model: LLM model name (default: deepseek-chat)
        base_url: API endpoint
        problems_filter: Regex pattern to filter problem IDs
        problems_limit: Max problems to attempt
    """
    agent = LeanSolveAgent(
        problems_dir=problems_dir,
        solve_prompt=solve_prompt,
        repair_prompt=repair_prompt,
        score_prompt=score_prompt,
        lean_project_dir=lean_project_dir,
        submissions_dir=submissions_dir,
        time_limit_hours=time_limit_hours,
        max_concurrent=max_concurrent,
        max_repair_attempts=max_repair_attempts,
        model=model,
        base_url=base_url,
        problems_filter=problems_filter,
        problems_limit=problems_limit,
    )
    asyncio.run(agent.run())


if __name__ == "__main__":
    fire.Fire(main)
