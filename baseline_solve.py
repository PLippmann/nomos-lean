#!/usr/bin/env python3
"""
Baseline Lean 4 Proof Generation - Single Shot, No Harness.

This script measures raw model performance without any of the nomos-lean harness:
- No repair loop
- No LLM scoring/feedback  
- No multiple attempts
- Just single-shot generation + verification

Usage:
    python baseline_solve.py PutnamBench/ --time_limit_hours=8.0 --max_concurrent=8
"""

import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import fire
import httpx
from openai import AsyncOpenAI

from lean_verifier import LeanVerifier, VerificationResult
from putnam_bench import PutnamBenchLoader, LeanProblem


@dataclass
class BaselineSubmission:
    """A single baseline submission."""
    problem_id: str
    lean_code: str
    verified: bool = False
    error_message: Optional[str] = None


@dataclass  
class BaselineProblemState:
    """State for a single problem."""
    problem: LeanProblem
    submission: Optional[BaselineSubmission] = None
    attempted: bool = False


class BaselineSolveAgent:
    """
    Minimal baseline agent - single shot generation only.
    
    No repair loop, no scoring, no multiple attempts.
    """

    def __init__(
        self,
        problems_dir: str,
        submissions_dir: Optional[str] = None,
        time_limit_hours: float = 8.0,
        max_concurrent: int = 8,
        model: str = "deepseek-reasoner",
        base_url: str = "https://api.deepseek.com/v1",
        problems_filter: Optional[str] = None,
        problems_limit: Optional[int] = None,
        verification_timeout: float = 120.0,
    ):
        self.problems_dir = Path(problems_dir)
        self.max_concurrent = max_concurrent
        self.model = model
        self.verification_timeout = verification_timeout
        self.problems_filter = problems_filter
        self.problems_limit = problems_limit

        # Timing
        self.time_limit_seconds = time_limit_hours * 3600
        self.start_time: float = 0

        # Output directory
        if submissions_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            submissions_dir = f"baseline_submissions/{timestamp}"
        self.submissions_dir = Path(submissions_dir)

        # Use the SAME prompt as the main harness for fair comparison
        # The only difference is: no repair loop, no scoring, single attempt
        base_dir = Path(__file__).parent
        self.prompt_template = (base_dir / "prompts/lean_solve.md").read_text()

        # Initialize verifier
        self.verifier = LeanVerifier("lean_project")

        # Initialize LLM client
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(600, connect=60.0),
        )

        # State
        self.problems: dict[str, BaselineProblemState] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.stopping = False

        # Stats
        self.stats = {
            "total_attempts": 0,
            "verified_proofs": 0,
            "api_calls": 0,
        }

    def _time_remaining(self) -> float:
        return max(0, self.time_limit_seconds - (time.time() - self.start_time))

    def _should_stop(self) -> bool:
        return self.stopping or self._time_remaining() <= 0

    async def _call_llm(self, prompt: str) -> str:
        """LLM call with basic retry for rate limits."""
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            async with self.semaphore:
                self.stats["api_calls"] += 1
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=8192,
                    )
                    message = response.choices[0].message
                    
                    # deepseek-reasoner: content has final answer, reasoning_content has CoT
                    content = message.content
                    if content and content.strip():
                        return content
                    
                    # If content is empty, extract from reasoning_content
                    reasoning = getattr(message, 'reasoning_content', None)
                    if reasoning:
                        return reasoning
                    
                    print(f"  Warning: API returned empty content")
                    return ""
                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()
                    print(f"  API exception (attempt {attempt+1}): {e}")
                    if "rate" in error_str or "429" in error_str or "timeout" in error_str or "connection" in error_str:
                        wait_time = (2 ** attempt) * 5
                        print(f"  Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    return ""
        
        print(f"  Failed after {max_retries} retries. Last error: {last_error}")
        return ""

    def _extract_lean_code(self, response: str) -> str:
        """Extract Lean code from response with multiple fallback strategies."""
        # Strategy 1: Find ```lean ... ``` code blocks (prefer last one)
        lean_blocks = list(re.finditer(r'```lean\s*(.*?)```', response, re.DOTALL))
        if lean_blocks:
            code = lean_blocks[-1].group(1).strip()
            if code.startswith('import') or 'theorem' in code or 'lemma' in code:
                return code
        
        # Strategy 2: Find ``` ... ``` blocks starting with import
        import_blocks = list(re.finditer(r'```\s*(import.*?)```', response, re.DOTALL))
        if import_blocks:
            code = import_blocks[-1].group(1).strip()
            return code
        
        # Strategy 3: Extract from \boxed{} with proper brace matching
        boxed_start = response.find('\\boxed{')
        if boxed_start != -1:
            content_start = boxed_start + 7  # len('\\boxed{')
            depth = 1
            end_idx = len(response)
            for i, char in enumerate(response[content_start:]):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        end_idx = content_start + i
                        break
            code = response[content_start:end_idx].strip()
            if code.startswith('import') or 'theorem' in code or 'lemma' in code:
                return code
        
        # Strategy 4: Smart line-by-line extraction as last resort
        lines = response.split('\n')
        lean_lines = []
        in_code = False
        
        lean_keywords = (
            'import', 'open', 'theorem', 'lemma', 'def', 'example', 
            'section', 'namespace', 'set_option', 'variable', 'structure', 
            'class', 'instance', 'attribute', '@[', 'noncomputable', 
            'abbrev', 'inductive', '#check', '#eval'
        )
        
        # Patterns that indicate prose (NOT Lean code) - both upper and lowercase
        prose_starters = (
            'We ', 'The ', 'This ', 'Let ', 'Now ', 'Since ', 'By ', 
            'Note ', 'First', 'Second', 'Third', 'Finally', 'Therefore',
            'However', 'Thus', 'Hence', 'Consider', 'Given', 'Suppose',
            'In ', 'For ', 'To ', 'If ', 'When ', 'As ', 'From ',
            'Here ', 'Step ', 'Case ', 'Proof', 'Solution', 'Answer',
            '**', '##', '# ', '1.', '2.', '3.', '- ', '* ',
            # Lowercase prose starters
            'where ', 'and ', 'so ', 'then ', 'but ', 'which ', 'that ',
            'with ', 'using ', 'because ', 'therefore ', 'hence ',
        )
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines unless we're in code
            if not stripped:
                if in_code:
                    lean_lines.append(line)
                continue
            
            # Check if line starts with prose indicators
            is_prose = any(stripped.startswith(p) or stripped.lower().startswith(p) for p in prose_starters)
            
            # Check if line looks like natural language (more aggressive detection)
            if not is_prose and len(stripped) > 10:
                # Sentence-like structure
                if stripped[-1] in '.?!:' and ' ' in stripped:
                    words = stripped.split()
                    # More than 4 words and not starting with Lean keyword
                    if len(words) > 4 and not any(stripped.startswith(k) for k in lean_keywords):
                        # Check for common English patterns
                        common_words = ['is', 'are', 'the', 'of', 'and', 'or', 'to', 'in', 'that', 'which']
                        word_list = [w.lower().strip('.,;:') for w in words]
                        if any(w in common_words for w in word_list):
                            is_prose = True
            
            if is_prose:
                if in_code and lean_lines:
                    # Hit prose after code - stop here
                    break
                continue
            
            # Check for Lean keywords to start code block
            if stripped.startswith(lean_keywords) or stripped.startswith('--') or stripped.startswith('/-'):
                in_code = True
                lean_lines.append(line)
            elif in_code:
                lean_lines.append(line)
        
        if lean_lines:
            result = '\n'.join(lean_lines).strip()
            # Post-process: remove trailing placeholder patterns
            result = re.sub(r'\s*:=\s*\.\.\.\s*$', ' := by sorry', result)
            result = re.sub(r'\s+\.\.\.\s*$', '', result)
            return result
        
        # Return empty rather than prose
        return ""



    async def solve_single(self, problem_state: BaselineProblemState) -> bool:
        """Single-shot solve attempt. No retries."""
        if self._should_stop() or problem_state.attempted:
            return False

        problem = problem_state.problem
        problem_state.attempted = True
        self.stats["total_attempts"] += 1

        print(f"Attempting {problem.problem_id}...")

        # Generate proof
        prompt = self.prompt_template.replace("{informal_statement}", problem.informal_statement)
        prompt = prompt.replace("{lean_statement}", problem.lean_statement)
        
        response = await self._call_llm(prompt)
        if not response:
            print(f"  ✗ {problem.problem_id}: No response from API")
            return False

        lean_code = self._extract_lean_code(response)
        
        # Ensure imports
        if not lean_code.startswith("import"):
            lean_code = "import Mathlib\n\n" + lean_code

        # Verify
        result = await self.verifier.verify_proof(
            lean_code, 
            timeout_seconds=self.verification_timeout
        )

        submission = BaselineSubmission(
            problem_id=problem.problem_id,
            lean_code=lean_code,
            verified=result.success,
            error_message=result.error_message,
        )
        problem_state.submission = submission

        if result.success:
            print(f"  ✓ {problem.problem_id}: VERIFIED!")
            self.stats["verified_proofs"] += 1
            return True
        else:
            print(f"  ✗ {problem.problem_id}: {result.error_message[:60]}...")
            return False

    async def run(self):
        """Main entry point."""
        self.start_time = time.time()

        print("=" * 60)
        print("BASELINE: Single-Shot Lean Proof Generation")
        print("=" * 60)
        print(f"Model: {self.model}")
        print(f"Time limit: {self.time_limit_seconds/3600:.1f} hours")
        print(f"Max concurrent: {self.max_concurrent}")
        print(f"Verification timeout: {self.verification_timeout}s")
        print("NO repair loop, NO scoring, NO multiple attempts")
        print("=" * 60)
        print()

        # Load problems
        try:
            loader = PutnamBenchLoader(self.problems_dir)
            problems = loader.load_problems(
                filter_pattern=self.problems_filter,
                limit=self.problems_limit
            )
        except Exception as e:
            print(f"Failed to load problems: {e}")
            return

        for p in problems:
            self.problems[p.problem_id] = BaselineProblemState(problem=p)

        print(f"Loaded {len(self.problems)} problems")

        # Check Lean environment
        print("Checking Lean environment...")
        try:
            env_ok = await self.verifier.check_environment()
            if env_ok:
                print("✓ Lean environment ready")
            else:
                print("✗ Lean environment check failed")
                return
        except Exception as e:
            print(f"✗ Lean environment error: {e}")
            return

        print()
        print("Starting single-shot solving...")
        print()

        # Process all problems
        async def worker():
            for pid, pstate in self.problems.items():
                if self._should_stop():
                    break
                if not pstate.attempted:
                    await self.solve_single(pstate)

        # Run workers
        tasks = [asyncio.create_task(worker()) for _ in range(self.max_concurrent)]
        
        try:
            await asyncio.wait(tasks, timeout=self.time_limit_seconds)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.stopping = True
            for task in tasks:
                if not task.done():
                    task.cancel()

        # Write results
        self.write_results()

        elapsed = time.time() - self.start_time
        print()
        print("=" * 60)
        print("BASELINE RESULTS")
        print("=" * 60)
        solved = self.stats["verified_proofs"]
        total = len(self.problems)
        print(f"Solved: {solved}/{total} ({100*solved/total:.1f}%)")
        print(f"API calls: {self.stats['api_calls']}")
        print(f"Time: {elapsed/60:.1f} minutes")
        print("=" * 60)

    def write_results(self):
        """Write results to disk."""
        self.submissions_dir.mkdir(parents=True, exist_ok=True)

        for pid, pstate in self.problems.items():
            if pstate.submission:
                # Write Lean file
                lean_file = self.submissions_dir / f"{pid}.lean"
                lean_file.write_text(pstate.submission.lean_code)

        # Write summary
        summary_path = self.submissions_dir / "SUMMARY.md"
        solved = self.stats["verified_proofs"]
        total = len(self.problems)

        summary = f"# Baseline Results (No Harness)\n\n"
        summary += f"- **Model**: {self.model}\n"
        summary += f"- **Solved**: {solved}/{total} ({100*solved/total:.1f}%)\n"
        summary += f"- **API Calls**: {self.stats['api_calls']}\n"
        summary += f"- **Method**: Single-shot generation, no repairs\n\n"
        summary += "## Problems\n\n"

        for pid, pstate in sorted(self.problems.items()):
            if pstate.submission:
                status = "✓" if pstate.submission.verified else "✗"
                summary += f"- {status} `{pid}`\n"
            else:
                summary += f"- ⊘ `{pid}` (not attempted)\n"

        summary_path.write_text(summary)
        print(f"Summary written to {summary_path}")


def main(
    problems_dir: str,
    submissions_dir: Optional[str] = None,
    time_limit_hours: float = 8.0,
    max_concurrent: int = 8,
    model: str = "deepseek-reasoner",
    base_url: str = "https://api.deepseek.com/v1",
    problems_filter: Optional[str] = None,
    problems_limit: Optional[int] = None,
    verification_timeout: float = 120.0,
):
    """
    Run baseline single-shot proof generation (no harness).
    
    Args:
        problems_dir: Path to PutnamBench directory
        submissions_dir: Output directory for results
        time_limit_hours: Time limit in hours
        max_concurrent: Max parallel API requests
        model: LLM model name
        base_url: API endpoint
        problems_filter: Regex pattern to filter problem IDs
        problems_limit: Max problems to attempt
        verification_timeout: Timeout for each Lean verification
    """
    agent = BaselineSolveAgent(
        problems_dir=problems_dir,
        submissions_dir=submissions_dir,
        time_limit_hours=time_limit_hours,
        max_concurrent=max_concurrent,
        model=model,
        base_url=base_url,
        problems_filter=problems_filter,
        problems_limit=problems_limit,
        verification_timeout=verification_timeout,
    )
    asyncio.run(agent.run())


if __name__ == "__main__":
    fire.Fire(main)
