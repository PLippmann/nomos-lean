# CLAUDE.md — Nomos-Lean Codebase Guide

This file provides context for AI assistants working in this repository.

## Project Overview

**Nomos-Lean** is a Lean 4 formal verification agent for Olympiad mathematics. It generates machine-verifiable proofs for competition problems (Putnam, IMO) using LLMs, then validates them with `lake build` for **deterministic 100% correctness guarantees** — replacing probabilistic LLM scoring with formal type checking.

**Benchmark results** (first 100 PutnamBench problems, `deepseek-reasoner`):
- Baseline (single-shot, no repair): 13/100 (13%)
- Agent with repair loop: 58/100 (58%)

## Repository Structure

```
nomos-lean/
├── lean_solve_agent.py     # Main agent: generate → verify → repair loop (841 lines)
├── lean_verifier.py        # Async subprocess wrapper for `lake env lean` (278 lines)
├── baseline_solve.py       # Single-shot baseline without repair loop (482 lines)
├── putnam_bench.py         # PutnamBench dataset loader and parser (280 lines)
├── solve_agent.py          # Alternative solver with judging/consolidation (780 lines)
├── prompts/
│   ├── lean_solve.md       # Proof generation prompt template
│   ├── lean_repair.md      # Error-based repair prompt template
│   ├── lean_score.md       # Partial proof scoring prompt (0.0–1.0)
│   ├── consolidation.md    # Consolidation prompt
│   ├── pairwise.md         # Pairwise comparison prompt
│   └── score.md            # General scoring prompt
├── lean_project/           # Lake project for Lean verification
│   ├── lakefile.toml       # Lake v2.0 manifest (depends on mathlib4)
│   ├── lean-toolchain      # Lean version: leanprover/lean4:v4.27.0-rc1
│   ├── LeanProject.lean    # Root module
│   └── Proofs/             # Temporary generated proof files (runtime)
├── problems/               # Problem datasets
│   ├── debug_problems/     # 3 test problems for development
│   ├── putnam-2024/        # Putnam 2024 a & b sets
│   ├── putnam-2025/        # Putnam 2025 a & b sets (6 problems each)
│   └── imo-2025/           # IMO 2025 problems
├── runbooks/               # Bash scripts for experiment runs
│   ├── run_debug.sh        # Quick test (3 problems, ~5 min)
│   └── run_putnam_*.sh     # Full benchmark runs (3+ hours)
├── submissions/            # Output: generated proofs per run
├── requirements.txt        # Python dependencies
├── setup_lean.sh           # Lean/elan environment setup script
└── .env.example            # API key template
```

**External dependency (submodule/clone)**: [PutnamBench](https://github.com/trishullab/PutnamBench) — 672 Lean 4 competition problems. Expected at `PutnamBench/lean4/src/*.lean`.

## Core Architecture

### Workflow

```
PutnamBench (.lean files)
    → PutnamBenchLoader: parse theorem + informal statement
    → LeanSolveAgent.generate_proof(): LLM generates Lean proof
    → LeanVerifier.verify_proof(): runs `lake env lean <file>`
    → if ✓ VERIFIED → save to submissions/
    → if ✗ FAILED → score_partial_proof() → repair_submission() → loop
```

### Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `LeanSolveAgent` | `lean_solve_agent.py` | Main orchestrator: parallel async workers |
| `LeanVerifier` | `lean_verifier.py` | Subprocess-based Lean type checker |
| `PutnamBenchLoader` | `putnam_bench.py` | Parser for PutnamBench `.lean` files |
| `LeanProblem` | `putnam_bench.py` | Dataclass: problem_id, informal/formal statement |
| `LeanSubmission` | `lean_solve_agent.py` | Dataclass: proof attempt with score/errors |
| `LeanProblemState` | `lean_solve_agent.py` | Per-problem state: submissions, solved flag |
| `VerificationResult` | `lean_verifier.py` | Verification output: success, error, sorry count |

### Concurrency Model

- **LLM workers**: Controlled by `asyncio.Semaphore(max_concurrent)` (default 32)
- **Lean verification workers**: Separate `asyncio.Semaphore(max_verify_concurrent)` (default 6, CPU-bound)
- **Problem selection**: Round-robin among unsolved problems, prioritizing highest-scoring partial proofs
- **Early stop**: Stops generating new work at 50% of time limit or 15 min before deadline; uses remaining time to finalize

### Verification Mechanism

`LeanVerifier.verify_proof()` in `lean_verifier.py`:
1. Writes proof to a temp file `lean_project/Proofs/Proof_<uuid8>.lean`
2. Runs `lake env lean <file>` as an async subprocess (timeout: 60s default)
3. Checks return code; also treats any `\bsorry\b` in code as failure even if rc=0
4. Parses Lean 4 error format: `file.lean:LINE:COL: error: MESSAGE`
5. Cleans up temp file in `finally` block

## Setup and Running

### Prerequisites

```bash
# Python 3.11+
pip install -r requirements.txt

# Lean 4 via elan
./setup_lean.sh          # installs elan + validates lake

# Initialize Lake project (downloads Mathlib cache, ~5GB, takes 5–10 min)
cd lean_project && lake build
```

### Environment Variables

Copy `.env.example` to `.env`:
```
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

The agent also accepts `OPENAI_API_KEY` as a fallback. Any OpenAI-compatible endpoint works via `--base_url`.

### Running the Agent

```bash
# Single problem test
python lean_solve_agent.py PutnamBench/ --problems_limit=1

# Filter by problem ID (regex)
python lean_solve_agent.py PutnamBench/ --problems_filter="1962_a[1-3]"

# Full benchmark (recommended config)
python lean_solve_agent.py PutnamBench/ \
    --problems_limit=100 \
    --max_concurrent=16 \
    --max_verify_concurrent=6 \
    --time_limit_hours=4.0

# Baseline (no repair loop, faster)
python baseline_solve.py PutnamBench/ --problems_limit=100 --max_concurrent=4

# Quick debug run
./runbooks/run_debug.sh
```

### Key CLI Parameters

| Flag | Default | Notes |
|------|---------|-------|
| `--problems_limit` | None | Cap on problems loaded |
| `--problems_filter` | None | Regex filter on problem IDs |
| `--time_limit_hours` | 3.0 | Hard wall-clock limit |
| `--max_concurrent` | 32 | Parallel LLM API workers (I/O-bound, can be high) |
| `--max_verify_concurrent` | 6 | Parallel Lean builds (CPU-bound, keep 4–8) |
| `--max_repair_attempts` | 3 | Repair iterations per initial attempt |
| `--verification_timeout` | 60 | Seconds per `lake env lean` call |
| `--model` | `deepseek-reasoner` | LLM model name |
| `--base_url` | DeepSeek API | OpenAI-compatible endpoint |
| `--submissions_dir` | auto-timestamped | Output directory |

## Python Code Conventions

### Style
- **Python version**: 3.11+ (uses `list[T]` generic syntax without `from __future__`)
- **Type hints**: All function signatures annotated; `Optional[T]` for nullable
- **Dataclasses**: `@dataclass` with `field(default_factory=...)` for mutable defaults
- **Async**: All I/O-bound operations use `async def` / `await` / `asyncio`
- **Error classes**: Custom exceptions at module top (`TokenLimitError`, `IncompleteResponseError`)
- **Naming**: `snake_case` for functions/variables, `CamelCase` for classes

### File Header Template
```python
#!/usr/bin/env python3
"""
One-line description.

Detailed description paragraph.

Usage:
    python module.py <args>
"""
```

### Retry Pattern
```python
@retry(
    stop=stop_after_delay(3 * 3600),
    wait=wait_exponential(multiplier=2, min=4, max=120),
    retry=retry_if_exception(_should_retry),
    reraise=True,
)
async def _call_llm(self, messages: list[dict]) -> str:
    ...
```

Do not retry `TokenLimitError` or `IncompleteResponseError` — these are unrecoverable for a given prompt.

### Output/Logging
Use `safeprint()` (defined in `lean_solve_agent.py`) instead of `print()` directly. It uses `rich.print` if available with colored markup like `[bold green]text[/bold green]`, falling back to plain `print`.

### Code Extraction from LLM Responses
`_extract_lean_code()` tries four strategies in order:
1. Last ` ```lean ... ``` ` code block (preferred)
2. Last ` ``` ` block starting with `import`
3. `\boxed{...}` with brace-matched extraction
4. Line-by-line heuristic: collect lines starting with Lean keywords, skip prose

Always prepend `import Mathlib\n\n` if extracted code doesn't start with `import`.

## Lean 4 Conventions

### Proof File Structure
Every generated proof file must follow this structure:
```lean
import Mathlib

-- optional: open namespaces
open Finset in

theorem problem_name (...) : statement := by
  tactic1
  tactic2
```

- Always start with `import Mathlib` (no partial imports)
- Never use `sorry` — it causes verification to fail even if `lake` returns 0
- Prefer `by` tactic mode over term-mode proofs for generated code

### Common Tactics Reference
| Tactic | Use case |
|--------|----------|
| `ring` | Polynomial/ring arithmetic equalities |
| `linarith` | Linear arithmetic inequalities |
| `omega` | Presburger arithmetic (integers/naturals) |
| `norm_num` | Concrete numeric computations |
| `simp [lemmas]` | Simplification with lemma hints |
| `exact h` | Goal matches hypothesis exactly |
| `apply f` | Apply lemma/function |
| `intro x` | Introduce universal/implication |
| `constructor` | Split `∧` or `∃` |
| `use witness` | Provide existential witness |
| `by_cases h : P` | Classical case split |
| `by_contra h` | Proof by contradiction |
| `push_neg` | Push negations inward |
| `induction n with \| zero => ... \| succ k ih => ...` | Structural induction |
| `calc` | Step-by-step calculation chains |
| `native_decide` / `decide` | Decision procedures for decidable props |
| `aesop` | Automation for structural goals |

### Mathlib Naming Conventions
- `Nat.add_comm`, `Int.mul_assoc`, `Real.sqrt_pos` (Namespace.property_name)
- Coercions: `↑n` casts `Nat` → `Int` or `Real`
- `Finset.sum`, `Finset.prod`, `Finset.card` for finite combinatorics
- `Function.Injective`, `Function.Surjective`, `Function.Bijective`

## Prompt Templates

All prompts live in `prompts/` as Markdown files with `{placeholder}` substitutions:

| File | Placeholders | Purpose |
|------|-------------|---------|
| `lean_solve.md` | `{informal_statement}`, `{lean_statement}` | Initial proof generation |
| `lean_repair.md` | `{informal_statement}`, `{lean_statement}`, `{previous_attempt}`, `{error_message}`, `{llm_feedback}` | Repair from error |
| `lean_score.md` | `{informal_statement}`, `{lean_code}`, `{error_message}`, `{sorry_count}` | Score partial proof 0.0–0.9 |

Score output format expected: `<score>0.7</score>` and `<feedback>...</feedback>` XML tags.

## Output Format

Submissions are written to `submissions/<run-name>/` (or `lean_submissions/<timestamp>/`):
- `<problem_id>.lean` — Best Lean proof (verified if solved, highest-scored otherwise)
- `<problem_id>.md` — Summary: status, attempt count, last error, code
- `SUMMARY.md` — Aggregate stats: solved/total, attempts, API calls, per-problem status

## Common Failure Patterns

| Pattern | Frequency | Example | Root cause |
|---------|-----------|---------|-----------|
| `sorry` placeholder | ~40% | `putnam_1962_b6` | LLM can't complete proof; leaves incomplete step |
| Truncated (`...`) | ~25% | `putnam_1966_b4` | Extraction failed to recover full proof |
| Missing Mathlib lemma | ~20% | `putnam_1970_b3` | LLM references nonexistent lemma name |
| Type mismatch | ~15% | `putnam_1964_a1` | Valid approach but types don't unify |

## Dependencies

### Python (`requirements.txt`)
- `openai` — OpenAI SDK used for DeepSeek API (OpenAI-compatible)
- `fire` — CLI argument parsing (`fire.Fire(main)`)
- `tenacity` — Retry with exponential backoff
- `httpx` — HTTP client (passed to `AsyncOpenAI` for custom timeout)
- `rich` — Colored terminal output (optional; graceful fallback)
- `python-dotenv` — Load `.env` file (optional; graceful fallback)

### Lean / System
- **elan** — Lean version manager (like `rustup`)
- **Lake** — Lean build system (like `cargo`)
- **mathlib4** — Community Lean 4 math library (~5GB cached)
- **Lean toolchain**: `leanprover/lean4:v4.27.0-rc1` (pinned in `lean_project/lean-toolchain`)

## Development Notes

### Adding a New Problem Set
1. Add `.lean` files to `problems/<set-name>/` following PutnamBench format:
   - `import Mathlib` header
   - `/-- informal statement -/` docstring
   - `theorem name ... :=` declaration (proof body is `sorry`)
2. Point `PutnamBenchLoader` at the directory, or adjust `src_dir` detection in `putnam_bench.py`

### Modifying Prompts
Edit files in `prompts/`. The `{placeholder}` strings are replaced via Python `.replace()` — not a templating engine. Keep placeholders exact.

### Testing Verification Locally
```bash
# Test that Lean environment works
python lean_verifier.py

# Verify a specific proof file
python -c "
import asyncio
from lean_verifier import verify_lean_proof
result = asyncio.run(verify_lean_proof(open('my_proof.lean').read()))
print('SUCCESS' if result.success else result.error_message)
"
```

### Adding a New Model
Pass `--model` and `--base_url` flags. Any OpenAI-compatible API endpoint works. The agent uses `response.choices[0].message.content`; for `deepseek-reasoner` it also falls back to `reasoning_content` if `content` is empty.

## No Automated Tests

There is no test suite. Validation is done by:
1. Running `python lean_verifier.py` directly (tests a simple proof)
2. Running `./runbooks/run_debug.sh` (solves 3 debug problems end-to-end)
3. Checking `lake build` succeeds in `lean_project/`
