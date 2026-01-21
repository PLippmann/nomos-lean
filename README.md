# Nomos-Lean

A Lean 4 formal verification agent for Olympiad mathematics. Generates machine-verifiable proofs and validates them with `lake build` for deterministic correctness.

## Overview

Nomos-Lean replaces LLM-based scoring with formal verification:

| Aspect | Traditional | Nomos-Lean |
|--------|-------------|------------|
| **Verification** | LLM scoring | Lean 4 `lake build` |
| **Certainty** | Probabilistic | Deterministic (100%) |
| **Input** | Markdown problems | [PutnamBench](https://github.com/trishullab/PutnamBench) Lean 4 |
| **Output** | Natural language | Compilable `.lean` files |


## Workflow

```mermaid
flowchart LR
    PB[PutnamBench<br/>672 problems] --> GEN[Generate Proof<br/>deepseek-reasoner]
    GEN --> VER{lake build}
    VER -->|✓| DONE[Verified!]
    VER -->|✗| SCORE[LLM Score + Feedback]
    SCORE --> REPAIR[Repair Attempt]
    REPAIR --> VER
```

1. **Generate**: LLM produces a complete Lean 4 proof
2. **Verify**: `lake build` checks type-correctness
3. **Repair**: If verification fails, LLM refines based on error feedback
4. **Repeat**: Until verified or max attempts reached

## Installation

### Prerequisites
- Python 3.11+
- [Lean 4 / elan](https://leanprover.github.io/lean4/doc/setup.html)

### Setup

```bash
# Clone with PutnamBench submodule
git clone --recursive https://github.com/your-repo/nomos-lean

# Install Python dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env with your DeepSeek API key

# Initialize Lean project (downloads Mathlib, takes 5-10 min)
cd lean_project && lake build
```

## Usage

```bash
# Single problem test
python lean_solve_agent.py PutnamBench/ --problems_limit=1

# Run specific problems
python lean_solve_agent.py PutnamBench/ --problems_filter="1962_a[1-3]"

# Full benchmark (672 problems)
python lean_solve_agent.py PutnamBench/ --time_limit_hours=3.0 --max_concurrent=8
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--problems_limit` | None | Max problems to attempt |
| `--problems_filter` | None | Regex filter for problem IDs |
| `--time_limit_hours` | `3.0` | Total time limit |
| `--max_concurrent` | `8` | Parallel workers |
| `--max_repair_attempts` | `3` | Repair loop depth |
| `--verification_timeout` | `60` | Seconds per verification |
| `--model` | `deepseek-reasoner` | LLM for proof generation |
| `--base_url` | DeepSeek API | OpenAI-compatible endpoint |

## Project Structure

```
nomos-lean/
├── lean_solve_agent.py   # Main agent (generate → verify → repair)
├── lean_verifier.py      # Subprocess wrapper for lake build
├── putnam_bench.py       # PutnamBench dataset loader
├── prompts/
│   ├── lean_solve.md     # Proof generation prompt
│   ├── lean_repair.md    # Error-based repair prompt
│   └── lean_score.md     # Partial proof scoring
├── lean_project/         # Lake project with Mathlib
│   ├── lakefile.toml
│   └── Proofs/           # Generated proof files
└── PutnamBench/          # Submodule: 672 competition problems
```

## Requirements

- **API**: DeepSeek API key (or any OpenAI-compatible endpoint)
- **Lean**: elan-managed toolchain (`leanprover/lean4:v4.x`)
- **Disk**: ~5GB for Mathlib cache

## License

MIT
