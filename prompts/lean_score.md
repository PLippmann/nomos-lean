You are evaluating an incomplete or failed Lean 4 proof attempt.

## Problem (Informal)
{informal_statement}

## Proof Attempt
```lean
{lean_code}
```

## Compilation Result
- **Status**: Failed
- **Error**: {error_message}
- **Remaining `sorry` count**: {sorry_count}

## Your Task
1. Score this proof attempt from 0.0 to 0.9 (1.0 is reserved for verified proofs)
2. Provide specific, actionable feedback on how to fix or improve the proof

## Scoring Guide
- **0.8-0.9**: Proof is nearly complete, minor tactic or syntax issue (e.g., wrong lemma name, missed case)
- **0.6-0.7**: Correct overall strategy, but contains errors that need fixing (e.g., type mismatches, missing steps)
- **0.4-0.5**: Reasonable approach, but significant gaps or fundamental errors in proof structure
- **0.2-0.3**: Some useful progress (e.g., correct setup, partial cases), but major rework needed
- **0.0-0.1**: Little to no meaningful progress, wrong approach entirely

## Evaluation Criteria
1. **Proof Strategy**: Is the overall approach sound for this problem?
2. **Tactic Usage**: Are the tactics appropriate for the goal types?
3. **Mathlib Knowledge**: Does it correctly use Mathlib lemmas and conventions?
4. **Completeness**: How close is it to a complete proof?
5. **Error Severity**: How hard is the error to fix?

## Output Format (STRICT)
You MUST use exactly this format:

<score>X.X</score>
<feedback>
Your specific feedback here. Include:
- What the proof attempt got right
- What specific error needs fixing and how
- Suggested tactics or lemmas to try
- Any structural changes needed
</feedback>
