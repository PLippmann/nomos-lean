You are debugging a Lean 4 proof that failed to compile.

## Problem (Informal Description)
{informal_statement}

## Formal Theorem Statement
```lean
{lean_statement}
```

## Your Previous Attempt
```lean
{previous_attempt}
```

## Compilation Error
```
{error_message}
```
{llm_feedback}

## Your Task
Fix the proof based on the error message and feedback above.

## Common Error Fixes

### Type Mismatch
- Check argument types carefully
- Use explicit type annotations: `(x : ℕ)`
- Check coercions: use `↑n` to cast Nat to Int/Real

### Unknown Identifier
- Verify import statements are complete
- Check mathlib naming: `Nat.add_comm` not `add_comm`
- Use `exact?` or `apply?` mentally to find the right lemma

### Tactic Failed
- `ring` fails: expression not a ring or contains non-ring operations
- `linarith` fails: not a linear arithmetic problem, or need more hypotheses
- `simp` fails: need to provide more lemmas `simp [lemma1, lemma2]`

### Missing Cases
- Ensure all constructors are covered in `cases` or `induction`
- Check if you need `| _ => ...` for remaining cases

### Goal Not Closed
- Proof is incomplete - need more tactics
- Check if you have leftover subgoals

## CRITICAL INSTRUCTIONS
1. Output a COMPLETE, COMPILABLE Lean 4 file
2. Do not output markdown formatting - ONLY the Lean code
3. Start directly with `import`
4. Address the specific error mentioned above

## OUTPUT FORMAT
Output ONLY valid Lean 4 code. No markdown code fences. No explanations.
