You are a Lean 4 theorem prover. Your task is to complete a formal proof for an Olympiad mathematics problem.

## Problem (Informal Description)
{informal_statement}

## Formal Theorem Statement (Lean 4)
```lean
{lean_statement}
```

## Lean 4 Syntax Reference

### Tactic Mode Structure
```lean
theorem name : statement := by
  tactic1
  tactic2
  -- proof continues...
```

### Essential Tactics
| Tactic | Use Case | Example |
|--------|----------|---------|
| `exact h` | Goal matches `h` exactly | `exact Nat.add_comm a b` |
| `apply f` | Apply function/lemma | `apply Nat.le_trans` |
| `intro x` | Introduce hypothesis | `intro n` for `∀ n, ...` |
| `have h : T := proof` | Intermediate result | `have h : n > 0 := by omega` |
| `constructor` | Split `∧` or `∃` goal | |
| `cases h with \| case1 => ... \| case2 => ...` | Case split | |
| `induction n with \| zero => ... \| succ k ih => ...` | Induction | |
| `simp [lemmas]` | Simplify | `simp [Nat.add_zero]` |
| `ring` | Ring equalities | Polynomial arithmetic |
| `linarith` | Linear arithmetic | Inequalities |
| `omega` | Presburger arithmetic | Integer constraints |
| `norm_num` | Numeric computation | `2 + 2 = 4` |
| `positivity` | Prove positivity | |
| `field_simp` | Clear denominators | |
| `nlinarith` | Nonlinear arithmetic | |
| `ext` | Extensionality | For functions, sets |
| `funext` | Function extensionality | |
| `contrapose` | Contrapositive | |
| `push_neg` | Push negation inside | |
| `by_contra h` | Proof by contradiction | |
| `by_cases h : P` | Classical case split | |
| `use witness` | Provide existential witness | `use 42` |
| `obtain ⟨a, b, hab⟩ := h` | Unpack existential | |
| `rcases h with ⟨x, hx⟩` | Recursive cases | |
| `refine ?_` | Leave hole to fill | |
| `rfl` | Reflexivity | |
| `rw [h]` | Rewrite with equality | `rw [Nat.add_comm]` |
| `conv => ...` | Conversion mode | For targeted rewriting |
| `calc` | Calculation chain | See below |

### Mathlib Naming Conventions
- Namespace format: `Nat.add_comm`, `Int.mul_assoc`, `Real.sqrt_pos`
- Coercions: `↑n` casts Nat to Int/Real
- Set notation: `s ∩ t`, `s ∪ t`, `x ∈ s`, `s ⊆ t`
- Functions: `Function.Injective`, `Function.Surjective`, `Function.Bijective`
- Finset operations: `Finset.sum`, `Finset.prod`, `Finset.card`

### Common Proof Patterns

**Existence proofs:**
```lean
use 42  -- provide witness
constructor
· exact proof_of_first_property
· exact proof_of_second_property
```

**Case analysis:**
```lean
rcases h with ⟨a, b, hab⟩  -- unpack existential
by_cases hP : P            -- classical case split
· -- case P is true
  exact ...
· -- case P is false
  exact ...
```

**Calculation chains:**
```lean
calc a = b := by ring
     _ = c := by simp [h]
     _ ≤ d := by linarith
```

**Induction:**
```lean
induction n with
| zero => 
  simp
| succ k ih =>
  rw [something]
  linarith [ih]
```

**Set proofs:**
```lean
ext x  -- for set equality
constructor
· intro hx  -- x ∈ LHS → x ∈ RHS
  ...
· intro hx  -- x ∈ RHS → x ∈ LHS
  ...
```

## Example Complete Proof
```lean
import Mathlib

open Finset in
theorem sum_first_n (n : ℕ) : 2 * (∑ i in range (n + 1), i) = n * (n + 1) := by
  induction n with
  | zero => simp
  | succ k ih =>
    rw [sum_range_succ, mul_add, ih]
    ring
```

---

## CRITICAL OUTPUT INSTRUCTIONS

You may think through the problem step by step above, but at the very end you MUST output your final Lean code in EXACTLY ONE of these two formats:

### Option 1: Use a ```lean code block
```lean
import Mathlib

theorem problem_name ... := by
  -- your proof tactics here
```

### Option 2: Use \boxed{}
\boxed{import Mathlib

theorem problem_name ... := by
  -- your proof tactics here
}

## ABSOLUTELY CRITICAL RULES:
1. Your final code MUST start with `import Mathlib`
2. Your final code MUST contain actual proof tactics - NEVER use `sorry`, `...`, or placeholders
3. Your final code MUST be COMPLETE and COMPILABLE - nothing can be left unfinished
4. Do NOT include any natural language text inside the code block or \boxed{}
5. The code block or \boxed{} MUST appear at the END of your response
6. If you cannot prove the theorem completely, use `native_decide`, `decide`, or powerful tactics like `aesop`, `simp_all`, `omega` to attempt to close goals

## FINAL REMINDER
Put your COMPLETE, COMPILABLE Lean 4 code in a ```lean code block at the END of your response. The code must start with `import Mathlib` and contain real proof tactics, not placeholders.
