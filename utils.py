#!/usr/bin/env python3
"""
Shared utilities for Lean proof generation agents.

Provides code extraction logic used by both the main agent and baseline.
"""

import re


def extract_lean_code(response: str) -> str:
    """
    Extract Lean code from an LLM response using multiple fallback strategies.

    Strategies (tried in order):
    1. Last ```lean ... ``` code block  (preferred — explicit fencing)
    2. Last ``` ... ``` block that starts with `import`
    3. \\boxed{...} with proper brace-matched extraction
    4. Line-by-line heuristic: collect Lean keyword lines, skip prose

    Returns:
        Extracted Lean code string, or "" if nothing is found.
    """
    # Strategy 1: ```lean ... ``` code blocks (prefer last one)
    lean_blocks = list(re.finditer(r'```lean\s*(.*?)```', response, re.DOTALL))
    if lean_blocks:
        code = lean_blocks[-1].group(1).strip()
        if code.startswith('import') or 'theorem' in code or 'lemma' in code:
            return code

    # Strategy 2: ``` ... ``` blocks starting with import
    import_blocks = list(re.finditer(r'```\s*(import.*?)```', response, re.DOTALL))
    if import_blocks:
        return import_blocks[-1].group(1).strip()

    # Strategy 3: \boxed{} with proper brace matching
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

    # Patterns that indicate prose (NOT Lean code)
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
        is_prose = any(
            stripped.startswith(p) or stripped.lower().startswith(p)
            for p in prose_starters
        )

        # Check if line looks like natural language (more aggressive detection)
        if not is_prose and len(stripped) > 10:
            if stripped[-1] in '.?!:' and ' ' in stripped:
                words = stripped.split()
                if len(words) > 4 and not any(stripped.startswith(k) for k in lean_keywords):
                    common_words = ['is', 'are', 'the', 'of', 'and', 'or', 'to', 'in', 'that', 'which']
                    word_list = [w.lower().strip('.,;:') for w in words]
                    if any(w in common_words for w in word_list):
                        is_prose = True

        if is_prose:
            if in_code and lean_lines:
                # Hit prose after code — stop
                break
            continue

        # Collect Lean keyword lines
        if stripped.startswith(lean_keywords) or stripped.startswith('--') or stripped.startswith('/-'):
            in_code = True
            lean_lines.append(line)
        elif in_code:
            lean_lines.append(line)

    if lean_lines:
        result = '\n'.join(lean_lines).strip()
        # Remove trailing placeholder patterns
        result = re.sub(r'\s*:=\s*\.\.\.\s*$', ' := by sorry', result)
        result = re.sub(r'\s+\.\.\.\s*$', '', result)
        return result

    return ""
