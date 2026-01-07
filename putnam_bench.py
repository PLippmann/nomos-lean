#!/usr/bin/env python3
"""
PutnamBench dataset loader for Lean 4 theorem proving.

Parses PutnamBench Lean files to extract theorem statements and informal descriptions.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class LeanProblem:
    """A single theorem proving problem from PutnamBench."""
    problem_id: str              # e.g., "putnam_1964_b2"
    informal_statement: str      # Human-readable problem description
    lean_statement: str          # Complete Lean theorem declaration
    imports: str                 # Import statements (typically "import Mathlib")
    file_path: Optional[Path] = None
    
    @property
    def full_code_template(self) -> str:
        """Returns the full Lean file with sorry placeholder."""
        return f"{self.imports}\n\n{self.lean_statement}\nsorry\n"


class PutnamBenchLoader:
    """
    Loader for PutnamBench Lean 4 problems.
    
    PutnamBench structure:
    - lean4/src/putnam_YYYY_XX.lean files
    - Each file contains:
      - import statements
      - docstring with informal problem statement
      - theorem declaration ending with := sorry
    """
    
    def __init__(self, bench_dir: str):
        """
        Initialize loader with path to PutnamBench repository.
        
        Args:
            bench_dir: Path to PutnamBench directory (containing lean4/src/)
        """
        self.bench_dir = Path(bench_dir).resolve()
        
        # Try different directory structures
        self.src_dir = None
        for candidate in [
            self.bench_dir / "lean4" / "src",
            self.bench_dir / "src",
            self.bench_dir,
        ]:
            if candidate.exists() and any(candidate.glob("*.lean")):
                self.src_dir = candidate
                break
        
        if self.src_dir is None:
            raise FileNotFoundError(
                f"Could not find Lean files in {bench_dir}. "
                "Expected structure: lean4/src/*.lean"
            )
    
    def _parse_lean_file(self, file_path: Path) -> Optional[LeanProblem]:
        """
        Parse a single Lean file to extract problem components.
        
        Args:
            file_path: Path to .lean file
            
        Returns:
            LeanProblem or None if parsing fails
        """
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return None
        
        # Extract problem ID from filename
        # e.g., "putnam_1964_b2.lean" -> "putnam_1964_b2"
        problem_id = file_path.stem
        
        # Extract imports (lines starting with 'import')
        import_lines = []
        for line in content.split('\n'):
            if line.strip().startswith('import '):
                import_lines.append(line)
            elif line.strip().startswith('open '):
                import_lines.append(line)
            elif import_lines and not line.strip():
                # Allow blank lines in import section
                continue
            elif import_lines:
                # First non-import, non-blank line after imports
                break
        imports = '\n'.join(import_lines) if import_lines else 'import Mathlib'
        
        # Extract docstring (informal statement)
        # Format: /-- ... -/
        docstring_match = re.search(
            r'/--\s*(.*?)\s*-/',
            content,
            re.DOTALL
        )
        informal_statement = ""
        if docstring_match:
            informal_statement = docstring_match.group(1).strip()
            # Clean up the docstring
            informal_statement = re.sub(r'\s+', ' ', informal_statement)
        
        # Extract theorem declaration
        # Match from 'theorem' to ':=' (excluding the proof/sorry)
        theorem_match = re.search(
            r'(theorem\s+\w+[\s\S]*?):=\s*$',
            content,
            re.MULTILINE
        )
        
        if not theorem_match:
            # Try alternative patterns
            theorem_match = re.search(
                r'(theorem\s+\w+[\s\S]*?)(?::=|:= by)',
                content
            )
        
        if not theorem_match:
            print(f"Warning: Could not find theorem in {file_path}")
            return None
        
        lean_statement = theorem_match.group(1).strip()
        
        # Ensure statement ends with := (for proof completion)
        if not lean_statement.endswith(':='):
            lean_statement += ' :='
        
        return LeanProblem(
            problem_id=problem_id,
            informal_statement=informal_statement,
            lean_statement=lean_statement,
            imports=imports,
            file_path=file_path
        )
    
    def load_problems(
        self, 
        filter_pattern: Optional[str] = None,
        limit: Optional[int] = None
    ) -> list[LeanProblem]:
        """
        Load all problems from PutnamBench.
        
        Args:
            filter_pattern: Optional regex pattern to filter problem IDs
            limit: Optional maximum number of problems to load
            
        Returns:
            List of LeanProblem instances
        """
        problems = []
        lean_files = sorted(self.src_dir.glob("*.lean"))
        
        for file_path in lean_files:
            # Skip non-problem files
            if file_path.stem.startswith('_'):
                continue
            
            # Apply filter
            if filter_pattern:
                if not re.search(filter_pattern, file_path.stem):
                    continue
            
            problem = self._parse_lean_file(file_path)
            if problem:
                problems.append(problem)
            
            # Check limit
            if limit and len(problems) >= limit:
                break
        
        return problems
    
    def get_problem(self, problem_id: str) -> Optional[LeanProblem]:
        """
        Get a specific problem by ID.
        
        Args:
            problem_id: Problem identifier (e.g., "putnam_1964_b2")
            
        Returns:
            LeanProblem or None if not found
        """
        file_path = self.src_dir / f"{problem_id}.lean"
        if file_path.exists():
            return self._parse_lean_file(file_path)
        return None
    
    def list_problem_ids(self) -> list[str]:
        """List all available problem IDs."""
        return sorted([
            f.stem for f in self.src_dir.glob("*.lean")
            if not f.stem.startswith('_')
        ])
    
    def __len__(self) -> int:
        """Return number of available problems."""
        return len(self.list_problem_ids())


def download_putnam_bench(target_dir: str = "PutnamBench") -> Path:
    """
    Clone or update PutnamBench repository.
    
    Args:
        target_dir: Directory to clone into
        
    Returns:
        Path to cloned repository
    """
    import subprocess
    
    target_path = Path(target_dir).resolve()
    
    if target_path.exists():
        print(f"PutnamBench already exists at {target_path}")
        # Try to update
        try:
            subprocess.run(
                ["git", "pull"],
                cwd=str(target_path),
                check=True,
                capture_output=True
            )
            print("Updated to latest version")
        except subprocess.CalledProcessError:
            print("Warning: Could not update repository")
        return target_path
    
    print(f"Cloning PutnamBench to {target_path}...")
    subprocess.run(
        [
            "git", "clone", "--depth=1",
            "https://github.com/trishullab/PutnamBench.git",
            str(target_path)
        ],
        check=True
    )
    print("Clone complete!")
    return target_path


if __name__ == "__main__":
    import sys
    
    # Demo usage
    if len(sys.argv) > 1:
        bench_dir = sys.argv[1]
    else:
        # Try to find or download PutnamBench
        default_path = Path("PutnamBench")
        if not default_path.exists():
            print("PutnamBench not found. Downloading...")
            bench_dir = download_putnam_bench()
        else:
            bench_dir = default_path
    
    loader = PutnamBenchLoader(bench_dir)
    print(f"Found {len(loader)} problems in {loader.src_dir}")
    print()
    
    # Show first few problems
    problems = loader.load_problems(limit=3)
    for p in problems:
        print(f"=== {p.problem_id} ===")
        print(f"Informal: {p.informal_statement[:100]}...")
        print(f"Statement: {p.lean_statement[:100]}...")
        print()
