#!/usr/bin/env python3
"""
Lean 4 proof verification engine using subprocess calls to Lake.

Provides async verification of Lean proofs with error parsing and sorry detection.
"""

import asyncio
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VerificationResult:
    """Result of verifying a Lean proof."""
    success: bool
    error_message: Optional[str] = None
    error_line: Optional[int] = None
    sorry_count: int = 0
    raw_output: str = ""


class LeanVerifier:
    """
    Subprocess-based Lean 4 verification engine.
    
    Uses `lake env lean` to verify individual proof files against a 
    pre-configured Lake project with mathlib4 dependencies.
    """
    
    def __init__(self, project_dir: str):
        """
        Initialize verifier with path to Lake project.
        
        Args:
            project_dir: Path to directory containing lakefile.toml
        """
        self.project_dir = Path(project_dir).resolve()
        self.proofs_dir = self.project_dir / "Proofs"
        self.proofs_dir.mkdir(exist_ok=True)
        
        # Verify project structure
        if not (self.project_dir / "lakefile.toml").exists():
            raise FileNotFoundError(
                f"No lakefile.toml found in {self.project_dir}. "
                "Run setup_lean.sh first."
            )
    
    def count_sorries(self, lean_code: str) -> int:
        """
        Count remaining 'sorry' placeholders in Lean code.
        
        Args:
            lean_code: Lean 4 source code
            
        Returns:
            Number of sorry occurrences
        """
        # Match 'sorry' as a word boundary (not inside identifiers)
        return len(re.findall(r'\bsorry\b', lean_code))
    
    def _parse_error(self, stderr: str) -> tuple[Optional[str], Optional[int]]:
        """
        Parse Lean compiler error output.
        
        Args:
            stderr: Standard error from lean command
            
        Returns:
            Tuple of (error_message, error_line)
        """
        if not stderr.strip():
            return None, None
        
        # Lean error format: "file.lean:line:col: error: message"
        error_match = re.search(
            r'\.lean:(\d+):\d+:\s*(error|warning):\s*(.+?)(?=\n\S|\Z)',
            stderr,
            re.DOTALL
        )
        
        if error_match:
            line_num = int(error_match.group(1))
            error_msg = error_match.group(3).strip()
            return error_msg, line_num
        
        # Fallback: return first non-empty line
        for line in stderr.strip().split('\n'):
            if line.strip():
                return line.strip(), None
        
        return stderr.strip()[:500], None
    
    async def verify_proof(
        self, 
        lean_code: str,
        timeout_seconds: float = 60.0
    ) -> VerificationResult:
        """
        Verify a Lean proof by compiling it.
        
        Args:
            lean_code: Complete Lean 4 file content (including imports)
            timeout_seconds: Maximum time to wait for compilation
            
        Returns:
            VerificationResult with success status and any errors
        """
        # Count sorries before verification
        sorry_count = self.count_sorries(lean_code)
        
        # Create temporary file in Proofs directory
        proof_id = f"Proof_{uuid.uuid4().hex[:8]}"
        proof_file = self.proofs_dir / f"{proof_id}.lean"
        
        try:
            # Write proof to file
            proof_file.write_text(lean_code)
            
            # Run `lake env lean <file>` to check the proof
            # This compiles the file in the context of the Lake project
            process = await asyncio.create_subprocess_exec(
                "lake", "env", "lean", str(proof_file),
                cwd=str(self.project_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "LEAN_ABORT_ON_PANIC": "1"}
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return VerificationResult(
                    success=False,
                    error_message=f"Verification timed out after {timeout_seconds}s",
                    sorry_count=sorry_count,
                    raw_output="TIMEOUT"
                )
            
            stdout_text = stdout.decode('utf-8', errors='replace')
            stderr_text = stderr.decode('utf-8', errors='replace')
            combined_output = f"{stdout_text}\n{stderr_text}".strip()
            
            # Check for success
            if process.returncode == 0:
                # Success, but check for sorry warnings
                if sorry_count > 0:
                    return VerificationResult(
                        success=False,
                        error_message=f"Proof contains {sorry_count} sorry placeholder(s)",
                        sorry_count=sorry_count,
                        raw_output=combined_output
                    )
                return VerificationResult(
                    success=True,
                    sorry_count=0,
                    raw_output=combined_output
                )
            
            # Parse error
            error_msg, error_line = self._parse_error(stderr_text)
            
            return VerificationResult(
                success=False,
                error_message=error_msg or "Unknown compilation error",
                error_line=error_line,
                sorry_count=sorry_count,
                raw_output=combined_output
            )
            
        finally:
            # Clean up temporary file
            if proof_file.exists():
                proof_file.unlink()
    
    async def check_environment(self) -> bool:
        """
        Verify that the Lean environment is properly set up.
        
        Returns:
            True if environment is ready, False otherwise
        """
        test_code = """\
import Mathlib

theorem test_env : 1 + 1 = 2 := by norm_num
"""
        result = await self.verify_proof(test_code, timeout_seconds=120.0)
        return result.success


# Convenience function for one-off verification
async def verify_lean_proof(
    lean_code: str,
    project_dir: str = "lean_project"
) -> VerificationResult:
    """
    Convenience function to verify a single proof.
    
    Args:
        lean_code: Complete Lean 4 file content
        project_dir: Path to Lake project directory
        
    Returns:
        VerificationResult
    """
    verifier = LeanVerifier(project_dir)
    return await verifier.verify_proof(lean_code)


if __name__ == "__main__":
    # Quick test
    import sys
    
    async def main():
        verifier = LeanVerifier("lean_project")
        
        # Test with a simple proof
        test_proof = """\
import Mathlib

theorem simple_test : ∀ n : ℕ, n + 0 = n := by
  intro n
  ring
"""
        print("Testing Lean verifier...")
        result = await verifier.verify_proof(test_proof)
        
        if result.success:
            print("✓ Verification succeeded!")
        else:
            print(f"✗ Verification failed: {result.error_message}")
            if result.error_line:
                print(f"  Error at line {result.error_line}")
        
        print(f"  Sorry count: {result.sorry_count}")
        return 0 if result.success else 1
    
    sys.exit(asyncio.run(main()))
