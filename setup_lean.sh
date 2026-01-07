#!/bin/bash
# Setup script for Lean 4 environment
# This script installs elan (if needed) and sets up the mathlib4 project

set -e

echo "=== Nomos-Lean Setup ==="
echo ""

# Check if elan is installed
if ! command -v elan &> /dev/null; then
    echo "Installing elan (Lean version manager)..."
    curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
    
    # Add to PATH for this session
    export PATH="$HOME/.elan/bin:$PATH"
    echo ""
    echo "elan installed. You may need to restart your shell or run:"
    echo "  export PATH=\"\$HOME/.elan/bin:\$PATH\""
    echo ""
else
    echo "✓ elan is already installed"
fi

# Check if lake is available
if ! command -v lake &> /dev/null; then
    echo "Error: lake not found. Please ensure elan is in your PATH."
    exit 1
fi

echo "✓ lake is available"
echo ""

# Navigate to the lean_project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/lean_project"

echo "=== Setting up Lean project ==="
echo "Working directory: $(pwd)"
echo ""

# Get the toolchain
echo "Toolchain: $(cat lean-toolchain)"
echo ""

# Update dependencies (this fetches mathlib4)
echo "Fetching mathlib4 dependencies..."
echo "This may take 10-15 minutes on first run..."
echo ""

lake update

echo ""
echo "=== Building project (caching mathlib) ==="
echo "This will take a while on first run but speeds up future verification..."
echo ""

# Build the project to cache mathlib
lake build

echo ""
echo "=== Setup Complete ==="
echo ""
echo "You can now run the verification agent with:"
echo "  python solve_agent.py <problems_dir> --model deepseek-chat --base_url https://api.deepseek.com/v1"
echo ""
echo "To test the Lean verifier:"
echo "  python lean_verifier.py"
