#!/bin/bash
# Script to install all requirements for File Organizer

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || exit 1

echo "=========================================="
echo "File Organizer - Requirements Installer"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo "✓ pip upgraded"
echo ""

# Check for tkinter support (macOS specific)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Checking for tkinter support..."
    if python -c "import tkinter" 2>/dev/null; then
        echo "✓ tkinter is available"
    else
        echo "⚠ Warning: tkinter is not available"
        echo "  Installing python-tk@3.13 via Homebrew..."
        if command -v brew &> /dev/null; then
            brew install python-tk@3.13 || echo "  ⚠ Failed to install python-tk. GUI may not work."
        else
            echo "  ⚠ Homebrew not found. Please install python-tk@3.13 manually for GUI support."
        fi
    fi
    echo ""
fi

# Install requirements
echo "Installing Python packages from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✓ All requirements installed successfully"
else
    echo "✗ Error: requirements.txt not found!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the GUI, use:"
echo "  ./manage_organizer.sh gui"
echo ""
