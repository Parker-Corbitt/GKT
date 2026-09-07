#!/usr/bin/env bash

set -euo pipefail

# Setup script to install dependencies for GKT project using Python 3.12.3

echo "Setting up the environment for Python 3.12..."

# Check if python3.12 is installed
if ! command -v python3.12 &> /dev/null; then
    echo "Error: python3.12 was not found in your PATH."
    echo "Please install Python 3.12 (e.g., via pyenv or your package manager) before running this script."
    exit 1
fi

# Create a virtual environment if it doesn't exist
if [ ! -d ".gkt" ]; then
    echo "Creating virtual environment (.gkt) using Python 3.12..."
    python3.12 -m venv .gkt
fi

# Activate the virtual environment
echo "Activating environment..."
source .gkt/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing compatible dependencies from requirements.txt..."
    python -m pip install -r requirements.txt
else
    echo "Error: requirements.txt not found!"
    exit 1
fi

echo "-----------------------------------------------------------"
echo "Installation complete!"
echo "To activate the environment, run: source .gkt/bin/activate"
echo "-----------------------------------------------------------"
