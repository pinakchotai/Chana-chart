#!/bin/bash

echo "Setting up Chickpeas Spot Price Chart application..."

# Check if Python is installed
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "Error: Python is not installed. Please install Python 3.7 or higher."
    exit 1
fi

echo "Python found: $($PYTHON --version)"

# Check if Excel file exists
if [ -f "CHANA R&D.xlsx" ]; then
    echo "Excel file found."
else
    echo "Warning: Excel file 'CHANA R&D.xlsx' not found in the current directory."
    echo "Please make sure to place the Excel file in this directory."
fi

# Create virtual environment
echo "Creating virtual environment..."
$PYTHON -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete!"
echo "To run the application:"
echo "1. Activate the virtual environment if not already activated:"
echo "   source venv/bin/activate"
echo "2. Run the application:"
echo "   python app.py"
echo "3. Open your browser and go to: http://localhost:8050" 