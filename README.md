# Play Whe AI Predictor

This Streamlit application predicts Trinidad & Tobago Play Whe lottery numbers using machine learning models trained on historical data.

## Features

- Predicts top 10 likely numbers for upcoming Play Whe draws
- Displays most/least frequent numbers and current streaks
- Validates user-chosen numbers against model predictions
- Automatically determines the current game period (Morning, Midday, Afternoon, Evening)
- Interactive UI with number animations and Plotly visualizations

## Requirements

- Python 3.8+
- `play_whe_results.csv` file in the same directory, containing at least:
  - `date` (in YYYY-MM-DD format)
  - `winning_number` (integers from 1 to 36)

## Installation

1. Clone the repository or copy the source files.
2. Install dependencies:

Play Whe AI Predictor - Complete Setup Guide
Prerequisites
Python 3.8 or higher installed on your system
Command line/terminal access
Step 1: Set Up Project Directory
bash
# Create a new directory for the project
mkdir playwhe-predictor
cd playwhe-predictor
Step 2: Create Virtual Environment (Recommended)
bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

Step 3: Create Requirements File
Create a file named requirements.txt with the following content:
txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.15.0
seaborn>=0.12.0
matplotlib>=3.7.0
Step 4: Install Dependencies
bash
pip install -r requirements.txt
Step 5: Prepare Data File
You need a CSV file named play_whe_results.csv in the same directory with columns:
date (YYYY-MM-DD format)
winning_number (integers 1-36)
Example CSV structure:
csv
date,winning_number
2024-01-01,15
2024-01-01,22
2024-01-01,8
2024-01-01,31
Step 6: Get the Streamlit App Code
You'll need the main Python file (app.py) containing the Streamlit application code.
Step 7: Run the Application
bash
# Run the Streamlit app
streamlit run app.py
