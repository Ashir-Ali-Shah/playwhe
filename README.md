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

```bash
pip install -r requirements.txt
