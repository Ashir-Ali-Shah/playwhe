# 🎰 Play Whe AI Predictor

A Streamlit-based machine learning app that predicts **Play Whe** numbers for Trinidad & Tobago lottery draws using historical data.

---

## ✨ Features

* 🔢 Predicts **Top 10 Likely Numbers** for upcoming Play Whe draws
* 📈 Displays **most/least frequent numbers** and **current number streaks**
* ✅ Validates **user-chosen numbers** against model predictions
* 🕒 Automatically detects the current **game period** (Morning, Midday, Afternoon, Evening)
* 🎨 Interactive UI with number animations and **Plotly visualizations**

---

## 📂 Project Structure

```
playwhe-predictor/
├── app.py                  # Main Streamlit application
├── play_whe_results.csv    # Historical draw data
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

---

## ⚙️ Requirements

* Python **3.8 or higher**
* `play_whe_results.csv` in the same directory with:

  * `date` (YYYY-MM-DD)
  * `winning_number` (1 to 36)

---

## 🚀 Installation & Setup Guide

### 🔧 Step 1: Set Up Project Directory

```bash
# Create a new project directory
mkdir playwhe-predictor
cd playwhe-predictor
```

### 🧪 Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 📦 Step 3: Create Requirements File

Create a file called `requirements.txt` and add the following:

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.15.0
seaborn>=0.12.0
matplotlib>=3.7.0
```

### 📥 Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### 📊 Step 5: Prepare Your Data File

You must have a file named **`play_whe_results.csv`** in the root directory. It should follow this structure:

```csv
date,winning_number
2024-01-01,15
2024-01-01,22
2024-01-01,8
2024-01-01,31
...
```

* `date`: Format must be `YYYY-MM-DD`
* `winning_number`: Must be between `1` and `36`

### 🧠 Step 6: Add the App Code

Ensure you have the main app file `app.py`. This should contain the Streamlit logic for your predictor.

### ▶️ Step 7: Run the App

```bash
streamlit run app.py
```


