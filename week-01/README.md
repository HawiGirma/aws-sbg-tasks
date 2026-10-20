# Week 01: Small Python Data-Processing App

## Overview
This project loads a raw CSV dataset, cleans the data with reusable Python functions, and uses NumPy to calculate summary statistics.

## Dataset
`sample_data.csv` contains sample student scores and attendance. It intentionally includes whitespace, a missing score, a non-numeric score, and an out-of-range score to demonstrate data cleaning.

## Features
- Loads CSV rows using Python's built-in `csv` module.
- Trims whitespace and validates numeric fields.
- Treats missing, non-numeric, and out-of-range scores as invalid.
- Uses NumPy to calculate mean, median, minimum, maximum, and mean attendance.
- Runs as a Python script or in Google Colab/Jupyter.

## Files
- `data_processing_app.py`: executable Python source.
- `data_processing_app.ipynb`: notebook with explanations, code, and outputs.
- `sample_data.csv`: raw sample dataset.
- `requirements.txt`: Python package dependency list.

## Setup with a virtual environment
Python's `venv` module creates an isolated environment. Run these commands from this folder.

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python data_processing_app.py
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python data_processing_app.py
```

## Google Colab
Upload `sample_data.csv` and the notebook to Colab, or upload the whole folder. The notebook includes a cell that installs NumPy if it is not already available. Run all cells from top to bottom, then download the notebook as an `.ipynb` file.

## Cleaning rules
1. Remove leading/trailing whitespace from student and department names.
2. Skip rows without a student name or department.
3. Skip rows with invalid or out-of-range attendance.
4. Convert invalid, missing, or out-of-range scores to `None`, so they are excluded from score statistics but the student's valid attendance remains available.

## Learning outcomes
- Read a CSV file.
- Write and call reusable functions.
- Handle missing and invalid values.
- Convert numeric data to NumPy arrays.
- Calculate descriptive statistics.
- Manage dependencies with `venv` and `requirements.txt`.
