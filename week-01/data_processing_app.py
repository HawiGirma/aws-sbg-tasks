"""
Small Python Data-Processing App
Loads a CSV dataset, cleans records, and summarizes numeric data with NumPy.
Run: python data_processing_app.py
"""

import csv
from pathlib import Path
import numpy as np


def load_dataset(file_path):
    """Load rows from a CSV file and return them as dictionaries."""
    with open(file_path, mode="r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def clean_dataset(rows):
    """
    Clean records:
    - Trim whitespace from text fields.
    - Convert score and attendance to numbers.
    - Replace missing/non-numeric scores with None.
    - Reject scores outside the valid 0-100 range.
    - Keep records with valid attendance between 0 and 100.
    """
    cleaned = []
    for row in rows:
        student = (row.get("student") or "").strip()
        department = (row.get("department") or "").strip()
        score_text = (row.get("score") or "").strip()
        attendance_text = (row.get("attendance") or "").strip()

        if not student or not department:
            continue

        try:
            attendance = float(attendance_text)
        except (TypeError, ValueError):
            continue

        if not 0 <= attendance <= 100:
            continue

        try:
            score = float(score_text)
        except (TypeError, ValueError):
            score = None

        if score is not None and not 0 <= score <= 100:
            score = None

        cleaned.append({
            "student": student,
            "department": department,
            "score": score,
            "attendance": attendance,
        })
    return cleaned


def process_dataset(rows):
    """Use NumPy to calculate summary statistics for valid scores and attendance."""
    valid_scores = np.array(
        [row["score"] for row in rows if row["score"] is not None],
        dtype=float,
    )
    attendance = np.array([row["attendance"] for row in rows], dtype=float)

    summary = {
        "records_after_cleaning": len(rows),
        "valid_scores_count": int(valid_scores.size),
        "missing_or_invalid_scores": int(
            sum(row["score"] is None for row in rows)
        ),
        "mean_score": float(np.mean(valid_scores)) if valid_scores.size else None,
        "median_score": float(np.median(valid_scores)) if valid_scores.size else None,
        "minimum_score": float(np.min(valid_scores)) if valid_scores.size else None,
        "maximum_score": float(np.max(valid_scores)) if valid_scores.size else None,
        "mean_attendance": float(np.mean(attendance)) if attendance.size else None,
    }
    return summary


def print_report(summary):
    """Print a readable report."""
    print("DATA-PROCESSING REPORT")
    print("=" * 30)
    for label, value in summary.items():
        if isinstance(value, float):
            print(f"{label.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"{label.replace('_', ' ').title()}: {value}")


def main():
    """Run the complete data-processing workflow."""
    data_path = Path(__file__).with_name("sample_data.csv")
    raw_rows = load_dataset(data_path)
    cleaned_rows = clean_dataset(raw_rows)
    summary = process_dataset(cleaned_rows)

    print(f"Raw records loaded: {len(raw_rows)}")
    print(f"Records retained after cleaning: {len(cleaned_rows)}")
    print()
    print_report(summary)


if __name__ == "__main__":
    main()
