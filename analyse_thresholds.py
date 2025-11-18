#!/usr/bin/env python3
"""
Analyze AWS Comprehend moderation results and recommend per-label score thresholds.

For each label (e.g. PROFANITY, HATE_SPEECH, INSULT, etc.), this script:

- Uses flag_count as ground truth (human judgment: flag_count > 0 ⇒ toxic).
- Treats the label's Score as the model output for that label.
- Finds two thresholds per label:
    1. review_threshold   – best F1 (good balance of precision/recall)
    2. autohide_threshold – high precision (>= 0.9) with best recall,
                            or highest precision if 0.9 is unreachable.
"""

import csv
import json
import sys
from collections import defaultdict
from typing import Dict, Tuple, List, Set
import statistics


class LabelThresholdAnalyzer:
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.results: List[Dict] = []
        self.flagged_posts: List[Dict] = []    # flag_count > 0
        self.unflagged_posts: List[Dict] = []  # flag_count == 0
        self.all_labels: Set[str] = set()      # e.g. PROFANITY, HATE_SPEECH, ...

    def load_data(self) -> None:
        """Load and parse the CSV data."""
        print(f"Loading data from {self.csv_file}...")

        try:
            with open(self.csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    # Parse flag_count (string → int)
                    try:
                        row["flag_count"] = int(row.get("flag_count") or 0)
                    except (ValueError, TypeError):
                        row["flag_count"] = 0

                    # Parse comprehend_labels JSON (list of {Name, Score})
                    raw_labels = row.get("comprehend_labels")
                    label_scores: Dict[str, float] = {}

                    if raw_labels:
                        try:
                            labels = json.loads(raw_labels)
                            if isinstance(labels, list):
                                for item in labels:
                                    name = str(item.get("Name", "")).strip().upper()
                                    score = float(item.get("Score", 0.0))
                                    if name:
                                        label_scores[name] = score
                                        self.all_labels.add(name)
                        except json.JSONDecodeError:
                            labels = []
                    else:
                        labels = []

                    # Store parsed structures
                    row["parsed_labels"] = labels          # original list
                    row["label_scores"] = label_scores     # dict: label -> score

                    self.results.append(row)

                    # Split into flagged/unflagged groups using human flags
                    if row["flag_count"] > 0:
                        self.flagged_posts.append(row)
                    else:
                        self.unflagged_posts.append(row)

            print(f"✓ Loaded {len(self.results)} posts")
            print(f"  - {len(self.flagged_posts)} flagged by users (flag_count > 0)")
            print(f"  - {len(self.unflagged_posts)} unflagged by users (flag_count = 0)")
            print(f"  - Labels observed: {', '.join(sorted(self.all_labels))}")
            print()

        except FileNotFoundError:
            print(f"Error: File '{self.csv_file}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading file: {e}")
            sys.exit(1)

    def _calculate_metrics_for_label(self, label: str, threshold: float) -> Dict:
        """
        Calculate precision, recall, F1, and accuracy for a given label + threshold.

        Ground truth:
            - Positive (toxic) if flag_count > 0
            - Negative (not toxic) if flag_count == 0

        Model prediction for this label:
            - Predict toxic if label_scores[label] >= threshold
            - Where missing, score is treated as 0.0
        """
        def score(post: Dict) -> float:
            return post["label_scores"].get(label, 0.0)

        true_positives = sum(1 for p in self.flagged_posts if score(p) >= threshold)
        false_positives = sum(1 for p in self.unflagged_posts if score(p) >= threshold)
        false_negatives = sum(1 for p in self.flagged_posts if score(p) < threshold)
        true_negatives = sum(1 for p in self.unflagged_posts if score(p) < threshold)

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        total = len(self.results)
        accuracy = (
            (true_positives + true_negatives) / total
            if total > 0
            else 0.0
        )

        return {
            "label": label,
            "threshold": threshold,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
        }

    def _find_thresholds_for_label(self, label: str) -> Tuple[Dict, Dict]:
        """
        For a single label, find:
            - review_threshold   (max F1)
            - autohide_threshold (precision >= 0.9, max recall; else highest precision)
        """
        # Gather all non-zero scores for this label
        scores = sorted({
            post["label_scores"].get(label, 0.0)
            for post in self.results
            if post["label_scores"].get(label, 0.0) > 0.0
        })

        if not scores:
            print(f"  [!] No scores found for label '{label}', skipping.")
            return {}, {}

        threshold_results = []
        for s in scores:
            metrics = self._calculate_metrics_for_label(label, s)
            threshold_results.append(metrics)

        # Review threshold: best F1
        best_review = max(threshold_results, key=lambda x: x["f1_score"])

        # Auto-hide threshold: precision >= 0.9, best recall
        high_precision = [t for t in threshold_results if t["precision"] >= 0.90]
        if high_precision:
            best_autohide = max(high_precision, key=lambda x: x["recall"])
        else:
            # Fallback: pick the threshold with highest precision
            best_autohide = max(threshold_results, key=lambda x: x["precision"])

        return best_review, best_autohide

    def run_analysis(self) -> Dict[str, Dict[str, Dict]]:
        """
        Run analysis for all labels seen in the data.

        Returns a nested dict:
        {
          "PROFANITY": {
             "review": {...metrics...},
             "autohide": {...metrics...}
          },
          "HATE_SPEECH": { ... },
          ...
        }
        """
        self.load_data()

        if not self.results:
            print("Error: No data to analyze.")
            return {}

        if not self.flagged_posts:
            print(
                "Warning: No flagged posts found. Cannot determine optimal thresholds "
                "without human ground truth (flag_count > 0)."
            )
            return {}

        if not self.all_labels:
            print("Warning: No labels found in comprehend_labels.")
            return {}

        recommendations: Dict[str, Dict[str, Dict]] = {}

        print("Finding thresholds per label...\n")

        for label in sorted(self.all_labels):
            print(f"▶ Label: {label}")
            review, autohide = self._find_thresholds_for_label(label)

            if not review or not autohide:
                print(f"  Skipping label '{label}' (insufficient data).")
                print()
                continue

            recommendations[label] = {
                "review": review,
                "autohide": autohide,
            }

            print(f"  Review threshold   : {review['threshold']:.4f}")
            print(
                f"    Precision: {review['precision']:.2%}, "
                f"Recall: {review['recall']:.2%}, F1: {review['f1_score']:.4f}"
            )
            print(f"  Auto-hide threshold: {autohide['threshold']:.4f}")
            print(
                f"    Precision: {autohide['precision']:.2%}, "
                f"Recall: {autohide['recall']:.2%}, F1: {autohide['f1_score']:.4f}"
            )
            print()

        # Summary table
        if recommendations:
            print("=" * 80)
            print("SUMMARY: RECOMMENDED THRESHOLDS PER LABEL")
            print("=" * 80)
            print(f"{'LABEL':25} {'REVIEW_THR':12} {'AUTOHIDE_THR':12} {'R_Prec':7} {'R_Rec':7} {'A_Prec':7} {'A_Rec':7}")
            print("-" * 80)
            for label, rec in sorted(recommendations.items()):
                r = rec["review"]
                a = rec["autohide"]
                print(
                    f"{label:25} "
                    f"{r['threshold']:.3f}       "
                    f"{a['threshold']:.3f}       "
                    f"{r['precision']:.2f}  "
                    f"{r['recall']:.2f}  "
                    f"{a['precision']:.2f}  "
                    f"{a['recall']:.2f}"
                )
            print()

        return recommendations


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_label_thresholds.py <output.csv>")
        print("\nExample: python analyze_label_thresholds.py output.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    analyzer = LabelThresholdAnalyzer(csv_file)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
