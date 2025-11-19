#!/usr/bin/env python3
"""
Simulate moderation impact given user-provided thresholds per label.

This script:

- Reads an output CSV with columns including `comprehend_labels`
  (list of {Name, Score}).
- Asks the user for review and auto-hide thresholds per label.
- Applies the following logic to each post:

    if ANY label score >= auto_hide_threshold[label] -> auto-hide
    elif ANY label score >= review_threshold[label] -> flag for review
    else -> unaffected

- Prints how many posts would be auto-hidden, flagged, and unaffected,
  plus some per-label stats.
"""

import csv
import json
import sys
from typing import Dict, List, Set


class ThresholdSimulator:
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.results: List[Dict] = []
        self.all_labels: Set[str] = set()

    def load_data(self) -> None:
        """Load posts and parse comprehend_labels into dicts."""
        print(f"Loading data from {self.csv_file}...")

        try:
            with open(self.csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for row in reader:
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
                            label_scores = {}
                    else:
                        label_scores = {}

                    row["label_scores"] = label_scores
                    self.results.append(row)

            print(f"✓ Loaded {len(self.results)} posts")
            print(f"  Labels observed: {', '.join(sorted(self.all_labels))}")
            print()

        except FileNotFoundError:
            print(f"Error: File '{self.csv_file}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading file: {e}")
            sys.exit(1)

    def _prompt_thresholds(self) -> (Dict[str, float], Dict[str, float]):
        """
        Prompt user for review and auto-hide thresholds for each label.

        Returns:
            review_thresholds:  dict[label] -> float
            autohide_thresholds: dict[label] -> float
        """
        review_thresholds: Dict[str, float] = {}
        autohide_thresholds: Dict[str, float] = {}

        print("Enter thresholds for each label.")
        print("Press Enter with no value to skip that threshold for a label.")
        print()

        for label in sorted(self.all_labels):
            # Review threshold
            while True:
                raw = input(
                    f"Review threshold for {label} (e.g. 0.15, or blank to skip): "
                ).strip()
                if not raw:
                    break
                try:
                    review_thresholds[label] = float(raw)
                    break
                except ValueError:
                    print("  Invalid number, please try again.")

            # Auto-hide threshold
            while True:
                raw = input(
                    f"Auto-hide threshold for {label} (e.g. 0.80, or blank to skip): "
                ).strip()
                if not raw:
                    break
                try:
                    autohide_thresholds[label] = float(raw)
                    break
                except ValueError:
                    print("  Invalid number, please try again.")

            print()

        # Warn if user skipped everything
        if not review_thresholds and not autohide_thresholds:
            print("Warning: no thresholds provided; nothing will be flagged or hidden.")
            print()

        return review_thresholds, autohide_thresholds

    def simulate(self) -> None:
        """Run the simulation using user-provided thresholds."""
        self.load_data()

        if not self.results:
            print("Error: No data to analyze.")
            return

        if not self.all_labels:
            print("Error: No labels found in comprehend_labels.")
            return

        review_thresholds, autohide_thresholds = self._prompt_thresholds()

        total_posts = len(self.results)
        auto_hidden = 0
        flagged_for_review = 0

        # For per-label stats: how many posts cross this label's thresholds
        per_label_review_counts: Dict[str, int] = {l: 0 for l in self.all_labels}
        per_label_autohide_counts: Dict[str, int] = {l: 0 for l in self.all_labels}

        for post in self.results:
            scores = post["label_scores"]

            # Track if this post crosses thresholds per label
            for label in self.all_labels:
                score = scores.get(label, 0.0)
                if label in autohide_thresholds and score >= autohide_thresholds[label]:
                    per_label_autohide_counts[label] += 1
                if label in review_thresholds and score >= review_thresholds[label]:
                    per_label_review_counts[label] += 1

            # Global moderation logic for this post
            # 1) Auto-hide if any label passes auto-hide threshold
            should_autohide = any(
                (label in autohide_thresholds) and
                (scores.get(label, 0.0) >= autohide_thresholds[label])
                for label in self.all_labels
            )

            if should_autohide:
                auto_hidden += 1
                continue

            # 2) Otherwise, flag for review if any label passes review threshold
            should_flag = any(
                (label in review_thresholds) and
                (scores.get(label, 0.0) >= review_thresholds[label])
                for label in self.all_labels
            )

            if should_flag:
                flagged_for_review += 1
                continue

            # 3) Else: unaffected

        unaffected = total_posts - auto_hidden - flagged_for_review

        # Overall summary
        print("=" * 80)
        print("SIMULATION RESULT (ALL LABELS COMBINED)")
        print("=" * 80)
        print(f"Total posts:             {total_posts:,}")
        print(f"Auto-hidden posts:       {auto_hidden:,}")
        print(f"Flagged for review:      {flagged_for_review:,}")
        print(f"Unaffected / normal:     {unaffected:,}")
        print()

        # Per-label stats
        print("=" * 80)
        print("PER-LABEL IMPACT")
        print("=" * 80)
        print(
            f"{'LABEL':25} "
            f"{'Review_thr':12} {'Auto_thr':12} "
            f"{'#>=Review':10} {'#>=Auto':10}"
        )
        print("-" * 80)
        for label in sorted(self.all_labels):
            r_thr = review_thresholds.get(label, None)
            a_thr = autohide_thresholds.get(label, None)
            r_cnt = per_label_review_counts.get(label, 0)
            a_cnt = per_label_autohide_counts.get(label, 0)
            r_thr_str = f"{r_thr:.3f}" if r_thr is not None else "-"
            a_thr_str = f"{a_thr:.3f}" if a_thr is not None else "-"
            print(
                f"{label:25} "
                f"{r_thr_str:12} {a_thr_str:12} "
                f"{r_cnt:10d} {a_cnt:10d}"
            )
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python simulate_thresholds.py <output.csv>")
        print("\nExample: python simulate_thresholds.py output.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    sim = ThresholdSimulator(csv_file)
    sim.simulate()


if __name__ == "__main__":
    main()
