#!/usr/bin/env python3
"""
Analyze AWS Comprehend moderation results and recommend score thresholds.

This script uses flag_count as ground truth (human judgment) to determine
optimal Comprehend score thresholds for:
1. Flagging posts for review (lower threshold, high sensitivity)
2. Auto-hiding posts (higher threshold, high confidence)
"""

import csv
import json
import sys
from collections import defaultdict
from typing import Dict, Tuple
import statistics


class ThresholdAnalyzer:
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.results = []
        self.flagged_posts = []    # Posts with flag_count > 0
        self.unflagged_posts = []  # Posts with flag_count == 0

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

                    # Normalise toxic result flag a bit
                    if "comprehend_toxic_result" in row and row["comprehend_toxic_result"] is not None:
                        row["comprehend_toxic_result"] = str(row["comprehend_toxic_result"]).strip().upper()

                    # Parse comprehend_labels JSON (if present) and derive score/label
                    raw_labels = row.get("comprehend_labels")
                    labels = []
                    if raw_labels:
                        try:
                            # raw_labels from CSV will be a string
                            labels = json.loads(raw_labels)
                        except json.JSONDecodeError:
                            labels = []

                    # Store parsed labels (even if empty)
                    row["comprehend_labels"] = labels

                    # If labels array is present, derive top label/score from it
                    if isinstance(labels, list) and labels:
                        top = max(labels, key=lambda x: x.get("Score", 0.0))
                        row["comprehend_score"] = float(top.get("Score", 0.0))
                        row["comprehend_label"] = str(top.get("Name", "UNKNOWN"))
                    else:
                        # Fall back to scalar columns if present
                        try:
                            row["comprehend_score"] = float(row.get("comprehend_score") or 0.0)
                        except (ValueError, TypeError):
                            row["comprehend_score"] = 0.0

                        if not row.get("comprehend_label"):
                            row["comprehend_label"] = "UNKNOWN"

                    self.results.append(row)

                    # Categorise by human flags
                    if row["flag_count"] > 0:
                        self.flagged_posts.append(row)
                    else:
                        self.unflagged_posts.append(row)

            print(f"✓ Loaded {len(self.results)} posts")
            print(f"  - {len(self.flagged_posts)} flagged by users (flag_count > 0)")
            print(f"  - {len(self.unflagged_posts)} unflagged by users (flag_count = 0)")
            print()

        except FileNotFoundError:
            print(f"Error: File '{self.csv_file}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading file: {e}")
            sys.exit(1)

    def calculate_threshold_metrics(self, threshold: float) -> Dict:
        """
        Calculate precision, recall, F1, and accuracy for a given threshold.
        Uses flag_count > 0 as ground truth for "toxic".
        """
        # Predict toxic if comprehend_score >= threshold
        true_positives = sum(
            1 for p in self.flagged_posts if p["comprehend_score"] >= threshold
        )
        false_positives = sum(
            1 for p in self.unflagged_posts if p["comprehend_score"] >= threshold
        )
        false_negatives = sum(
            1 for p in self.flagged_posts if p["comprehend_score"] < threshold
        )
        true_negatives = sum(
            1 for p in self.unflagged_posts if p["comprehend_score"] < threshold
        )

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
        accuracy = (
            (true_positives + true_negatives) / len(self.results)
            if self.results
            else 0.0
        )

        return {
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

    def find_optimal_thresholds(self) -> Tuple[Dict, Dict]:
        """
        Find optimal thresholds for:
        1. Review threshold: maximize F1 (balanced precision/recall)
        2. Auto-hide threshold: high precision (>=0.9) with best recall
        """
        # Unique non-zero scores as candidate thresholds
        all_scores = sorted(
            {p["comprehend_score"] for p in self.results if p["comprehend_score"] > 0}
        )

        if not all_scores:
            print("Warning: No Comprehend scores found in data.")
            return {}, {}

        threshold_results = []
        for score in all_scores:
            metrics = self.calculate_threshold_metrics(score)
            threshold_results.append(metrics)

        # Review threshold: best F1
        best_review = max(threshold_results, key=lambda x: x["f1_score"])

        # Auto-hide threshold: precision >= 0.9, best recall
        high_precision = [t for t in threshold_results if t["precision"] >= 0.90]
        if high_precision:
            best_autohide = max(high_precision, key=lambda x: x["recall"])
        else:
            # Fall back to highest precision if nothing hits 0.9
            best_autohide = max(threshold_results, key=lambda x: x["precision"])

        return best_review, best_autohide

    def analyze_score_distribution(self) -> Dict:
        """Analyze score distributions for flagged vs unflagged posts."""
        flagged_scores = [
            p["comprehend_score"]
            for p in self.flagged_posts
            if p["comprehend_score"] > 0
        ]
        unflagged_scores = [
            p["comprehend_score"]
            for p in self.unflagged_posts
            if p["comprehend_score"] > 0
        ]

        stats = {"flagged": {}, "unflagged": {}}

        if flagged_scores:
            stats["flagged"] = {
                "count": len(flagged_scores),
                "mean": statistics.mean(flagged_scores),
                "median": statistics.median(flagged_scores),
                "min": min(flagged_scores),
                "max": max(flagged_scores),
                "stdev": statistics.stdev(flagged_scores)
                if len(flagged_scores) > 1
                else 0.0,
            }

        if unflagged_scores:
            stats["unflagged"] = {
                "count": len(unflagged_scores),
                "mean": statistics.mean(unflagged_scores),
                "median": statistics.median(unflagged_scores),
                "min": min(unflagged_scores),
                "max": max(unflagged_scores),
                "stdev": statistics.stdev(unflagged_scores)
                if len(unflagged_scores) > 1
                else 0.0,
            }

        return stats

    def analyze_by_flag_count(self) -> Dict:
        """Analyze Comprehend scores grouped by flag_count."""
        flag_groups = defaultdict(list)

        for post in self.results:
            if post["comprehend_score"] > 0:
                flag_groups[post["flag_count"]].append(post["comprehend_score"])

        analysis = {}
        for flag_count, scores in sorted(flag_groups.items()):
            if scores:
                analysis[flag_count] = {
                    "count": len(scores),
                    "mean": statistics.mean(scores),
                    "median": statistics.median(scores),
                    "min": min(scores),
                    "max": max(scores),
                }

        return analysis

    def analyze_label_distribution(self) -> Dict:
        """Analyze toxicity labels for flagged vs unflagged posts."""
        flagged_labels = defaultdict(int)
        unflagged_labels = defaultdict(int)

        for post in self.flagged_posts:
            label = post.get("comprehend_label") or "UNKNOWN"
            flagged_labels[label] += 1

        for post in self.unflagged_posts:
            label = post.get("comprehend_label") or "UNKNOWN"
            # Only count where Comprehend thinks it's toxic
            if post.get("comprehend_toxic_result") == "Y":
                unflagged_labels[label] += 1

        return {
            "flagged_by_humans": dict(flagged_labels),
            "unflagged_by_humans": dict(unflagged_labels),
        }

    def print_report(
        self,
        review_threshold: Dict,
        autohide_threshold: Dict,
        score_dist: Dict,
        flag_analysis: Dict,
        label_dist: Dict,
    ) -> None:
        """Print comprehensive analysis report."""
        print("=" * 80)
        print("AWS COMPREHEND THRESHOLD ANALYSIS REPORT")
        print("=" * 80)
        print()

        # Dataset Overview
        print("📊 DATASET OVERVIEW")
        print("-" * 80)
        total = len(self.results)
        flagged = len(self.flagged_posts)
        unflagged = len(self.unflagged_posts)
        print(f"Total Posts:                    {total:,}")
        if total > 0:
            print(
                f"Flagged by Users:               {flagged:,} ({flagged/total*100:.1f}%)"
            )
            print(
                f"Unflagged by Users:             {unflagged:,} ({unflagged/total*100:.1f}%)"
            )
        print()

        # Score Distribution
        print("📈 COMPREHEND SCORE DISTRIBUTION")
        print("-" * 80)
        if score_dist.get("flagged"):
            print("Flagged Posts (Human Ground Truth = Toxic):")
            print(f"  Count:                        {score_dist['flagged']['count']}")
            print(f"  Mean Score:                   {score_dist['flagged']['mean']:.4f}")
            print(f"  Median Score:                 {score_dist['flagged']['median']:.4f}")
            print(
                "  Range:                        "
                f"{score_dist['flagged']['min']:.4f} - {score_dist['flagged']['max']:.4f}"
            )
            print(
                f"  Std Deviation:                {score_dist['flagged']['stdev']:.4f}"
            )

        print()
        if score_dist.get("unflagged"):
            print("Unflagged Posts (Human Ground Truth = Not Toxic):")
            print(
                f"  Count:                        {score_dist['unflagged']['count']}"
            )
            print(
                f"  Mean Score:                   {score_dist['unflagged']['mean']:.4f}"
            )
            print(
                f"  Median Score:                 {score_dist['unflagged']['median']:.4f}"
            )
            print(
                "  Range:                        "
                f"{score_dist['unflagged']['min']:.4f} - {score_dist['unflagged']['max']:.4f}"
            )
            print(
                f"  Std Deviation:                {score_dist['unflagged']['stdev']:.4f}"
            )
        print()

        # Score by Flag Count
        if flag_analysis:
            print("📊 COMPREHEND SCORES BY FLAG COUNT")
            print("-" * 80)
            for flag_count, stats in flag_analysis.items():
                print(
                    f"Flag Count = {flag_count}: "
                    f"n={stats['count']}, "
                    f"mean={stats['mean']:.4f}, "
                    f"median={stats['median']:.4f}, "
                    f"range=[{stats['min']:.4f}, {stats['max']:.4f}]"
                )
            print()

        # Label Distribution
        if label_dist["flagged_by_humans"] or label_dist["unflagged_by_humans"]:
            print("🏷️  TOXICITY LABEL DISTRIBUTION")
            print("-" * 80)
            print("Labels in Flagged Posts:")
            for label, count in sorted(
                label_dist["flagged_by_humans"].items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {label}: {count}")

            print("\nLabels in Unflagged Posts (False Positives from Comprehend):")
            for label, count in sorted(
                label_dist["unflagged_by_humans"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                print(f"  {label}: {count}")
            print()

        # Recommended Thresholds
        print("=" * 80)
        print("🎯 RECOMMENDED THRESHOLDS")
        print("=" * 80)
        print()

        # Review Threshold
        print("1️⃣  REVIEW THRESHOLD (Flag for Manual Review)")
        print("-" * 80)
        print(f"Recommended Score:              {review_threshold['threshold']:.4f}")
        print()
        print("Performance Metrics:")
        print(
            f"  Precision:                    {review_threshold['precision']:.2%} "
            "(of flagged posts, what % are truly toxic)"
        )
        print(
            f"  Recall:                       {review_threshold['recall']:.2%} "
            "(of toxic posts, what % are caught)"
        )
        print(
            f"  F1 Score:                     {review_threshold['f1_score']:.4f} "
            "(balance of precision/recall)"
        )
        print(f"  Accuracy:                     {review_threshold['accuracy']:.2%}")
        print()
        print("Expected Outcomes:")
        print(
            f"  True Positives:               {review_threshold['true_positives']} "
            "(toxic posts correctly flagged)"
        )
        print(
            f"  False Positives:              {review_threshold['false_positives']} "
            "(safe posts incorrectly flagged)"
        )
        print(
            f"  False Negatives:              {review_threshold['false_negatives']} "
            "(toxic posts missed)"
        )
        print()
        print("Use Case: Flag posts for moderator review queue")
        print()

        # Auto-hide Threshold
        print("2️⃣  AUTO-HIDE THRESHOLD (Automatic Removal)")
        print("-" * 80)
        print(f"Recommended Score:              {autohide_threshold['threshold']:.4f}")
        print()
        print("Performance Metrics:")
        print(
            f"  Precision:                    {autohide_threshold['precision']:.2%} "
            "(of auto-hidden posts, what % are truly toxic)"
        )
        print(
            f"  Recall:                       {autohide_threshold['recall']:.2%} "
            "(of toxic posts, what % are auto-hidden)"
        )
        print(
            f"  F1 Score:                     {autohide_threshold['f1_score']:.4f}"
        )
        print(f"  Accuracy:                     {autohide_threshold['accuracy']:.2%}")
        print()
        print("Expected Outcomes:")
        print(
            f"  True Positives:               {autohide_threshold['true_positives']} "
            "(toxic posts correctly hidden)"
        )
        print(
            f"  False Positives:              {autohide_threshold['false_positives']} "
            "(safe posts incorrectly hidden)"
        )
        print(
            f"  False Negatives:              {autohide_threshold['false_negatives']} "
            "(toxic posts not auto-hidden)"
        )
        print()
        print("Use Case: Automatically hide posts with high confidence of toxicity")
        print()

        # Implementation Guide
        print("=" * 80)
        print("🔧 IMPLEMENTATION RECOMMENDATIONS")
        print("=" * 80)
        print(
            f"""
Recommended Configuration:

1. REVIEW THRESHOLD: {review_threshold['threshold']:.4f}
   - Posts scoring >= {review_threshold['threshold']:.4f} are flagged for manual review
   - Balances catching toxic content while keeping the review queue manageable
   - Expected: ~{review_threshold['true_positives']} true flags, ~{review_threshold['false_positives']} false flags

2. AUTO-HIDE THRESHOLD: {autohide_threshold['threshold']:.4f}
   - Posts scoring >= {autohide_threshold['threshold']:.4f} are automatically hidden
   - High precision minimises incorrectly hidden posts
   - Expected: ~{autohide_threshold['true_positives']} correctly hidden, ~{autohide_threshold['false_positives']} incorrectly hidden

Workflow:
┌─────────────────────────────────────────────────────────────┐
│ Post submitted                                              │
│ ↓                                                           │
│ AWS Comprehend analyzes content                             │
│ ↓                                                           │
│ Score >= {autohide_threshold['threshold']:.4f}? → YES → Auto-hide + notify moderators    │
│ ↓ NO                                                        │
│ Score >= {review_threshold['threshold']:.4f}? → YES → Flag for review queue              │
│ ↓ NO                                                        │
│ Post published normally                                     │
└─────────────────────────────────────────────────────────────┘

Next Steps:
1. Implement these thresholds in your moderation pipeline.
2. Monitor false positive/negative rates for a few weeks.
3. Collect moderator feedback on flagged content accuracy.
4. Fine-tune thresholds based on operational experience.
5. (Optional) Later, consider per-label thresholds for nuanced moderation.
"""
        )
        print("=" * 80)

    def export_results(
        self, review_threshold: Dict, autohide_threshold: Dict, output_file: str
    ) -> None:
        """Export recommendations to JSON file."""
        output = {
            "dataset_summary": {
                "total_posts": len(self.results),
                "flagged_posts": len(self.flagged_posts),
                "unflagged_posts": len(self.unflagged_posts),
                "flagged_percentage": (
                    len(self.flagged_posts) / len(self.results) * 100
                    if self.results
                    else 0
                ),
            },
            "recommended_thresholds": {
                "review_threshold": {
                    "score": review_threshold["threshold"],
                    "description": "Flag for manual review",
                    "precision": review_threshold["precision"],
                    "recall": review_threshold["recall"],
                    "f1_score": review_threshold["f1_score"],
                    "expected_true_positives": review_threshold["true_positives"],
                    "expected_false_positives": review_threshold["false_positives"],
                },
                "autohide_threshold": {
                    "score": autohide_threshold["threshold"],
                    "description": "Automatically hide post",
                    "precision": autohide_threshold["precision"],
                    "recall": autohide_threshold["recall"],
                    "f1_score": autohide_threshold["f1_score"],
                    "expected_true_positives": autohide_threshold["true_positives"],
                    "expected_false_positives": autohide_threshold["false_positives"],
                },
            },
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        print(f"\n✓ Results exported to {output_file}")

    def run_analysis(self, output_json: str = "threshold_recommendations.json") -> None:
        """Run complete analysis pipeline."""
        self.load_data()

        if not self.results:
            print("Error: No data to analyze.")
            return

        if not self.flagged_posts:
            print(
                "Warning: No flagged posts found. Cannot determine optimal thresholds "
                "without ground truth."
            )
            return

        review_threshold, autohide_threshold = self.find_optimal_thresholds()

        if not review_threshold or not autohide_threshold:
            print("Error: Could not compute thresholds (no candidate scores).")
            return

        score_dist = self.analyze_score_distribution()
        flag_analysis = self.analyze_by_flag_count()
        label_dist = self.analyze_label_distribution()

        self.print_report(
            review_threshold,
            autohide_threshold,
            score_dist,
            flag_analysis,
            label_dist,
        )
        self.export_results(review_threshold, autohide_threshold, output_json)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_thresholds.py <output.csv> [recommendations.json]")
        print("\nExample: python analyze_thresholds.py output.csv")
        print(
            "\nThis script analyzes AWS Comprehend moderation results and recommends\n"
            "score thresholds based on human flag_count as ground truth."
        )
        sys.exit(1)

    csv_file = sys.argv[1]
    json_file = sys.argv[2] if len(sys.argv) > 2 else "threshold_recommendations.json"

    analyzer = ThresholdAnalyzer(csv_file)
    analyzer.run_analysis(json_file)


if __name__ == "__main__":
    main()
