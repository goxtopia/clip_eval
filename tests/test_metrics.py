
import unittest
import torch
import numpy as np
from src.metrics import compute_metrics, compute_t2i_metrics

class TestMetrics(unittest.TestCase):
    def test_compute_metrics_i2t(self):
        # Scenario: 3 items, 2 classes ["cat", "dog"]
        # Item 0: GT {"cat"}, Text "cat" -> Index 0
        # Item 1: GT {"dog"}, Text "dog" -> Index 1
        # Item 2: GT {"cat", "feline"}, Text "cat" -> Index 0

        # Candidate texts: ["cat", "dog"]
        candidate_texts = ["cat", "dog"]

        # GT Sets (expanded)
        gt_class_sets = [
            {"cat"},
            {"dog"},
            {"cat", "feline"}
        ]

        # Predictions (indices into candidate_texts)
        # Item 0: Pred [0, 1] ("cat", "dog") -> Top1 Hit
        # Item 1: Pred [0, 1] ("cat", "dog") -> Top1 Miss, Top5 Miss (if only 2 cands, wait. dog is at 1. Miss)
        # Item 2: Pred [1, 0] ("dog", "cat") -> Top1 Miss, Top5 Hit (cat is at 1 which is 2nd pos)

        pred_classes = torch.tensor([
            [0, 1],
            [0, 1],
            [1, 0]
        ])

        # Class to images map
        class_to_images = {
            "cat": [0, 2],
            "dog": [1]
        }

        metrics, per_class, bad_cases = compute_metrics(
            pred_classes, gt_class_sets, candidate_texts, class_to_images
        )

        # Verify Global Stats
        # Hits Top1: 1.0 (Item 0), 0.0 (Item 1), 0.0 (Item 2) -> Avg 1/3 = 0.333
        # Hits Top5: 1.0 (Item 0), 1.0 (Item 1), 1.0 (Item 2) -> Avg 3/3 = 1.0
        # (Item 1 preds: [0, 1] -> "cat", "dog". "dog" is in GT {"dog"}. So it's a hit.)

        self.assertAlmostEqual(metrics["global_top1"], 1/3)
        self.assertAlmostEqual(metrics["global_top5"], 1.0)

        # Verify Bad Cases
        # All hit Top5, so bad_cases should be empty
        self.assertEqual(len(bad_cases), 0)
        # Item 2 missed Top1 but hit Top5, so it shouldn't be in bad_cases (which is for failure to retrieve)
        # Wait, compute_metrics logic: "if hit5 == 0.0: bad_cases[i] = ..."
        self.assertNotIn(2, bad_cases)

        # Verify Per Class
        # Cat: Item 0 (1/1), Item 2 (0/1). Avg Top1: 0.5. Avg Top5: 1.0.
        # Dog: Item 1 (0/0). Avg Top1: 0.0. Avg Top5: 0.0.

        cat_stats = next(s for s in per_class if s["label"] == "cat")
        self.assertAlmostEqual(cat_stats["top1"], 0.5)
        self.assertAlmostEqual(cat_stats["top5"], 1.0)

        dog_stats = next(s for s in per_class if s["label"] == "dog")
        self.assertAlmostEqual(dog_stats["top1"], 0.0)

    def test_compute_t2i_metrics(self):
        # Scenario: 2 queries
        # Query 0: "cat"
        # Query 1: "dog"
        query_texts = ["cat", "dog"]

        # Images (GT sets)
        # Img 0: {"cat"}
        # Img 1: {"dog"}
        gt_class_sets = [{"cat"}, {"dog"}]

        # Predictions (indices into images)
        # Query 0: [0, 1] -> Img 0 (cat) HIT, Img 1 (dog)
        # Query 1: [0, 1] -> Img 0 (cat) MISS, Img 1 (dog) HIT (at rank 2)
        pred_images = torch.tensor([
            [0, 1],
            [0, 1]
        ])

        metrics, per_class, bad_cases = compute_t2i_metrics(
            pred_images, gt_class_sets, query_texts
        )

        # Global
        # Q0: Top1=1.0, Top5=1.0
        # Q1: Top1=0.0, Top5=1.0
        self.assertAlmostEqual(metrics["global_top1"], 0.5)
        self.assertAlmostEqual(metrics["global_top5"], 1.0)

        # Bad cases (Top 5 miss) -> None
        self.assertEqual(len(bad_cases), 0)

if __name__ == "__main__":
    unittest.main()
