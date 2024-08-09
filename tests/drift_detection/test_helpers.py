import unittest
from driftbench.drift_detection.helpers import (
    find_drift_segments,
    transform_drift_segments_into_binary,
    binarize_scores,
    find_changing_scores_in_ground_truth,
    find_complementary_segments,
    find_evaluation_segments,
)


class TestFindDriftSegments(unittest.TestCase):
    def test_no_drift_segment(self):
        prediction = [0, 0, 0, 0]
        expected = []
        actual = find_drift_segments(prediction)
        self.assertEqual(expected, actual)

    def test_one_drift_segment(self):
        prediction = [0, 0, 1, 1]
        expected = [(2, 3)]
        actual = find_drift_segments(prediction)
        self.assertEqual(expected, actual)

    def test_multitple_drift_segments(self):
        prediction = [0, 0, 1, 1, 0, 1, 1, 1, 0]
        expected = [(2, 3), (5, 7)]
        actual = find_drift_segments(prediction)
        self.assertEqual(expected, actual)


class TestTransformSegementsIntoBinary(unittest.TestCase):
    def test_no_segments_in_drift_segments(self):
        segments = []
        expected = []
        actual = transform_drift_segments_into_binary(segments, 0).tolist()
        self.assertEqual(expected, actual)

    def test_segment_in_drift_segments(self):
        segments = [(2, 4), (6, 9)]
        expected = [0, 0, 1, 1, 1, 0, 1, 1, 1, 1]
        actual = transform_drift_segments_into_binary(segments, 10).tolist()
        self.assertEqual(expected, actual)


class TestBinarizeScores(unittest.TestCase):
    def test_invalid_pair_of_scores_and_thresholds(self):
        scores = [0.2, 0.1, 0.01, 0.3]
        thresholds = [0.5, 0.5, 0.5]
        self.assertRaises(ValueError, binarize_scores, scores, thresholds)

    def test_no_scores_and_thresholds_provided(self):
        scores = []
        thresholds = []
        expected = []
        actual = binarize_scores(scores, thresholds).tolist()
        self.assertEqual(expected, actual)

    def test_no_score_above_threshold(self):
        scores = [0, 0, 0]
        thresholds = [1, 1, 1]
        expected = [0, 0, 0]
        actual = binarize_scores(scores, thresholds).tolist()
        self.assertEqual(expected, actual)

    def test_all_scores_above_threshold(self):
        scores = [1, 1, 1]
        thresholds = [0.5, 0.5, 0.5]
        expected = [1, 1, 1]
        actual = binarize_scores(scores, thresholds).tolist()
        self.assertEqual(expected, actual)

    def test_some_scores_above_threshold(self):
        scores = [0.2, 0.1, 0.01, 0.6, 0.7, 0.9, 0.1, 0.05, 0.02, 0.05]
        thresholds = [0.5] * 10
        expected = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        actual = binarize_scores(scores, thresholds).tolist()
        self.assertEqual(expected, actual)


class TestFindChangingScoresInGroundTruth(unittest.TestCase):
    def test_invalid_pair_of_scores_and_targets(self):
        scores = [0.2, 0.1, 0.01, 0.3]
        ground_truth = [0, 1, 1]
        self.assertRaises(ValueError, find_changing_scores_in_ground_truth, scores, ground_truth)

    def test_one_drift_in_ground_truth(self):
        scores = [0.2, 0.3, 0.6, 0.7, 0.8, 0.4, 0.1]
        ground_truth = [0, 0, 1, 1, 1, 0, 0]
        expected = [0.2, 0.3, 0.6, 0.7]
        actual = find_changing_scores_in_ground_truth(scores, ground_truth).tolist()
        self.assertEqual(expected, actual)

    def test_duplicate_scores(self):
        scores = [0.3, 0.3, 0.6, 0.7, 0.8, 0.5, 0.1]
        ground_truth = [0, 0, 1, 1, 1, 0, 0]
        expected = [0.3, 0.6, 0.7]
        actual = find_changing_scores_in_ground_truth(scores, ground_truth).tolist()
        self.assertEqual(expected, actual)

    def test_multiple_drifts_in_ground_truth(self):
        scores = [0.1, 0.5, 0.6, 0.7, 0.3, 0.4, 0.9, 0.8, 0.05, 0.2]
        ground_truth = [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
        expected = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        actual = find_changing_scores_in_ground_truth(scores, ground_truth).tolist()
        self.assertEqual(expected, actual)


class TestFindComplementarySegments(unittest.TestCase):
    def test_no_drifts_in_input(self):
        segments = []
        expected = [(0, 9)]
        actual = find_complementary_segments(segments, 10)
        self.assertEqual(expected, actual)

    def test_only_drifts_in_input(self):
        segments = [(0, 9)]
        expected = []
        actual = find_complementary_segments(segments, 10)
        self.assertEqual(expected, actual)

    def test_one_drift_segment_in_input(self):
        segments = [(3, 5)]
        expected = [(0, 2), (6, 9)]
        actual = find_complementary_segments(segments, 10)
        self.assertEqual(expected, actual)

    def test_multiple_drift_segments_in_input(self):
        segments = [(5, 9), (15, 17)]
        expected = [(0, 4), (10, 14), (18, 19)]
        actual = find_complementary_segments(segments, 20)
        self.assertEqual(expected, actual)


class TestFindEvaluationSegments(unittest.TestCase):
    def test_whole_input_as_evaluation_segment(self):
        pred_segments = [(0, 9)]
        target_segments = [(3, 6)]
        expected = [(0, 9)]
        actual = find_evaluation_segments(pred_segments, target_segments)
        self.assertEqual(expected, actual)

    def test_target_segment_as_evaluation_segment(self):
        pred_segments = [(4, 6)]
        target_segments = [(3, 7)]
        expected = [(3, 7)]
        actual = find_evaluation_segments(pred_segments, target_segments)
        self.assertEqual(expected, actual)

    def test_overlapping_evaluation_segments(self):
        pred_segments = [(5, 9), (13, 17)]
        target_segments = [(8, 14)]
        expected = [(5, 17)]
        actual = find_evaluation_segments(pred_segments, target_segments)
        self.assertEqual(expected, actual)

    def test_multiple_evaluation_segments(self):
        pred_segments = [(8, 9), (15, 18)]
        target_segments = [(6, 9), (13, 17)]
        expected = [(6, 9), (13, 18)]
        actual = find_evaluation_segments(pred_segments, target_segments)
        self.assertEqual(expected, actual)
