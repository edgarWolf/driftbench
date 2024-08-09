import numpy as np
import unittest

from driftbench.drift_detection.metrics import (
    JaccardScore,
    JaccardOverThresholdScore,
    DistanceScore,
    ContinousSegmentRewardScore,
    OverlapScore,
    TemporalAUC,
)


class TestJaccardIndex(unittest.TestCase):
    def test_jaccard_index_equals_0(self):
        prediction = [0, 0, 0, 0]
        targets = [1, 1, 1, 1]
        expected = 0.0
        jaccard_index = JaccardScore()
        actual = jaccard_index(prediction, targets)
        self.assertAlmostEqual(expected, actual)

    def test_jaccard_index_equals_1(self):
        prediction = [0, 1, 0, 1]
        targets = [0, 1, 0, 1]
        expected = 1.0
        jaccard_index = JaccardScore()
        actual = jaccard_index(prediction, targets)
        self.assertAlmostEqual(expected, actual)

    def test_jaccard_index_equals_0_5(self):
        prediction = [0, 1, 1, 0, 0, 0, 0]
        targets = [0, 1, 1, 0, 1, 1, 0]
        expected = 0.5
        jaccard_index = JaccardScore()
        actual = jaccard_index(prediction, targets)
        self.assertAlmostEqual(expected, actual)

    def test_additional_score_yields_other_score(self):
        targets = np.array([0, 1, 1, 1, 0])
        prediction = np.array([0, 1, 1, 0, 0])
        jaccard_index1 = JaccardScore(reward_score=DistanceScore())
        jaccard_index2 = JaccardScore()
        better_score = jaccard_index1(prediction, targets)
        worse_score = jaccard_index2(prediction, targets)
        self.assertTrue(better_score, worse_score)


class TestJaccardOverThreshold(unittest.TestCase):
    def test_no_score_above_threshold_with_wrong_prediction(self):
        scores = np.array([0, 0, 0, 0])
        targets = np.array([0, 1, 1, 0])
        thresholds = np.array([1.0])
        agg_func = np.mean
        expected = 0.0
        jaccard_over_threshold = JaccardOverThresholdScore(thresholds, agg_func)
        actual = jaccard_over_threshold(scores, targets)
        self.assertEqual(expected, actual)

    def test_score_above_threshold_with_correct_prediction(self):
        scores = np.array([0, 1, 1, 0])
        targets = np.array([0, 1, 1, 0])
        thresholds = np.array([0.5])
        agg_func = np.mean
        expected = 1.0
        jaccard_over_threshold = JaccardOverThresholdScore(thresholds, agg_func)
        actual = jaccard_over_threshold(scores, targets)
        self.assertEqual(expected, actual)

    def test_score_above_threshold_with_partial_correct_prediction(self):
        scores = np.array([0, 0, 1, 1])
        targets = np.array([0, 1, 1, 0])
        thresholds = np.array([0.5])
        agg_func = np.mean
        expected = 1 / 3
        jaccard_over_threshold = JaccardOverThresholdScore(thresholds, agg_func)
        actual = jaccard_over_threshold(scores, targets)
        self.assertEqual(expected, actual)


class TestDistanceScore(unittest.TestCase):
    def test_perfect_score(self):
        targets = np.array([0, 1, 1, 1, 0])
        scores = np.array([0, 1, 1, 1, 0])
        expected = 1.0
        distance_score = DistanceScore()
        actual = distance_score(scores, targets)
        self.assertEqual(expected, actual)

    def test_wrong_prediction_score(self):
        targets = np.array([0, 1, 1, 1, 0])
        scores = np.array([1, 0, 0, 0, 1])
        expected = 0.0
        distance_score = DistanceScore()
        actual = distance_score(scores, targets)
        self.assertEqual(expected, actual)

    def test_no_continous_prediction_score_equals_zero(self):
        targets = np.array([0, 1, 1, 1, 0])
        scores = np.array([0, 1, 0, 1, 0])
        expected = 0.0
        distance_score = DistanceScore()
        actual = distance_score(scores, targets)
        self.assertEqual(expected, actual)

    def test_equal_number_of_ones_different_score(self):
        targets = np.array([0, 1, 1, 1, 0])
        scores1 = np.array([0, 1, 1, 0, 0])
        scores2 = np.array([0, 1, 0, 1, 0])
        distance_score = DistanceScore()
        better_score = distance_score(scores1, targets)
        worse_score = distance_score(scores2, targets)
        self.assertTrue(better_score > worse_score)

    def test_multiple_segments(self):
        targets = np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0])
        scores1 = np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 0])
        scores2 = np.array([0, 1, 0, 1, 0, 0, 0, 1, 1, 0])
        distance_score = DistanceScore()
        better_score = distance_score(scores1, targets)
        worse_score = distance_score(scores2, targets)
        self.assertTrue(better_score > worse_score)


class TestContinousSegmentsRewardScore(unittest.TestCase):
    def test_perfect_score(self):
        targets = np.array([0, 1, 1, 1, 0])
        scores = np.array([0, 1, 1, 1, 0])
        expected = 1.0
        continuous_segments_reward_score = ContinousSegmentRewardScore()
        actual = continuous_segments_reward_score(scores, targets)
        self.assertEqual(expected, actual)

    def test_wrong_prediction_score(self):
        targets = np.array([0, 1, 1, 1, 0])
        scores = np.array([1, 0, 0, 0, 1])
        expected = 0.0
        continuous_segments_reward_score = ContinousSegmentRewardScore()
        actual = continuous_segments_reward_score(scores, targets)
        self.assertEqual(expected, actual)

    def test_no_continous_prediction_score_equals_zero(self):
        targets = np.array([0, 1, 1, 1, 0])
        scores = np.array([0, 1, 0, 1, 0])
        expected = 0.0
        continuous_segments_reward_score = ContinousSegmentRewardScore()
        actual = continuous_segments_reward_score(scores, targets)
        self.assertEqual(expected, actual)

    def test_equal_number_of_ones_different_score(self):
        targets = np.array([0, 1, 1, 1, 0])
        scores1 = np.array([0, 1, 1, 0, 0])
        scores2 = np.array([0, 1, 0, 1, 0])
        continuous_segments_reward_score = ContinousSegmentRewardScore()
        better_score = continuous_segments_reward_score(scores1, targets)
        worse_score = continuous_segments_reward_score(scores2, targets)
        self.assertTrue(better_score > worse_score)

    def test_lower_alpha_equals_higher_score_with_same_non_perfect_prediction(self):
        """
        This test ensures, that a lower alpha value returns a higher reward score
        than a higher alpha value. This is due to the maximum reward being a
        multiple of alpha, and a non-perfect prediction will always miss such a multiple
        of alpha. The higher the alpha, the higher multiple will be missing, and thus
        returning a lower score.
        """
        targets = np.array([0, 1, 1, 1, 0])
        scores = np.array([0, 1, 1, 0, 0])
        higher_continuous_segments_reward_score = ContinousSegmentRewardScore(alpha=1)
        lower_continuous_segments_reward_score = ContinousSegmentRewardScore(alpha=2)
        higher_score = higher_continuous_segments_reward_score(scores, targets)
        lower_score = lower_continuous_segments_reward_score(scores, targets)
        self.assertTrue(higher_score > lower_score)

    def test_multiple_segments(self):
        targets = np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0])
        scores1 = np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 0])
        scores2 = np.array([0, 1, 0, 1, 0, 0, 0, 1, 1, 0])
        continuous_segments_reward_score = ContinousSegmentRewardScore()
        better_score = continuous_segments_reward_score(scores1, targets)
        worse_score = continuous_segments_reward_score(scores2, targets)
        self.assertTrue(better_score > worse_score)


class TestOverlapScore(unittest.TestCase):
    def test_prediction_always_drift(self):
        prediction = np.ones(10)
        targets = np.zeros(10)
        targets[3:6] = 1
        expected = 3 / 10
        actual = OverlapScore()(prediction, targets)
        self.assertEqual(expected, actual)

    def test_prediction_no_drift(self):
        prediction = np.zeros(10)
        targets = np.zeros(10)
        targets[3:6] = 1
        expected = 0.0
        actual = OverlapScore()(prediction, targets)
        self.assertEqual(expected, actual)

    def test_perfect_prediction(self):
        targets = np.zeros(10)
        targets[3:6] = 1
        prediction = targets[:]
        expected = 1.0
        actual = OverlapScore()(prediction, targets)
        self.assertEqual(expected, actual)

    def test_prediction_overlaps_targets(self):
        prediction = np.zeros(10)
        prediction[3:8] = 1
        targets = np.zeros(10)
        targets[4:7] = 1
        expected = 3 / 5
        actual = OverlapScore()(prediction, targets)
        self.assertEqual(expected, actual)

    def test_prediction_contained_in_targets(self):
        prediction = np.zeros(10)
        prediction[4:7] = 1
        targets = np.zeros(10)
        targets[3:8] = 1
        expected = 3 / 5
        actual = OverlapScore()(prediction, targets)
        self.assertEqual(expected, actual)

    def test_multiple_drifts_in_targets(self):
        prediction = np.zeros(20)
        prediction[4:7] = 1
        prediction[10:16] = 1
        targets = np.zeros(20)
        targets[3:8] = 1
        targets[12:18] = 1
        expected = ((3 / 5) + (1 / 2)) / 2
        actual = OverlapScore()(prediction, targets)
        self.assertEqual(expected, actual)

class TestTemporalAUC(unittest.TestCase):

    def test_invalid_rule(self):
        rule = "lagrange"
        self.assertRaises(ValueError, TemporalAUC, rule)

    def test_prediction_only_drifts(self):
        prediction = np.ones(10)
        targets = np.zeros(10)
        targets[3:6] = 1
        score = TemporalAUC(rule="trapez")(prediction, targets)
        actual = 0 < score < 1 /3
        expected = True
        self.assertEqual(expected, actual)
        expected = TemporalAUC(rule="step")(prediction, targets)
        actual = 0
        self.assertEqual(expected, actual)

    def test_prediction_no_drifts(self):
        prediction = np.zeros(10)
        targets = np.zeros(10)
        targets[3:6] = 1
        expected = (3 / 10) / 2
        actual = TemporalAUC(rule="trapez")(prediction, targets)
        self.assertEqual(expected, actual)
        expected = TemporalAUC(rule="step")(prediction, targets)
        actual = 0
        self.assertEqual(expected, actual)

    def test_ordinary_prediction(self):
        prediction = [0.2, 0.4, 0.7, 0.7, 0.7, 0.6, 0.6, 0.5, 0.3, 0.1]
        targets = np.zeros(10)
        targets[3:6] = 1
        score = TemporalAUC(rule="trapez")(prediction, targets)
        expected = True
        actual = 0 < score < 1
        self.assertEqual(expected, actual)
        score = TemporalAUC(rule="step")(prediction, targets)
        expected = True
        actual = 0 < score < 1
        self.assertEqual(expected, actual)

    def test_perfect_prediction(self):
        targets = np.zeros(10)
        targets[3:6] = 1
        prediction = targets[:]
        score1 = TemporalAUC(rule="trapez")(prediction, targets)
        expected = 0 < score1 < 1
        actual = True
        self.assertEqual(expected, actual)
        score2 = TemporalAUC(rule="step")(prediction, targets)
        self.assertTrue(score2 > score1)
        self.assertTrue(0 <= score2 <= 1)
