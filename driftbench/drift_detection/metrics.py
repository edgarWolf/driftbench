"""The metrics module."""

from abc import (
    ABCMeta,
    abstractmethod,
)
import numpy as np
from sklearn import metrics
from driftbench.drift_detection.helpers import (
    find_drift_segments,
    find_evaluation_segments,
    has_overlap,
)


class Metric(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, prediction, targets):
        pass

    @property
    def name(self):
        return self.__class__.__name__

class RewardScore(Metric):
    pass


class JaccardScore(Metric):
    def __init__(self, reward_score=None):
        if reward_score is not None and not isinstance(reward_score, RewardScore):
            raise ValueError("The reward score has to be an instance of the RewardScore-class.")
        self.reward_score = reward_score

    def __call__(self, prediction, targets):
        score = metrics.jaccard_score(targets, prediction)
        if self.reward_score:
            score = np.mean([score, self.reward_score(prediction, targets)])
        return score


class JaccardOverThresholdScore(Metric):
    def __init__(self, thresholds, score_agg_func, reward_score=None):
        self.thresholds = thresholds
        self.score_agg_func = score_agg_func
        self.reward_score = reward_score

    def __call__(self, prediction, targets):
        jaccard_score = JaccardScore(reward_score=self.reward_score)
        jaccards = [jaccard_score((prediction > threshold).astype(int), targets)
                    for threshold in self.thresholds]
        return self.score_agg_func(jaccards)


class DistanceScore(RewardScore):
    def __call__(self, prediction, targets):
        target_indices = np.where(targets == 1)[0]
        pred_indices = np.where(prediction == 1)[0]
        pred_intersect = np.intersect1d(target_indices, pred_indices)
        target_diff = np.diff(target_indices)
        prediction_diff = np.diff(pred_intersect)
        prediction_distance_score = np.where(prediction_diff == 1)[0].shape[0]
        target_distance_score = np.where(target_diff == 1)[0].shape[0]
        return prediction_distance_score / target_distance_score


class ContinousSegmentRewardScore(RewardScore):
    def __init__(self, alpha=2):
        self.alpha = alpha

    def __call__(self, prediction, targets):
        gt_segments = find_drift_segments(targets)
        max_reward = np.sum(
            [self.alpha * (end - start + 1) - (self.alpha - 1) if (end - start > 0) else 0 for start, end in
             gt_segments])
        correct_predictions = np.logical_and(prediction, targets).astype(int)
        correct_pred_segments = find_drift_segments(correct_predictions)
        pred_reward = np.sum(
            [self.alpha * (end - start + 1) - (self.alpha - 1) if (end - start > 0) else 0 for start, end in
             correct_pred_segments])
        return pred_reward / max_reward


class OverlapScore(Metric):
    def __call__(self, prediction, targets):
        drift_segments = find_drift_segments(targets)
        pred_segments = find_drift_segments(prediction)
        scores = []
        for start, end in drift_segments:
            # Go through all ground truth segments in isolation, as evaluation segments may overlap.
            segment = find_evaluation_segments([(start, end)], pred_segments)
            if not segment:
                scores.append(0.0)
            else:
                # Always take the first one, since only one segment per drift segment is constructed
                seg_start, seg_end = segment[0]
                N = seg_end - seg_start + 1
                for pred_start, pred_end in pred_segments:
                    if has_overlap((seg_start, seg_end), (pred_start, pred_end)):
                        overlap_start, overlap_end = np.max([start, pred_start]), np.min([end, pred_end])
                        overlap_len = np.max([0, overlap_end - overlap_start + 1])
                        scores.append(overlap_len / N)
        return np.mean(scores)


class TemporalAUC(Metric):
    """The temporal area under the curve."""
    _supported_integration_rules = ["step", "trapez"]

    def __init__(self, rule="step"):
        if not self._is_valid_rule(rule):
            raise ValueError(
                f"Unknown rule {rule}: Supported integration rules are {TemporalAUC._supported_integration_rules}.")
        self.rule = rule

    def _is_valid_rule(self, rule):
        return rule in TemporalAUC._supported_integration_rules

    def __call__(self, prediction, targets, return_scores=False):
        overlap_score = OverlapScore()
        thresholds = np.unique(prediction)
        thresholds = np.append(thresholds, np.inf)
        thresholds.sort()
        scores = np.zeros((thresholds.shape[0], 2))
        for i, threshold in enumerate(thresholds):
            bin_predictions = (prediction >= threshold).astype(int)
            fpr = (bin_predictions[targets == 0] == 1).sum() / (targets == 0).sum()
            scores[i] = [overlap_score(bin_predictions, targets), fpr]
        os, fpr = scores[:, 0], scores[:, 1]

        if return_scores:
            return thresholds, fpr, os

        if self.rule == "trapez":
            return metrics.auc(fpr, os)
        elif self.rule == "step":
            return np.sum(np.diff(fpr[::-1]) * os[::-1][:-1])

    @property
    def name(self):
        return f'TAUC-{self.rule}'


class SoftOverlapScore(Metric):
    def __call__(self, prediction, targets):
        drift_segments = find_drift_segments(targets)
        pred_segments = find_drift_segments(prediction)
        scores = []
        for start, end in drift_segments:
            # Go through all pred segments
            T = [pred_segment for pred_segment in pred_segments if has_overlap(pred_segment, [start, end])]

            b_min = min([ps[0] for ps in T] + [start])
            b_max = max([ps[1] for ps in T] + [end])

            overlap = sum([pred_segment[1]-pred_segment[0]+1 for pred_segment in T])
            scores.append(overlap / (b_max - b_min+1))
        return np.mean(scores)

    def _compute_overlap(self, segment_a, segment_b):
        return min([segment_a[1], segment_b[1]]) - max([segment_a[0], segment_b[0]])


class SoftTAUC(Metric):
    """A softened version of the TAUC."""
    _supported_integration_rules = ["step", "trapez"]

    def __init__(self, rule="step"):
        self.rule = rule

    def __call__(self, prediction, targets, return_scores=False):
        overlap_score = SoftOverlapScore()
        thresholds = np.unique(prediction)
        thresholds = np.append(thresholds, np.inf)
        thresholds.sort()
        scores = np.zeros((thresholds.shape[0], 2))
        for i, threshold in enumerate(thresholds):
            bin_predictions = (prediction >= threshold).astype(int)
            fpr = (bin_predictions[targets == 0] == 1).sum() / (targets == 0).sum()
            scores[i] = [overlap_score(bin_predictions, targets), fpr]
        os, fpr = scores[:, 0], scores[:, 1]

        if return_scores:
            return thresholds, fpr, os

        if self.rule == "trapez":
            return metrics.auc(fpr, os)
        elif self.rule == "step":
            return np.sum(np.diff(fpr[::-1]) * os[::-1][:-1])


class AUC(Metric):
    """The area under the curve."""

    def __call__(self, prediction, target):
        prediction = np.nan_to_num(prediction, 0)
        return metrics.roc_auc_score(target, prediction)
