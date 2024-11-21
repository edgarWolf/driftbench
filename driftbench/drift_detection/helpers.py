import numpy as np


def find_drift_segments(prediction):
    # Handle case when no timestamp indicates drift
    if np.all(prediction == 0):
        return []
    # Handle case when every timestep indicates drift
    if np.all(prediction == 1):
        return [(0, prediction.shape[0] - 1)]
    diff = np.diff(prediction)
    starts, ends = np.where(diff == 1)[0] + 1, np.where(diff == -1)[0]
    # Handle case when drift is until end of prediction
    if starts.shape[0] > ends.shape[0]:
        return [(start, end) for start, end in zip(starts, ends)] + [(starts[-1], prediction.shape[0] - 1)]
    # Handle case when drift is right at the beginning of the prediction
    if starts.shape[0] < ends.shape[0]:
        return [(0, ends[0])] + [(start, end) for start, end in zip(starts, ends[1:])]
    return [(start, end) for start, end in zip(starts, ends)]


def transform_drift_segments_into_binary(drift_segments, input_len):
    # Validation
    for start, end in drift_segments:
        if not (0 <= start < input_len and 0 <= end < input_len):
            raise ValueError(
                f"""Invalid drift segment {start, end}: 
                All drift bounds have to be greater than or equal to 0 and less than the input length.""")
        if start > end:
            raise ValueError(
                f"Invalid drift segment {start, end}: End index must be greater than the start index.")

    drifts_binary = np.zeros(input_len, dtype=int)
    for start, end in drift_segments:
        drifts_binary[start:end + 1] = 1

    return drifts_binary


def binarize_scores(scores, thresholds):
    # Validation
    if len(scores) != len(thresholds):
        raise ValueError(
            f"""Invalid pair of scores and thresholds. 
            The lenght of the scores and the thresholds must match.
            Length of scores: {len(scores)}; Length of thresholds: {len(thresholds)}""")
    scores, thresholds = np.asarray(scores), np.asarray(thresholds)
    return (scores > thresholds).astype(int)


def find_changing_scores_in_ground_truth(scores, ground_truth):
    if len(scores) != len(ground_truth):
        raise ValueError(
            f"""Invalid pair of scores and targets. 
                The length of the scores and the ground truth must match.
                Length of scores: {len(scores)}; Length of ground truth: {len(ground_truth)}""")
    scores_sorted = np.sort(scores)
    drift_segments = find_drift_segments(ground_truth)
    changing_indices = [(start - 1, start, end, end + 1) for start, end in drift_segments]
    changing_indices = list(sum(changing_indices, ()))
    return np.unique(scores_sorted[changing_indices])


def has_overlap(segment_a, segment_b):
    (s1, e1), (s2, e2) = segment_a, segment_b
    return s1 <= s2 <= e1 or s1 <= e2 <= e1 or s2 <= s1 <= e2 or s2 <= e1 <= e2


def find_complementary_segments(segments, N):
    segments_sorted = np.sort(segments)
    complementary_segments = []

    if segments_sorted.size == 0:
        complementary_segments = [(0, N - 1)]
    else:
        # Check the gap before the first tuple
        if segments_sorted[0][0] > 0:
            complementary_segments.append((0, segments_sorted[0][0] - 1))

        # Check the gap between consecutive tuples
        for i in range(len(segments_sorted) - 1):
            start_gap = segments_sorted[i][1] + 1
            end_gap = segments_sorted[i + 1][0] - 1
            if start_gap <= end_gap:
                complementary_segments.append((start_gap, end_gap))

        # Check the gap after the last tuple
        if segments_sorted[-1][1] < N - 1:
            complementary_segments.append((segments_sorted[-1][1] + 1, N - 1))
    return complementary_segments


def find_evaluation_segments(prediction_segments, target_segments):
    segments = []
    for start, end in target_segments:
        overlapped = False
        seg_start, seg_end = start, end
        for s, e in prediction_segments:
            if has_overlap((seg_start, seg_end), (s, e)):
                overlapped = True
                seg_start = np.min([s, seg_start])
                seg_end = np.max([e, seg_end])
        if overlapped:
            segments.append((seg_start, seg_end))
    return segments
