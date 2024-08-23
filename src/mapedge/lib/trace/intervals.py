import unittest
import numpy as np


# FIXME: instead of the threshold inside, make it a replacement of num_stddev arg
#
def generate_intervals__num_std_dev__above(arr, num_std_dev):
    avg = np.average(arr)
    std_dev = np.std(arr)
    # print(avg, std_dev)
    threshold = avg + num_std_dev * std_dev
    return _generate_intervals(arr, threshold)


def generate_intervals_above_threshold(arr, threshold):
    # avg = np.average(arr)
    # std_dev = np.std(arr)
    # print(avg, std_dev)
    # threshold = avg + num_std_dev * std_dev
    return _generate_intervals(arr, threshold)


def _generate_intervals(arr, threshold):
    intervals = []
    start_idx = None

    # FIXME: the > and <= comparisions make that we can only search for above a threshold
    for end_idx in range(len(arr)):
        if (arr[end_idx] > threshold) and start_idx is None:
            start_idx = end_idx
        elif (arr[end_idx] <= threshold) and start_idx is not None:
            intervals.append((start_idx, end_idx))
            start_idx = None

    if start_idx is not None:
        intervals.append((start_idx, len(arr)))

    return intervals


def closest_interval(intervals, value):
    """interval that is closest to a value in distance"""
    # Initialize minimum distance as infinity
    min_distance = float("inf")
    closest = None

    # Iterate over all intervals
    for interval in intervals:
        # Calculate distance from the value to the current interval
        if value < interval[0]:
            distance = interval[0] - value
        elif value > interval[1]:
            distance = value - interval[1]
        else:
            distance = 0

        # Update minimum distance and closest interval
        if distance < min_distance:
            min_distance = distance
            closest = interval

    return closest


class TestGenerateIntervals(unittest.TestCase):
    def setUp(self):
        self.arr1 = np.array(
            [1, 2, 3, 7, 8, 9, 2, 3, 4, 1, 2, 3, 7, 8, 9, 2, 3, 4, 8, 9]
        )
        self.arr2 = np.array([1, 1, 1, 1, 1])
        self.arr3 = np.array([])

    def test_normal_case(self):
        result = generate_intervals__num_std_dev__above(self.arr1, 1)
        self.assertEqual(result, [(4, 6), (13, 15), (18, 20)])

    def test_all_same_values(self):
        result = generate_intervals__num_std_dev__above(self.arr2, 1)
        self.assertEqual(result, [])

    def test_empty_array(self):
        result = generate_intervals__num_std_dev__above(self.arr3, 1)
        self.assertEqual(result, [])


def largest_interval(lst):
    for interval in sorted(lst, key=lambda x: x[1] - x[0], reverse=True):
        return interval


def similar_sized_interval(intervals, size):
    min_diff = float("inf")
    closest_interval = None
    for interval in intervals:
        interval_size = interval[1] - interval[0]
        diff = abs(interval_size - size)
        # print(f"{interval_size=} {diff=}")
        if diff < min_diff:
            min_diff = diff
            closest_interval = interval
    return closest_interval


class TestSimilarSizedInterval(unittest.TestCase):
    def test_similar_sized_interval1(self):
        self.assertEqual(
            similar_sized_interval([(1, 3), (2, 5), (4, 8), (6, 10)], 6), (4, 8)
        )

    def test_similar_sized_interval2(self):
        self.assertEqual(
            similar_sized_interval([(1, 3), (2, 5), (4, 8), (6, 10)], 2), (1, 3)
        )

    def test_similar_sized_interval3(self):
        self.assertEqual(
            similar_sized_interval([(1, 3), (2, 5), (4, 8), (6, 10)], 3), (2, 5)
        )

    def test_similar_sized_interval4(self):
        self.assertEqual(similar_sized_interval([(1, 3)], 2), (1, 3))

    def test_similar_sized_interval5(self):
        self.assertEqual(similar_sized_interval([], 2), None)


class TestClosestInterval(unittest.TestCase):
    def setUp(self):
        self.intervals = [(1, 2), (4, 5), (8, 9)]

    def test_value_less_than_all_intervals(self):
        value = 0
        self.assertEqual(closest_interval(self.intervals, value), (1, 2))

    def test_value_greater_than_all_intervals(self):
        value = 10
        self.assertEqual(closest_interval(self.intervals, value), (8, 9))

    def test_value_within_interval(self):
        value = 4.5
        self.assertEqual(closest_interval(self.intervals, value), (4, 5))

    def test_value_equal_to_interval_boundary(self):
        value = 2
        self.assertEqual(closest_interval(self.intervals, value), (1, 2))


if __name__ == "__main__":
    unittest.main(verbosity=5)
