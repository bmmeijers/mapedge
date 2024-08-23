import unittest


def find_closest(sorted_list, target):
    """in a sorted list,
    find the index of the `target` value that is close
    """
    if len(sorted_list) == 0:
        return None
    low = 0
    high = len(sorted_list) - 1
    while low < high:
        mid = (low + high) // 2
        if sorted_list[mid] < target:
            low = mid + 1
        else:
            high = mid
    # At this point, low and high should be equal.
    if low == 0:
        return sorted_list[0]
    if low == len(sorted_list):
        return sorted_list[-1]
    before = sorted_list[low - 1]
    after = sorted_list[low]
    if after - target < target - before:
        return after
    else:
        return before


class TestFindApproximateValue(unittest.TestCase):
    def setUp(self):
        self.lst = [1, 2, 4, 5, 6, 8, 9]

    def test_value_in_list(self):
        self.assertEqual(find_closest(self.lst, 5), 5)

    def test_value_not_in_list(self):
        self.assertEqual(find_closest(self.lst, 7), 6)

    def test_value_outside_units(self):
        self.assertEqual(find_closest(self.lst, 10), 9)

    def test_empty_list(self):
        self.assertEqual(find_closest([], 5), None)


if __name__ == "__main__":
    unittest.main()
