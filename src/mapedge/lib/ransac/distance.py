# distance.py
import unittest
import numpy as np


def calculate_distances(points1, points2):
    """
    Calculates the Euclidean distance between corresponding points in two arrays.

    Args:
        points1 (np.array): Array of shape (n, 2) representing the first set of 2D points.
        points2 (np.array): Array of shape (n, 2) representing the second set of 2D points.

    Returns:
        np.ndarray: A vector of distances between the points.
    """
    assert points1.shape == points2.shape, "Input arrays must have the same shape"
    assert points1.shape[1] == 2, "Input arrays must have 2 columns (x, y)"

    # Calculate the squared Euclidean distance
    squared_distances = np.sum((points1 - points2) ** 2, axis=1)

    # Take the square root to get the actual distances
    distances = np.sqrt(squared_distances)

    return distances


class TestCalculateDistances(unittest.TestCase):
    def test_same_points(self):
        # Test when both arrays have the same points
        points = np.array([[1, 2], [3, 4], [5, 6]])
        self.assertTrue(np.allclose(calculate_distances(points, points), 0.0))

    def test_different_points(self):
        # Test with different points
        points1 = np.array([[1, 2], [3, 4], [5, 6]])
        points2 = np.array([[7, 8], [9, 10], [11, 12]])
        expected_distances = np.array([8.48528137, 8.48528137, 8.48528137])
        self.assertTrue(
            np.allclose(calculate_distances(points1, points2), expected_distances)
        )
        self.assertTrue(
            np.allclose(calculate_distances(points2, points1), expected_distances)
        )

    def test_empty_arrays(self):
        # Test with empty arrays
        empty_points = np.empty((0, 2))
        self.assertEqual(calculate_distances(empty_points, empty_points).shape, (0,))


def example_usage():
    # Example usage:
    points1 = np.array([[1, 2], [3, 4], [5, 6]])
    points2 = np.array([[7, 8], [9, 10], [11, 12]])

    distances = calculate_distances(points1, points2)
    print("Distances between points:")
    print(distances)


if __name__ == "__main__":
    unittest.main()
    # example_usage()
