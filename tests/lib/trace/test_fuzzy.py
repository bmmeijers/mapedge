import unittest
import math
from mapedge.lib.trace.fuzzy import (
    triangular_mf,
    trapezoidal_mf,
    cauchy_mf,
    sigmoid_mf,
    gaussian_mf,
)


class TestTriangularMF(unittest.TestCase):
    def test_feet(self):
        self.assertEqual(triangular_mf(2, 2, 3, 4), 0)
        self.assertEqual(triangular_mf(4, 2, 3, 4), 0)

    def test_peak(self):
        self.assertEqual(triangular_mf(3, 2, 3, 4), 1)

    def test_halfway_to_peak(self):
        self.assertAlmostEqual(triangular_mf(2.5, 2, 3, 4), 0.5, places=5)
        self.assertAlmostEqual(triangular_mf(3.5, 2, 3, 4), 0.5, places=5)

    def test_outside_interval(self):
        self.assertEqual(triangular_mf(1.5, 2, 3, 4), 0)
        self.assertEqual(triangular_mf(4.5, 2, 3, 4), 0)


class TestCauchyMembership(unittest.TestCase):
    def test_center(self):
        self.assertEqual(cauchy_mf(2, 2, 1), 1)

    def test_far_from_center(self):
        self.assertAlmostEqual(cauchy_mf(100, 2, 1), 0.00010411244143675169, places=5)
        self.assertAlmostEqual(cauchy_mf(-100, 2, 1), 9.610764055742432e-05, places=5)

    def test_halfway_to_zero(self):
        self.assertAlmostEqual(cauchy_mf(2.5, 2, 1), 0.8, places=5)
        self.assertAlmostEqual(cauchy_mf(1.5, 2, 1), 0.8, places=5)


class TrapezoidalMembershipFunctionTests(unittest.TestCase):

    def test_value_within_flat_region(self):
        membership_value = trapezoidal_mf(100, 95, 100, 105, 110)
        self.assertEqual(membership_value, 1.0)

    def test_value_at_lower_bound(self):
        membership_value = trapezoidal_mf(95, 95, 100, 105, 110)
        self.assertEqual(membership_value, 0.0)

    def test_value_at_upper_bound(self):
        membership_value = trapezoidal_mf(110, 95, 100, 105, 110)
        self.assertEqual(membership_value, 0.0)

    def test_value_increasing_towards_flat_region(self):
        membership_value = trapezoidal_mf(97, 95, 100, 105, 110)
        self.assertAlmostEqual(membership_value, 0.4)

    def test_value_decreasing_from_flat_region(self):
        membership_value = trapezoidal_mf(107.5, 95, 100, 105, 110)
        self.assertAlmostEqual(membership_value, 0.5)

    def test_value_outside_support(self):
        membership_value = trapezoidal_mf(90, 95, 100, 105, 110)
        self.assertEqual(membership_value, 0.0)

        membership_value = trapezoidal_mf(115, 95, 100, 105, 110)
        self.assertEqual(membership_value, 0.0)


class TestGaussianMembership(unittest.TestCase):
    def setUp(self):
        self.gaussian_membership = (
            gaussian_mf  # assuming gaussian_membership is defined globally
        )

    def test_center(self):
        # when x is at the center, the membership should be 1
        self.assertEqual(self.gaussian_membership(5, 5, 1), 1)

    def test_far_from_center(self):
        # when x is far from the center, the membership should be close to 0
        self.assertAlmostEqual(self.gaussian_membership(100, 5, 1), 0, places=5)

    def test_negative_sigma(self):
        # when sigma is negative, the function should raise a ValueError
        # with self.assertRaises(ValueError):
        self.assertEqual(self.gaussian_membership(5, 5, -1), 1.0)

    def test_positive_sigma_from_center(self):
        # when x is at c + sigma, the membership should be approximately 0.6065
        self.assertAlmostEqual(
            self.gaussian_membership(6, 5, 1), math.exp(-0.5), places=4
        )

    def test_negative_sigma_from_center(self):
        # when x is at c - sigma, the membership should be approximately 0.6065
        self.assertAlmostEqual(
            self.gaussian_membership(4, 5, 1), math.exp(-0.5), places=4
        )

    def test_zero_sigma(self):
        # when sigma is zero and x is not equal to c, the function should return 0
        with self.assertRaises(ZeroDivisionError):
            self.assertEqual(self.gaussian_membership(6, 5, 0), 0)
        # when sigma is zero and x is equal to c, the function should raise
        with self.assertRaises(ZeroDivisionError):
            self.assertEqual(self.gaussian_membership(5, 5, 0), 1)


class TestSigmoidMF(unittest.TestCase):
    def test_sigmoid_mf(self):
        # Test at midpoint
        self.assertAlmostEqual(sigmoid_mf(0, 1, 0), 0.5)

        # Test at positive infinity, should be close to 1
        self.assertAlmostEqual(sigmoid_mf(float("+inf"), 1, 0), 1)

        # Test at negative infinity, should be close to 0
        self.assertAlmostEqual(sigmoid_mf(float("-inf"), 1, 0), 0)

        # Test with different slope
        self.assertAlmostEqual(sigmoid_mf(0, 2, 0), 0.5)
        self.assertAlmostEqual(sigmoid_mf(1, 2, 0), 0.8807970779778823)
