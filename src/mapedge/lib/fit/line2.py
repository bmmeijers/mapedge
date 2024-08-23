import numpy as np
import unittest


# def distance(pt_a, pt_b):


class Line2:
    """A 2D line"""

    def __init__(self, normal, distance):
        """Init a Line2"""
        vector = np.array(normal)
        assert vector.shape == (2,), vector.shape
        magnitude = np.linalg.norm(vector)
        # Normalize, makes sure that the normal is a unit vector
        self.normal = vector / magnitude
        self.distance = distance / magnitude

    def __str__(self):
        return f"{self.normal=}, {self.distance=}\n{self.normal[0]}·x + {self.normal[1]}·y + {self.distance} = 0\n\n"

    @classmethod
    def from_normal_and_point(cls, normal, point):
        """
        Create a Line2 instance from a given normal vector and a point on the line.

        Args:
            normal (np.ndarray): A 1D numpy array representing the normal vector (a, b).
            point (np.ndarray): A 1D numpy array representing a point (x0, y0) on the line.

        Returns:
            Line2: A Line2 instance.
        """
        distance = np.dot(-normal, point)
        return cls(normal, distance)

    @classmethod
    def from_points(cls, p1, p2):
        """
        https://math.stackexchange.com/a/422608

        Given two points [x1, y1] [x2, y2]

        The following is the equation we are looking for:

        (y1 - y2) * x + (x2 - x1) * y + (x1 * y2 - x2 * y1) = 0
        """
        x1, y1 = p1
        x2, y2 = p2
        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        normal = np.array([a, b])
        distance = c
        return cls(normal, distance)

    # @classmethod
    # def from_points(cls, p1, p2):
    #     x1, y1 = p1
    #     x2, y2 = p2
    #     a = y2 - y1
    #     b = x1 - x2
    #     distance = x2 * y1 - x1 * y2
    #     normal = np.array([a, b])
    #     return cls(normal, distance)

    def signed_distance(self, point):
        """
        Compute the signed distance from a point to the line.

        Args:
            point (np.ndarray): A 1D numpy array representing the point (x0, y0).

        Returns:
            float: The signed distance from the point to the line.
        """
        return np.dot(self.normal, point) + self.distance
        # no need for this, unit vector: / np.linalg.norm(self.normal)

    def project_points_deprecated(self, pts):
        normal, dist = self.normal, self.distance
        a, b = normal
        denom = a**2 + b**2
        # assert denom == 1
        # c = dist
        projected = []
        for x0, y0 in pts:
            # - dist or + dist ???
            d = a * x0 + b * y0 + dist / denom
            xp = x0 - a * d
            yp = y0 - b * d
            # plt.plot([x0, xp], [y0, yp], 'g:')
            projected.append((xp, yp))
        return np.array(projected)

    def project_points(self, pts):
        """
        Project an array of points onto the line.

        Args:
            pts (np.ndarray): A 2D numpy array representing points (x, y).

        Returns:
            np.ndarray: An array of projected points.
        """
        # denom = np.sum(self.normal**2)
        # Equivalent to a**2 + b**2 = 1 (as normal is unit vec)
        d = np.dot(pts, self.normal) + self.distance
        projected = pts - np.outer(d, self.normal)
        return projected

    def intersect(self, other):
        # url = f"https://mathsolver.microsoft.com/nl/solve-problem/%60left.%20%60begin%7Bcases%7D%20%7B%20{self.normal[0]}x%2B{self.normal[1]}y%20%3D%20%20{-self.distance}%20%20%7D%20%60%60%20%7B%20{other.normal[0]}x%2B{other.normal[1]}y%20%3D%20%20{-other.distance}%20%20%7D%20%60end%7Bcases%7D%20%60right."
        # print(url)

        # return None
        A = np.array([self.normal, other.normal])
        # if np.linalg.det(A) == 0:
        #     return None
        det = np.linalg.det(A)
        if np.allclose(det, 0.0):
            return None
            # raise ValueError(
            #     "determinant close to 0.0, no solution: parallel / overlapping"
            # )
        b = np.array([-self.distance, -other.distance])
        return np.linalg.solve(A, b)

    def as_dict(self):
        tmp = self.normal.tolist()
        tmp.append(self.distance)
        return {
            "Line": {"dimension": 2, "equation": tmp}
            # "Line2": {
            #     "normal": self.normal.tolist(),
            #     "distance": self.distance,
            # }
        }

    @classmethod
    def from_dict(cls, dct):
        params = dct["Line"]["equation"]
        return cls(params[:2], params[2])


class TestLine2_dict(unittest.TestCase):
    def test_from_dict(self):
        d = {
            "Line": {
                "equation": [
                    -0.00013139035285430288,
                    -0.9999999913682877,
                    894.0606707810485,
                ],
                "dimension": 2,
                # "normal": [-0.00013139035285430288, -0.9999999913682877],
                # "distance": 894.0606707810485,
            }
        }
        ln = Line2.from_dict(d)
        self.assertEqual(
            ln.normal.tolist(), [-0.00013139035285430288, -0.9999999913682877]
        )
        self.assertEqual(ln.distance, 894.0606707810485)

    def test_roundtrip_dict(self):
        d1 = {
            "Line": {
                "equation": [
                    -0.00013139035285430288,
                    -0.9999999913682877,
                    894.0606707810485,
                ],
                "dimension": 2,
                # "normal": [-0.00013139035285430288, -0.9999999913682877],
                # "distance": 894.0606707810485,
            }
        }
        ln = Line2.from_dict(d1)
        d2 = ln.as_dict()
        self.assertEqual(d1, d2)


class TestLine2_from_normal_and_point(unittest.TestCase):
    def test_from_normal_and_point(self):
        # Create a line using the from_normal_and_point method
        normal = np.array([5.0, 0.0])
        point = np.array([10, 10])
        ln = Line2.from_normal_and_point(normal, point)

        # Check if the normal and distance are set correctly
        self.assertTrue(np.array_equal(ln.normal, [1.0, 0.0]))
        self.assertEqual(ln.distance, -10.0)


class TestLine2_Distance(unittest.TestCase):
    def setUp(self):
        # Create a line using the from_normal_and_point method
        normal = np.array([1.0, 0.0])
        point = np.array([10, 10])
        self.ln = Line2.from_normal_and_point(normal, point)

    def test_signed_distance_on(self):
        point1 = np.array([10, 10])
        distance1 = self.ln.signed_distance(point1)
        self.assertAlmostEqual(distance1, 0.0, places=6)  # Point on the line

    def test_signed_distance_negative(self):
        point2 = np.array([0, 0])
        distance2 = self.ln.signed_distance(point2)
        self.assertAlmostEqual(distance2, -10.0, places=6)  # Distance from origin

    def test_signed_distance_positive(self):
        point3 = np.array([15, 15])
        distance3 = self.ln.signed_distance(point3)
        self.assertAlmostEqual(distance3, 5.0, places=6)  # Distance from the line


def test():
    from distance import calculate_distances

    pts = []
    for x in range(-10, 10):
        pts.append((x, x))

    ln = Line2.from_normal_and_point(np.array((0, 1)), np.array((10, 10)))
    proj1 = ln.project_points(np.array(pts))
    proj2 = ln.project_points2(np.array(pts))
    import pprint

    pprint.pprint(list(zip(pts, proj1)))
    pprint.pprint(list(zip(pts, proj2)))

    dists1 = calculate_distances(proj1, np.array(pts))
    pprint.pprint(dists1)

    dists2 = calculate_distances(proj2, np.array(pts))
    pprint.pprint(dists2)


def point4():
    points = [(2, 2), (-3, 1), (-1, -2), (3, -4)]

    for centroid, normal in zip(
        [(0, 1), (0, 1), (1, 0), (1, 0)], [(0, 10), (0, -10), (-10, 0), (10, 0)]
    ):
        print(centroid, normal)
        ln = Line2.from_normal_and_point(np.array(normal), np.array(centroid))
        print(ln)

    ln = Line2.from_normal_and_point(np.array((0, 1)), np.array((10, 10)))

    dists1 = np.array([abs(ln.signed_distance(pt)) for pt in points])

    from distance import calculate_distances

    # for pt in points:
    projected = ln.project_points2(np.array(points))
    print(projected)

    dists2 = calculate_distances(np.array(points), projected)
    print(dists1, dists2)


class TestIntersect(unittest.TestCase):
    def test_parallel_lines(self):
        a = Line2(np.array([1, 0]), 1)
        b = Line2(np.array([1, 0]), 2)
        self.assertIsNone(a.intersect(b))

    def test_identical_lines(self):
        a = Line2(np.array([1, 0]), 1)
        b = Line2(np.array([1, 0]), 1)
        self.assertIsNone(a.intersect(b))

    def test_intersecting_lines(self):
        a = Line2(np.array([1, 0]), 1)
        b = Line2(np.array([0, 1]), 1)
        # self.assertEqual(a.intersect(b), (-1, -1))
        self.assertTrue(np.array_equal(a.intersect(b), [-1, -1]))

    def test_lines_in_each_quadrantI(self):
        # Quadrant I
        a = Line2(np.array([1, 1]), 1)
        b = Line2(np.array([-1, 1]), 1)
        # self.assertEqual(a.intersect(b), (0, -1))
        self.assertTrue(
            np.array_equal(a.intersect(b), [0, -1]), f"got: {a.intersect(b)}"
        )

    def test_lines_in_each_quadrantII(self):
        # Quadrant II
        a = Line2(np.array([-1, 1]), 1)
        b = Line2(np.array([-1, -1]), 1)
        # self.assertEqual(a.intersect(b), (1, 0))
        self.assertTrue(
            np.array_equal(a.intersect(b), [1, 0]), f"got: {a.intersect(b)}"
        )

    def test_lines_in_each_quadrantIII(self):
        # Quadrant III
        a = Line2(np.array([-1, -1]), 1)
        b = Line2(np.array([1, -1]), 1)
        # self.assertEqual(a.intersect(b), (0, 1))
        self.assertTrue(
            np.array_equal(a.intersect(b), [0, 1]), f"got: {a.intersect(b)}"
        )

    def test_lines_in_each_quadrantIV(self):
        # Quadrant IV
        a = Line2(np.array([1, -1]), 1)
        b = Line2(np.array([1, 1]), 1)
        # print(a.intersect(b))
        self.assertTrue(
            np.array_equal(a.intersect(b), [-1, 0]), f"got: {a.intersect(b)}"
        )


class TestLine2(unittest.TestCase):
    def test_from_points1(self):
        # Test 1: line passing through (0, 0) and (1, 1)
        line = Line2.from_points((0, 0), (1, 1))
        np.testing.assert_almost_equal(line.normal, np.array([-1, 1]) / np.sqrt(2))
        self.assertAlmostEqual(line.distance, 0)

    def test_from_points2(self):
        # Test 2: line passing through (1, 0) and (0, 1)
        line = Line2.from_points((1, 0), (0, 1))
        np.testing.assert_almost_equal(line.normal, np.array([-1, -1]) / np.sqrt(2))
        self.assertAlmostEqual(line.distance, 0.5 * np.sqrt(2))

    def test_from_points3(self):
        # Test 3: line passing through (0, 0) and (0, 1)
        line = Line2.from_points((0, 0), (0, 1))
        np.testing.assert_almost_equal(line.normal, np.array([-1, 0]))
        self.assertAlmostEqual(line.distance, 0)

    def test_from_points4(self):
        # Test 4: line passing through (0, 0) and (1, 0)
        line = Line2.from_points((0, 0), (1, 0))
        np.testing.assert_almost_equal(line.normal, np.array([0, 1]))
        self.assertAlmostEqual(line.distance, 0)

    def test_from_points5(self):
        # Test 5: line passing through (1, 0) and (0, 0)
        line = Line2.from_points((1, 0), (0, 0))
        np.testing.assert_almost_equal(line.normal, np.array([0, -1]))
        self.assertAlmostEqual(line.distance, 0)


if __name__ == "__main__":
    unittest.main()
    # test()
    # point4()
