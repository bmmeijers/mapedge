import numpy as np
import math


def centroid(pts):
    # calculate centroid of the cloud
    return np.mean(pts, axis=0)


def mean_free(pts):
    # mean free data,
    # i.e. center on mean (subtract the centroid from all points)
    mean = centroid(pts)
    return pts - mean


def design_matrix(pts):
    return mean_free(pts)


def design_matrix_orthogonal(pts):
    tmp_pts = np.array(pts, copy=True)
    tmp_pts[:, [1, 0]] = tmp_pts[:, [0, 1]]
    tmp_pts[:, 1] *= -1
    return mean_free(tmp_pts)


def design_matrix_part_centroid(pts):
    # calculate centroid of the cloud
    centroid = np.mean(pts, axis=0)

    # center on mean (subtract the centroid from all points)
    centered = pts - centroid

    # the design matrix part and the centroid
    return centered, centroid


def apply_svd(design_matrix):
    _U, S, Vt = np.linalg.svd(design_matrix)

    # print(f"{U=}")
    # print(f"{S=}")
    # print(f"{Vt=}")

    # # U, S, Vt = np.linalg.svd(data)

    # # Print the singular values
    # print("Singular values:", S)

    # # Extract the smallest singular value and its index
    # smallest_singular_value = S[-1]
    # smallest_singular_value_idx = len(S) - 1

    # # # Print information about the smallest singular value and vector
    # print("Smallest singular value:", smallest_singular_value)
    # print(
    #     "Right singular vector associated with the smallest singular value:",
    #     Vt[:, smallest_singular_value_idx],
    # )

    # # right_singular_vector_for_smallest_singular_value = Vt.T[:, 0]
    # # print(right_singular_vector_for_smallest_singular_value)
    # print(np.argmin(S))
    return Vt[np.argmin(S)]


def project_points(line, pts):
    normal, dist = line
    a, b = normal
    denom = a**2 + b**2
    # assert denom == 1
    # c = dist
    projected = []
    for x0, y0 in pts:
        # - dist or + dist ???
        d = (a * x0 + b * y0 - dist) / denom
        xp = x0 - a * d
        yp = y0 - b * d
        # plt.plot([x0, xp], [y0, yp], 'g:')
        projected.append((xp, yp))
    return projected


def convert_line_to_points(line, size):
    # convert line: a*x + b*y + c = 0
    # over the `rectangle`` with size
    # into 2 points where this rectangle is intersected by the line
    # returns: [origin], [destination]
    # as 2-tuple
    width, height = size
    n, c = line
    xrange, yrange = (0, width), (0, height)

    a, b = n
    # depending on whether the line is:
    # - horizontal
    # - vertical
    # - slanted
    # get points where the line intersects the boundaries
    # of the rectangle indicated by `size`
    if a == 0:
        #
        x1 = xrange[0]
        x2 = xrange[1]

        y1 = c / b
        y2 = y1

    elif b == 0:  # % c+n1*x=0  => x = -c/n(1)

        x1 = -c / a
        x2 = x1

        y1 = yrange[0]
        y2 = yrange[1]

    else:
        # do not take care of whether line goes through area
        x1 = xrange[0]
        x2 = xrange[1]

        y1 = -(c + a * x1) / b
        y2 = -(c + a * x2) / b

    orig, dest = (x1, y1), (x2, y2)
    # print(f"LINESTRING({orig[0]} {orig[1]}, {dest[0]} {dest[1]})")

    # import matplotlib.pyplot as plt

    # plt.xlim(*xrange)
    # plt.ylim(*yrange)
    # plt.plot([x1, x2], [y1, y2], linestyle="dotted")
    # plt.show()
    return [orig, dest]


def distance(orig, dest):
    return math.hypot(dest, orig)


# def my_intersect(h1: HyperPlane, h2: HyperPlane):
#     """Get point in 2D for lines where they intersect (as column vector)"""
#     A = np.array([h1.n, h2.n])
#     b = np.array([[h1.d], [h2.d]])
#     return np.linalg.solve(A, b)
#     # h2.d


def intersect_2d(line0, line1):
    # Stack the normal vectors into a matrix A
    A = np.stack((line0[0], line1[0]))

    # Create a column vector b with the negative distances
    b_vec = np.array([(line0[1], line1[1])]).T

    det = np.linalg.det(A)
    if np.allclose(det, 0.0):
        raise ValueError(
            "determinant close to 0.0, no solution: parallel / overlapping"
        )
    # print(det)
    # Obtain the intersection point
    pt = np.linalg.solve(A, b_vec)

    return pt


def test_intersect():
    line0 = (np.array([-1, 0]), +10)
    line1 = (np.array([0, -1]), -20)
    xpt = intersect_2d(line0, line1)
    print(f"{xpt=}")


if __name__ == "__main__":
    test_intersect()
