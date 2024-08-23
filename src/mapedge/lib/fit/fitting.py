# fitting.py
from .lib import apply_svd, centroid, design_matrix, design_matrix_orthogonal
from .line2 import Line2
import numpy as np


def swizzle(n):
    return np.array([n[1], n[0]])


def fit_line(points):
    c = centroid(points)
    D = design_matrix(points)
    n = apply_svd(D)
    return Line2.from_normal_and_point(n, c)


def fit_parallel_lines(sets):
    centroids = []
    matrices = []
    for points in sets:
        c = centroid(points)
        D = design_matrix(points)
        centroids.append(c)
        matrices.append(D)
    D = np.vstack(matrices)
    n = apply_svd(D)
    lines = [Line2.from_normal_and_point(n, c) for c in centroids]
    return lines


def fit_perpendicular_lines(sets, is_perp):
    centroids = []
    matrices = []
    for points, perp in zip(sets, is_perp):
        c = centroid(points)
        if perp:
            # interpret this set of points as the perpendicular set
            D = design_matrix_orthogonal(points)
        else:
            D = design_matrix(points)
        centroids.append(c)
        matrices.append(D)
    D = np.vstack(matrices)
    n = apply_svd(D)
    n_ortho = swizzle(n)
    lines = []
    for c, perp in zip(centroids, is_perp):
        if perp:
            ln = Line2.from_normal_and_point(n_ortho, c)
        else:
            ln = Line2.from_normal_and_point(n, c)
        lines.append(ln)
    return lines
