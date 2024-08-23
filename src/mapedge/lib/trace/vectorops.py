# -*- coding: utf-8 -*-
"""Operations that allow tuples/lists (or any type that implements __getitem__
and __iter__) to be used as vectors"""

import logging
import math
from operator import sub as _sub, mul as _mul, truediv as _div, add as _add


def proj(u, v):
    """Returns projection of vector u on vector v"""
    # finding norm of the vector v
    v_norm = norm(v)

    # Apply the formula as mentioned above
    # for projecting a vector onto another vector
    # find dot product
    proj_of_u_on_v = mul(v, dot(u, v) / v_norm**2)
    return proj_of_u_on_v


def sub(a, b):
    """Subtract a vector b from a, or subtract a scalar"""
    if hasattr(b, "__iter__"):
        assert len(a) == len(b), "Vector dimensions should be equal"
        return tuple(map(_sub, a, b))
    else:
        return tuple(ai - b for ai in a)


def add(a, b):
    """Add a vector b to a, or add a scalar"""
    if hasattr(b, "__iter__"):
        assert len(a) == len(b), "Vector dimensions should be equal"
        return tuple(map(_add, a, b))
    else:
        return tuple(ai + b for ai in a)


def mul(a, b):
    """Multiply a vector either element-wise with another vector, or with a
    scalar."""
    if hasattr(b, "__iter__"):
        assert len(a) == len(b), "Vector dimensions should be equal"
        return tuple(map(_mul, a, b))
    else:
        return tuple(ai * b for ai in a)


def div(a, b):
    """Element-wise division with another vector, or with a scalar."""
    if hasattr(b, "__iter__"):
        assert len(a) == len(b), "Vector dimensions should be equal"
        return tuple(map(_div, a, b))
    else:
        return tuple(ai / b for ai in a)


def make_vector(end, start):
    """Creates a vector from the start to the end.

    Vector is made based on two points: end -(minus) start.
    """
    return sub(end, start)


def dot(v1, v2):
    """Returns dot product of v1 and v2"""
    assert len(v1) == len(v2), "Vector dimensions should be equal"
    return sum(p * q for p, q in zip(v1, v2))


def norm2(v):
    """Returns the norm of v, *squared*."""
    return dot(v, v)


def norm(a):
    """L2 norm ('distance')"""
    return math.sqrt(norm2(a))


# def length(v):
#     """Euclidean length of vector"""
#     return sqrt(sum([x**2 for x in v]))
#
#
# def length2(v):
#     """*squared* Euclidean length of vector"""
#     return sum([x**2 for x in v])
#
#
# def dist(start, end):
#     """distance between two positons"""
#     return length(map(sub, start, end))
#
#
# def dist2(start, end):
#     """squared distance between two positons"""
#     return length2(map(sub, start, end))


def dist(start, end):
    """Distance between two positons"""
    return norm(make_vector(end, start))


def unit(v):
    """Returns the unit vector in the direction of v."""
    return div(v, norm(v))


def cross(a, b):
    """Cross product between a 3-vector or a 2-vector"""
    assert len(a) == len(b), "Vector dimensions should be equal"
    if len(a) == 3:
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )
    elif len(a) == 2:
        return a[0] * b[1] - a[1] * b[0]
    else:
        raise ValueError("Vectors must be 2D or 3D")


# def distance2(a, b):
#     """*Squared* Euclidean distance between 2 points"""
#     if len(a) != len(b):
#         raise ValueError('Point dimensions should be equal')
#     s = 0
#     for i in range(len(a)):
#         s += (a[i] - b[i]) * (a[i] - b[i])
#     return s
#
#
# def distance(a, b):
#     """Euclidean distance between 2 points"""
#     return math.sqrt(distance2(a, b))


def angle(v1, v2):
    """angle between 2 vectors"""
    return math.acos(dot(v1, v2) / (norm(v1) * norm(v2)))


def angle_unit(v1, v2):
    """angle between 2 *unit* vectors

    does not compute the norm(v1)*norm(v2), as it is assumed to be 1
    """
    d = dot(v1, v2)
    if d > 1.0 or d < -1.0:
        logging.debug("dot not in [-1, 1] -- clamp")
    d = max(-1.0, min(1.0, d))
    acos_d = math.acos(d)
    logging.debug(" d : {}".format(d))
    logging.debug(" acos(d) : {}".format(acos_d))
    logging.debug(" degrees(acos(d)): {}°".format(math.degrees(acos_d)))
    return d, acos_d


# def bisector(u1, u2):
#     """Based on two unit vectors perpendicular to the wavefront,
#     get the bisector

#     The magnitude of the bisector vector represents the speed
#     in which a vertex has to move to keep up (stay at the intersection of)
#     the 2 wavefront edges
#     """
#     direction = add(u1, u2)
#     logging.debug(" direction: {}".format(direction))
#     d, acos_d = angle_unit(u1, u2)

#     #    if all(map(near_zero, direction)) or near_zero(acos_d - math.pi): #
#     if all(map(near_zero, direction)) or (
#         near_zero(acos_d - math.pi) or d < math.cos(math.radians(179.999))
#     ):
#         logging.debug(
#             " vectors cancel each other out / angle ~180° -> parallel wavefront!"
#         )
#         return (0, 0)
#         # raise ValueError("parallel wavefront")
#     ###    logging.debug(" unit(direction): {}".format(unit(direction)))
#     alpha = 0.5 * math.pi + 0.5 * acos_d
#     logging.debug(" degrees(alpha): {}°".format(math.degrees(alpha)))
#     magnitude = math.sin(alpha)
#     logging.debug(" magnitude: {}".format(magnitude))
#     # print magnitude
#     bisector = div(unit(direction), magnitude)
#     logging.debug(" bisector: {}".format(bisector))
#     return bisector


def rotate90ccw(v):
    """Rotate 2d vector 90 degrees counter clockwise

    (x, y) -> (-y, x)
    """
    return (-(v[1]), v[0])


def rotate90cw(v):
    """Rotate 2d vector 90 degrees clockwise

    (x, y) -> (y, -x)
    """
    return (v[1], -v[0])


def test():
    v1, v2 = list(map(float, list(range(0, 3)))), list(map(float, list(range(4, 7))))
    # v1 = [0,1,2]
    # v2 = [4,5,6]

    assert sub(v2, v1) == (4, 4, 4), sub(v2, v1)
    assert sub(v1, v2) == (-4, -4, -4), sub(v1, v2)
    assert sub(v1, 5) == (-5, -4, -3), sub(v1, 5)

    assert add(v1, v2) == (4, 6, 8)
    assert add(v1, 10) == (10, 11, 12)

    assert div((4, 4, 4), 4) == (1, 1, 1)
    assert div(v1, v2) == (0, 1.0 / 5.0, 2.0 / 6.0), div(v1, v2)

    assert mul(v1, v2) == (0, 1 * 5, 2 * 6), mul(v1, v2)
    assert mul(v2, 10) == (40, 50, 60)

    assert angle((1, 1), (-1, 1)) == math.pi * 0.5

    # test almost equal
    print(unit((1, 1)) == (0.5 * math.sqrt(2.0), 0.5 * math.sqrt(2.0)))


#     assert distance((0, 0), (3, 4)) == 5
#     assert distance2((0, 0), (3, 4)) == 25


def test_proj():
    u, v = (-1, -1), (10, 0)
    proj_u_on_v = proj(u, v)
    print(proj_u_on_v)

    proj_v_on_u = proj(v, u)
    print(proj_v_on_u)

    p0 = (0, 5)
    p1 = (2, 0)

    v = make_vector(p1, p0)
    n = unit(rotate90ccw(v))
    print(f"unit vec {n}")

    origin = (0, 0)
    v = make_vector(origin, p0)
    print(f"v {v}")
    proj_v_on_n = proj(v, n)
    print(
        f"proj_v_on_n {proj_v_on_n}, with norm: |{norm(proj_v_on_n)}| (which is param b in Ax + b = 0)"
    )


if __name__ == "__main__":
    test()
    test_proj()
