import math
import unittest
import itertools
from enum import IntFlag
import numpy

from mapedge.lib.trace.fuzzy import trapezoidal_mf

import mapedge.lib.trace.vectorops as vec
import pprint


def compensate_outside(v):
    """get vector for compensating if a vector is outside of the image size

    returns (0, 0) if no compensation is required
    otherwise returns +outside for a component of the vector how far it is outside
    """
    return tuple([abs(x) if x < 0 else 0 for x in v])


def rim_side_as_str(rim_side):
    if rim_side & RimSide.LEFT:
        return "left"
    elif rim_side & RimSide.RIGHT:
        return "right"
    elif rim_side & RimSide.TOP:
        return "top"
    elif rim_side & RimSide.BOTTOM:
        return "bottom"


class RimSide(IntFlag):
    LEFT = 1
    RIGHT = 2
    TOP = 4
    BOTTOM = 8


def get_part_count(region_image: numpy.ndarray, rim_side: RimSide, part_size: int):
    # this only works for BW image (otherwise we have more channels)
    rows, cols = region_image.shape
    # apparently under IIIF it is allowed to scale down the image in size
    # if the request is too large to service
    # check if the image size we think we requested is also the one we obtain
    # (otherwise we get a scaling effect, imposed by the server,
    # in where items occur, and they finally end up in the wrong place)
    # assert cols == w
    # assert rows == h

    # when the image is too limited in size (not so tall)
    # confusion can easily happen
    # because more other peaks are above the threshold that is set
    # for black lines, and spots in the image can fullfill the
    # black line criterion (has been improved a bit by better ranking
    # criterion
    # -> apparently there is more uncertainty in the a-priori determined size
    # in pixels of the rim)

    # FIXME: should be setting (in high res image space)
    # part_size = 250
    if rim_side & RimSide.LEFT or rim_side & RimSide.RIGHT:
        parts = rows // part_size
    elif rim_side & RimSide.BOTTOM or rim_side & RimSide.TOP:
        parts = cols // part_size

    print(
        f"{rim_side_as_str(rim_side):<7} {rows} × {cols} with {part_size=}, we split in {parts=}"
    )
    return parts


def extract_subregion(region_image, part_size, i, rim_side):
    """
    returns subregion and top left coordinate
    """
    start_part_index = part_size * i
    end_part_index = part_size * (i + 1)

    # print(part_size * i, part_size * (i + 1))

    ### copy the subregion from the image

    # all columns / rows we need to copy from the image
    to_slice_indices = list(range(start_part_index, end_part_index))
    if rim_side & RimSide.LEFT or rim_side & RimSide.RIGHT:
        sub_region = region_image[to_slice_indices, :]
        tl_of_slice = [0, start_part_index]
    elif rim_side & RimSide.BOTTOM or rim_side & RimSide.TOP:
        sub_region = region_image[:, to_slice_indices]
        tl_of_slice = [start_part_index, 0]

    return sub_region, tl_of_slice


def determine_one_rim_point(
    intervals, rim_side, sub_center, settings
) -> tuple[float] | None:
    """
    given:
    - a list of intervals
    - the rim side we are processing
    - and an approximate for the center of this subregion

    return:
    - one point that lies on the rim we are processing
    """

    ranked = []
    expected_size = settings["rim_width_fuzzy"]
    for interval in intervals:
        # ranking criteria
        ranking = [
            # the width of the black peak should be similar to what we expect (e.g. 5px)
            trapezoidal_mf(size(interval), *expected_size),
        ]
        # print('trapezoidal shape params', size, expected_size -2.5, expected_size-1, expected_size+1, expected_size + 2.5)
        # print('gaussian shape params', sub_center[sub_ax], 0.5 * sub_center[sub_ax])
        # print('triangle shape params', 0, sub_center[sub_ax], sub.shape[1 - sub_ax])
        # pprint.pprint([interval, ranking])
        # print(
        #     size,
        #     expected_size - 4,
        #     expected_size - 2,
        #     expected_size + 2,
        #     expected_size + 4,
        # )
        ranked.append([math.prod(ranking), interval])

    # --
    # FIXME: when all solutions are ranked 0.0 we get garbage!
    # what to do? ignore points? warn user?
    # handle when fitting lines to the points (e.g. filter unsure points)?
    # --
    if ranked:
        ranked.sort(reverse=True, key=lambda x: x[0])
        print("ranked alternatives 1 point")
        print("-" * 20)
        pprint.pprint(ranked[:3])
        how_many = len(list(filter(lambda x: x[0] != 0.0, ranked)))
        print(how_many)
        # we should have 1 ranked solution that is not 0.0 to accept it
        if how_many == 1:
            # determine the exact point coordinate
            # we override the coordinate axis we are looking at
            # with the right (local) value
            # we can do this with the first or second value of the interval
            # FIXME: we can also take the center of the peak
            # (which might be more like what we want for a thin line)
            best = ranked[0]
            # best_interval_rank = best[0]
            # if best_interval_rank == 0.0:  # minimum rank to accept the point
            #     return None
            best_interval = best[1]
            if rim_side & RimSide.LEFT:
                ax = 0
                # first_second = 1
            elif rim_side & RimSide.RIGHT:
                ax = 0
                # first_second = 0
            elif rim_side & RimSide.BOTTOM:
                ax = 1
                # first_second = 1
            elif rim_side & RimSide.TOP:
                ax = 1
                # first_second = 0

            to_adapt = sub_center[:]
            # best_interval[first_second]
            to_adapt[ax] = mid(best_interval)
            return to_adapt
        else:
            return None

    else:
        return None


# def one(to_rank):
#     to_rank.sort(key=lambda x: -x[0])

#     pair = list(filter(lambda x: x[0] != 0, to_rank))
#     # pprint.pprint(to_rank)

#     # FIXME: this is a somewhat bold assumption:
#     # the one that is ranked top is the one we need
#     # (if we do not have a rank value of 1.0, and the second value is not 0.0,
#     # it means that the ranking can be confused)
#     # on the other hand, if we just found 1 interval, and all other intervals do not match the fuzzy size...
#     # then this is an indication that we can continue with this one...)
#     # why would we make it special?
#     # can't we go with just the one for ranking multiple?

#     best_combi = pair[0]
#     best_combi_annotated_intervals = best_combi[1]
#     print(best_combi_annotated_intervals)
#     related = [annotated[1] for annotated in best_combi_annotated_intervals]
#     # for interval in related:
#     #     to_adapt = sub_center[:]
#     #     to_adapt[ax] = mid(interval)
#     #     if do_interactive_plot:
#     #         axB.plot(*to_adapt, "cx")
#     # FIXME :
#     # verify sizes of peaks <-> did we find a line of the size we would expect to find
#     print("We found exactly 1 match based on distance between peaks, sizes:")
#     matching_expectations = True
#     # check if the two related intervals contain the sizes as expected
#     for expected_size in [1, 18]:
#         found = False
#         for interval in related:
#             fuzzy_klass = trapezoidal_mf(
#                 size(interval),
#                 expected_size - 3,
#                 expected_size - 1,
#                 expected_size + 1,
#                 expected_size + 3,
#             )
#             print(" -", size(interval), fuzzy_klass)
#             if fuzzy_klass > 0:
#                 found = True
#         if not found:
#             matching_expectations = False
#     if matching_expectations:
#         print("USE:", best_combi_annotated_intervals)
#         # unpack the annotated interval
#         return [mid(interval) for rank_, interval in best_combi_annotated_intervals]
#     else:
#         print("do not use and continue")
#         return None


def decide_best_match(ranked_pairs, rim_side, sub_center, settings):
    to_process = [
        [rank, pair, distance] for rank, pair, distance in ranked_pairs if rank != 0.0
    ]
    print(f"To find a best match, we have #{len(to_process)} candidate pairs:")
    pprint.pprint(to_process)
    matching = []
    # large = 18
    # small = 1

    for rank, pair, distance in to_process:

        related = [c[1] for c in pair]
        print("---", related)
        # let's see if we can find exactly one interval/peak
        # more or less of size 1 and one peak with size ~18
        # depending on which side of the map frame we are
        # we encounter the fat line, and then the thin line
        # (or vice versa)
        found = [False, False]
        if rim_side & RimSide.LEFT or rim_side & RimSide.BOTTOM:
            # peak_order = [large, small]

            peak_order = [
                settings["rim_width_outer_fuzzy"],
                settings["rim_width_inner_fuzzy"],
            ]
        elif rim_side & RimSide.RIGHT or rim_side & RimSide.TOP:
            # peak_order = [small, large]
            peak_order = [
                settings["rim_width_inner_fuzzy"],
                settings["rim_width_outer_fuzzy"],
            ]
        for index, expected_size, interval in zip([0, 1], peak_order, related):
            fuzzy_klass = trapezoidal_mf(
                size(interval),
                *expected_size,
                # expected_size - 3,
                # expected_size - 1,
                # expected_size + 5,  # upto 22 / 23
                # expected_size + 8,
            )
            print("🎯", fuzzy_klass, size(interval), interval, expected_size)
            if fuzzy_klass > 0:
                found[index] = True
        # for index, expected_size, _ in zip([0, 1], [18, 1], ["outer", "inner"]):
        #     for interval in related:
        #         fuzzy_klass = trapezoidal_mf(
        #             size(interval),
        #             expected_size - 3,
        #             expected_size - 1,
        #             expected_size + 5,  # upto 22 / 23
        #             expected_size + 8,
        #         )
        #         print(expected_size, rank, size(interval), fuzzy_klass)
        #         if fuzzy_klass > 0:
        #             found[index] = True
        has_expected_sizes = all(found)
        if has_expected_sizes:
            pprint.pprint(pair)
            matching.append(pair)
            print("✅", pair)
        else:
            print("❌", pair)

    if len(matching) == 1:

        to_use = matching[0]
        print("GO GO USE:", to_use)
        # return [mid(interval) for rank_, interval in [pair for pair in matching]]

        if rim_side & RimSide.LEFT:
            ax = 0
        elif rim_side & RimSide.RIGHT:
            ax = 0
        elif rim_side & RimSide.BOTTOM:
            ax = 1
        elif rim_side & RimSide.TOP:
            ax = 1

        result = []
        for annotated in to_use:
            rank_, interval = annotated
            to_adapt = sub_center[:]
            to_adapt[ax] = mid(interval)
            result.append(to_adapt)
        print("FOUND POINTS [local]", result)
        return result
    else:
        print(f"Not found what we were looking for {len(matching)} is not exactly 1")
        print(f"{matching=}")
        return None


def rank_pairs(pairs, settings):
    to_rank = []
    for pair in pairs:
        distance = abs(pair[0][0] - pair[1][0])
        to_rank.append(
            [
                trapezoidal_mf(
                    distance, *settings["rim_outer_inner_distance_fuzzy"]
                ),  # 210 - 15, 210 - 2, 210 + 2, 210 + 15),
                pair,
                distance,
            ]
        )
    to_rank.sort(key=lambda x: -x[0])
    print("ranked pairs of peaks:")
    for pair in to_rank:
        print(f"- {pair}")
    return to_rank


def determine_multiple_rim_points(intervals, rim_side, sub_center, settings):
    print(intervals, rim_side, rim_side_as_str(rim_side), sub_center)
    centers = decorate_peaks_with_center(intervals)
    pairs = calculate_combinations(centers)
    ranked_pairs = rank_pairs(pairs, settings)
    return decide_best_match(ranked_pairs, rim_side, sub_center, settings)


def get_translate_vector(rect, tl_of_slice):
    # the translation vector for putting the
    # found point in the global system
    displace_vecs = list(
        map(
            tuple,
            [
                rect.center,
                vec.mul(rect.half_size, -1),
                tl_of_slice,
                compensate_outside(rect.tl()),
            ],
        )
    )
    displacement = [0, 0]
    for v in displace_vecs:
        # print("VECTOR", v, "POINT", point)
        displacement = vec.add(displacement, v)
    return displacement


# a peak in the histogram is an interval on the 1D number line


def mid(interval):
    half_size = size(interval) * 0.5
    return interval[0] + half_size


def size(interval):
    lower, upper = interval
    return upper - lower


def decorate_peaks_with_center(intervals):
    """Calculate the center of each interval."""
    return [(mid(interval), interval) for interval in intervals]


def calculate_combinations(centers):
    """Form all pair of centers."""
    return list(itertools.combinations(centers, 2))


class TestIntervalFunctions(unittest.TestCase):
    def test_mid(self):
        # Test with an interval [3, 9]
        self.assertAlmostEqual(mid([3, 9]), 6.0, places=2)

        # Test with an interval [0, 10]
        self.assertAlmostEqual(mid([0, 10]), 5.0, places=2)

        # Test with an interval [-5, 5]
        self.assertAlmostEqual(mid([-5, 5]), 0.0, places=2)

    def test_size(self):
        # Test with an interval [3, 9]
        self.assertEqual(size([3, 9]), 6)

        # Test with an interval [0, 10]
        self.assertEqual(size([0, 10]), 10)

        # Test with an interval [-5, 5]
        self.assertEqual(size([-5, 5]), 10)

    def test_decorate_centers(self):
        # Test with a list of intervals
        intervals = [[3, 9], [0, 10], [-5, 5]]
        centers = decorate_peaks_with_center(intervals)
        expected_centers = [(6.0, [3, 9]), (5.0, [0, 10]), (0.0, [-5, 5])]
        self.assertEqual(centers, expected_centers)

    def test_calculate_combinations(self):
        # Test with a list of decorated intervals
        centers = [(6.0, [3, 9]), (5.0, [0, 10]), (0.0, [-5, 5])]
        combinations = calculate_combinations(centers)
        expected_combinations = [
            ((6.0, [3, 9]), (5.0, [0, 10])),
            ((6.0, [3, 9]), (0.0, [-5, 5])),
            ((5.0, [0, 10]), (0.0, [-5, 5])),
        ]
        self.assertEqual(combinations, expected_combinations)


###
# [(541, 543), (562, 580), (594, 598), (658, 660), (677, 679), (697, 699), (786, 789), (990, 993)] 1 [565, 125]
#
# [
#   [0.6538461538461539, ([571.0, (562, 580)], [787.5, (786, 789)])],
#   [0.6923076923076923, ([787.5, (786, 789)], [991.5, (990, 993)])]
# ]

if __name__ == "__main__":
    unittest.main()
    # pass
