# detect_rim.py

# iiif_url = f"{url}/{rect.iiif_region()}/full/0/default.jpg"
#         print(iiif_url)
# try:

# _ = stats_per_axis(iiif_url, f"iiif_cache/stats__sheet_{sheet_id:02d}__r{rect_id}.png", window_size=35, do_morph=True)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from .detect_fetch import fetch_image
from .detect_image_pipeline import apply_pipeline
from .intervals import (
    generate_intervals_above_threshold,
    # largest_interval,
    closest_interval,
    similar_sized_interval,
)


LEFT = 1
RIGHT = 2
TOP = 4
BOTTOM = 8

RIM_TYPES = {
    1: "left",
    2: "right",
    4: "top",
    8: "bottom",
}


def grayscale_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def bgr2rgb_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def threshold_image(grey, threshold):
    return cv2.threshold(grey, threshold, 255, cv2.THRESH_BINARY)[1]


def adaptive_threshold_image(grey):
    return cv2.adaptiveThreshold(
        grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )


def invert_image(thresholded_image):
    return cv2.bitwise_not(thresholded_image)


def center_coordinates(image):
    rows, cols = image.shape[0], image.shape[1]

    # print(rows, " (y) ×", cols, "(x)")
    center_rows = rows // 2
    center_cols = cols // 2

    xfound = center_cols
    yfound = center_rows

    return [xfound, yfound]


def count_pixel_values_along_axis(image, pixel_value, axis):
    counts = np.count_nonzero(image == pixel_value, axis=axis)
    return counts


def determine_threshold(image, axis, rim_fraction_filled_setting):
    row, col = image.shape
    if axis == 0:
        threshold = int(rim_fraction_filled_setting * row)
    elif axis == 1:
        threshold = int(rim_fraction_filled_setting * col)
    return threshold


def search_rim(image, pixel_value, axis, settings, figaxis):
    # find the peak locations (indices along along the axis)
    # where the value of black pixels is maximal
    row, col = image.shape
    if axis == 0:
        # x = np.array(range(col))
        # pixel_count = col
        title_part = "cols (horizontal)"
        threshold = settings["rim_fraction_filled"] * row

    elif axis == 1:
        # x = np.array(range(row))
        # pixel_count = row
        threshold = settings["rim_fraction_filled"] * col
        title_part = "rows (vertical)"
    else:
        raise ValueError(f"unhandled axis {axis} arg")
    # print(image == value)

    counts = np.count_nonzero(image == pixel_value, axis=axis)
    # print(counts)
    # get counts as fraction [0-1] = percentage value / 100.0
    # counts = counts / pixel_count
    print(f"{threshold=}")

    if figaxis:
        figaxis.set_title(title_part)
        figaxis.axhline(threshold, color="g")
        figaxis.plot(list(range(len(counts))), counts)

    intervals = generate_intervals_above_threshold(counts, threshold)

    return intervals, counts


def process_intervals(intervals, counts, settings):
    expected_size = settings["rim_size_pixels"]
    diffs = []
    black_pixel_diff = []
    mids = []
    for lower, upper in intervals:
        print(lower, upper)
        interval_size = upper - lower

        mid = lower + interval_size // 2
        diff = abs(156 - mid)
        mids.append(diff)

        black_pixel_count = sum(counts[i] for i in range(lower, upper))  # / 64.0
        print([lower, upper], black_pixel_count)
        ## FIXME: hard coded size here for image dimension
        ## should be passed in (is known)
        full_black_count = expected_size * 64

        diff = abs(interval_size - expected_size)
        diffs.append(diff)

        diff = abs(full_black_count - black_pixel_count)

        black_pixel_diff.append(diff)
    l = list(zip(intervals, diffs, mids, black_pixel_diff))
    # l.sort(key=lambda x: x[1])
    print(l)

    # D = sum(diffs)
    # C = sum(black_pixel_diff)
    to_sort = [
        (diff * 0.5 + count * 0.5 + mid * 0.5, interval)
        for interval, diff, mid, count in zip(intervals, diffs, mids, black_pixel_diff)
    ]
    to_sort.sort()
    print(to_sort)
    return to_sort[0][1]


def detect_peaks_in_detail(iiif_url, rim_type, settings):  # output_file):
    print("@ rim we are at @@", RIM_TYPES[rim_type])
    image = fetch_image(iiif_url)

    xfound, yfound = center_coordinates(image)
    rim_point = [xfound, yfound]

    # gray = grayscale_image(image)

    # FIXME: make how to threshold detail image a parameter
    # bw = threshold_image(gray, settings['rim_threshold'])
    # bw = adaptive_threshold_image(gray)

    # WSK
    if rim_type & LEFT or rim_type & RIGHT:
        pipeline = "gray|adaptive_threshold|dilate_vert(8)"
    elif rim_type & TOP or rim_type & BOTTOM:
        pipeline = "gray|adaptive_threshold|dilate_hori(8)"

    # pipeline = "gray|threshold(178)"
    bw = apply_pipeline(image, pipeline, inspect=False)

    # interactive figure for plot, layout
    if settings["interactive_plots"]:
        fig = plt.figure(layout="constrained")
        if rim_type & LEFT or rim_type & RIGHT:
            axis_dict = fig.subplot_mosaic(mosaic="AAAABB;CCCCDE")
        elif rim_type & TOP or rim_type & BOTTOM:
            axis_dict = fig.subplot_mosaic(mosaic="AACCBB;AACCDE")

        ax1 = axis_dict["A"]
        axB = axis_dict["B"]
        axC = axis_dict["C"]
        axD = axis_dict["D"]
        axE = axis_dict["E"]

        axC.imshow(bw, cmap=mpl.colormaps["gray"])

    intervals = None

    offset_setting = settings["rim_offset"]

    # FIXME: do the if here inside (where needed), then we reduce duplication of code
    if rim_type & LEFT or rim_type & RIGHT:
        figaxis = None
        if settings["interactive_plots"]:
            figaxis = axB
        intervals, counts = search_rim(
            bw, pixel_value=0, axis=0, settings=settings, figaxis=figaxis
        )
        # process_intervals(intervals, counts, settings)
        if intervals:
            # print(res)

            # FIXME:
            # a metric for confusion here would be good
            # how many similar sized intervals are there to pick
            # the smallest diff from
            #
            # if there is only one: no problem
            # if there is multiple: we can pick the wrong one
            # - we could rank on distance from center
            # - we could rank on magnitude of the peak
            #   (number of black pixels in the interval)
            # most_similar = process_intervals(intervals, counts, settings)
            most_similar = similar_sized_interval(
                intervals, settings["rim_size_pixels"]
            )
            print(f"{most_similar=}")
            lower, upper = most_similar
            if settings["interactive_plots"]:
                axB.axvline(lower, color="r")
                axB.axvline(upper, color="r")

            if rim_type & LEFT:
                rim_side = most_similar[1]
            elif rim_type & RIGHT:
                rim_side = most_similar[0]
                offset_setting *= -1

            rim_point = [int(rim_side), yfound]
            if False:
                expected_location = rim_side + offset_setting
                if settings["interactive_plots"]:
                    axC.plot(most_similar[0], yfound, "ro")  # x unknown, y known
                    axC.plot(most_similar[1], yfound, "ro")
                    axC.plot(expected_location, yfound, "go")
                    # axB.plot(expected_location, yfound, "go")
                    axB.plot(expected_location, 0, "go")
                    # axB.axvline(expected_location, color="g", linestyle="-.")

                closest = closest_interval(intervals, expected_location)
                lower, upper = closest
                if settings["interactive_plots"]:
                    axC.axvline(lower, color="b")  # v instead of h
                    axC.axvline(upper, color="b")  # v instead of h
                    axB.axvline(lower, color="b")
                    axB.axvline(upper, color="b")

                final_location = lower + (upper - lower) * 0.5
                rim_point = [int(final_location), yfound]

    elif rim_type & TOP or rim_type & BOTTOM:
        figaxis = None
        if settings["interactive_plots"]:
            figaxis = axB
        intervals, counts = search_rim(
            bw, pixel_value=0, axis=1, settings=settings, figaxis=figaxis
        )

        if intervals:
            # most_similar = process_intervals(intervals, counts, settings) # waterstaat try
            most_similar = similar_sized_interval(
                intervals, settings["rim_size_pixels"]
            )
            print(f"{most_similar=}")
            lower, upper = most_similar
            if settings["interactive_plots"]:
                axB.axvline(lower, color="r")
                axB.axvline(upper, color="r")

            if rim_type & TOP:
                rim_side = most_similar[1]
            elif rim_type & BOTTOM:
                rim_side = most_similar[0]
                offset_setting *= -1

            rim_point = [xfound, int(rim_side)]
            if False:
                expected_location = rim_side + offset_setting
                if settings["interactive_plots"]:
                    axC.plot(xfound, most_similar[0], "ro")
                    axC.plot(xfound, most_similar[1], "ro")
                    axC.plot(xfound, expected_location, "go")
                    axB.plot(expected_location, 0, "go")
                    # axB.axvline(expected_location, color="g", linestyle="-.")

                closest = closest_interval(intervals, expected_location)
                lower, upper = closest
                if settings["interactive_plots"]:
                    axC.axhline(lower, color="b")
                    axC.axhline(upper, color="b")
                    axB.axvline(lower, color="b")
                    axB.axvline(upper, color="b")

                final_location = lower + (upper - lower) * 0.5
                rim_point = [xfound, int(final_location)]

    try:
        delta = 40
        slicedbw = bw[
            int(rim_point[1]) - delta : int(rim_point[1]) + delta,
            int(rim_point[0]) - delta : int(rim_point[0]) + delta,
        ]
        if rim_type & LEFT or rim_type & RIGHT:
            pixels = slicedbw[:, delta]
        elif rim_type & TOP or rim_type & BOTTOM:
            pixels = slicedbw[delta, :]

        visibility = np.count_nonzero(pixels == 0) / (delta * 2)
        print("RIM VISIBILITY", visibility)
    except IndexError:
        visibility = 0.0

    if settings["interactive_plots"]:
        ax1.imshow(bw, cmap=mpl.colormaps["gray"])
        small_image = image[
            int(rim_point[1]) - delta : int(rim_point[1]) + delta,
            int(rim_point[0]) - delta : int(rim_point[0]) + delta,
        ]
        try:
            axD.imshow(slicedbw, cmap=mpl.colormaps["gray"])
            axE.imshow(small_image)
        except:
            pass
        if rim_type & LEFT or rim_type & RIGHT:
            axE.axvline(delta, color="orange")
        elif rim_type & TOP or rim_type & BOTTOM:
            axE.axhline(delta, color="orange")
        plt.show()
        plt.close()

    return [rim_point, visibility]


if __name__ == "__main__":
    samples = [
        [
            "https://stacks.stanford.edu/image/iiif/yj390bt2192/yj390bt2192_00_0001/2484,1123,166,1246/full/0/default.jpg",
            TOP,
        ],
        [
            "https://stacks.stanford.edu/image/iiif/yj390bt2192/yj390bt2192_00_0001/2484,10315,166,1246/full/0/default.jpg",
            BOTTOM,
        ],
        [
            "https://stacks.stanford.edu/image/iiif/yj390bt2192/yj390bt2192_00_0001/1435,10392,1246,166/full/0/default.jpg",
            LEFT,
        ],
        [
            "https://stacks.stanford.edu/image/iiif/yj390bt2192/yj390bt2192_00_0001/11633,2122,1246,166/full/0/default.jpg",
            RIGHT,
        ],
    ]

    for sample in samples:
        print(detect_peaks_in_detail(*sample))
