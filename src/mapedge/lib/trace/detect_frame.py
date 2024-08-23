import glob
import itertools
import pprint
import cv2

import json
import numpy as np

# from .find_approx_away import find_closest

import matplotlib.pyplot as plt
import matplotlib as mpl

from .intervals import generate_intervals__num_std_dev__above

# from peak_find import get_persistent_homology as find_peaks

from .detect_fetch import fetch_image

import string

from .fuzzy import trapezoidal_mf, triangular_mf


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


def erode_dilate(image, kernel):
    # pattern = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
    its = 2
    eroded_image = cv2.erode(image, kernel, iterations=its)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=its)
    return dilated_image


# def erode_dilate_horizontal(image, size):
# return erode_dilate(image, np.array([[1]*size]))


def erode_dilate_horizontal_vertical(image, size):
    # return erode_dilate(image, np.array([[1]*size]))

    eroded_dilated_horizontal = erode_dilate(image, np.array([[1] * size]))
    eroded_dilated_vertical = erode_dilate(image, np.array([[1]] * size))
    combined = combine_eroded_images(eroded_dilated_horizontal, eroded_dilated_vertical)
    return combined


# gray = grayscale_image()


def pipeline(url):
    image = fetch_image(url)
    gray = grayscale_image(image)


def info_json(file_name):
    with open(file_name) as fh:
        info = json.load(fh)
    return info


def get_url(info):
    url = info["@id"]
    return url


def iiif_overview_image(url, overview_width):
    # width, height = info['width'], info['height']
    iiif_url_overview = f"{url}/full/{overview_width},/0/default.jpg"
    return iiif_url_overview


def canny_edge_detect(img_gray):
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    return img_edges


def combine_eroded_images(one, two):
    combined_image = cv2.add(one, two)
    return combined_image


def dilate_combined_image_to_make_lines_thicker(combined_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    its = 10
    dilated = cv2.dilate(combined_image, kernel, iterations=its)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(dilated, kernel, iterations=its)
    return eroded


def make_fat(combined_image, size, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    eroded = cv2.erode(combined_image, kernel, iterations=iterations)
    blowup = cv2.dilate(eroded, kernel, iterations=iterations)
    return blowup


def find_contours(image):
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return cnts


def determine_sizes(cnts):
    areas = np.array([])

    for cc in range(len(cnts)):
        # if len(cnts[cc]) < 6:
        # area = cv2.contourArea(cnts[cc])
        _, _, w, h = cv2.boundingRect(cnts[cc])
        area = w * h
        areas = np.append(areas, area)

    return areas


def largest_contour(cnts, sizes):
    return cnts[np.argmax(sizes)]


def largest_rect(cnts, sizes):
    rect = cv2.minAreaRect(cnts[np.argmax(sizes)])
    return rect


def pixel_count(image, value, axis):
    return np.count_nonzero(image == value, axis=axis)


def moving_average(arr, window_size):
    return np.convolve(arr, np.ones((window_size,)) / window_size, mode="same")


def detect_lines(image, min_line_length=50):
    # Apply HoughLinesP method to
    # to directly obtain line end points
    lines_list = []
    # lines = cv2.HoughLinesP(
    #     image, # Input edge image
    #     10, # Distance resolution in pixels
    #     1* (np.pi/180.0), # Angle resolution in radians
    #     threshold=100, # Min number of votes for valid line
    #     minLineLength=400, # Min allowed length of line
    #     maxLineGap=10 # Max allowed gap between line for joining them
    # )
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, 20, None, min_line_length, 10)
    # Iterate over points
    if lines is not None:
        for points in lines:
            # Extracted points nested in the list
            x1, y1, x2, y2 = points[0]
            # Draw the lines joing the points
            # On the original image
            # cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
            # Maintain a simples lookup list for points
            lines_list.append([(x1, y1), (x2, y2)])
    print(lines_list)
    return lines_list


def approximate_map_content(iiif_image_url, settings):
    image = fetch_image(iiif_image_url)
    gray = grayscale_image(image)

    if True:
        # preprocessing strategy
        # gray = threshold_image(gray, 127)
        gray = adaptive_threshold_image(gray)
        # - edge detect by canny filter
        # - erode dilate horizontal and vertical lines separately
        # - combine the erode / dilated images
        edge_detected = canny_edge_detect(gray)
        size = settings["erode_dilate_kernel"]
        eroded_dilated_horizontal = erode_dilate(edge_detected, np.array([[1] * size]))
        # plt.imshow(eroded_dilated_horizontal, cmap=mpl.colormaps["gray"])
        # plt.show()

        eroded_dilated_vertical = erode_dilate(edge_detected, np.array([[1]] * size))
        # plt.imshow(eroded_dilated_vertical, cmap=mpl.colormaps["gray"])
        # plt.show()

        combined_pre_dilate = invert_image(
            combine_eroded_images(eroded_dilated_horizontal, eroded_dilated_vertical)
        )

        # combined = erode_dilate_lines_found(combined_pre_dilate)

        combined = combined_pre_dilate
    else:
        # preprocessing strategy
        # - threshold the gray image into binary image
        combined = threshold_image(gray, 127)  # 178)

        # combined = adaptive_threshold_image(gray)
    # plt.imshow(combined, cmap=mpl.colormaps["gray"])
    # plt.show()

    # thick_lines = dilate_combined_image_to_make_lines_thicker(combined)

    # # plt.show()
    # # lines = detect_lines(eroded_dilated_horizontal)
    # # for line in lines:
    # #     print(line)
    # #     plt.plot([pt[0] for pt in line], [pt[1] for pt in line], 'bo', linestyle="--")
    # # plt.show()

    # # lines = detect_lines(eroded_dilated_vertical)
    # # for line in lines:
    # #     print(line)
    # #     plt.plot([pt[0] for pt in line], [pt[1] for pt in line], 'ro', linestyle="--")
    # # plt.show()

    # lines = detect_lines(thick_lines, min_line_length=100)
    # for line in lines:
    #     print(line)
    #     plt.plot([pt[0] for pt in line], [pt[1] for pt in line], 'go', linestyle="--")
    # plt.show()

    # # Create blank image
    # blank = np.zeros(image.shape, gray.dtype)
    # # Draw filtered lines on blank image
    # for line in lines:
    #     cv2.line(blank,
    #         line[0], line[1],
    #         (255, 255, 255), 1, cv2.LINE_AA)

    # blank = threshold_image(grayscale_image(blank))

    # # blank = erode_dilate_lines_found(blank)

    # plt.imshow(blank)
    # plt.show()

    # contours = find_contours(blank)
    # sizes = determine_sizes(contours)

    # # rect = largest_rect(contours, sizes)

    # # box = np.int0(cv2.boxPoints(rect))

    # # # order the points in the contour such that they appear
    # # # in top-left, top-right, bottom-right, and bottom-left
    # # # order, then draw the outline of the rotated bounding
    # # # box
    # # print(box)

    # image_with_contours = cv2.drawContours(image, [largest_contour(
    #     contours, sizes
    # )], -1, (0,255,0), 3)
    # plt.imshow(image_with_contours)
    # plt.show()

    # plt.imshow(thick_lines, cmap=mpl.colormaps["gray"])
    # plt.show()

    if settings["interactive_plots"]:
        fig = plt.figure(layout="constrained")
        # mosaic allows to specify in easy way the layout of the charts
        ax_dict = fig.subplot_mosaic(mosaic="AADD;AADD;BECF")
        # fig = plt.figure()
        figaxisA = ax_dict["A"]
        figaxisB = ax_dict["B"]
        figaxisC = ax_dict["C"]
        figaxisD = ax_dict["D"]
        figaxisE = ax_dict["E"]
        figaxisF = ax_dict["F"]

        figaxisA.imshow(bgr2rgb_image(image))
        # figaxisD.imshow(combined_pre_dilate, cmap=mpl.colormaps['gray'])
        figaxisD.imshow(combined, cmap=mpl.colormaps["gray"])

    bounds = []

    for ax in (0, 1):
        count = pixel_count(combined, value=0, axis=ax)

        if settings["interactive_plots"]:
            if ax == 0:
                plotax = figaxisB
            #     figaxisE.hist(count, bins = 100)
            elif ax == 1:
                plotax = figaxisC
            #     figaxisF.hist(count, bins = 100)

        # peaks = find_peaks(count)
        # xs = [peak.born for peak in peaks]
        # ys = [count[x] for x in xs]
        # if settings['interactive_plots']:
        #     plotax.plot(xs, ys, "ro")
        #     plotax.set_label(ax)

        mu = np.mean(count)
        sigma = np.std(count)

        # FIXME: should be setting
        numstddev = 1.5

        intervals = generate_intervals__num_std_dev__above(count, numstddev)
        print(f"{intervals=}")
        interval_stats = {}
        for interval in intervals:
            lower, upper = interval
            size = upper - lower
            mid = lower + size // 2
            values = [count[i] for i in range(lower, upper)]
            values.sort()
            median = values[len(values) // 2]
            total = sum(values)
            avg = total / size
            interval_stats[mid] = [interval, size, total, avg, size * avg, median]
            if settings["interactive_plots"]:
                plotax.axvline(mid, color="purple")
                if ax == 0:
                    figaxisD.axvline(mid, color="purple", linestyle=":")
                elif ax == 1:
                    figaxisD.axhline(mid, color="purple", linestyle=":")
        mid_pts = list(interval_stats.keys())
        print(f"{mid_pts=}")
        print(f"{interval_stats=}")
        img_center = image.shape[1 - ax] // 2
        alternatives = []
        # expected_rim_pixels = settings['expected_rim_pixels']
        for layout in settings["expected_layouts"]:
            ranking_matrix = []

            expected_start, expected_end = layout[ax]
            # print(ax, expected_start, expected_end)
            expected_span = expected_end - expected_start

            # expected_blacks = expected_rim_pixels * expected_span

            # FIXME: this should be refactored to
            # separate functionality that can be re-used
            # for ranking the different alternatives

            # create the solutions
            # a solution is a start and a end of the map frame
            # along one dimension

            # solutions = []
            # for start in mid_pts:
            #     end = find_closest(mid_pts, start + expected_span)
            #     solutions.append((start, end))

            solutions = [valid for valid in itertools.combinations(mid_pts, 2)]
            print(f"{solutions=}")
            for start, end in solutions:
                span = end - start
                center = start + span // 2
                # pixel_size = interval_stats[start][1]
                total_black_pixels = interval_stats[start][2]
                ranking_matrix.append(
                    [
                        # abs(expected_rim_pixels - pixel_size),
                        pow(expected_span - total_black_pixels, 2),
                        pow(
                            img_center - center, 2
                        ),  # center should be roughly in the center
                        pow(
                            expected_span - span, 2
                        ),  # the width of the frame should correspond with the map frame
                        pow(
                            expected_start - start, 2
                        ),  # the start should be at the expected start location
                        pow(
                            expected_end - end, 2
                        ),  # the end should be at the expected end location
                    ]
                )
            # print(f"{ranking_matrix=}")
            no_solutions = len(ranking_matrix)
            no_criteria = len(ranking_matrix[0])

            transposed = []
            for criterion_idx in range(no_criteria):
                col = []
                for row in ranking_matrix:
                    # print(criterion_idx, row[criterion_idx])
                    col.append(row[criterion_idx])
                s = sum(col)
                if s == 0:
                    # all values will be 0.0
                    # we can just use the column as-is
                    # this prevents division by zero
                    normalized_col = col
                else:
                    normalized_col = [v / s for v in col]
                # print(normalized_col, sum(normalized_col))

                transposed.append(normalized_col)

            normalized_matrix = []
            for sol_idx in range(no_solutions):
                row = []
                for cri_idx in range(no_criteria):
                    val = transposed[cri_idx][sol_idx]
                    row.append(val)
                normalized_matrix.append(row)

            # for row in normalized_matrix:
            #     print(f"{row=}")

            # weights for the ranking criteria
            weights = [
                # 0.1, 0.1,
                0.05,
                0.05,
                0.85,
                0.025,
                0.025,
            ]
            min_weights = [0, 1, 80, 8, 8]

            # min_weights = [1 / w for w in weights]

            # weights applied
            weighted_criteria = [
                [min_weights[i] * val for i, val in enumerate(alternative)]
                for alternative in normalized_matrix
            ]
            # for w, sol in zip(weighted_criteria, solutions):
            #     print(sum(w), w, sol)
            # sum the weighted criteria per solution
            summed_criteria = [sum(alternative) for alternative in weighted_criteria]
            # combine score / solution
            ranked_solutions = list(zip(summed_criteria, solutions))

            # we sort the scored solution
            # the first solution in this list is the best ranked solution
            ranked_solutions.sort()
            print("-------------------=--------------------")
            for sol in ranked_solutions:
                print(f" {sol[0]} = {sol[1]}")
            print("FIT of solution", ranked_solutions[0])
            alternatives.append(ranked_solutions[0])

        alternatives.sort()
        best = alternatives[0]
        # print("GLOBAL BEST", best)
        bounds.append(best[-1])
        if settings["interactive_plots"]:
            if ax == 0:
                figaxisA.axvline(bounds[-1][0])
                figaxisA.axvline(bounds[-1][1])
            elif ax == 1:
                figaxisA.axhline(bounds[-1][0])
                figaxisA.axhline(bounds[-1][1])

        # for first, second in zip(intervals[:], intervals[1:]):
        #     span = second[0] - first[1]
        # if span > 100:
        #     print(f"{first[1]=}, {second[0]=}, {span=}")

        if settings["interactive_plots"]:
            plotax.axhline(mu + numstddev * sigma, color="g")

            plotax.plot(list(range(len(count))), count)
            # plt.show()

    # plt.gcf().set_size_inches(10, 10)

    # fig.savefig(output_fig_name)
    ranked_solutions = []
    if len(bounds) == 2:
        ranked_solutions.extend(bounds[0][:])
        ranked_solutions.extend(bounds[1][:])

    print(ranked_solutions)
    if settings["interactive_plots"]:
        plt.show()
    # plt.savefig("iiifcache")
    return ranked_solutions
    # plt.close()

    # print(bounds)

    # input('paused')

    # contours = find_contours(dilate_combined_image_to_make_lines_thicker(combined))
    # print(contours)

    # sizes = determine_sizes(contours)
    # print(sizes)

    # largest = largest_rect(contours, sizes)
    # print(largest)

    # plt.imshow(combined, cmap=mpl.colormaps["gray"])
    # plt.show()

    # image_with_contours = cv2.drawContours(image, contours, -1, (0,255,0), 3)
    # plt.imshow(image_with_contours)
    # plt.show()

    # thicker = dilate_combined_image_to_make_lines_thicker(combined)
    # plt.imshow(thicker, cmap=mpl.colormaps["gray"])
    # plt.show()


import unittest


def parse_pipeline_steps(input_string):
    steps = input_string.replace("\n", "")
    steps = steps.replace(" ", "")
    steps = steps.split("|")
    parsed_steps = []
    for step in steps:
        step = step.strip()
        if not step:
            continue
        if step.startswith("#"):
            continue
        if "(" in step and ")" in step:
            function_name, args_str = step.split("(")
            args_str = args_str.strip(")")
            # args = [int(arg) for arg in args_str.split(',')]
            args = []
            for arg in args_str.split(","):
                # if "=" in arg:
                #     _, val = arg.split("=")
                # else:
                #     val = arg
                args.append(int(arg))
            parsed_steps.append((function_name, args))
        else:
            parsed_steps.append((step, []))
    return parsed_steps


class TestParseSteps(unittest.TestCase):
    def test_parse_steps1(self):
        self.assertEqual(
            parse_pipeline_steps("fetch|gray|threshold(140, 123)"),
            [("fetch", []), ("gray", []), ("threshold", [140, 123])],
        )

    def test_parse_steps2(self):
        self.assertEqual(
            parse_pipeline_steps("step1|step2|step3(1, 2, 3)"),
            [("step1", []), ("step2", []), ("step3", [1, 2, 3])],
        )

    def test_parse_steps3(self):
        self.assertEqual(parse_pipeline_steps("single_step"), [("single_step", [])])

    def test_parse_steps4(self):
        self.assertEqual(
            parse_pipeline_steps("step_with_args(100)"), [("step_with_args", [100])]
        )


import math


def transform_to_square(lst):
    n = len(lst)
    size = math.isqrt(n)
    if size * size != n:
        size += 1
    matrix = [lst[i * size : i * size + size] for i in range(size)]

    for row in matrix:
        while len(row) != size:
            row.append(None)
    return matrix


class TestTransformToSquare(unittest.TestCase):
    def test_transform_to_square(self):
        self.assertEqual(transform_to_square([1, 2, 3, 4]), [[1, 2], [3, 4]])
        self.assertEqual(
            transform_to_square([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        )
        self.assertEqual(
            transform_to_square([1, 2, 3, 4, 5, 6, 7]),
            [[1, 2, 3], [4, 5, 6], [7, None, None]],
        )
        self.assertEqual(transform_to_square([1]), [[1]])


def unsharp_mask(img, blur_size=(5, 5), imgWeight=1.5, gaussianWeight=-0.5):
    gaussian = cv2.GaussianBlur(img, blur_size, 0)
    return cv2.addWeighted(img, imgWeight, gaussian, gaussianWeight, 0)


def mean_shift(img):
    return cv2.pyrMeanShiftFiltering(img, sp=5, sr=40)


def equalize_hist(img):
    return cv2.equalizeHist(img)


def equalize_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def otsu_threshold(image):
    otsu_threshold, image_result = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return image_result


def dilate_vert(image, size):
    kernel = np.ones((size, 1), np.uint8)
    return cv2.dilate(image, kernel)


def dilate_hori(image, size):
    kernel = np.ones((1, size), np.uint8)
    return cv2.dilate(image, kernel)


def dilate_hori_vert(image, size):
    # first dilate
    kernel = np.ones((size, 1), np.uint8)
    vert = cv2.dilate(image, kernel)
    # second dilate
    kernel = np.ones((1, size), np.uint8)
    hori = cv2.dilate(image, kernel)
    vert = invert_image(vert)
    hori = invert_image(hori)
    # combine and return
    return invert_image(vert + hori)


def erode2x2(image):
    kernel = np.ones((2, 2), np.uint8)
    return cv2.erode(image, kernel)


def erode_square(image, size):
    kernel = np.ones((size, size), np.uint8)
    return cv2.erode(image, kernel)


def dilate_square(image, size):
    kernel = np.ones((size, size), np.uint8)
    return cv2.dilate(image, kernel)


def apply_pipeline(input_image, pipeline, inspect=False):
    funcs = {
        "bgr2rgb": bgr2rgb_image,
        "fetch": fetch_image,
        "gray": grayscale_image,
        "threshold": threshold_image,
        "adaptive_threshold": adaptive_threshold_image,
        "erode_dilate_horizontal_vertical": erode_dilate_horizontal_vertical,
        "invert": invert_image,
        "canny": canny_edge_detect,
        "make_fat": make_fat,
        "unsharp_mask": unsharp_mask,
        "mean_shift": mean_shift,
        "equalize_hist": equalize_hist,
        "equalize_clahe": equalize_clahe,
        "otsu_threshold": otsu_threshold,
        "dilate_vert": dilate_vert,
        "dilate_hori": dilate_hori,
        "erode2x2": erode2x2,
        "dilate_hori_vert": dilate_hori_vert,
        "erode_square": erode_square,
        "dilate_square": dilate_square,
    }

    # pipeline from text to functions and additional arguments
    # next to image
    steps = parse_pipeline_steps(pipeline)
    functions = [(funcs[label], label, args) for label, args in steps]

    # plotting
    if inspect:
        fig = plt.figure(layout="constrained")
        mosaic = [string.ascii_uppercase[i] for i in range(len(steps) + 1)]
        mosaic = transform_to_square(mosaic)
        ax_dict = fig.subplot_mosaic(
            mosaic=mosaic, sharex=True, sharey=True, empty_sentinel=None
        )
        cmap = None

    def plot(image, ax_dict, i, label, args, cmap):
        if label == "gray":
            cmap = mpl.colormaps["gray"]
        axis = ax_dict[string.ascii_uppercase[i]]
        axis.imshow(image, cmap=cmap)
        title = f"[{i}] {label}"
        if args:
            title += "::" + ", ".join(map(str, args))
        axis.set_title(title)
        return cmap

    image = input_image
    if inspect:
        cmap = plot(image, ax_dict, 0, "original", None, cmap)
    for i, (func, label, args) in enumerate(functions, start=1):
        #

        all_args = [image] + args
        image = func(*all_args)
        if inspect:
            cmap = plot(image, ax_dict, i, label, args, cmap)

    # cmap = plot(image, ax_dict, "input for " +label, cmap)
    if inspect:
        plt.show()
        plt.close()
    return image


def approximate_map_frame_sides(url: str, settings):
    # url = "https://stacks.stanford.edu/image/iiif/qg824rx0608/qg824rx0608_00_0001/full/1024,/0/default.jpg"

    # pipeline = "fetch|bgr2rgb|gray|threshold(127)|canny|invert|erode_dilate_horizontal_vertical(3)"

    #     pipeline = """
    # fetch |
    # bgr2rgb |
    # gray |
    # adaptive_threshold |
    # # threshold(134) |
    # canny |
    # # invert |
    # erode_dilate_horizontal_vertical(5) |
    # invert |
    # make_fat(3, 4)
    # """

    print(url)
    input_image = fetch_image(
        url, settings["enable_caching"]
    )  # [150:600, 160:860] # [y/rows : x/cols]

    #

    # cropped_image = input_image

    # pipeline = "mean_shift|gray|adaptive_threshold|canny|erode_dilate_horizontal_vertical(5) |invert"
    # pipeline = "gray|equalize_hist|adaptive_threshold|canny|erode_dilate_horizontal_vertical(5) |invert"

    # This seems a sensible pre-processing pipeline for the Waterstaatskaart
    # pipeline = "bgr2rgb|gray|equalize_clahe|otsu_threshold|make_fat(2,4)|canny|invert"

    # pipeline = "bgr2rgb|gray|equalize_clahe|otsu_threshold|make_fat(2,4)|canny|invert|dilate_hori_vert(5)|erode_square(3)"
    # pipeline = "bgr2rgb|gray|adaptive_threshold"

    # bonnebladen
    # pipeline = "bgr2rgb|gray|threshold(50)"

    # image = apply_pipeline(input_image, pipeline, inspect=True)
    image = apply_pipeline(
        input_image,
        settings["pipeline_overview"],
        inspect=settings["interactive_plots"],
    )

    # # Convert image to grayscale
    # gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # # Use canny edge detection
    # edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # lines = cv2.HoughLinesP(
    #     edges,  # Input edge image
    #     1,  # Distance resolution in pixels
    #     np.pi / 180,  # Angle resolution in radians
    #     threshold=100,  # Min number of votes for valid line
    #     minLineLength=200,  # Min allowed length of line
    #     maxLineGap=5,  # Max allowed gap between line for joining them
    # )

    # if lines.any():
    #     # Iterate over points

    #     lines_list = []
    #     for points in lines:
    #         # Extracted points nested in the list
    #         x1, y1, x2, y2 = points[0]
    #         # Draw the lines joing the points
    #         # On the original image
    #         cv2.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #         # Maintain a simples lookup list for points
    #         lines_list.append([(x1, y1), (x2, y2)])

    # # Save the result image
    # cv2.imwrite("detectedLines.png", edges)

    # linesP = cv2.HoughLinesP(image, 1, np.pi / 180, 50, None, 50, 10)
    # tmp = []
    # for line in linesP:
    #     tmp.append([[line[0][0], line[0][1]], [line[0][2], line[0][3]]])
    # plt.imshow(apply_pipeline(input_image, "bgr2rgb"))
    # import pprint

    # pprint.pprint(tmp)
    # from matplotlib import collections as mc

    # # lines = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
    # # c = np.array([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])

    # lc = mc.LineCollection(tmp, linewidths=2)
    # fig, ax = plt.subplots()
    # ax.add_collection(lc)

    # plt.show()
    # input("pause")

    # data = [None, None]
    peak_mid_points = [None, None]
    stats = [None, None]
    for ax in (0, 1):
        count = pixel_count(image, value=0, axis=ax)
        num_std_dev = 0.5
        if settings["interactive_plots"]:
            mu = np.mean(count)
            # FIXME: this sigma is not related to the setting below of 0.5
            sigma = np.std(count)
            plt.plot(list(range(len(count))), count)
            plt.axhline(mu, color="g")
            plt.axhline(mu + num_std_dev * sigma, color="g", linestyle=":")
            plt.show()
        # avg = moving_average(count, 40)
        # values = np.diff(count - avg)
        # mu = np.mean(values)
        # sigma = np.std(values)
        # plt.plot(list(range(len(values))), values, 'r')
        # plt.plot(list(range(len(count))), count)
        # plt.axhline(mu, color='g')
        # plt.axhline(mu+sigma, color='g', linestyle=":")

        # FIXME: this stddev multiplier (1.0 by default), should be part of the settings for finding the overview
        # so it can be overwritten, case for which this applies:
        # - [ ] Stanford, Japan, 926, wj252dn6880
        intervals = generate_intervals__num_std_dev__above(count, num_std_dev)
        interval_stats = {}
        for interval in intervals:
            lower, upper = interval
            size = upper - lower
            mid = lower + size // 2
            values = [count[i] for i in range(lower, upper)]
            values.sort()
            median = values[len(values) // 2]
            total = sum(values)
            avg = total / size
            interval_stats[mid] = {
                "interval": interval,
                "values": values,
                "size": size,
                "sum": total,
                "avg": avg,
                "median": median,
                "max": max(values),
            }
        # print("## Processing", ax, "stats:")
        # pprint.pprint(interval_stats)

        mid_pts = []
        for interval in intervals:
            (
                lower,
                upper,
            ) = interval
            half = (upper - lower) // 2
            mid = lower + half
            # plt.axvline(mid, color="purple", linestyle=":")
            mid_pts.append(mid)
        # plt.show()
        peak_mid_points[ax] = mid_pts
        stats[ax] = interval_stats
        # data[ax] = count

    # plt.show()

    # expected_layouts = [
    #     [[200, 820], [210, 600]], # wsk, at size width=1024
    #     # [[140, 890], [135, 790]], # nk

    # ]

    size = image.shape
    size = [size[1], size[0]]  # swap x with y

    to_consider_peaks = [[], []]
    for ax in [0, 1]:
        print(f"Going over axis: {ax}")
        mid_pts = peak_mid_points[ax]
        for peak_mid_point in mid_pts:
            peak_stats = stats[ax][peak_mid_point]
            print(peak_mid_point, "->", peak_stats)
            print(
                size[(ax + 1) % 2]
            )  # to see the occupancy, we need the other side of the frame
            percentage_filled = peak_stats["max"] / size[(ax + 1) % 2]
            ###  ###  ###
            ### FIXME: the following values for how much the line is filled are hard coded and should be placed in settings.json
            ###  ###  ###
            print(
                percentage_filled,
                trapezoidal_mf(percentage_filled, 0.05, 0.65, 0.8, 0.85),
            )
            if trapezoidal_mf(percentage_filled, 0.05, 0.65, 0.8, 0.85) > 0:
                to_consider_peaks[ax].append(peak_mid_point)
    # input("paused")
    layout = settings["expected_layouts"][0]

    outcomes = []
    for layout in settings["expected_layouts"]:
        if settings["interactive_plots"]:
            plt.imshow(apply_pipeline(input_image, "bgr2rgb"))
            for ax in [0, 1]:
                for pt in to_consider_peaks[ax]:
                    # depending on which axis, plot horizontal or vertical line
                    if ax == 1:
                        axline = plt.axhline
                    elif ax == 0:
                        axline = plt.axvline
                    axline(pt, color="purple", linestyle="dashdot", linewidth=0.25)
        center = image.shape
        print("image shape", image.shape)
        expected_center = [center[1] // 2, center[0] // 2]  # swaps x|y
        print("expected center ⊚", expected_center)
        to_return = []
        ranks = 0
        for ax in [0, 1]:
            expected_start, expected_end = layout[ax]

            expected_span = expected_end - expected_start
            print(f"## {ax=} {expected_start=} {expected_end=} {expected_span=}")

            # based on the mid points of the peaks
            solutions = [
                valid
                for valid in itertools.combinations(
                    to_consider_peaks[ax], 2
                )  # to_consider_peaks
            ]
            # print(f"{solutions=}")
            ranked_solutions = []
            for start, end in solutions:
                span = end - start
                center = start + span // 2
                # print("⊚ -- ⊚ [", start, "-", end, "]::", center, "versus",expected_center[ax], triangular_membership_function(center, expected_center[ax] - 100, expected_center[ax], expected_center[ax] + 100))
                # print("⊚ -- ⊚")
                # print(stats[ax][start])
                # print(stats[ax][end])
                # print()
                ranked_solutions.append(
                    [
                        # criteria
                        [
                            trapezoidal_mf(
                                span,
                                expected_span - 30,
                                expected_span - 10,
                                expected_span + 10,
                                expected_span + 30,
                            ),
                            # triangular_mf(start, expected_start - 50, expected_start, expected_start + 50),
                            triangular_mf(
                                end, expected_end - 50, expected_end, expected_end + 50
                            ),
                            triangular_mf(
                                start + expected_span,
                                expected_end - 20,
                                expected_end,
                                expected_end + 20,
                            ),
                            # triangular_mf(
                            #     center,
                            #     expected_center[ax] - 100,
                            #     expected_center[ax],
                            #     expected_center[ax] + 100,
                            # ),
                        ],
                        # the interval
                        start,
                        end,
                    ]
                )

            final = [[np.sum(sol[0]), sol] for sol in ranked_solutions]
            # final = [[sol[0:3], sol] for sol in ranked_solutions]

            final.sort(reverse=True)
            # this gives 2 scalars for a location (either left /right or top/ bottom)
            to_return.extend([final[0][1][1], final[0][1][2]])
            ranks += final[0][0]

            if settings["interactive_plots"]:
                pprint.pprint(final)

                # best = final[0]
                # print(best[1][3])
                # print(best[1][4])
                colours = ["green", "blue", "red"]
                for i in range(min(len(final), 1)):
                    best = final[i]
                    for pt in [best[1][1], best[1][2]]:
                        if ax == 1:
                            plt.axhline(
                                pt,
                                color=colours[i],
                                linestyle="dotted",
                                linewidth=3 - i,
                            )
                        elif ax == 0:
                            plt.axvline(
                                pt,
                                color=colours[i],
                                linestyle="dotted",
                                linewidth=3 - i,
                            )
                        # pprint.pprint(stats[ax][pt])
        if settings["interactive_plots"]:
            plt.show()
            plt.close()
        outcomes.append([to_return, ranks])

    outcomes.sort(key=lambda x: -x[1])
    print("⭐", outcomes)
    # input('paused detect_frame.py')
    # FIXME: incorrect (will return last)
    return outcomes[0][0]


# if __name__ == '__main__':
#     unittest.main()

if __name__ == "__main__":
    url = "https://stacks.stanford.edu/image/iiif/hb236xh2949/hb236xh2949_00_0001/full/1024,/0/default.jpg"
    image = approximate_map_frame_sides(url)

    # settings = {}
    # settings['overview_width'] = 2048
    # settings['glob_pattern'] = "iiif_cache/waterstaatskaart/*.json"
    # settings['erode_dilate_kernel'] = 5
    # settings['expected_size'] = [1235, 760]

    # for file_name in sorted(glob.glob(settings['glob_pattern'])):
    #     info = info_json(file_name)
    #     with open(file_name) as fh:
    #         info = json.load(fh)
    #         iiif_image_url = iiif_overview_image(get_url(info), settings['overview_width'])
    #         print(iiif_image_url)
    #         left, right, top, bottom = approximate_map_content(iiif_image_url, settings)
    #         print(left, right, top, bottom)
    #         input('paused')
