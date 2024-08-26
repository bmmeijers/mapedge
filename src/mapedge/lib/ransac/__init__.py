import random
import numpy as np
import unittest

import math

from matplotlib import pyplot as plt

from mapedge.lib.fit.fitting import fit_line
from .distance import calculate_distances

# for stats / distances
# from itertools import combinations
# from math import hypot

# import sys


def pixels_cm(pix, dpi=200):
    # 600 DPI = 236.22047244094 Pixels per cm.\
    # dpi = 200
    dots_per_cm = 2.54 / dpi
    # print(dots_per_cm)

    return pix * dots_per_cm  # 0.0264583333  # dots_per_cm


random.seed(2023)


def ransac(points, max_iterations=50, threshold=0.5, sample_size=2, do_plot=False):
    # total_fitting_pts = 0
    best_model = None
    inlier_indices = None
    if len(points) < sample_size:
        # print("early return, point count under sample size")
        return best_model, inlier_indices

    inlier_ratio = 0.0
    consider_all_input = True
    for _ in range(max_iterations + 1):
        if inlier_ratio == 1.0:
            # we cannot get better than including all source points
            break

        # modified ransac
        # try with all sampled points in first iteration before we go to sampling
        if consider_all_input:
            # print("fit all points")
            # first iteration, we take all points
            sampled = points
            consider_all_input = False  # next iteration, do take sample

        else:
            #  pick randomly `sample_size` items to fit model on
            # print("fit on sample")
            sample_indices = random.sample(range(0, len(points)), sample_size)
            sample_indices.sort()
            sampled = points[sample_indices]

        line = fit_line(sampled)
        projected = line.project_points(points)
        # projected = project_points(points, normal, dist)

        dists = calculate_distances(points, projected)
        inliers = points[dists < threshold]

        # print(np.mean(dists))

        current_ratio = (len(inliers)) / len(points)

        if do_plot:
            # TODO: create a matrix of sqrt(max_iterations) of subplots
            # and only show once this matrix of small multiples (when we are done)

            # highest = np.max(points)
            plt.plot(projected[:, 0], projected[:, 1], "r")
            # orth_vectors = projected - points
            for point, proj_point in zip(points, projected):
                plt.quiver(
                    *point,
                    *(proj_point - point),
                    color="k",
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    headwidth=2.5,
                    width=0.0025,
                )

            plt.scatter(
                points[:, 0],
                points[:, 1],
                s=5,
                facecolors="black",
                edgecolors="black",
            )
            plt.scatter(
                inliers[:, 0],
                inliers[:, 1],
                s=15.0,
                facecolors="green",
                edgecolors="green",
            )

            # plt.xlim([0, highest])
            # plt.ylim([0, highest])
            plt.axis("equal")
            plt.show()
            plt.close()

        if len(inliers) and current_ratio >= inlier_ratio:
            # FIXME: accept a better solution only if it:
            # - improves on the inlier ratio
            # - and when it has lower error metric (mean dist smaller)
            # - i.e. we should compute the distances and take average of
            #        these distances also into account?
            # print('accept better solution with inlier ratio', current_ratio)
            # update inlier ratio to only get better results later
            # than what we achieved here
            inlier_ratio = current_ratio
            total_fitting_pts = len(inliers)
            # re-fit the model with all the inlying data points
            best_model = fit_line(inliers)
            # print("fitting count     : ", total_fitting_pts)
            # print("best model set to : ", best_model)
            inlier_indices = np.argwhere((dists < threshold) == True)
            # print(len(inlier_indices), len(points))
    return best_model, inlier_indices.T.tolist()[0]


class TestRansac(unittest.TestCase):
    def test_ransac1(self):
        # Test 1: Check if the function handles an empty array
        # FIXME: it does not
        points = np.array([])
        self.assertEqual(ransac(points), (None, 0))

    def test_ransac2(self):
        # Test 2: Check if the function handles a single point
        # FIXME: it does not
        points = np.array([[1, 1]])
        self.assertEqual(ransac(points), (None, 0))

    def test_ransac3(self):
        # Test 3: Check if the function handles two points
        points = np.array([[1, 1], [2, 2]])
        model, fitting_pts = ransac(points)
        self.assertIsNotNone(model)
        self.assertEqual(fitting_pts, 2)

    def test_ransac4(self):
        # Test 4: Check if the function handles multiple points
        points = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        model, fitting_pts = ransac(points)
        self.assertIsNotNone(model)
        self.assertEqual(fitting_pts, 4)

    def test_ransac5(self):
        # Test 5: 100 points on a straight line (along x-axis)
        pts = []
        for x in range(0, 100):
            y = 0
            pt = [x, y]
            pts.append(pt)
        pts = np.array(pts)
        model, fitting_pts = ransac(pts)
        self.assertIsNotNone(model)
        self.assertEqual(fitting_pts, len(pts))


def calculate_ransac_iterations(w, p):
    """
    Calculate the number of iterations for RANSAC.

    Parameters:
    w (float): The inlier ratio in the data.
    p (float): The required confidence level.

    Returns:
    int: The calculated number of iterations.
    """
    if w <= 0:
        raise ValueError("Inlier ratio must be greater than 0")
    if not (0 < p < 1):
        raise ValueError("Confidence level must be between 0 and 1")

    return math.ceil(math.log(1 - p) / math.log(1 - w**2))


def example_calc_its():
    iterations = calculate_ransac_iterations(0.3, 0.99999)
    print(f"The number of iterations is {iterations}")


# def get_data(input_filename):
#     """
#     """
#     with open(input_filename) as fh:
#         lines = fh.readlines()
#     for line in lines:
#         j = json.loads(line.strip())
#         yield j

# def fit_rim_points():
#     for j in get_data("north_korea_rim_points.ndjson"):
#         # print(j['samples_per_rim'])
#         for rim in j['samples_per_rim']:
#             print(len(rim))
#             best_model, inlier_count = ransac(np.array(rim))
#             print(best_model, inlier_count, len(rim), "{:.2f}".format(inlier_count / len(rim)))

import unittest


class TestRansacIterations(unittest.TestCase):
    def test_inlier_ratio_zero(self):
        with self.assertRaises(ValueError):
            calculate_ransac_iterations(0, 0.99)

    def test_inlier_ratio_negative(self):
        with self.assertRaises(ValueError):
            calculate_ransac_iterations(-0.5, 0.99)

    def test_inlier_ratio_greater_than_one(self):
        with self.assertRaises(ValueError):
            calculate_ransac_iterations(1.5, 0.99)

    def test_confidence_level_zero(self):
        with self.assertRaises(ValueError):
            calculate_ransac_iterations(0.5, 0)

    def test_confidence_level_negative(self):
        with self.assertRaises(ValueError):
            calculate_ransac_iterations(0.5, -0.5)

    def test_confidence_level_greater_than_one(self):
        with self.assertRaises(ValueError):
            calculate_ransac_iterations(0.5, 1.5)

    def test_valid_parameters(self):
        iterations = calculate_ransac_iterations(0.5, 0.99)
        self.assertEqual(iterations, 17)


def example_ransac():
    points = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    model, fitting_pts = ransac(points, do_plot=True)
    print(model)
    print(fitting_pts)


# def make_simple_svg_mask(img_size, points):
#     svg_points = [f"{x},{y}" for (x, y) in points]

#     wh = 'width="' + str(img_size[0]) + '" height="' + str(img_size[1]) + '"'
#     svg_mask = f"<svg {wh}>"
#     svg_mask += f'<polygon points="{" ".join(svg_points)}" />'
#     svg_mask += "</svg>"
#     return svg_mask


# def process_sheet(
#     J,
#     max_iterations,
#     threshold_pixel_dist,
#     sample_point_count,
#     threshold_count_confidence,
#     do_plot=False,
#     output_debug=sys.stdout,
# ):

#     # print(f"<h1>{number}</h1> <a id='{sheet_id}'></a>", file=output_debug)
#     print(
#         f"""<img loading="lazy" src="{J['iiif_end_point']}/full/512,/0/default.jpg" width="384" alt="IIIF image">""",
#         file=output_debug,
#     )
#     print("", file=output_debug)
#     uuid = J["uuid"]
#     print(
#         f'<img loading="lazy" src="{uuid}.svg" width="384" alt="Point Samples">',
#         file=output_debug,
#     )
#     print("", file=output_debug)

#     sufficiently_sampled_outer = True
#     sufficiently_sampled_both = True

#     for which in ["outer", "outer-inner"]:
#         counts = []
#         print(f"<p>{which}</p>", file=output_debug)
#         print("<ul>", file=output_debug)
#         for side in ["left", "right", "top", "bottom"]:
#             pts = np.array(J["samples"][which]["outer"][side])
#             # def ransac(points, max_iterations=50, threshold=0.5, sample_size=2, do_plot=False):
#             #
#             best_fitted_line, inlier_count = ransac(
#                 pts,
#                 max_iterations,
#                 threshold_pixel_dist,
#                 sample_point_count,
#                 do_plot,
#             )
#             if best_fitted_line:
#                 projected = best_fitted_line.project_points(pts)
#                 # projected = project_points(points, normal, dist)

#                 dists = calculate_distances(pts, projected)
#                 # inlier_indices = np.arange(0, len(pts))[
#                 #     dists < threshold_pixel_dist
#                 # ]
#                 inlier_indices = []
#                 for i, d in enumerate(dists):
#                     if d < threshold_pixel_dist:
#                         inlier_indices.append(i)
#             else:
#                 inlier_indices = []
#             perc = 0
#             if len(pts):
#                 perc = round(inlier_count / len(pts) * 100.0, 1)
#             print(
#                 "<li>",
#                 inlier_count,
#                 "/",
#                 len(pts),
#                 " &middot; ",
#                 f"{perc}% - {side} <small>{inlier_indices}</small></li>",
#                 file=output_debug,
#             )
#             counts.append(inlier_count)

#             if best_fitted_line:
#                 projected = best_fitted_line.project_points(pts)
#                 dists = calculate_distances(pts, projected)
#                 print(
#                     """
# <details>
# <summary>
# <small>
#     ê distance (against inliers line fit)
# </small>
# </summary>
# """,
#                     file=output_debug,
#                 )

#                 print("<table>", file=output_debug)
#                 print(
#                     "<tr><th>index</th><th>pt</th><th>dist (px)</th></tr>",
#                     file=output_debug,
#                 )
#                 for i, (pt, dist) in enumerate(zip(pts, dists)):
#                     print(
#                         f"<tr><td>{i}</td><td>{pt}</td><td>{dist}</td></tr>",
#                         file=output_debug,
#                     )
#                 print("</table>", file=output_debug)
#                 print(
#                     """
# </details>
# """,
#                     file=output_debug,
#                 )
#         if all(count >= threshold_count_confidence for count in counts):
#             print("✅", file=output_debug)
#         else:
#             print("❌", file=output_debug)
#             sufficiently_sampled_both = False
#             if which == "outer":
#                 sufficiently_sampled_outer = False

#         print("</ul>", file=output_debug)

# if sufficiently_sampled_outer:
#     can_run_automated_outer += 1
#     sheets_outer.append(sheet_id)
# if sufficiently_sampled_both:
#     can_run_automated_both += 1
#     sheets_both.append(sheet_id)

# if not sufficiently_sampled_outer or not sufficiently_sampled_both:
#     should_harvest_again.append(sheet_id)


# if __name__ == "__main__":
#     # unittest.main(verbosity=5)

#     # example_calc_its()
#     # import sys
#     import glob
#     import os

#     # sys.exit()
#     import json

#     # folder_name = "/scratch/iiif_inspect/north_korea_individual_run/"
#     folder_name = "/scratch/iiif_inspect/north_korea/"

#     # will store for all sheets in the series the output
#     output_filename = "/tmp/rim.ndjson"

#     output_debug = open("/tmp/nk/ransac.html", "w")

#     # with open(output_filename, "w") as fh:
#     #     pass

#     ###########################
#     ## PHASE I : the 'reliability' of the points
#     ###########################

#     # how many tries to fit a line
#     max_iterations = calculate_ransac_iterations(0.5, 0.99999)
#     threshold_pixel_dist = 5  # 0.8  # 0.5
#     sample_point_count = 2
#     do_plot = False

#     threshold_count_confidence = 8
#     can_run_automated_outer = 0
#     can_run_automated_both = 0

#     sheets_both = []
#     sheets_outer = []
#     should_harvest_again = []
#     file_names = glob.glob(os.path.join(folder_name, "*.json"))

#     total_sheet_count = 0
#     for filename in file_names:
#         if "~" in filename:
#             continue
#         sheet_id = int(filename.split("_")[-1].replace(".json", ""))
#         number = f"{sheet_id:04}"
#         total_sheet_count += 1

#         # for sheet_id, filename in enumerate(sorted(file_names)):
#         # for sheet_id in range(181):
#         # folder_name = "/scratch/iiif_inspect/north_korea/"

#         # i = 39
#         number = f"{sheet_id:04}"

#         # filename = f"/scratch/iiif_inspect/waterstaatskaart_edition_1/point_sample_{number}.json"
#         with open(filename) as fp:
#             J = json.load(fp)

#         print(f"<h1>{number}</h1> <a id='{sheet_id}'></a>", file=output_debug)
#         print(
#             f"""<img loading="lazy" src="{J['iiif_end_point']}/full/512,/0/default.jpg" width="384" alt="IIIF image">""",
#             file=output_debug,
#         )
#         print("", file=output_debug)
#         uuid = J["uuid"]
#         print(
#             f'<img loading="lazy" src="{uuid}.svg" width="384" alt="Point Samples">',
#             file=output_debug,
#         )
#         print("", file=output_debug)

#         sufficiently_sampled_outer = True
#         sufficiently_sampled_both = True

#         for which in ["outer", "outer-inner"]:
#             counts = []
#             print(f"<p>{which}</p>", file=output_debug)
#             print("<ul>", file=output_debug)
#             for side in ["left", "right", "top", "bottom"]:
#                 pts = np.array(J["samples"][which]["outer"][side])
#                 # def ransac(points, max_iterations=50, threshold=0.5, sample_size=2, do_plot=False):
#                 #
#                 best_fitted_line, inlier_count = ransac(
#                     pts,
#                     max_iterations,
#                     threshold_pixel_dist,
#                     sample_point_count,
#                     do_plot,
#                 )
#                 if best_fitted_line:
#                     projected = best_fitted_line.project_points(pts)
#                     # projected = project_points(points, normal, dist)

#                     dists = calculate_distances(pts, projected)
#                     # inlier_indices = np.arange(0, len(pts))[
#                     #     dists < threshold_pixel_dist
#                     # ]
#                     inlier_indices = []
#                     for i, d in enumerate(dists):
#                         if d < threshold_pixel_dist:
#                             inlier_indices.append(i)
#                 else:
#                     inlier_indices = []
#                 perc = 0
#                 if len(pts):
#                     perc = round(inlier_count / len(pts) * 100.0, 1)
#                 print(
#                     "<li>",
#                     inlier_count,
#                     "/",
#                     len(pts),
#                     " &middot; ",
#                     f"{perc}% - {side} <small>{inlier_indices}</small></li>",
#                     file=output_debug,
#                 )
#                 counts.append(inlier_count)

#                 if best_fitted_line:
#                     projected = best_fitted_line.project_points(pts)
#                     dists = calculate_distances(pts, projected)
#                     print(
#                         """
# <details>
#   <summary>
#     <small>
#      ê distance (against inliers line fit)
#     </small>
#   </summary>
#   """,
#                         file=output_debug,
#                     )

#                     print("<table>", file=output_debug)
#                     print(
#                         "<tr><th>index</th><th>pt</th><th>dist (px)</th></tr>",
#                         file=output_debug,
#                     )
#                     for i, (pt, dist) in enumerate(zip(pts, dists)):
#                         print(
#                             f"<tr><td>{i}</td><td>{pt}</td><td>{dist}</td></tr>",
#                             file=output_debug,
#                         )
#                     print("</table>", file=output_debug)
#                     print(
#                         """
# </details>
# """,
#                         file=output_debug,
#                     )
#             if all(count >= threshold_count_confidence for count in counts):
#                 print("✅", file=output_debug)
#             else:
#                 print("❌", file=output_debug)
#                 sufficiently_sampled_both = False
#                 if which == "outer":
#                     sufficiently_sampled_outer = False

#             print("</ul>", file=output_debug)

#         if sufficiently_sampled_outer:
#             can_run_automated_outer += 1
#             sheets_outer.append(sheet_id)
#         if sufficiently_sampled_both:
#             can_run_automated_both += 1
#             sheets_both.append(sheet_id)

#         if not sufficiently_sampled_outer or not sufficiently_sampled_both:
#             should_harvest_again.append(sheet_id)

#     print("<h1><a id='confidence'></a>Confidence</h1>", file=output_debug)

#     can_run_automated_outer_perc = round(
#         can_run_automated_outer / total_sheet_count * 100, 1
#     )
#     can_run_automated_both_perc = round(
#         can_run_automated_both / total_sheet_count * 100, 1
#     )
#     print(
#         f"Sheets [{total_sheet_count}] that have at all rims {threshold_count_confidence} points sampled, on a straight line, with distance threshold ε = {threshold_pixel_dist} px",
#         file=output_debug,
#     )
#     print(
#         f"<p>{can_run_automated_outer} ({can_run_automated_outer_perc}%) can run fully automated [outer rim only]</p><small>sheets: {sheets_outer}</small>",
#         file=output_debug,
#     )

#     print(
#         f"<p>{can_run_automated_both} ({can_run_automated_both_perc}%) can run fully automated [both rims]</p><small>sheets: {sheets_both}</small>",
#         file=output_debug,
#     )
#     print(
#         f"<p>To harvest again (either outer or outer-inner having too little points):</p><small>sheets: {should_harvest_again}</small>",
#         file=output_debug,
#     )
#     tmp = ",".join([f"<a href='#{s_id}'>{s_id}</a>" for s_id in should_harvest_again])
#     print(f"<p>{tmp}</p>", file=output_debug)
#     ###
#     # FIXME:
#     # do a check here on whether outer::outer / outer-inner::outer correspond to the same line
#     # inliers of outer::outer should be close to outer-inner::outer and vice versa
#     # {
#     #   ...
#     # }

#     ###
#     # FIXME:
#     # distances between outer-inner, does that correspond to distances between lines?
#     # import sys

#     # sys.exit()
#     ###########################
#     ## PHASE II : the corners
#     ###########################
#     for sheet_id in sheets_both:
#         # print(sheet_id)
#         number = f"{sheet_id:04}"
#         print("<h2>", sheet_id, "</h2>", file=output_debug)
#         filename = os.path.join(folder_name, f"point_sample_{number}.json")
#         with open(filename) as fp:
#             J = json.load(fp)

#         lines = []
#         for side in ["left", "right", "top", "bottom"]:
#             print("<p><b>", side, "</b></p>", file=output_debug)

#             ## FIXME:
#             ## this is *all* unfiltered outer-inner points
#             ## we should base this on filter'ed ones (inliers)
#             ##
#             ## or, we could select only these points from dual
#             ## that are close to the outer rim fit lines
#             tmp = []
#             for pair in zip(
#                 J["samples"]["outer-inner"]["outer"][side],
#                 J["samples"]["outer-inner"]["inner"][side],
#             ):
#                 start, end = pair
#                 deltas = [end[i] - start[i] for i in range(2)]
#                 size = hypot(*deltas)

#                 tmp.append(size)
#             print(
#                 "<small><details><summary>distance between pairs</summary>",
#                 file=output_debug,
#             )
#             print("<ul>", file=output_debug)
#             for size in tmp:
#                 print("<li>", size, "</li>", file=output_debug)
#             print("</ul>", file=output_debug)
#             print("</details></small>", file=output_debug)
#             print("<dl>", file=output_debug)
#             print("<dt>median</dt><dd>", np.median(tmp), "</dd>", file=output_debug)
#             print("<dt>μ</dt><dd>", np.mean(tmp), "</dd>", file=output_debug)
#             print("<dt>σ</dt><dd>", np.std(tmp), "</dd>", file=output_debug)
#             print("</dl>", file=output_debug)
#             pts = np.array(J["samples"]["outer-inner"]["inner"][side])
#             # def ransac(points, max_iterations=50, threshold=0.5, sample_size=2, do_plot=False):
#             #
#             best_fitted_line, inlier_count = ransac(
#                 pts,
#                 max_iterations,
#                 threshold_pixel_dist,
#                 sample_point_count,
#                 do_plot,
#             )
#             lines.append(best_fitted_line)
#         left, right, top, bottom = lines

#         if None in lines:
#             continue

#         corners = [
#             # FIXME: top / bottom naming is wrong in the .json file (due to y-down coordinate axis)
#             left.intersect(top),
#             right.intersect(top),
#             right.intersect(bottom),
#             left.intersect(bottom),
#         ]
#         # print(corners)

#         # geometry = [bl, br, tr, tl]
#         pixel_corner_points = [list(map(int, pt)) for pt in corners]
#         # print(pixel_corner_points)

#         ### distances on sides and diagonals

#         # FIXME: confirm that the real world distances calculated here are okay...
#         # - measure some sheets?
#         # [hori, diag, vert, vert, diag, hori]
#         # dists = []
#         for pair in combinations(corners, 2):
#             start, end = pair
#             deltas = [end[i] - start[i] for i in range(2)]
#             size = hypot(*deltas)
#             # dists.append(size)
#             print(
#                 f"<pre>{size:.2f} -- {start}, {end} {size:.0f} pix ~ {pixels_cm(size, dpi=600):.2f} cm</pre>",
#                 file=output_debug,
#             )

#         # print()
#         # print(f"# {J['sheet_id']}")
#         # print("-" * 60)
#         uuid = J["uuid"]
#         iiif_end_point = J["iiif_end_point"]

#         svg_mask = make_simple_svg_mask(J["image_size"], pixel_corner_points)
#         record = {
#             # FIXME: this is specific for NK
#             "uuid": uuid.replace("_00_0001", ""),
#             "iiif_end_point": iiif_end_point,
#             "pixel_corner_points": pixel_corner_points,
#             "svg_mask": svg_mask,
#         }

#         with open(output_filename, "a") as fh:
#             json.dump(record, fh)
#             fh.write("\n")
#     print(
#         "<strong>1 pixel :=", round(pixels_cm(1), 4), "cm</strong>", file=output_debug
#     )
