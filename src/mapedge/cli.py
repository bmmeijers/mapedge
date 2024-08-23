import argparse
from pathlib import Path
import sys, os

from collections import defaultdict
from mapedge.lib.report import create_quality_report
from mapedge.lib.fit.line2 import Line2

from mapedge.lib.trace import trace
from mapedge.lib.visualize.visualize_sampled_points import main as visualize
import json

from mapedge.lib.visualize import make_simple_svg_mask


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


class CLI:
    def __init__(self):
        pass

    def trace(self, args):
        """
        E.g.

        $ mapedge trace -s ~/Documents/work/2023-08_synthesis/mapedge/settings/tmk.json
        """
        # FIXME: the URL we should visit (info.json) + setting we should apply
        # If we remove the for loop inside `mapedge trace` and leave that to the
        # surrounding os, we can do something with xargs, and run even in parallel easily,
        # as a combination like this:
        # cat file | xargs -I %1 echo http://example.com/%1.tar
        # printf %s\\n {0..99} | xargs -n 1 -P 8 script-to-run.sh input/ output/
        print(f"trace, {args}")
        # settings = json.load(
        #     args.settings
        # )  # FIXME: try/except, mention invalid json or something

        all_settings = [json.load(f) for f in args.settings]
        settings = all_settings[0]
        for override in all_settings[1:]:
            settings.update(override)

        # override settings for interactively displaying the result
        settings["interactive_plots"] = args.i
        settings["produce_output"] = args.d
        print(settings)
        # import sys

        # sys.exit()

        trace(file_name=args.input_file, sheet_id=args.sheet_id, settings=settings)

    def visualize(self, args):
        print(f"visualize, {args}")
        visualize(args.input_folder, args.output_folder)

    # def fit(self, args):
    #     print(f"fit, {args}")
    #     # J = json.load(args.point_sample_file)
    #     # fit(J)

    def annotate(self, args):
        # print(f"annotate, {args}")
        from mapedge.lib.annotate import annotate

        # annotate()

        index_map_geojson = json.load(args.geojson_sheet_index)
        # corners = json.load(args.corners_file)

        corner_files = [json.load(f) for f in args.corners_file]
        # import pprint

        # pprint.pprint(corner_files)

        annotate(
            index_map_geojson,
            corner_files,
            args.sheet_column,
            args.annotation_page_file,
        )

    # def bar(self, args):
    #     print("bar {}".format(args.context))

    def report(self, args):
        create_quality_report(args.input_folder, args.output_file_name)

    def ransac(self, args):
        # print(f"ransac, {args}")
        import numpy as np
        from mapedge.lib.ransac import ransac, calculate_ransac_iterations

        max_iterations = calculate_ransac_iterations(0.5, 0.99999)
        threshold_pixel_dist = 10.0  # 1.5  # 0.8  # 0.5
        sample_point_count = 2
        do_plot = False

        lines = {}
        inliers = {}
        for file in args.point_sample_file:
            content = file.read()

            output_filename = "ransac__" + Path(file.name).stem + ".json"

            # sys.exit()

            # print(content)
            J = json.loads(content)
            # hexdigest = hashlib.md5(
            #     content.encode("utf-8"), usedforsecurity=False
            # ).hexdigest()

            for sample_strategy in [
                "outer",
                # "outer-inner",
            ]:
                # sample strategy >  > frame_side
                if sample_strategy not in lines:
                    lines[sample_strategy] = {}
                    inliers[sample_strategy] = {}

                if sample_strategy == "outer":
                    rims = ["outer"]
                elif sample_strategy == "outer-inner":
                    rims = ["outer", "inner"]
                for rim in rims:
                    if rim not in lines[sample_strategy]:
                        lines[sample_strategy][rim] = {}
                        inliers[sample_strategy][rim] = {}

                    for side in ["left", "right", "top", "bottom"]:
                        # print(sample_strategy, side)
                        # if side not in lines[sample_strategy][rim]:
                        #     lines[sample_strategy][rim][side] = {}
                        #     inliers[sample_strategy][rim][side] = {}
                        pts = np.array(J["samples"][sample_strategy][rim][side])
                        # def ransac(points, max_iterations=50, threshold=0.5, sample_size=2, do_plot=False):
                        #
                        best_fitted_line, inlier_indices = ransac(
                            pts,
                            max_iterations,
                            threshold_pixel_dist,
                            sample_point_count,
                            do_plot,
                        )

                        # FIXME: start / end | divide line into regions, e.g. 10 regions, 1/10 and 10/10 to be used?
                        # print(best_fitted_line.as_dict())
                        # print(inlier_indices)
                        # print("ε :=", threshold_pixel_dist)
                        lines[sample_strategy][rim][
                            side
                        ] = best_fitted_line  # .as_dict()
                        inliers[sample_strategy][rim][side] = inlier_indices
            # J["lines"] = lines
            # with open("/tmp/fitted_lines.json", "w") as fh:
            #     # lines["md5"] = hexdigest
            #     json.dump(J, fh)

            jsonified = defaultdict(lambda: {})
            for sample_strategy in [
                "outer",
                # "outer-inner",
            ]:
                jsonified[sample_strategy] = defaultdict(lambda: {})
                if sample_strategy == "outer":
                    rims = ["outer"]
                elif sample_strategy == "outer-inner":
                    rims = ["outer", "inner"]
                for rim in rims:
                    jsonified[sample_strategy][rim] = defaultdict(lambda: {})
                    for side in ["left", "right", "top", "bottom"]:
                        if lines[sample_strategy][rim][side]:
                            jsonified[sample_strategy][rim][side] = lines[
                                sample_strategy
                            ][rim][side].as_dict()
                        else:
                            jsonified[sample_strategy][rim][side] = None

            # Try to use the fitted lines in the following order:
            # 4 lines fitted by:
            #  outer-inner -> inner
            #  outer-inner -> outer
            #  outer -> outer
            # if 1 of them gives a result for 4 lines, quit the loop
            # this way we have hierarchy in fitting, and max chance to find corners
            #
            # FIXME:
            # - assumes that both strategies were executed for finding borders
            visit_next = True
            for tmp_strategy in ["outer"]:  # ["outer-inner", "outer"]:
                if not visit_next:
                    break
                if tmp_strategy == "outer-inner":
                    tmp_which = ["inner", "outer"]
                else:
                    tmp_which = ["outer"]
                for w in tmp_which:
                    if not visit_next:
                        break
                    tmp = []
                    for side in ["left", "right", "top", "bottom"]:
                        tmp.append(lines[tmp_strategy][w][side])

                    visit_next = None in tmp
                    # print(tmp_strategy, w, tmp, visit_next)
                    # if None in tmp:
                    #     continue
                    # else:
                    #     break

            if visit_next:
                print("WARNING : No option with all 4 lines present found")
                print(Path(file.name).stem)

            left, right, top, bottom = tmp

            # there is no guarantee that all 4 lines are found
            # (they can still be None at this point)
            # let's deal with this explicitly
            corners = []

            if left is not None and top is not None:
                corners.append(left.intersect(top))
            else:
                corners.append(None)

            if right is not None and top is not None:
                corners.append(right.intersect(top))
            else:
                corners.append(None)

            if right is not None and bottom is not None:
                corners.append(right.intersect(bottom))
            else:
                corners.append(None)

            if left is not None and bottom is not None:
                corners.append(left.intersect(bottom))
            else:
                corners.append(None)

            # print(corners)

            # geometry = [bl, br, tr, tl]
            # pixel_corner_points = [list(map(int, pt)) for pt in corners]
            pixel_corner_points = []
            for pt in corners:
                if pt is not None:
                    pixel_corner_points.append(list(map(int, pt)))
                # else:
                #     pixel_corner_points.append(None)

            # pprint.pprint(lines)
            # uuid = J["uuid"]
            # iiif_end_point = J["iiif_end_point"]

            svg_mask = make_simple_svg_mask(J["image_size"], pixel_corner_points)
            record = {
                # "uuid": uuid,
                # "iiif_end_point": iiif_end_point,
                "pixel_corner_points": pixel_corner_points,
                "svg_mask": svg_mask,
                "fitted_lines": jsonified,
                "inliers": inliers,
                "threshold": threshold_pixel_dist,
            }

            J["corners"] = record

            out_name = os.path.join(args.output_folder, output_filename)
            print(out_name)
            with open(out_name, "w") as fh:
                json.dump(J, fh)

    def test_argp(self, args):
        print(args)
        # corner_jsons = [json.load(f) for f in args.path]
        # print(corner_jsons)


#     def fit_stats(self, args):
#         with open("/tmp/fitted_lines.json") as fh:
#             J = json.load(fh)
#             print(J)

#         # having the points content in a different file
#         # than the fitted lines will be error prone!
#         #
#         # let's *not* do that...
#         #
#         # let's make an enriched point_sample file to which we add:
#         # - the fitted lines
#         # - the indices of points to preserve
#         #
#         # we save these files in a different folder (phase1 / phase2 / or more descriptive)
#         # we do this with the same json structure, and we are good to go in a sequential phase
#         #
#         # can we also handle then how we fit lines at the start or at the end?
#         # with open("/scratch/iiif_inspect/tmk_hires/point_sample_0020.json") as fh:
#         #     pts_sample = json.load(fh)
#         #     print(pts_sample)

#         for which in ["outer", "outer-inner"]:
#             for side in ["left", "right", "top", "bottom"]:
#                 d = J["lines"][which]["outer"][side]
#                 print(
#                     """
# """
#                 )
#                 print("## ", which, "-", side)
#                 print(Line2.from_dict(d))
#                 print(J["samples"][which]["outer"][side])


def main():
    cli = CLI()

    parser = argparse.ArgumentParser(
        # description="MapEdge, user-assisted geo-referencing of historic map sheets, based on the map frame its edges",
        description="MapEdge, geo-referencing historic map sheets, based on the map frame its extent",
        epilog="""
        TODO: Text to show after the commands
        """,
    )
    parser.add_argument(
        "-v",
        "--version",
        help="show version and exit",
        action="version",
        version="0",  # FIXME: deduplicate with version in setup.py
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Actions to perform", required=True
    )

    trace = subparsers.add_parser("trace", help="Harvest points on the rim")
    trace.set_defaults(func=cli.trace)
    # points.add_argument(
    #     "-s",
    #     help="settings",
    #     # action="version",
    #     # version="0",  # ?FIXME: deduplicate with version in setup.py
    # )
    trace.add_argument("input_file", help="info.json")
    trace.add_argument("sheet_id", type=int, help="sheet identifier")
    trace.add_argument(
        "-s",
        "--settings",
        help="Settings file",
        # nargs="+",
        action="append",
        required=True,
        type=argparse.FileType("r", encoding="utf-8"),
    )
    trace.add_argument("-i", help="Interactive", action=argparse.BooleanOptionalAction)
    trace.add_argument(
        "-d",
        "--dry-run",
        dest="d",
        action="store_false",
        help="Do not store output",
        # action=argparse.BooleanOptionalAction,
    )

    # fit = subparsers.add_parser("fit", help="Fit lines and find corners")
    # fit.add_argument(
    #     "point_sample_file",
    #     help="File with sampled points",
    #     type=argparse.FileType("r", encoding="utf-8"),
    # )

    # fit.set_defaults(func=cli.fit)

    # fit = subparsers.add_parser("fit", help="Fit lines and find corners")
    # fit.add_argument(
    #     "point_sample_file",
    #     help="File with sampled points",
    #     type=argparse.FileType("r", encoding="utf-8"),
    # )
    # fit.set_defaults(func=cli.fit)

    ransac = subparsers.add_parser(
        "ransac",
        help="Perform ransac to determine which points to keep and calculate corner points",
    )
    ransac.add_argument(
        "point_sample_file",
        nargs="+",
        help="File with sampled points",
        type=argparse.FileType("r", encoding="utf-8"),
    )
    ransac.add_argument(
        "-o",
        "--output",
        dest="output_folder",
        help="Output folder",
        type=dir_path,
    )
    ransac.set_defaults(func=cli.ransac)

    # fit_stats = subparsers.add_parser(
    #     "fit_stats", help="Perform ransac to determine which points to keep"
    # )
    # fit_stats.set_defaults(func=cli.fit_stats)

    annotate = subparsers.add_parser(
        "annotate",
        help="Based on calculated corner points, make an annotation page (join to sheet index)",
    )
    annotate.add_argument(
        "geojson_sheet_index",
        help="geojson File with sheet index",
        type=argparse.FileType("r", encoding="utf-8"),
    )
    annotate.add_argument(
        "corners_file",
        nargs="+",
        help="File(s) with sampled points *and* lines added",
        type=argparse.FileType("r", encoding="utf-8"),
    )
    annotate.add_argument(
        "annotation_page_file",
        help="Output filename for annotation page",
        type=argparse.FileType("w", encoding="utf-8"),
    )
    annotate.add_argument(
        "-s",
        dest="sheet_column",
        required=True,
    )
    annotate.set_defaults(func=cli.annotate)

    visualize = subparsers.add_parser(
        "visualize",
        help="Visualize the found points as SVG",
    )
    visualize.add_argument("input_folder")
    visualize.add_argument("output_folder")
    visualize.set_defaults(func=cli.visualize)

    report = subparsers.add_parser(
        "report",
        help="Quality report for various aspects on the processed sheets as CSV file",
    )
    report.add_argument("input_folder")
    report.add_argument(
        "output_file_name",
        help="Output filename for report CSV",
        type=argparse.FileType("w", encoding="utf-8"),
    )
    report.set_defaults(func=cli.report)

    # bar = subparsers.add_parser("bar", help="bar some Bars")
    # bar.add_argument("context", help="context for bar")
    # bar.set_defaults(func=cli.bar)

    # tmp = subparsers.add_parser("test")
    # tmp.add_argument(
    #     "path",
    #     nargs="+",
    #     help="Path of a file or a folder of files.",
    #     type=argparse.FileType("r", encoding="utf-8"),
    # )
    # parser.add_argument("output", type=dir_path)
    # tmp.set_defaults(func=cli.test_argp)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
