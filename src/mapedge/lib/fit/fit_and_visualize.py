# import pprint


from typing import List
from .svg_visualizer import SVGVisualizer
import numpy as np
from matplotlib import pyplot as plt

# def intersect(self, line):
#     a, b, c = line.normal[0], line.normal[1], line.distance
#     width, height = self.size

#     # Determine the intersection points with the rectangle boundaries
#     if a == 0:
#         # Horizontal line: y = -c/b
#         y1, y2 = -c / b, -c / b
#         x1, x2 = 0, width
#     elif b == 0:
#         # Vertical line: x = -c/a
#         x1, x2 = -c / a, -c / a
#         y1, y2 = 0, height
#     else:
#         # General case: compute intersection points
#         x1 = max(0, min(width, -c / a))
#         y1 = max(0, min(height, -(c + a * x1) / b))
#         x2 = max(0, min(width, -c / a + width * abs(b) / (a * a + b * b)))
#         y2 = max(
#             0, min(height, -(c + a * x2) / b + height * abs(a) / (a * a + b * b))
#         )

#     return [x1, x2], [y1, y2]


def make_simple_svg_mask(img_size, points):
    svg_points = [f"{int(x)},{int(y)}" for (x, y) in points]

    wh = 'width="' + str(img_size[0]) + '" height="' + str(img_size[1]) + '"'
    svg_mask = f"<svg {wh}>"
    svg_mask += f'<polygon points="{" ".join(svg_points)}" />'
    svg_mask += "</svg>"
    return svg_mask


def main():
    import json
    import numpy as np

    # filename = "/scratch/iiif_inspect/tmk_hires/point_sample_0008.json"

    i = 0

    for i in range(181):
        try:
            number = f"{i:04}"
            print()
            print("<hr>")
            print()
            print(f"<h1>Map sheet {number}</h1>")

            filename = f"/scratch/iiif_inspect/waterstaatskaart_edition_2/point_sample_{number}.json"
            # filename = "/scratch/iiif_inspect/waterstaatskaart_edition_1/point_sample_0001.json"
            with open(filename) as fp:
                J = json.load(fp)
            print(
                f"""<img src="{J['iiif_end_point']}/full/512,/0/default.jpg" width="384" alt="IIIF image">"""
            )
            print("")
            print(
                f'<img src="/home/martijn/tmp/ed2_svg_{number}.svg" width="384" alt="Point Samples">'
            )
            print("")
            # print(
            #     f'<img src="/home/martijn/tmp/hist_{number}.png" width="384" alt="Histogram">'
            # )
            # print("")

            # print(J["samples"].keys())  # ['outer-inner']
            dualrim_inner_left = np.array(J["samples"]["outer-inner"]["inner"]["left"])
            dualrim_inner_top = np.array(J["samples"]["outer-inner"]["inner"]["top"])
            dualrim_inner_bottom = np.array(
                J["samples"]["outer-inner"]["inner"]["bottom"]
            )
            dualrim_inner_right = np.array(
                J["samples"]["outer-inner"]["inner"]["right"]
            )

            dualrim_outer_left = np.array(J["samples"]["outer-inner"]["outer"]["left"])
            dualrim_outer_top = np.array(J["samples"]["outer-inner"]["outer"]["top"])
            dualrim_outer_bottom = np.array(
                J["samples"]["outer-inner"]["outer"]["bottom"]
            )
            dualrim_outer_right = np.array(
                J["samples"]["outer-inner"]["outer"]["right"]
            )

            # print(J["samples"].keys())  # ['outer-inner']

            outer_left = np.array(J["samples"]["outer"]["outer"]["left"])
            outer_top = np.array(J["samples"]["outer"]["outer"]["top"])
            outer_bottom = np.array(J["samples"]["outer"]["outer"]["bottom"])
            outer_right = np.array(J["samples"]["outer"]["outer"]["right"])

            # print(j.keys())
            iiif_url_overview = f"{J['iiif_end_point']}/full/4096,/0/default.jpg"

            vis = SVGVisualizer(J["image_size"])

            vis.add_xlink_image(iiif_url_overview)

            radius = 8
            vis.add_points(dualrim_inner_left, radius=radius)
            vis.add_points(dualrim_outer_left, radius=radius)
            vis.add_points(dualrim_inner_right, radius=radius)
            vis.add_points(dualrim_outer_right, radius=radius)
            vis.add_points(dualrim_inner_bottom, radius=radius)
            vis.add_points(dualrim_outer_bottom, radius=radius)
            vis.add_points(dualrim_inner_top, radius=radius)
            vis.add_points(dualrim_outer_top, radius=radius)

            radius = 30
            vis.add_points(outer_left, radius=radius)
            vis.add_points(outer_right, radius=radius)
            vis.add_points(outer_bottom, radius=radius)
            vis.add_points(outer_top, radius=radius)

            print("<h2>single points</h2>")
            print("<pre>")
            for points, side in zip(
                [
                    outer_left,
                    outer_top,
                    outer_right,
                    outer_bottom,
                ],
                ["left", "bottom", "right", "top"],
            ):
                print(f"{side:>7}", "▣" * len(points), f" ({len(points)})")
            # print("--")
            print("</pre>")
            print("<h2>dual points</h2>")
            print("<pre>")
            for points, side in zip(
                [
                    dualrim_outer_left,
                    dualrim_outer_top,
                    dualrim_outer_right,
                    dualrim_outer_bottom,
                ],
                ["left", "bottom", "right", "top"],
            ):
                print(f"{side:>7}", "▣" * len(points), f" ({len(points)})")
            print("</pre>")
            # n = np.array([0.99998829, 0.00483941])
            # p0 = np.array([966.1015625, 5358.4015625])
            # p1 = np.array([749.3046875, 5358.4015625])

            # ln0 = Line2.from_normal_and_point(n, p0)
            # ln1 = Line2.from_normal_and_point(n, p1)

            ## fit individual lines
            # for pts in [
            #     inner_left,
            #     outer_left,
            #     inner_right,
            #     outer_right,
            #     inner_bottom,
            #     outer_bottom,
            #     inner_top,
            #     outer_top,
            # ]:
            #     ln = fit_line(pts)
            #     # vis.add_line(ln)

            ## fit lines at start / end of each rim frame
            if True:
                lines = []
                how_many_are_relevant = 4
                for points in [
                    # outer_left,
                    # outer_right,
                    # outer_bottom,
                    # outer_top,
                    dualrim_inner_left,
                    dualrim_inner_right,
                    dualrim_inner_bottom,
                    dualrim_inner_top,
                ]:
                    for start in [True, False]:
                        if start:
                            pts = points[:how_many_are_relevant]
                        else:
                            pts = points[-how_many_are_relevant:]
                        # try:
                        ln = fit_line(pts)
                        lines.append(ln)
                        # except:
                        #     lines.append(None)
                        #     pass
                        # vis.add_points(pts, radius=18)
                        vis.add_line(ln)

                ###################
                #          l r b t
                # starts: [0,2,4,6]
                # ends:   [1,3,5,7]

                index_pairs = [
                    (0, 4),  # left start, bottom start
                    (1, 7),  # left end, top end
                    (2, 6),  # right start, top end
                    (3, 5),  # right start, bottom end
                ]
                xpts = []
                for i, j in index_pairs:
                    # if lines[i] is not None and lines[j] is not None:
                    pt = lines[i].intersect(lines[j])
                    xpts.append(pt)
                vis.add_points(xpts, radius=18)

                # visualize the corner points as closed polygon
                vis.add_polygon(xpts)

                # print(xpts)

                if True:
                    ###
                    # do check stats on 'errors', i.e. signed distances of the point samples to the 4 lines of the mask
                    ###
                    poly_sides: List[Line2] = []
                    for side in zip(xpts, xpts[1:] + [xpts[0]]):
                        # print(side)
                        try:
                            ln = Line2.from_points(*side)
                            # print(ln)
                            poly_sides.append(ln)
                            vis.add_line(ln)
                        except TypeError:
                            continue

                    fig = plt.figure(layout="constrained")
                    # axis_dict = fig.subplot_mosaic(mosaic="AAABBC;AAABBC")
                    # axis_dict = fig.subplot_mosaic(mosaic="BBBC;BBBC")
                    axis_dict = fig.subplot_mosaic(
                        mosaic="ABCD",
                        # sharex=True,
                        # sharex=bool(rim_side & RimSide.LEFT or rim_side & RimSide.RIGHT),
                        # sharey=bool(rim_side & RimSide.BOTTOM or rim_side & RimSide.TOP),
                        # sharey=True,
                    )
                    axA = axis_dict["A"]
                    axB = axis_dict["B"]
                    axC = axis_dict["C"]
                    axD = axis_dict["D"]

                    for line, pts, title, axis in zip(
                        poly_sides,
                        [
                            # inner_left,
                            # inner_top,
                            # inner_right,
                            # inner_bottom,
                            outer_left,
                            outer_top,
                            outer_right,
                            outer_bottom,
                        ],
                        ["left", "bottom", "right", "top"],
                        [axA, axB, axC, axD],
                    ):
                        dists = [line.signed_distance(pt) for pt in pts]
                        # print(dists)
                        axis.set_title(title)
                        # axis.set_xlim(-100, 100)
                        axis.set_ylim(0, 25)

                        axis.hist(dists, bins=5)
                    fig.suptitle("Distance to fitted line in pixel (ê)")
                    plt.tight_layout()
                    # plt.show()
                    plt.savefig(f"/home/martijn/tmp/hist_{number}.png")
                    plt.close()

                # FIXME:
                # this closed polygon has 4 sides

                # to assess the quality of the result, there is a few things to consider:
                # - the distances from the measured points towards these sides
                # - how these distances are 'distributed' (same for all sides?)

                # if we would fit a big rectangle, then the total errors are minimized by definition

                # we can also compare against a straight line fit per side, considering all points per side

            # vis.add_points(inner_left[:10], radius=15)
            # vis.add_line(fit_line(inner_left[:10]))

            # vis.add_points(inner_top[:10], radius=15)
            # vis.add_line(fit_line(inner_top[:10]))

            # vis.add_points(outer_top[:10], radius=15)
            # vis.add_line(fit_line(outer_top[:10]))

            # ###
            # how_many_are_relevant = 6
            # point_sets = [inner_top[:how_many_are_relevant], outer_top[:how_many_are_relevant]]
            # lines0 = fit_parallel_lines(point_sets)
            # for points in point_sets:
            #     vis.add_points(points, radius=18)
            # for line in lines0:
            #     vis.add_line(line)

            # point_sets = [
            #     inner_right[:how_many_are_relevant],
            #     outer_right[:how_many_are_relevant],
            # ]
            # lines1 = fit_parallel_lines(point_sets)
            # for points in point_sets:
            #     vis.add_points(points, radius=18)
            # for line in lines1:
            #     vis.add_line(line)

            # xpt = []
            # for ln0 in lines0:
            #     for ln1 in lines1:
            #         xpt.append(ln0.intersect(ln1))

            # vis.add_points(xpt, radius=3.5)

            ####
            # Try to fit 2x two parallel lines at rim corners that are forming rectangular
            # shape at corner
            # ###
            if False:
                # how many points to include in the fitting process
                how_many_are_relevant = 3
                corners = [
                    [
                        inner_top[:how_many_are_relevant],
                        outer_top[:how_many_are_relevant],
                        inner_right[:how_many_are_relevant],
                        outer_right[:how_many_are_relevant],
                    ],
                    [
                        inner_top[-how_many_are_relevant:],
                        outer_top[-how_many_are_relevant:],
                        inner_left[-how_many_are_relevant:],
                        outer_left[-how_many_are_relevant:],
                    ],
                    [
                        inner_bottom[-how_many_are_relevant:],
                        outer_bottom[-how_many_are_relevant:],
                        inner_right[-how_many_are_relevant:],
                        outer_right[-how_many_are_relevant:],
                    ],
                    [
                        inner_bottom[:how_many_are_relevant],
                        outer_bottom[:how_many_are_relevant],
                        inner_left[:how_many_are_relevant],
                        outer_left[:how_many_are_relevant],
                    ],
                ]
                for point_sets in corners:
                    # how to interpret the point set, as perpendicular or not
                    perpendicular = [False, False, True, True]

                    lines = fit_perpendicular_lines(point_sets, perpendicular)
                    for line in lines:
                        vis.add_line(line)

                    for points in point_sets:
                        vis.add_points(points, radius=18)

                    lines0 = [line for line, perp in zip(lines, perpendicular) if perp]
                    lines1 = [
                        line for line, perp in zip(lines, perpendicular) if not perp
                    ]
                    xpts = []
                    for ln0 in lines0:
                        for ln1 in lines1:
                            xpts.append(ln0.intersect(ln1))

                    vis.add_points(xpts, radius=23.5)

            svg = vis.show()
            with open(f"/home/martijn/tmp/ed2_svg_{number}.svg", "w") as fh:
                fh.write(svg)

            if False:
                import os

                os.system(
                    f"flatpak run org.inkscape.Inkscape /home/martijn/tmp/svg_{number}.svg &"
                )

            ##
            # Produce annotation page, pixel coordinates and mask,
            # to be joined with world coordinates from sheet index
            ##

            # use these corners for the annotation page
            corners_to_use = [(int(pt[0]), int(pt[1])) for pt in xpts]
            corners_to_use = corners_to_use[1:] + [corners_to_use[0]]

            # mask for use in annotation
            svg_mask = make_simple_svg_mask(J["image_size"], xpts)

            uuid = J["uuid"]
            # uuid = uuid.replace("_00_0001", "")
            record = {
                "uuid": uuid,
                "iiif_end_point": J["iiif_end_point"],
                "pixel_corner_points": corners_to_use,
                "svg_mask": svg_mask,
            }
            ndjson_output_filename = f"/tmp/tmk_rim_{number}.ndjson"
            with open(ndjson_output_filename, "w") as fh:
                json.dump(record, fh)
                fh.write("\n")
        except:
            continue
    # input("paused")


if __name__ == "__main__":
    main()
