from mapedge.lib.fit.line2 import Line2
from mapedge.lib.visualize.svg_visualizer import SVGVisualizer
import glob
import os
import json
import numpy as np

import os


# FIXME: just transforming the sampled points to svg could be its own script?


# folder_name = f"/scratch/iiif_inspect/waterstaatskaart_edition_2"
# output_folder_name = "/tmp/waterstaatskaart__edition_2"

# folder_name = "/scratch/iiif_inspect/north_korea_individual_run/"
# output_folder_name = "/tmp/north_korea_tmp"


def main(folder_name, output_folder_name):

    # folder_name = "/scratch/iiif_inspect/north_korea/"
    # folder_name = "/scratch/iiif_inspect/tmk_hires/"  # point_sample_0008.json"
    # output_folder_name = "/tmp/tmk"

    # make sure we can write into output_folder_name
    os.makedirs(output_folder_name, exist_ok=True)
    glob_pattern = os.path.join(folder_name, "*.json")
    files = glob.glob(glob_pattern)
    files.sort()

    output_filenames = []
    for filename in files:
        # skip backup files
        if "~" in filename:
            continue
        i = int(filename.split("_")[-1].replace(".json", ""))
        number = f"{i:04}"
        with open(filename) as fp:
            J = json.load(fp)

        iiif_url_overview = f"{J['iiif_end_point']}/full/1024,/0/default.jpg"

        vis = SVGVisualizer(J["image_size"])

        vis.add_xlink_image(iiif_url_overview)

        radius = 8
        for sampling_strategy in ["outer", "outer-inner"]:
            if sampling_strategy == "outer":
                frames = ["outer"]
                text_anchor = "start"
            elif sampling_strategy == "outer-inner":
                frames = ["outer", "inner"]
                text_anchor = "end"

            for map_side in ["left", "right", "top", "bottom"]:
                color = vis.propose_color()
                for frame in frames:
                    try:
                        pts = np.array(J["samples"][sampling_strategy][frame][map_side])
                    except:
                        print(f"WARN: no points found under this key [{filename}]")
                        continue
                    vis.add_points(
                        pts, radius=radius, color=color, text_anchor=text_anchor
                    )
            # if sampling_strategy == "outer-inner":
            #     for side in ["left", "right", "top", "bottom"]:
            #         pts = np.array(J["samples"][sampling_strategy]["inner"][side])
            #         vis.add_points(pts, radius=radius)
        if "approximate_sides" in J:
            xmin, xmax, ymin, ymax = J["approximate_sides"]
            w, h = J["image_size"]

            vis.add_line(
                [[0, ymin], [w, ymin]],
                extra_attribs='stroke-width="2.5" stroke-dasharray="10,5"',
            )
            vis.add_line(
                [[0, ymax], [w, ymax]],
                extra_attribs='stroke-width="2.5" stroke-dasharray="10,5"',
            )
            vis.add_line(
                [[xmin, 0], [xmin, h]],
                extra_attribs='stroke-width="2.5" stroke-dasharray="10,5"',
            )
            vis.add_line(
                [[xmax, 0], [xmax, h]],
                extra_attribs='stroke-width="1.25" stroke-dasharray="10,5"',
            )

        # if this is a file to which the corners have been added
        # we will visualize a bit more (corners, fitted lines)
        if "corners" in J:
            # print(J["corners"]["inliers"])

            # print(J["corners"]["fitted_lines"])
            for sampling_strategy in ["outer", "outer-inner"]:
                if sampling_strategy == "outer":
                    frames = ["outer"]
                elif sampling_strategy == "outer-inner":
                    frames = ["outer", "inner"]

                for map_side in ["left", "right", "top", "bottom"]:
                    for frame in frames:

                        ln = J["corners"]["fitted_lines"][sampling_strategy][frame][
                            map_side
                        ]
                        # print(ln)
                        if ln is not None:
                            vis.add_line_equation(line=Line2.from_dict(ln))

                        pts = J["samples"][sampling_strategy][frame][map_side]
                        indices = J["corners"]["inliers"][sampling_strategy][frame][
                            map_side
                        ]
                        # print(pts, indices)
                        if pts and indices:
                            selected = [pts[i] for i in indices]
                            for pt in selected:
                                # vis.add_circle(pt, radius=20)
                                from mapedge.lib.trace.vectorops import add, mul

                                # make a centered rectangle
                                size = (30, 30)
                                to_add = mul(size, (-0.5, -0.5))
                                tl = add(pt, to_add)
                                vis.add_rect(
                                    topleft=tl,
                                    size=size,
                                    attributes='style="fill:none;stroke-width:0.5;stroke:darkgreen"',
                                )

        if "corners" in J and "pixel_corner_points" in J["corners"]:
            # print(J["corners"]["pixel_corner_points"])
            for pt in J["corners"]["pixel_corner_points"]:
                vis.add_crop_mark(pt)

        svg = vis.show()
        with open(os.path.join(output_folder_name, f"{J['uuid']}.svg"), "w") as fh:
            fh.write(svg)
        output_filenames.append(
            [f"{J['uuid']}.svg", iiif_url_overview, number, J["uuid"]]
        )

    m = map(
        lambda x: f"<h1>{x[2]} - {x[3]}</h1><p><img src='{x[1]}' loading='lazy' width='480'><img src='{x[0]}' width='480'></p>",
        output_filenames,
    )
    with open(os.path.join(output_folder_name, "index.html"), "w") as fh:
        fh.write("<!doctype html><html>")
        fh.write("<head><title>Traced map sheets</title></head>")
        fh.write("\n".join(m))
        fh.write("</body></html>")
