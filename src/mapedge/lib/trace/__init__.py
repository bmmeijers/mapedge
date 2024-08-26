from enum import IntFlag
import os
import aiohttp
import asyncio

from matplotlib import pyplot as plt
import matplotlib as mpl

# from itertools import combinations
# import itertools
import json
from typing import List
import time

from .detect_frame import (
    apply_pipeline,
    approximate_map_frame_sides,
)  # , approximate_map_content

# import detect_rim

# from approximate_corners import detect_peaks_in_detail, LEFT, RIGHT, BOTTOM, TOP
# from approximate_corners_overview import get_approximate_corners
# import math

# from fuzzy import gaussian_mf, trapezoidal_mf, triangular_mf, cauchy_mf
from .samples_make_mask import make_svg

import mapedge.lib.trace.vectorops as vec
from .centered_rect import CenteredRectangle
from .iiif_image_load import iiif_image_from_url


from .detect_rim import center_coordinates, search_rim

from .ranking import (
    determine_multiple_rim_points,
    get_part_count,
    extract_subregion,
    # size,
    determine_one_rim_point,
    get_translate_vector,
    # compensate_outside,
    RimSide,
    # mid,
)

from .open_with_move import OpenWithMove

# from settings import settings

# if settings["produce_output"]:
#     output_ndjson_file_name = settings["output_file_name"]
#     with open(output_ndjson_file_name, "w") as fh_samples:
#         pass


# def get_json(url):
#     import requests

#     print(url)
#     r = requests.get(url)
#     return r.json()


def rim_side_as_str(rim_side):
    if rim_side & RimSide.LEFT:
        return "left"
    elif rim_side & RimSide.RIGHT:
        return "right"
    elif rim_side & RimSide.TOP:
        return "top"
    elif rim_side & RimSide.BOTTOM:
        return "bottom"


# urls = ["http://localhost:8182/iiif/2/testbeeld.png/"]
# for sheet_id, url in enumerate(sorted(urls), start=1):

#     info = get_json(url)


class FrameType(IntFlag):
    OUTER = 1
    INNER = 2


class RimPoint:
    def __init__(
        self,
        x: int,
        y: int,
        setting_type: str,
        frame_type: FrameType,
        rim_side: RimSide,
    ):
        self.x = x
        self.y = y
        self.setting_type = setting_type
        self.frame_type = frame_type
        self.rim_side = rim_side

    def __str__(self):
        return f"{self.x} | {self.y} | {self.setting_type} | {self.frame_type} | {self.rim_side}"


def trace(file_name, sheet_id, settings):
    """Traces the map sheet image to find points on its edges"""
    # make the output folder, if it does not yet exist
    os.makedirs(settings["output_folder"], exist_ok=True)
    # load the IIIF image info json
    with open(file_name) as fh:
        try:
            info = json.load(fh)
        except Exception as e:
            print(f"FAILURE IN {file_name}")
            raise ValueError(f"not able to load json from {file_name}: {e}")

    url = info["@id"]
    width = info["width"]

    iiif_url_overview = f"{url}/full/{settings['overview_width']},/0/default.jpg"

    print()
    print(f"{sheet_id} {url}")
    print(iiif_url_overview)
    print("-" * 60)

    # find appropriate 'peaks' in the small image
    approximate_corners = approximate_map_frame_sides(iiif_url_overview, settings)
    scale_small2large = width / settings["overview_width"]
    if not approximate_corners:
        print(f"No corners found for sheet {sheet_id}")
        return
    (
        left,
        right,
        top,
        bottom,
    ) = approximate_corners

    # peaks in the small image
    approximate_sides = [left, right, top, bottom]
    # peaks in the high resolution image
    approximate_sides_large = [
        int(side * scale_small2large) for side in approximate_sides
    ]

    print(approximate_sides_large)
    print("# ", sheet_id)
    xmin, xmax, ymin, ymax = approximate_sides_large
    img_size = info["width"], info["height"]
    half_height_rect = int(40 * scale_small2large)

    cx = xmin + (xmax - xmin) // 2
    cy = ymin + (ymax - ymin) // 2

    percentage_to_skip = -0.15  # extend 15%?

    half_size_left_right = (ymax - ymin) // 2
    half_size_left_right *= (
        1 - percentage_to_skip
    )  # ratio to skip (I think we should halve the setting for total side)
    center_left = [xmin, cy]
    center_right = [xmax, cy]

    half_size_bottom_top = (xmax - xmin) // 2
    half_size_bottom_top *= 1 - percentage_to_skip

    center_bottom = [cx, ymin]
    center_top = [cx, ymax]

    rects = [
        CenteredRectangle(*side)
        for side in zip(
            [center_left, center_right, center_bottom, center_top],
            [
                [half_height_rect, half_size_left_right],
                [half_height_rect, half_size_left_right],
                [half_size_bottom_top, half_height_rect],
                [half_size_bottom_top, half_height_rect],
            ],
        )
    ]

    do_interactive_plot = settings["interactive_plots"]
    points_found: List[RimPoint] = []

    for rect, rim_side in zip(
        rects, [RimSide.LEFT, RimSide.RIGHT, RimSide.BOTTOM, RimSide.TOP]
    ):

        print(f"will fetch rectangle: {rect.iiif_region()}")

        # fetch the big rectangle that covers the rim
        x, y, w, h = rect.svg_region()
        # queried_rects.append(rect.svg_region())
        iiif_url = f"{url}/{rect.iiif_region()}/full/0/default.jpg"
        print(iiif_url)
        if False:
            from .detect_fetch import fetch_image

            region_image = fetch_image(iiif_url, use_cache=False)
        else:
            # Initial wait time in seconds
            wait_time = 2

            for i in range(5):  # Number of attempts
                try:
                    region_image = iiif_image_from_url(url, x, y, w, h)
                    break
                except (
                    aiohttp.client_exceptions.ClientOSError,
                    aiohttp.client_exceptions.ContentTypeError,
                    asyncio.exceptions.TimeoutError,
                ) as e:
                    print("💩 FAILURE in retrieving region image")
                    print(f"Attempt {i+1} failed with error: {e}")
                    print(f"Waiting for {wait_time} seconds before retrying...\n")
                    time.sleep(wait_time)
                    wait_time *= 2  # Double the wait time for the next attempt
                # relevant links:
                # - https://note.nkmk.me/en/python-pillow-concat-images/
                # - https://note.nkmk.me/en/python-opencv-hconcat-vconcat-np-tile/
        # check if the region image retrieved has the size we expect
        # (some image servers scale the image a small bit down, if they feel the request is too large)
        assert (w, h) == (
            region_image.shape[1],
            region_image.shape[0],
        ), f"({w=} {h=}) != {(region_image.shape[1], region_image.shape[0])}"

        for setting in settings["find_rims"]:
            print(setting)
            print("💡", setting["rim_to_find"])
            print(setting.get("pipeline_detail", ""))

            # do we process the image in horizontal / vertical direction differently?
            # e.g. if we want to apply erosion / dilation in specific direction to make
            # lines more visible
            if setting["pipeline_detail_split_horizontal_vertical"]:
                # FIXME: 2 settings, different for left and right | bottom and top
                # (most 'flexible' 4 different settings)
                if rim_side & RimSide.LEFT or rim_side & RimSide.RIGHT:
                    region_image_processed = apply_pipeline(
                        region_image,
                        setting["pipeline_detail_vertical"],
                        inspect=do_interactive_plot,
                    )
                elif rim_side & RimSide.BOTTOM or rim_side & RimSide.TOP:
                    region_image_processed = apply_pipeline(
                        region_image,
                        setting["pipeline_detail_horizontal"],
                        inspect=do_interactive_plot,
                    )
            else:
                region_image_processed = apply_pipeline(
                    region_image,
                    setting["pipeline_detail"],
                    inspect=do_interactive_plot,
                )

            # in how many parts we subdivide the region image
            # for each part, we get an estimate where the rim should be
            part_size = settings["part_size"]
            # FIXME:
            # - [ ] check: is this a setting of global part_size or per rim_find setting?
            #       `settings`` seem to be global
            part_ct = get_part_count(region_image_processed, rim_side, part_size)

            for i in range(part_ct):
                print(f"\n- processing sub-region #{i=}")
                sub_region, tl_of_slice = extract_subregion(
                    region_image_processed, part_size, i, rim_side
                )
                # the vector to position points in global reference frame of image
                # instead of in clipped position frame of sub region
                transformation_vector = get_translate_vector(rect, tl_of_slice)

                sub_center = center_coordinates(sub_region)

                if do_interactive_plot:
                    fig = plt.figure(layout="constrained")
                    axis_dict = fig.subplot_mosaic(
                        mosaic="BBBCC;BBBCC",
                    )
                    axB = axis_dict["B"]
                    axC = axis_dict["C"]
                    axB.imshow(sub_region, cmap=mpl.colormaps["gray"])
                else:
                    axC = None

                # within this sub-region apply the histogram thresholding
                # this leads to intervals that are more black, each interval
                # having a certain width, center, etc.

                # - [ ] FIXME: update search_rim to do less things (e.g. no plotting)
                # - [ ] functionality is a bit opaque here:
                #       threshold > | >= above / < | <= below, threshold could all come from settings
                if rim_side & RimSide.LEFT or rim_side & RimSide.RIGHT:
                    black_pixels__intervals, _ = search_rim(
                        sub_region, 0, 0, setting, axC
                    )
                    white_pixels__intervals, _ = search_rim(
                        sub_region, 255, 0, {"rim_fraction_filled": 0.999}, axC
                    )
                    # x=0 or y=0
                    # this depends on which axis we are working on
                    # sub_ax = 0  # For the cauchy mf (close to center)
                    # tl_of_slice = [0, start_part_index]
                elif rim_side & RimSide.BOTTOM or rim_side & RimSide.TOP:
                    # sub_ax = 1
                    black_pixels__intervals, _ = search_rim(
                        sub_region, 0, 1, setting, axC
                    )
                    white_pixels__intervals, _ = search_rim(
                        sub_region, 255, 1, {"rim_fraction_filled": 0.999}, axC
                    )

                print(f"{black_pixels__intervals=}")
                print(f"{white_pixels__intervals=}")

                ## Find 2 lines, 1 fat line and 1 skinny line that are $n$ pixels apart
                # - [ ] we find exactly 2 points with this method, not 'multiple' -> rename
                # - [ ] separate the 2 points found into outer and inner frame
                if setting["rim_to_find"] == "outer-inner":
                    # find the points by thresholding the peaks
                    # these points have coordinates that are locally
                    # (against top-left of subregion)
                    rim_points_local = determine_multiple_rim_points(
                        black_pixels__intervals, rim_side, sub_center, setting
                    )
                    if rim_points_local is not None:
                        print("FOUND MULTIPLE RIMS - POINTS", rim_points_local)
                        global_pts = [
                            vec.add(pt, transformation_vector)
                            for pt in rim_points_local
                        ]
                        # all_found_points_multiple.extend(global_pts)
                        if do_interactive_plot:
                            for pt in rim_points_local:
                                axB.plot(*pt, "r^")

                        if rim_side & RimSide.LEFT or rim_side & RimSide.BOTTOM:
                            # peak_order = [large, small]
                            outer, inner = global_pts

                        elif rim_side & RimSide.RIGHT or rim_side & RimSide.TOP:
                            # peak_order = [small, large]
                            inner, outer = global_pts

                        points_found.append(
                            RimPoint(
                                *outer,
                                setting_type=setting["rim_to_find"],
                                frame_type=FrameType.OUTER,
                                rim_side=rim_side,
                            )
                        )
                        points_found.append(
                            RimPoint(
                                *inner,
                                setting_type=setting["rim_to_find"],
                                frame_type=FrameType.INNER,
                                rim_side=rim_side,
                            )
                        )

                elif setting["rim_to_find"] == "outer":
                    pt = determine_one_rim_point(
                        black_pixels__intervals, rim_side, sub_center, setting
                    )
                    if pt is not None:
                        if do_interactive_plot:
                            # plot with local coordinates
                            axB.plot(*pt, "cX")
                        # produce point with global coordinates and add to result
                        pt = vec.add(pt, transformation_vector)
                        points_found.append(
                            RimPoint(
                                *pt,
                                setting_type=setting["rim_to_find"],
                                frame_type=FrameType.OUTER,
                                rim_side=rim_side,
                            )
                        )

                if do_interactive_plot:
                    plt.show()
                    plt.close()

    # produce the output (if wanted)
    if settings["produce_output"]:
        uuid = info["@id"].split("/")[-1]
        iiif_full_url = f"{url}/full/full/0/default.jpg"
        iiif_end_point = info["@id"]
        uuid = info["@id"].split("/")[-1]

        rimside_labels = {
            RimSide.LEFT: "left",
            RimSide.RIGHT: "right",
            RimSide.TOP: "top",
            RimSide.BOTTOM: "bottom",
        }
        frame_type_labels = {
            FrameType.INNER: "inner",
            FrameType.OUTER: "outer",
        }
        # organise the points found in the samples dictionary
        # -- with nested dicts per rim (outer|inner), then per side (left|right|top|bottom)
        samples = {}
        for setting in settings["find_rims"]:
            # which type of rim finding has been applied (outer or outer-inner)
            rim_to_find = setting["rim_to_find"]
            samples[rim_to_find] = {}
            points_to_consider = list(
                filter(lambda pt: pt.setting_type == rim_to_find, points_found)
            )
            if rim_to_find == "outer":
                frame_types = [FrameType.OUTER]
            elif rim_to_find == "outer-inner":
                frame_types = [FrameType.OUTER, FrameType.INNER]

            for frame_type in frame_types:
                samples[rim_to_find][frame_type_labels[frame_type]] = {}
                for side in [RimSide.LEFT, RimSide.RIGHT, RimSide.TOP, RimSide.BOTTOM]:
                    to_iter = filter(
                        lambda x: x.rim_side == side and x.frame_type == frame_type,
                        points_to_consider,
                    )
                    # if the side is the right / top, we reverse the points
                    # (to store them in counter? clockwise order from center)
                    if side in [RimSide.RIGHT, RimSide.TOP]:
                        to_iter = reversed(list(to_iter))
                    rim_pts = [[round(pt.x, 1), round(pt.y, 1)] for pt in to_iter]
                    samples[rim_to_find][frame_type_labels[frame_type]][
                        rimside_labels[side]
                    ] = rim_pts
        # write output as json, moving aside a possible existing file (and keeping a backup)
        j = {
            "sheet_id": sheet_id,
            "uuid": uuid,
            "iiif_end_point": iiif_end_point,
            "iiif_full_url": iiif_full_url,
            "image_size": img_size[:],
            "approximate_sides": approximate_sides_large[:],
            "samples": samples,  # FIXME: organise as dict with 'l':[], 'r':[], 'b':[], 't':[] ?
        }
        output_file_name = f"point_sample_{sheet_id:04d}.json"
        file_name = os.path.join(settings["output_folder"], output_file_name)
        with OpenWithMove(file_name, "w") as fh_samples:
            json.dump(j, fp=fh_samples)
