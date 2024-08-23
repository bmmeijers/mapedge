from .annotation_page import make_annotation_feature, make_annotation_item, make_page
import json


def sort_points(points):
    # Sort the points based on x-coordinate
    points.sort(key=lambda point: (point[0]))

    # we now have two on the left of the rectangle
    leftx = [points[0], points[1]]
    rightx = [points[2], points[3]]

    # sort both sides on y increasing
    leftx.sort(key=lambda point: (point[1]))
    rightx.sort(key=lambda point: (point[1]))
    # print(leftx, rightx)

    lower_left = leftx[0]
    upper_left = leftx[1]
    lower_right = rightx[0]
    upper_right = rightx[1]

    # The point with the smallest y and x coordinate is the lower left
    # lower_left = points[0]

    # The point with the largest x and y coordinate is the upper right
    # upper_right = points[3]

    # The remaining two points are the lower right and upper left,
    # but we need to determine which is which
    # point1, point2 = points[1], points[2]

    # The point with the larger x coordinate is the lower right
    # lower_right = point1 if point1[0] > point2[0] else point2

    # The other point is the upper left
    # upper_left = point2 if point1[0] > point2[0] else point1

    # Return the points in counter-clockwise order starting from lower left
    return [lower_left, lower_right, upper_right, upper_left]


def min_max_geo_polygon(feature):
    """gets first ring from multipoly,
    without first and last point duplicated"""
    for ring in feature["geometry"]["coordinates"]:
        # for ring in geom:
        return sort_points(ring[:-1])


def annotate(
    index_map_geojson,
    corner_json,  # one or more files with sample point json + lines fitted
    # sheet_geojson_filename,
    sheet_id_column,
    annotation_page_file,
):

    # open the geojson with the sheet index
    # with open(sheet_geojson_filename) as fh:  # "tmk_index_map.geojson") as fh:
    #     index_map_geojson = json.load(fh)

    # make a lookup table, so that we can get for each sheet the world extent
    # by using the sheet index
    # sheet_id -> feature (includes geometry in wgs'84)
    lut = {}
    for feature in index_map_geojson["features"]:
        oid = feature["properties"][sheet_id_column]  # "sheet_id"
        lut[oid] = feature

    # import pprint

    # pprint.pprint(lut)
    # now process the sheets, for which we
    # have pixel coordinates detected (and a mask)
    # with open("/tmp/corner_find.json") as fh:

    items = []
    for record in corner_json:

        # FIXME: this should not be necessary
        # we should base the join on 'uuid'
        # (and get it from inside of the file, and not a filename)

        oid = record[sheet_id_column]
        #  + 1 # FIXME: be able to flexible with the name of the attribute in corner.json and sheet-index.geojson

        # the world feature
        try:
            f = lut[oid]
        except KeyError:
            print(f"WARN: {oid} not found in sheet index (world side)")
            continue
        # the world coordinates
        world = min_max_geo_polygon(f)
        # bl, tl, tr, br
        bl, br, tr, tl = world[0], world[1], world[2], world[3]
        world_corner_points = [bl, br, tr, tl]

        # the pixel coordinates
        uuid = record["uuid"] + "_" + str(oid)
        svg_mask = record["corners"]["svg_mask"]
        pixel_corner_points = record["corners"]["pixel_corner_points"]
        # pixel_corner_points = record["corners"][
        #     "refined_pixel_corner_points"
        # ]  # Temp hack for refining corners
        iiif_end_point = iiif_full_url = record["iiif_end_point"]
        iiif_full_url += "/full/full/0/default.jpg"

        # # tie them together
        features = [
            make_annotation_feature(*pixel, *world)
            for pixel, world in zip(pixel_corner_points, world_corner_points)
        ]
        if len(features) == 4:

            item = make_annotation_item(
                uuid, iiif_end_point, iiif_full_url, svg_mask, features
            )

            items.append(item)
        else:
            print(f"WARNING: {oid} *not* enough GCPs")
    # break
    # else:
    #     print(f"skipped {sheet_id}")
    # from the annotation items, make an annotation page
    j = make_page(items)

    # annotation_page_filename = "tmk_annotation_page.json"
    # with open(annotation_page_filename, "w") as fh:
    print(json.dumps(j, indent=0), file=annotation_page_file)

    print(f"written {annotation_page_file.name}, done.")


# if __name__ == "__main__":
#     annotate(
#         "/home/martijn/Documents/work/2023-08_synthesis/tmk/tmk_index_map.geojson",
#         "sheet_id",
#         annotation_page_filename="/tmp/annotation_page.json",
#     )
