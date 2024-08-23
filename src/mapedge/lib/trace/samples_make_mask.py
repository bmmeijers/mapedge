import json
from itertools import chain


def get_x(pt):
    return pt[0]


def get_y(pt):
    return pt[1]


def make_simple_svg_mask(img_size, points):
    svg_points = [f"{x},{y}" for (x, y) in points]

    wh = 'width="' + str(img_size[0]) + '" height="' + str(img_size[1]) + '"'
    svg_mask = f"<svg {wh}>"
    svg_mask += f'<polygon points="{" ".join(svg_points)}" />'
    svg_mask += "</svg>"
    return svg_mask


def make_svg(img_size, points, rects, iiif_full_url, extra_points=None):
    svg_points = [f"{x},{y}" for (x, y) in points]

    # make rects, where we sampled
    svg_rects = "".join(
        [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="green" fill-opacity="0.25" />'
            for (x, y, w, h) in rects
        ]
    )

    wh = 'width="' + str(img_size[0]) + '" height="' + str(img_size[1]) + '"'
    svg_mask = f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" {wh}>'
    svg_mask += "\n"
    svg_mask += f'<image x="0" y="0" {wh} xlink:href="{iiif_full_url}" />'
    svg_mask += "\n"
    svg_mask += svg_rects
    svg_mask += "\n"
    svg_mask += f'<polygon points="{" ".join(svg_points)}" style="stroke:#ff0000;fill:#ff0000;fill-opacity:0.333333" />'
    svg_mask += "\n"
    if extra_points:
        for pt in extra_points:
            svg_mask += f'<circle cx="{pt[0]}" cy="{pt[1]}" r="15" stroke="#0000ff"  stroke-width="1.25" fill="#0000ff" fill-opacity="0.333333" />'
            svg_mask += "\n"

    svg_mask += "</svg>"
    return svg_mask


if __name__ == "__main__":
    with open("sheet_sampled_rims.json") as fh:
        j = json.load(fh)
        print("w x h", j[0]["image_size"])
        print(j)
        left, right, top, bottom = j[0]["samples_per_rim"]
        print(left)
        print(list(sorted(left, key=get_y)))
        print(right)
        print(list(sorted(right, key=get_y)))
        print(top)

        print(list(sorted(top, key=get_x)))

        print(bottom)
        print(list(sorted(bottom, key=get_x)))

        pts = list(
            chain(
                sorted(left, key=get_y),
                sorted(bottom, key=get_x),
                sorted(right, key=get_y, reverse=True),
                sorted(top, key=get_x, reverse=True),
            )
        )
        mask = make_svg(
            j[0]["image_size"],
            pts,
            j[0]["sampled_rects"],
            j[0]["iiif_end_point"] + "/full/full/0/default.jpg",
        )
        with open("/tmp/mask.svg", "w") as fh:
            print(mask, file=fh)
