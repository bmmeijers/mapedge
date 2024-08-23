# make_svg_image.py
import requests
import os


# def calc(width, height, tw, th, s, n=0, m=0):
#### from iiif image api spec
#     # Calculate region parameters /xr,yr,wr,hr/
#     xr = n * tw * s
#     yr = m * th * s
#     wr = tw * s
#     if xr + wr > width:
#         wr = width - xr
#     hr = th * s
#     if yr + hr > height:
#         hr = height - yr
#     # Calculate size parameters /ws,hs/
#     ws = tw
#     if xr + tw * s > width:
#         ws = (width - xr + s - 1) // s  # +s-1 in numerator to round up
#     hs = th
#     if yr + th * s > height:
#         hs = (height - yr + s - 1) // s

#     print(s, xr, yr, wr, hr, ws, hs)


def get_json(url):
    r = requests.get(url)
    return r.json()


url = "https://dlc.services/iiif-img/7/4/8e8a6139-2700-4d8b-8ed0-815f60c1ab36/"  # including trailing slash
url = "https://dlc.services/iiif-img/7/32/068764d0-fe90-4aed-b8d2-e63b0be4e213/"
url = "https://dlc.services/iiif-img/7/32/bcc82e21-e28b-4a04-bceb-f9b579f7f447/"
url = "https://dlc.services/iiif-img/7/32/1d4b4c01-6938-49e6-bc3b-aa9cd5a28cf1/"


J = get_json(os.path.join(url, "info.json"))
size = [J["width"], J["height"]]
# print(j)

if "tiles" in J:
    # print(J["tiles"])
    # FIXME: if there is multiple tile sizes, pick largest?
    for tile_setting in J["tiles"]:
        # print(tile_setting)
        twidth = tile_setting["width"]
        if "height" in tile_setting:
            theight = tile_setting["height"]
        else:
            theight = twidth
        tile_size = [twidth, theight]
        # print(tile_setting["scaleFactors"])
        assert 1 in tile_setting["scaleFactors"]
        # for s in tile_setting["scaleFactors"]:
        #     calc(J["width"], J["height"], twidth, theight, s)

# import sys

# sys.exit()


# tile_size = [256, 256]  ##512, 512]  #

fitting_remainder = []
for i in range(2):
    fitting_remainder.append(divmod(size[i], tile_size[i]))

for i in range(2):
    times, remain = fitting_remainder[i]

# we add 1 to the number of times the size fits along the dimension
# if there is a remainder
# moreover, we set up the lastsize in each dimension
# so we can use it for the last row / column
timesx, remainx = fitting_remainder[0]
timesy, remainy = fitting_remainder[1]

smaller_lastx = False
smaller_lasty = False
if remainx != 0:
    timesx += 1
    lastsizex = remainx
else:
    lastsizex = tile_size[0]

if remainy != 0:
    timesy += 1
    lastsizey = remainy
else:
    lastsizey = tile_size[1]


svg = []
svg.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{size[0]}" height="{size[1]}">'
)


for x in range(timesx):
    is_last_x = x == timesx - 1
    for y in range(timesy):
        is_last_y = y == timesy - 1

        image_x = x * tile_size[0]
        image_y = y * tile_size[1]
        image_width = tile_size[0]
        image_height = tile_size[1]

        if is_last_x:
            image_width = lastsizex
        if is_last_y:
            image_height = lastsizey

        size_str = ",".join(
            map(str, (x * tile_size[0], y * tile_size[1], tile_size[0], tile_size[1]))
        )
        xlink = url + size_str + "/full/0/default.jpg"
        image_tag = f'<image x="{image_x}" y="{image_y}" width="{image_width}" height="{image_height}" xlink:href="{xlink}" />'
        svg.append(image_tag)
        #
        # rect_tag = f'<rect x="{image_x}" y="{image_y}" width="{image_width}" height="{image_height}" style="stroke:#ac0d0d; stroke-width:0.5; fill: none;" />'
        # print(rect_tag)
svg.append("</svg>")

with open("/home/martijn/Documents/tmp/svg.svg", "w") as fh:
    fh.write("\n".join(svg))
