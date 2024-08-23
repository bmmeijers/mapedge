from mapedge.lib.fit.fitting import Line2
import numpy as np


class SVGVisualizer:
    def __init__(self, size):
        self.size = size
        # self.colors = [
        #     "#1b9e77",
        #     "#d95f02",
        #     "#7570b3",
        #     "#e7298a",
        #     "#66a61e",
        #     "#e6ab02",
        #     "#a6761d",
        #     "#666666",
        # ]
        self.colors = [
            "#a6cee3",
            "#1f78b4",
            "#b2df8a",
            "#33a02c",
            "#fb9a99",
            "#e31a1c",
            "#fdbf6f",
            "#ff7f00",
        ]
        self.current_color = 0
        self.buffer = []

    def propose_color(self):
        self.current_color += 1
        self.current_color %= len(self.colors)
        return self.colors[self.current_color]

    def add_xlink_image(self, url):
        xlink = url
        image_tag = f'<image x="0" y="0" width="{self.size[0]}" height="{self.size[1]}" xlink:href="{xlink}" />'
        self.buffer.append(image_tag)

    def add_crop_mark(self, pt):
        self.buffer.append(
            f'<circle cx="{pt[0]}" cy="{pt[1]}" r="0.5" stroke="#000000" stroke-width="1" />'
        )

        size = 23
        self.buffer.append(
            f'<circle cx="{pt[0]}" cy="{pt[1]}" r="{size}" stroke="#000000" fill="none" stroke-width="1" />'
        )

        self.buffer.append(
            f'<line x1="{pt[0]}" y1="{pt[1] - size}" x2="{pt[0]}" y2="{pt[1] - size + 10}" stroke-width="1" stroke="#000000" />'
        )
        self.buffer.append(
            f'<line x1="{pt[0]}" y1="{pt[1] + size}" x2="{pt[0]}" y2="{pt[1] + size - 10}" stroke-width="1" stroke="#000000" />'
        )

        self.buffer.append(
            f'<line x1="{pt[0] - size}" y1="{pt[1]}" x2="{pt[0] - size + 10}" y2="{pt[1]}" stroke-width="1" stroke="#000000" />'
        )
        self.buffer.append(
            f'<line x1="{pt[0] + size - 10}" y1="{pt[1]}" x2="{pt[0] + size}" y2="{pt[1]}" stroke-width="1" stroke="#000000" />'
        )

        self.buffer.append(
            f'<line x1="{pt[0] -5}" y1="{pt[1] -5 }" x2="{pt[0] - size }" y2="{pt[1] - size}" stroke-width="1" stroke="#000000" />'
        )
        self.buffer.append(
            f'<line x1="{pt[0] + 5}" y1="{pt[1] -5 }" x2="{pt[0] + size }" y2="{pt[1] - size}" stroke-width="1" stroke="#000000" />'
        )
        self.buffer.append(
            f'<line x1="{pt[0] - 5}" y1="{pt[1] + 5 }" x2="{pt[0] - size }" y2="{pt[1] + size}" stroke-width="1" stroke="#000000" />'
        )

        self.buffer.append(
            f'<line x1="{pt[0] + 5}" y1="{pt[1] + 5 }" x2="{pt[0] + size }" y2="{pt[1] + size}" stroke-width="1" stroke="#000000" />'
        )

    def add_points(self, pts, color=None, radius=15.5, text_anchor=None):
        if color is None:
            color = self.propose_color()
        if not text_anchor:
            text_anchor = "start"
        self.buffer.append("<g>")
        for i, pt in enumerate(pts):
            if pt is not None:
                self.buffer.append(
                    f'<circle cx="{pt[0]}" cy="{pt[1]}" r="{radius}" stroke="{color}" stroke-width="0.75" fill="{self.colors[self.current_color]}" fill-opacity="0.333333" />'
                )
                self.buffer.append(
                    f'<text text-anchor="{text_anchor}" x="{pt[0]}" y="{pt[1]}" class="smallText" fill="#000" font-weight="bold">{i}\u00a0</text>'
                )
        self.buffer.append("</g>")
        # self.current_color += 1
        # self.current_color %= len(self.colors)

    def add_circle(self, center, radius=15.5, attributes=None):
        text_attribs = ""
        if attributes is not None:
            text_attribs = attributes
        self.buffer.append(
            f'<circle cx="{center[0]}" cy="{center[1]}" r="{radius}" {text_attribs} />'
        )

    def add_rect(self, topleft, size, attributes=None):
        text_attribs = ""
        if attributes is not None:
            text_attribs = attributes
        self.buffer.append(
            f'<rect x="{topleft[0]}" y="{topleft[1]}" width="{size[0]}" height="{size[1]}" {text_attribs} />'
        )

    def add_polygon(self, pts):
        svg_points = [f"{x},{y}" for (x, y) in pts]
        self.buffer.append(
            f'<polygon points="{" ".join(svg_points)}" style="stroke:#ff00ff;fill:#ff00ff;fill-opacity:0.1" />'
        )

    def show(self):
        l = [
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{self.size[0]}" height="{self.size[1]}">',
            "<style>.smallText { font: 12px sans-serif; }</style>",
        ]
        l.extend(self.buffer)
        l.append("</svg>")
        return "\n".join(l)

    def add_line(self, line, extra_attribs=""):
        [x1, y1], [x2, y2] = line
        svg_line = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{self.colors[self.current_color]}" {extra_attribs} />'
        self.buffer.append(svg_line)
        self.current_color += 1
        self.current_color %= len(self.colors)

    def add_line_equation(self, line):
        [x1, x2], [y1, y2] = self.intersect(line)
        svg_line = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{self.colors[self.current_color]}" stroke-width="0.25" />'
        self.buffer.append(svg_line)
        self.current_color += 1
        self.current_color %= len(self.colors)

    def intersect(self, line):
        a, b, c = [line.normal[0], line.normal[1], line.distance]
        width, height = self.size
        # Determine the intersection points with the rectangle boundaries
        if a == 0:
            # Horizontal line: y = -c/b
            y1, y2 = -c / b, -c / b
            x1, x2 = 0, width
        elif b == 0:
            # Vertical line: x = -c/a
            x1, x2 = -c / a, -c / a
            y1, y2 = 0, height
        else:

            sides = [
                # top / bottom
                Line2.from_normal_and_point(np.array([0, 1]), (0, 0)),
                Line2.from_normal_and_point(np.array([0, -1]), (0, height)),
                # left / right
                Line2.from_normal_and_point(np.array([1, 0]), (width, 0)),
                Line2.from_normal_and_point(np.array([-1, 0]), (0, 0)),
            ]

            # get the points that are 'closest' to the rectangle
            # note, not sure if this always works
            # aim is to not use the large numbers (points far away), when we have intersected
            # the rectangle with 'near' vertical or 'near' horizontal lines
            side_points = [line.intersect(side) for side in sides]
            tmp = [(np.sum(np.abs(pt)), pt) for pt in side_points]
            tmp.sort()
            p1, p2 = tmp[0][1], tmp[1][1]

            x1 = p1[0]
            x2 = p2[0]
            y1 = p1[1]
            y2 = p2[1]
            # # General case: compute intersection points
            # # -- note, this leads to large values / numbers
            # x1 = 0
            # y1 = -(c + a * x1) / b
            # x2 = width
            # y2 = -(c + a * x2) / b

        # return [p1[0], p2[0]], [p1[1], p2[1]]  #
        return [x1, x2], [y1, y2]
