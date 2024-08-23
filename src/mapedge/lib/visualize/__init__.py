def make_simple_svg_mask(img_size, points):
    svg_points = [f"{x},{y}" for (x, y) in points]

    wh = 'width="' + str(img_size[0]) + '" height="' + str(img_size[1]) + '"'
    svg_mask = f"<svg {wh}>"
    svg_mask += f'<polygon points="{" ".join(svg_points)}" />'
    svg_mask += "</svg>"
    return svg_mask
