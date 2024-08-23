import mapedge.lib.trace.vectorops as vec


class CenteredRectangle:
    def __init__(self, center, half_size):
        self.center = center
        self.half_size = half_size

    # def corners(self):
    #     bl = vec.sub(self.center, self.half_size)
    #     tr = vec.add(self.center, self.half_size)
    #     print(bl)
    #     print(tr)

    # def as_wkt2d(self):
    #     return f"{self.center} {self.half_size}"

    def tl(self):  # as y-axis is top down, tl uses vec.sub
        return vec.sub(self.center, self.half_size)

    def iiif_region(self):
        # x, y = self.tl()
        # w, h = vec.mul(self.half_size, 2)
        #
        x, y, w, h = self.svg_region()
        if y < 0:
            y = 0
        if x < 0:
            x = 0
        iiif = f"{x},{y},{w},{h}"
        return iiif

    def svg_region(self):
        x, y = self.tl()
        w, h = vec.mul(self.half_size, 2)
        return map(int, [x, y, w, h])

    # iiif region=125,15,200,200  dx,dy,w,h
    # dx against top -->
    #
    # # dy against top (image coordinate tl = (0,0))
    # |
    # v
    #
    # width
    # height


# center = vec.make_vector((10, 10), (0,0))
# print(center)
# half_size = (5, 2.5)

# r = CenteredRectangle( (10, 10), (5, 2.5))
# r.corners()
