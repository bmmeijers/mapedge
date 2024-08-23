import math


def linear_mf(x, a, b) -> float:
    res = 0.0
    if x < a:
        res = 0.0
    elif x <= b:
        res = (x - a) / (b - a)
    else:
        res = 1.0
    return res


def triangular_mf(x, a, b, c) -> float:
    res = 0.0
    if x < a:
        res = 0.0
    elif x <= b:
        res = (x - a) / (b - a)
    elif x <= c:
        res = 1.0 - (x - b) / (c - b)
    else:
        res = 0.0
    return res


def trapezoidal_mf(x, a, b, c, d) -> float:
    res = 0.0
    if x < a:
        res = 0.0
    elif x <= b:
        res = (x - a) / (b - a)
    elif x <= c:
        res = 1.0
    elif x <= d:
        res = 1.0 - (x - c) / (d - c)
    else:
        res = 0.0
    return res


def cauchy_mf(x, a, b):
    return 1 / (1 + ((x - a) / b) ** 2)


def gaussian_mf(x, mu, sigma):
    return math.exp(-((x - mu) ** 2) / (2 * sigma**2))


def gaussian_trapezoidal_mf(x, a, b, c, d):
    return max(gaussian_mf(x, a, b), trapezoidal_mf(x, c, d, a, b))


def sigmoid_mf(x, a, c):
    return 1 / (1 + math.exp(-a * (x - c)))


def gbell_mf(x, a, b, c):
    return 1.0 / (1 + (abs((x - c) / a) ** (2 * b)))


def plot():
    from matplotlib import pyplot

    # trapezoidal -- shoulders at 95 and 110, neck at 100 and 105
    # x = range(90, 115)
    # y = [trapezoidal_mf(val, 95, 100, 105, 110) for val in x]

    x = range(0, 100)
    y = [cauchy_mf(val, 50, 20) for val in x]

    # x = range(0, 1112)
    # y = [triangular_mf(val, 0, 556, 1112) for val in x]

    # x = range(0, 1112)
    # y = [gaussian_mf(val, 556.0, 278.0) for val in x]

    # import numpy as np
    x = range(0, 100)

    # x = range(0, 100)
    # y = [linear_mf(val, 25.0, 75.0) for val in x]

    # for i in [0.1, 0.2, 0.4, 0.8, 1.0]:
    #     y = [sigmoid_mf(val, i, 50) for val in x]
    #     pyplot.plot(x,y)

    # x = [i * 0.01 for i in range(-1000, 1000)]
    # y = [gaussian_trapezoidal_mf(i, 0, 1, -2, 2) for i in x]
    # x = range(0,10)
    x = [i * 0.1 for i in range(0, 100)]
    y = [trapezoidal_mf(i, 2, 4, 6, 8) for i in x]
    pyplot.plot(x, y)
    pyplot.show()


def how_does_this():
    # [(352, 354), [0.0]]
    # 2 14.5 16 20 21.5
    # [(442, 444), [0.0]]
    # 2 14.5 16 20 21.5
    # [(460, 462), [0.0]]
    # 2 14.5 16 20 21.5
    # [(476, 477), [0.0]]
    # 1 14.5 16 20 21.5
    # [(544, 545), [0.0]]
    # 1 14.5 16 20 21.5
    # [(561, 576), [0.0]]
    # 15 14.5 16 20 21.5
    # [(592, 594), [0.0]]
    # 2 14.5 16 20 21.5
    # ranked alternatives
    # --------------------
    # [[0.0, (352, 354)],
    #  [0.0, (442, 444)],
    #  [0.0, (460, 462)],
    #  [0.0, (476, 477)],
    #  [0.0, (544, 545)],
    #  [0.0, (561, 576)],
    #  [0.0, (592, 594)]]

    res = trapezoidal_mf(15, 14.5, 16, 20, 21.5)
    print(res)


if __name__ == "__main__":
    plot()
