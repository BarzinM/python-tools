import matplotlib.pyplot as plt
from numpy import average, std, arange, cumsum, concatenate


def errorPlot(array_list):
    avg = average(array_list, axis=0)
    covar = std(array_list, axis=0)

    base_line = plt.plot(avg)
    color = base_line[0].get_color()
    upper = avg + covar
    lower = avg - covar
    plt.fill_between(arange(0, len(avg)), upper, lower, color=color, alpha=.2)

import numpy as np


def runningAverage(data, look_around):
    csum = cumsum(data)
    start = []
    end = []
    for i in range(look_around):
        start.append(csum[i * 2] / (2 * i + 1))
        end.append((csum[-1] - csum[-2 * i - 2]) / (2 * i + 1))
    start.append(csum[look_around * 2] / (2 * look_around + 1))
    window = 1 + look_around * 2
    ret = (csum[window:] - csum[:-window]) / window
    return concatenate([start, ret, end[::-1]])

if __name__ == "__main__":
    ar = np.sin(np.arange(0, 3, .1))
    plt.plot(ar)
    plt.plot(runningAverage(ar, 3))
    plt.show()
