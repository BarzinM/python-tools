import matplotlib.pyplot as plt
from numpy import average, arange, cumsum, concatenate
from numpy import std as _std
import numpy as np


def errorPlot(array_list, axis=0, std=True, label='', fill=False):
    avg = average(array_list, axis=axis)
    if std:
        covar = _std(array_list, axis=axis)
        upper = avg + covar
        lower = avg - covar
    else:
        upper = np.max(array_list, axis=axis)
        lower = np.min(array_list, axis=axis)

    base_line = plt.plot(avg, linewidth=3, label=label)
    color = base_line[0].get_color()
    plt.fill_between(arange(0, len(avg)), upper, lower, color=color, alpha=.05)
    plt.plot(upper, color=color, alpha=.5)
    plt.plot(lower, color=color, alpha=.5)


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
