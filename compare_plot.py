import matplotlib.pyplot as plt
from numpy import average, arange, cumsum, concatenate, max, min
from numpy import std as standard_dev


def errorPlot(array_list, x_axis=None, smooth=0, std=True):
    avg = average(array_list, axis=0)
    if std:
        covar = standard_dev(array_list, axis=0)
        upper = avg + covar
        lower = avg - covar
    else:
        upper = max(array_list, axis=0)  # avg + covar
        lower = min(array_list, axis=0)  # avg - covar

    if smooth:
        avg = runningAverage(avg, smooth)
        upper = runningAverage(upper, smooth)
        lower = runningAverage(lower, smooth)

    if x_axis is None:
        base_line = plt.plot(avg)
        color = base_line[0].get_color()
        plt.fill_between(arange(0, len(avg)), upper,
                         lower, color=color, alpha=.2)
    else:
        base_line = plt.plot(x_axis, avg)
        color = base_line[0].get_color()
        plt.fill_between(x_axis, upper,
                         lower, color=color, alpha=.2)
    return base_line


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
    import numpy as np
    ar = np.sin(np.arange(0, 3, .1))
    plt.plot(ar)
    plt.plot(runningAverage(ar, 3))
    plt.show()
