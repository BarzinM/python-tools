import matplotlib.pyplot as plt
import numpy as np


def errorPlot(array_list, x_axis=None, smooth=0, std=True, label=None):
    avg = np.average(array_list, axis=0)
    if std:
        covar = np.std(array_list, axis=0)
        upper = avg + covar
        lower = avg - covar
    else:
        upper = np.max(array_list, axis=0)
        lower = np.min(array_list, axis=0)

    if smooth:
        avg = exponential_moveing_average(avg, smooth)
        upper = exponential_moveing_average(upper, smooth)
        lower = exponential_moveing_average(lower, smooth)

    if x_axis is None:
        base_line = plt.plot(avg, label=label)
        color = base_line[0].get_color()
        plt.fill_between(np.arange(0, len(avg)), upper,
                         lower, color=color, alpha=.2)
    else:
        base_line = plt.plot(x_axis, avg, label=label)
        color = base_line[0].get_color()
        plt.fill_between(x_axis, upper,
                         lower, color=color, alpha=.2)
    return base_line


def runningAverage(data, look_around):
    length = len(data)
    avg = data * 0.
    for i in range(length):
        avg[i] = np.mean(
            data[max(0, i - look_around):min(length, i + look_around + 1)])
    return avg


def exponential_moveing_average(data, alpha=.9):
    m = data * 0.
    m[0] = data[0]
    for i in range(1, len(data)):
        m[i] = m[i - 1] * alpha + (1 - alpha) * data[i]
    return m


if __name__ == "__main__":
    ar = np.sin(np.arange(0, 3, .1)) + np.sin(np.arange(0, 300, 10))
    plt.plot(ar)
    plt.plot(runningAverage(ar, 3))
    plt.show()
