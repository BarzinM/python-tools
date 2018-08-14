from collections import deque
import numpy as np


class RunningStats(object):

    def __init__(self, tau=None, *args):
        if len(args):
            print("WARNING: Remove extra argumetns when initializing RunningStates.")
        self.tau = tau
        self._func = self._first

    def __call__(self, *args):
        return self._func(*args)

    def _first(self, value=None, axis=0):
        self.avg = np.mean(value, axis=0)
        self.var = np.var(value, axis=0)
        self.data_count = len(value)

        if self.tau is None:
            self._func = self._exact
        else:
            self._func = self._running

        return self.avg, self.var

    def _exact(self, value=None, axis=0):
        if value is not None:
            n = len(value)

            new_data_mean = np.mean(value, axis=0)

            temp = value - new_data_mean
            new_data_var = np.dot(temp, temp) / n

            new_data_mean_sq = np.square(new_data_mean)

            new_means = ((self.avg * self.data_count) +
                         (new_data_mean * n)) / (self.data_count + n)

            self.var = (((self.data_count * (self.var + np.square(self.avg))) +
                         (n * (new_data_var + new_data_mean_sq))) / (self.data_count + n) -
                        np.square(new_means))

            # occasionally goes negative, clip
            self.var = np.maximum(0.0, self.var)
            self.avg = new_means
        return self.avg, self.var

    def _running(self, value=None, axis=0):
        if value is not None:
            var = np.var(value - self.avg, axis=0)
            self.var = self.var * (1 - self.tau) + var * self.tau
            self.avg = self.avg * (1 - self.tau) + value * self.tau
        return self.avg, self.var

    def normalize(self, values):
        return (values - self.avg) / (np.sqrt(self.var) + 1e-6)


class RunningAverage(RunningStats):

    def _first(self, value=None, axis=0):
        if type(value) in [int, float]:
            self.avg = value
            self.data_count = 1
        else:
            self.avg = np.mean(value, axis=0)
            self.data_count = len(value)

        if self.tau is None:
            self._func = self._exact
        else:
            self._func = self._running

        return self.avg

    def _exact(self, value=None, axis=0):
        if value is not None:
            n = len(value)

            new_data_mean = np.mean(value, axis=0)

            new_means = ((self.avg * self.data_count) +
                         (new_data_mean * n)) / (self.data_count + n)

            self.avg = new_means
        return self.avg

    def _running(self, value=None, axis=0):
        if value is not None:
            self.avg = self.avg * (1 - self.tau) + value * self.tau
        return self.avg


class Normality(object):

    def __init__(self, tol, window):
        self.window = window
        self.tol = tol
        self.var = 0.
        self.avg = 0.
        self.avg_var = 0.
        self.dist = [0, 0, 0, 0]
        self.queue = deque(maxlen=window)

    def update(self, value):
        total = len(self.queue)
        if total < self.window:
            var = np.linalg.norm(value - self.avg)
            self.avg = (self.avg * total + value) / \
                (total + 1)
            self.var = (self.var * total + var) / \
                (total + 1)
            z_score = abs(value - self.avg) / self.var

            if z_score < 1.:
                self.dist[0] += 1
                self.queue.append((0, value, var))
            elif z_score < 2.:
                self.dist[1] += 1
                self.queue.append((1, value, var))
            elif z_score < 3.:
                self.dist[2] += 1
                self.queue.append((2, value, var))
            else:
                self.dist[3] += 1
                self.queue.append((3, value, var))

            return False

        d, old_value, old_var = self.queue.popleft()
        self.dist[d] -= 1

        var = np.linalg.norm(value - self.avg)
        self.avg += (value - old_value) / total
        self.var += (var - old_var) / total

        z_score = abs(value - self.avg) / self.var

        if z_score < 1.:
            self.dist[0] += 1
            self.queue.append((0, value, var))
        elif z_score < 2.:
            self.dist[1] += 1
            self.queue.append((1, value, var))
        elif z_score < 3.:
            self.dist[2] += 1
            self.queue.append((2, value, var))
        else:
            self.dist[3] += 1
            self.queue.append((3, value, var))

        if abs(self.dist[0] / total - .6827) < self.tol and \
                abs(sum(self.dist[:2]) / total - .9545) < self.tol and \
                abs(sum(self.dist[:3]) / total - .9973) < self.tol:
            return True
        return False


def cosineSimilarity(a, b):
    return np.dot(a, b) / (np.norm(a) * np.norm(b))


if __name__ == "__main__":
    from time import time
    m = RunningStats()
    for _ in range(2):
        print(m([1, 2]))
