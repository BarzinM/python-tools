from collections import deque
import numpy as np


class RunningAverage(object):

    def __init__(self, tau, initial_value=0.):
        self.tau = tau
        self.avg = initial_value

    def __call__(self, value=None):
        if value is not None:
            self.avg = self.avg * (1 - self.tau) + value * self.tau
        return self.avg


class RunningStats(object):

    def __init__(self, tau, initial_values):
        pass


class RunningStd(object):

    def __init__(self, tau, initial_avg=0., initial_var=0.):
        self.tau = tau
        self.avg = initial_avg
        self.var = initial_var

    def __call__(self, value=None):
        if value is not None:
            var = np.linalg.norm(value - self.avg)
            self.var = self.var * (1 - self.tau) + var * self.tau
            self.avg = self.avg * (1 - self.tau) + value * self.tau
        return self.avg, self.var


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
    m = RunningAverage(.01)
    print(m(1.))
    print(m())
    m.value = 1.
    print(m.value)
