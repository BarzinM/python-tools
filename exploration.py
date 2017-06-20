from numpy.random import randn
from numpy import ones


class OUExploration(object):
    def __init__(self, action_dim=1, sigma=.3, mu=0., theta=.15):
        self.action_dim = action_dim
        self.sigma = sigma
        self.mu = mu
        self.theta = theta
        self.reset()

    def noise(self):
        self.state += self.theta * (self.mu - self.state) + \
            self.sigma * randn(len(self.state))
        return self.state

    def reset(self):
        self.state = ones(self.action_dim) * self.mu


class LinearDecay(object):
    def __init__(self, action_dim=1, deviation=1.):
        self.action_dim = action_dim
        self.deviation = deviation
        self.reset()

    def noise(self, step=None):
        self.step = (step or self.step) + 1
        return self.deviation * randn(self.action_dim) / self.step

    def reset(self):
        self.step = 0
