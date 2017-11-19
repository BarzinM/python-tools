from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from monitor import Figure


def randomPose(height, width):
    return np.array([np.random.randint(height), np.random.randint(width)])


def maze(width=81, height=51, complexity=.75, density=.75, wall_size=2):
    # Only odd shapes
    shape = ((height // wall_size) * wall_size + 1,
             (width // wall_size) * wall_size + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density = int(
        density * ((shape[0] // wall_size) * (shape[1] // wall_size)))
    # Build actual maze
    Z = np.ones(shape, dtype=bool)
    # Fill borders
    Z[0, :] = Z[-1, :] = 0
    Z[:, 0] = Z[:, -1] = 0
    # Make aisles
    complexity = 1
    # density = 0
    for i in range(density):
        x, y = rand(0, shape[1] // wall_size) * \
            wall_size, rand(0, shape[0] // wall_size) * wall_size
        for j in range(complexity):
            neighbours = []
            if x > 1:
                neighbours.append((y, x - wall_size))
            if x < shape[1] - wall_size:
                neighbours.append((y, x + wall_size))
            if y > 1:
                neighbours.append((y - wall_size, x))
            if y < shape[0] - wall_size:
                neighbours.append((y + wall_size, x))
            if len(neighbours):
                y_, x_ = neighbours[rand(0, len(neighbours) - 1)]
                if Z[y_, x_]:
                    Z[y, x] = 0
                    Z[y_, x:x_ + 1] = 0
                    Z[y_, x_:x + 1] = 0
                    Z[y:y_ + 1, x_] = 0
                    Z[y_:y + 1, x_] = 0
            x, y = x_, y_
    return Z


class Object(object):

    def __init__(self, pose=0., bounds=None, bounce=None):
        if bounce is not None:
            assert 0. <= bounce <= 1.
            self.bounce_coef = -bounce
        self.velocity = 0.
        self.pose = pose
        self.max_velocity = 1.
        if bounds is None and bounce is None:
            self.force = self._force
        elif bounce is not None:
            self.max = max(bounds)
            self.min = min(bounds)
            self.force = self._force_bouncy
        else:
            self.max = max(bounds)
            self.min = min(bounds)
            self.force = self._force_crash

    def _force(self, value):
        new_velocity = min(self.max_velocity,
                           max(-self.max_velocity, self.velocity + value))
        self.velocity = new_velocity

        velocity = (self.velocity + new_velocity) * .5

        self.pose += velocity
        return self.pose

    def _force_bouncy(self, value):
        new_velocity = min(self.max_velocity,
                           max(-self.max_velocity, self.velocity + value))
        self.velocity = new_velocity

        velocity = (self.velocity + new_velocity) * .5

        self.pose += velocity
        if self.pose < self.min:
            self.pose = self.min
            self.velocity *= self.bounce_coef
        elif self.pose > self.max:
            self.pose = self.max
            self.velocity *= self.bounce_coef
        return self.pose

    def _force_crash(self, value):
        new_velocity = min(self.max_velocity,
                           max(-self.max_velocity, self.velocity + value))
        self.velocity = new_velocity

        velocity = (self.velocity + new_velocity) * .5

        self.pose += velocity
        if self.pose < self.min:
            self.pose = self.min
            self.velocity *= self.bounce_coef
        elif self.pose > self.max:
            self.pose = self.max
            self.velocity *= self.bounce_coef
        return self.pose

    def bounce(self):
        self.velocity *= self.bounce_coef


class MultiDimObject(Object):

    def __init__(self, pose=(0., 0.), bounds=None, bounce=None):
        if bounds is None:
            self.object = [Object(p) for p in pose]
        else:
            self.object = [Object(p, b, bounce) for p, b in zip(pose, bounds)]

    def force(self, force):
        return [o.force(f) for o, f in zip(self.object, force)]

    def bounce(self, bounce):
        [o.bounce() for o in self.object]


class Simple(object):

    def __init__(self, height, width, obs_length, etype='random'):
        self.height = height
        self.width = width

        if etype == 'random':
            self.area = np.random.randint(10, 60, size=(height, width))
            self.area[self.area == 10] = 1
            self.area[self.area > 10] = 0
        elif etype == 'obs':
            self.area = np.Ones(size=(height, width))

        self.obs = np.zeros(shape=(1 + 2 * obs_length, 1 + 2 * obs_length))
        self.obs_length = obs_length

        self.target = randomPose(height, width)
        space = 5
        while self._check_around(self.target, space):
            self.target = randomPose(height, width)

        self.area[self.target[0], self.target[1]] = 4
        self.env_plot = Figure()
        self.env_plot.imshow(self.area)
        self.agent_view = Figure()

        self.pose = randomPose(height, width)
        self.agent = MultiDimObject(self.pose)
        self.terminal = False

    def _check_around(self, pose, space):
        return np.any(self.area[pose[0] - space:
                                pose[0] + space + 1,
                                pose[1] - space:
                                pose[1] + space + 1])

    def getPose(self):
        diff = self.target - self.pose
        distance = np.linalg.norm(diff)
        angle = np.arctan2(diff[0], diff[1])
        return distance, np.sin(angle), np.cos(angle)

    def observe(self):
        self.obs[:] = 1
        top = self.pose[0] - self.obs_length
        top = max(0, self.pose[0] - self.obs_length)
        left = max(0, self.pose[1] - self.obs_length)
        view = self.area[top: self.pose[0] + self.obs_length + 1,
                         left: self.pose[1] + self.obs_length + 1]
        top = max(0, self.obs_length - self.pose[0])
        left = max(0, self.obs_length - self.pose[1])
        self.obs[top:top + view.shape[0],
                 left:left + view.shape[1]] = view
        return self.obs

    def plot(self):
        self.env_plot.imshow(self.area)
        self.env_plot.plot(self.pose[1], self.pose[0], 'o')  # pose[1] -> x

    def plotObservation(self):
        self.agent_view.imshow(self.obs)

    def act(self, up, right):
        y, x = self.agent.force((up, right))
        self.pose[0] = int(y)
        self.pose[1] = int(x)

        self.terminal = self._crashed()

    def _crashed(self):
        t = (self.height <= self.pose[0] < 0) or (
            self.width <= self.pose[1] < 0)
        if t:
            return True
        else:
            return self.area[self.pose[0], self.pose[1]] > 0


if __name__ == "__main__":
    plt.ion()
    env = Simple(50, 70, 10)
    env.plot()
    terminal = False
    while not terminal:
        up, right = np.random.randn(2)
        env.act(up, right)
        obs = env.observe()
        terminal = env.terminal
        env.plot()
        env.plotObservation()
        # sleep(.05)
    # sleep(10)
    # print(env.getPose())
    # sleep(10)
