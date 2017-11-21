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

    def __init__(self, height, width, obs_length, etype='empty'):
        self.height = height
        self.width = width

        if etype == 'random':
            self.area = np.random.randint(10, 60, size=(height, width))
            self.area[self.area == 10] = 1
            self.area[self.area > 10] = 0
        elif etype == 'empty':
            self.area = np.zeros((height, width))

        elif etype == 'obs':
            self.area = np.Ones(size=(height, width))

        self.obs = np.zeros(shape=(1 + 2 * obs_length, 1 + 2 * obs_length))
        self.obs_length = obs_length

        # self.env_plot = Figure()
        # self.env_plot.imshow(self.area, vmin=0, vmax=5)
        # self.agent_view = Figure()
        # self.agent = MultiDimObject(self.pose)

    def reset(self):
        self.pose = randomPose(self.height, self.width)# * 0 + 10

        self.target = randomPose(self.height, self.width)# * 0 + [25, 25]
        # space = 5
        # while self._check_around(self.target, space):
        #     self.target = randomPose(self.height, self.width)

        self.area[self.target[0], self.target[1]] = 4

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

    def getState(self):
        diff = self.target - self.pose
        distance = np.linalg.norm(diff)
        angle = np.arctan2(diff[0], diff[1])
        return np.array([*self.pose, *self.target])

    def observe(self):
        self.obs[:] = 1
        y, x = round(self.pose[0]), round(self.pose[1])
        top = y - self.obs_length
        top = max(0, y - self.obs_length)
        left = max(0, x - self.obs_length)
        view = self.area[top: y + self.obs_length + 1,
                         left: x + self.obs_length + 1]
        top = max(0, self.obs_length - y)
        left = max(0, self.obs_length - x)
        self.obs[top:top + view.shape[0],
                 left:left + view.shape[1]] = view
        return self.obs

    def plot(self):
        self.env_plot.imshow(self.area, vmin=0, vmax=5)
        self.env_plot.plot(self.pose[1], self.pose[0], 'o')  # pose[1] -> x

    def plotObservation(self):
        self.agent_view.imshow(self.obs, vmin=0, vmax=4)

    def step(self, up, right):
        self.pose[:] = self.agent.force((up, right))
        # self.pose[0] = int(y)
        # self.pose[1] = int(x)

        self.terminal = self._crashed()

    def _crashed(self):
        y, x = round(self.pose[0]), round(self.pose[1])
        # or self.area[y, x] > 0
        return self.height <= y or y < 0 or self.width <= x or x < 0


class Discrete(Simple):

    def step(self, action):
        if action == 0:
            self.pose[0] += 1
        elif action == 1:
            self.pose[0] -= 1
        elif action == 2:
            self.pose[1] += 1
        elif action == 3:
            self.pose[1] -= 1
        else:
            raise ValeError("Undefined action %i" % action)
        self.terminal = self._crashed()


if __name__ == "__main__":
    plt.ion()
    env = Discrete(50, 70, 10, "empty")
    env.reset()
    env.plot()
    terminal = False
    while not terminal:
        env.step(np.random.randint(4))
        # obs = env.observe()
        print(env.getPose())
        terminal = env.terminal
        env.plot()
        # env.plotObservation()
        # sleep(.05)
    sleep(100)
    # print(env.getPose())
    # sleep(10)
