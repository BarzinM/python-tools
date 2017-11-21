import numpy as np


class TMaze(object):

    def __init__(self):
        self.agent_size = 5
        self.target_size = 5
        self.neck_width = 5
        self.height = 5.
        self.scale = 30
        self.vis_edge = int(.2 * self.scale)
        self.vis_width = int(self.scale)
        self.vis_corridor = int(self.scale * .2)
        self.vis_height = int((self.height) * self.scale) + self.vis_edge
        self.accelarations = ((.01, 0.), (-.01, 0.), (0., .01), (0., -.01))
        self.env = np.zeros(
            (self.vis_height + self.vis_edge, 2*self.vis_edge+self.vis_width * 2), dtype=np.uint8)

    def get_possible_actions(self):
        return len(self.accelarations)

    def _block(self, env, x, y, value):
        x = int((x + 1) * self.scale)+self.vis_edge
        y = int((y) * self.scale) + self.vis_edge
        print(x)
        env[y - self.vis_edge:y + self.vis_edge, x -
            self.vis_edge:x + self.vis_edge] = value

    def vis(self):
        self.env[:] = 0
        self.env[-2 * self.vis_edge:, :] = 50
        self.env[:, self.vis_width:self.vis_width+self.vis_edge*2] = 50
        self._block(self.env, self.x, self.y, 255)

        return self.env[::-1, :]

    def observe(self):
        o = np.zeros(3, dtype=np.int8)
        if self.y < .2:
            o[1] = self.signal
        elif self.y > self.height - .2:
            o[0] = 1  # self.loc
            o[2] = -1  # self.loc

        return o

    def reset(self):
        self.step = 0
        self.vel_x = 0.
        self.vel_y = 0.
        self.signal = 1#1 if np.random.randint(2) else -1
        # self.loc = 2*np.random.randint(2)-1
        self.visible = True
        self.terminal = False
        self.x, self.y = 0, 0
        return self.observe()

    def reward(self):
        if self.x > .2:
            if self.signal < 0:
                return 1
            else:
                return -1
        elif self.x < -.2:
            if self.signal < 0:
                return -1
            else:
                return 1
        return 0

    def act(self, action):
        if self.terminal:
            raise ValueError
        acc_x, acc_y = self.accelarations[action]
        self.vel_x = min(max(self.vel_x + acc_x, -5.), 5.)
        self.vel_y = min(max(self.vel_y + acc_y, -5.), 5.)
        self.x += self.vel_x
        self.y += self.vel_y
        # if action == 0:
        #     self.x += 1
        # elif action == 1:
        #     self.x -= 1
        # elif action == 2:
        #     self.y += 1
        # elif action == 3:
        #     self.y -= 1
        # else:
        #     raise

        self._bound()

        return self.observe(), self.reward()  # , self.terminal

    def _bound(self):
        if self.y < self.height:
            # min(max(self.x,self.size-self.neck_width),self.size+self.neck_width)
            self.x = 0.
        else:
            if self.x > .2 or self.x < .2:
                self.y = max(self.y, .1)

        if self.x > 1.:
            self.x = 1.
            self.vel_x = 0.
        elif self.x < -1.:
            self.x = -1.
            self.vel_x = 0.
        if self.y > self.height:
            self.y = self.height
            self.vel_y = 0.
        elif self.y < 0.:
            self.y = 0.
            self.vel_y = 0.

    def _reached(self, x, y):
        return abs(x - self.x) < self.agent_size and abs(y - self.y) < self.agent_size

    def distance(self, x, y):
        return np.linalg.norm([self.x - x, self.y - y], ord=2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from time import sleep
    env = TMaze()
    plt.ion()
    fig, ax = plt.subplots(1, 1)
    env.reset()
    p1 = ax.imshow(env.vis(), cmap='gray')
    fig.canvas.draw()
    for _ in range(100):
        print(env.act(3))
        p1.set_data(env.vis())
        fig.canvas.draw()

    for _ in range(100):
        print(env.act(0))
        p1.set_data(env.vis())
        fig.canvas.draw()
