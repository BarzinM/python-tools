import numpy as np


class SumTree(object):

    def __init__(self, size):
        self.size = size
        self.tree = np.zeros(2 * size - 1, dtype=float)
        self.data = np.zeros(size, dtype=int)
        self.pointer = 0

    def add(self, content, probability):
        idx = self.pointer + self.size - 1
        self.data[self.pointer] = content
        self.update(idx, probability)
        self.pointer = (self.pointer + 1) % self.size

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def total(self):
        return self.tree[0]

    def get(self, s):
        idx = self._get(0, s)
        data_idx = idx - self.size + 1

        return idx, self.tree[idx], self.data[data_idx]

    def _get(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._get(left, s)
        else:
            return self._get(right, s - self.tree[left])
