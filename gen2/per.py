import numpy as np
import random
from collections import namedtuple

# Standard transition tuple (gamma added for n-step consistency across stage changes)
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'gamma'))

class SumTree:
    """
    SumTree structure for efficient storage and sampling of prioritized experience.
    Leaf nodes store priorities. Internal nodes store sum of children.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        # Tree size is 2 * capacity - 1
        # Indices for leaves start at capacity - 1
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.count = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.count < self.capacity:
            self.count += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1 # Start at 1 to avoid div by zero if logic changes

    def push(self, *args):
        """Save a transition"""
        # Max priority for new items ensures they get replayed at least once
        max_p = np.max(self.tree.tree[-self.capacity:])
        if max_p == 0:
            max_p = 1.0

        self.tree.add(max_p, Transition(*args))

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        # Calculate current beta
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += batch_size # Advance frame count

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.total() * sampling_probabilities, -beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (error + 1e-5) ** self.alpha
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.count
