from hyperparameters import *
from collections import deque
import random
import numpy as np


class PrioritizedExperienceReplay:
    def __init__(self):
        self.experience_replay = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.priorities = deque(maxlen=REPLAY_MEMORY_SIZE)

    def add_experience(self, experience):
        self.experience_replay.append(experience)
        self.priorities.append(max(self.priorities, default=1))

    def get_sample_weight(self, probs):
        sample_weights = 1 / len(self.experience_replay) * 1 / probs
        sample_weights /= np.max(sample_weights)
        return sample_weights

    def sample(self):
        probs = np.array(self.priorities) ** EXPONENT_A
        probs = probs / sum(probs)
        indices = sorted(random.choices(range(len(self.experience_replay)), k=BATCH_SIZE, weights=probs))
        indices_clone = indices.copy()
        indices_clone.reverse()
        res = []
        for (i, e) in enumerate(self.experience_replay):
            while i == indices_clone[-1]:
                indices_clone.pop()
                res.append(e)
                if len(indices_clone) == 0:
                    return res, self.get_sample_weight(probs[indices]), indices
        assert False

    def update_priorities(self, indices, errors):
        for i, e in zip(indices, errors):
            self.priorities[i] = e + OFFSET

    def __len__(self):
        return len(self.experience_replay)

    def __getitem__(self, key):
        return self.experience_replay[key]
