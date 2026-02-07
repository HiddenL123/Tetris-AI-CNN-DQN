import numpy as np
from collections import deque
import torch

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.pos = 0
        self.max_priority = 1.0

    def add(self, transition, priority=1.0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample with replacement using probabilities
        if len(self.buffer) == 0:
            return [], [], torch.tensor([], dtype=torch.float32)

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=True)
        samples = [self.buffer[i] for i in indices]

        # Importance sampling weights (recommended)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        # avoid division by zero
        max_w = weights.max() if weights.max() > 0 else 1.0
        weights = weights / (max_w + 1e-8)

        # return weights as column vector to match (batch, 1) losses
        return samples, indices, torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

    def update_priorities(self, indices, priorities):
        # Ensure indices are integers and priorities are floats
        for idx, prio in zip(map(int, indices), priorities):
            prio = float(prio)
            # clamp priority to a small positive value to avoid zeros
            prio = max(prio, 1e-6)
            self.priorities[idx] = prio
            self.max_priority = max(self.max_priority, prio)

    def __len__(self):
        return len(self.buffer)
    
    def state_dict(self):
        return {
            "buffer": self.buffer,
            "priorities": list(self.priorities),
            "pos": self.pos,
            "capacity": self.capacity,
            "alpha": self.alpha,
        }

    def load_state_dict(self, state):
        self.buffer = state["buffer"]
        self.priorities = deque(state["priorities"], maxlen=state["capacity"])
        self.pos = state["pos"]
        self.capacity = state["capacity"]
        self.alpha = state["alpha"]
