"""
Trajectory Buffer for PPO RL agent.
"""
import numpy as np

class TrajectoryBuffer:
    def __init__(self, buffer_size, obs_dim, action_dim):
        self.states = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.bool_)
        self.log_probs = np.zeros((buffer_size,), dtype=np.float32)
        self.ptr = 0
        self.max_size = buffer_size

    def add(self, state, action, reward, done, log_prob):
        if self.ptr < self.max_size:
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.dones[self.ptr] = done
            self.log_probs[self.ptr] = log_prob
            self.ptr += 1

    def reset(self):
        self.ptr = 0

    def get(self):
        return {
            'states': self.states[:self.ptr],
            'actions': self.actions[:self.ptr],
            'rewards': self.rewards[:self.ptr],
            'dones': self.dones[:self.ptr],
            'log_probs': self.log_probs[:self.ptr],
        } 