import random
from collections import deque

class DQNAgent:
    def __init__(self):
        self.state_size = 16   
        self.action_size = 4   
        self.memory = deque(maxlen=500000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    