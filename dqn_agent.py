import random
import torch
import torch.nn
from collections import deque


# Will need to create a QNetwork class
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super.__init__()


        self


class DQNAgent:
    def __init__(self):
        self.state_size = 16   
        self.action_size = 4   
        self.memory = deque(maxlen=500000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.q_network = QNetwork(self.state_size, self.action_size)
        self.q_network.eval()


    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    