import numpy as np
import torch
from dqn_agent import DQNAgent
from game_2048 import Game2048

# Training Hyperparameters
NUM_EPISODES = 1000        
MAX_STEPS = 10000          
INVALID_MOVE_PENALTY = -5

SAVE_MODEL_PATH = "dqn_2048.pth"

def train():
    agent = DQNAgent()
    game = Game2048()

    scores = []

    for episode in range(NUM_EPISODES):
        grid = game.reset()
        state = grid.flatten()
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            action = agent.action(state)
            next_grid, reward, done, moved = game.step(action)

            if not moved:
                reward = INVALID_MOVE_PENALTY
            next_state = next_grid.flatten()
            agent.store(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            steps += 1

        scores.append(game.score)
        print(
            f"Episode {episode:4d} | "
            f"Score: {game.score:6d} | "
            f"Epsilon: {agent.epsilon:.3f} | "
            f"Steps: {steps}"
        )

    torch.save(agent.q_network.state_dict(), SAVE_MODEL_PATH)
    print(f"\nModel saved to {SAVE_MODEL_PATH}")


if __name__ == "__main__":
    train()