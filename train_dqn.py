import numpy as np
import torch
from dqn_agent import DQNAgent
from game_2048 import Game2048

NUM_EPISODES = 500
MAX_STEPS = 2000
INVALID_MOVE_PENALTY = -5

SAVE_MODEL_PATH = "dqn_2048.pth"
LOG_FILE = "training_log.txt"

def preprocess_state(grid):
    state = np.zeros_like(grid, dtype=np.float32)
    mask = grid > 0
    state[mask] = np.log2(grid[mask])
    return state.flatten()

def train():
    agent = DQNAgent()
    game = Game2048()

    with open(LOG_FILE, "w") as f:
        f.write("episode,score,epsilon,steps\n")  

        for episode in range(NUM_EPISODES):
            grid = game.reset()
            state = preprocess_state(grid)
            done = False
            steps = 0

            while not done and steps < MAX_STEPS:
                action = agent.action(state)
                next_grid, reward, done, moved = game.step(action)

                if not moved:
                    reward = INVALID_MOVE_PENALTY
                else:
                    if reward > 0:
                        reward = np.log2(reward)

                    empty_tiles = np.sum(next_grid == 0)
                    reward += 0.01 * empty_tiles
                
                if done:
                    reward += 0.1 * np.log2(np.max(next_grid))

                next_state = preprocess_state(next_grid)
                agent.store(state, action, reward, next_state, done)
                agent.train()

                state = next_state
                steps += 1

            agent.epsilon = max(
            agent.epsilon * agent.epsilon_decay,
            agent.epsilon_min
            )
            line = f"{episode},{game.score},{agent.epsilon:.3f},{steps}\n"
            f.write(line)

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