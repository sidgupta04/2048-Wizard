import numpy as np
import torch
from dqn_agent import DQNAgent
from game_2048 import Game2048

NUM_EPISODES = 500
MAX_STEPS = 2000
INVALID_MOVE_PENALTY = -5

SAVE_MODEL_PATH = "dqn_hybrid.pth"
LOG_FILE = "training_log_hybrid.txt"

def preprocess_state(grid):
    state = np.zeros_like(grid, dtype=np.float32)
    mask = grid > 0
    state[mask] = np.log2(grid[mask])
    return state.flatten()


def monotonicity_bonus(grid):
    """
    Reward the agent for keeping tiles in a monotonic (snake-like) order.
    A monotonic board means large tiles cluster in a corner, which is the
    key strategy for reaching high tiles in 2048.
    """
    bonus = 0.0
    n = grid.shape[0]
    for row in grid:
        bonus += max(
            sum(row[i] >= row[i+1] for i in range(n - 1)),
            sum(row[i] <= row[i+1] for i in range(n - 1)),
        )
    for col in grid.T:
        bonus += max(
            sum(col[i] >= col[i+1] for i in range(n - 1)),
            sum(col[i] <= col[i+1] for i in range(n - 1)),
        )
    return bonus / (2 * n * (n - 1))


def corner_bonus(grid):
    """
    Reward the agent for keeping the highest tile in the top-left corner.
    The bonus scales with the value of the max tile so it stays meaningful
    as the game progresses.
    """
    max_val = grid.max()
    if max_val == 0:
        return 0.0
    if grid[0][0] == max_val:
        return np.log2(max_val) / 11.0
    return 0.0

def train():
    agent = DQNAgent()

    agent.q_network.load_state_dict(torch.load("dqn_pretrained.pth"))
    agent.target_network.load_state_dict(agent.q_network.state_dict())

    agent.epsilon = 0.05
    agent.epsilon_min = 0.05

    for param_group in agent.optimizer.param_groups:
        param_group['lr'] = 5e-5
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

                    # Monotonicity bonus: encourages snake/corner strategy
                    reward += 0.5 * monotonicity_bonus(next_grid)

                    # Corner bonus: reward for keeping max tile in top-left
                    reward += 0.5 * corner_bonus(next_grid)
                
                if done:
                    reward += 0.1 * np.log2(np.max(next_grid))

                next_state = preprocess_state(next_grid)
                agent.store(state, action, reward, next_state, done)
                if episode > 20:
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