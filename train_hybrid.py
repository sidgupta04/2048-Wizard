import numpy as np
import torch
from dqn_agent import DQNAgent
from game_2048 import Game2048

NUM_EPISODES = 2000
MAX_STEPS = 2000
GRID_SIZE = 4
SAVE_MODEL_PATH = "dqn_hybrid.pth"
LOG_FILE = "training_log_hybrid.txt"


def preprocess_state(grid):
    """Must match training preprocessing exactly: log2 normalized by 11 (log2(2048)=11)."""
    state = np.where(grid == 0, 0, np.log2(np.maximum(grid, 1))).flatten().astype(np.float32)
    return state / 11.0


def monotonicity_bonus(grid):
    """Reward keeping tiles in a monotonic snake-like order."""
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
    """Reward keeping the highest tile in the top-left corner."""
    max_val = grid.max()
    if max_val == 0:
        return 0.0
    if grid[0][0] == max_val:
        return np.log2(max_val) / 11.0
    return 0.0


def train():
    agent = DQNAgent()

    # Load pretrained Expectimax-imitation weights
    agent.q_network.load_state_dict(torch.load("dqn_pretrained.pth", map_location=agent.device))
    agent.target_network.load_state_dict(agent.q_network.state_dict())

    # Low epsilon since we start from a good policy
    agent.epsilon = 0.05
    agent.epsilon_min = 0.01
    agent.epsilon_decay = 0.9977  # reaches ~0.01 by episode 2000

    # Lower LR for fine-tuning — don't want to overwrite pretrained weights too fast
    for param_group in agent.optimizer.param_groups:
        param_group['lr'] = 5e-5

    game = Game2048(grid_size=GRID_SIZE)

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
                # Don't store invalid moves — they pollute the replay buffer.
                continue

            # ── Reward shaping ──────────────────────────────────────────
            # 1. Merge reward: log2 keeps scale manageable
            if reward > 0:
                reward = np.log2(reward)

            # 2. Empty tile bonus: more open space = more future options
            empty_tiles = np.sum(next_grid == 0)
            reward += 0.1 * empty_tiles

            # 3. Monotonicity bonus: encourages snake/corner strategy
            reward += 0.5 * monotonicity_bonus(next_grid)

            # 4. Corner bonus: reward for keeping max tile in top-left
            reward += 0.5 * corner_bonus(next_grid)

            # 5. Terminal bonus: reward proportional to best tile achieved
            if done:
                reward += np.log2(max(next_grid.max(), 1))
            # ────────────────────────────────────────────────────────────

            next_state = preprocess_state(next_grid)
            agent.store(state, action, reward, next_state, done)

            # Wait 20 episodes before training so buffer has enough data
            if episode > 20:
                agent.train()

            state = next_state
            steps += 1

        # Decay once per episode
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

        line = f"{episode},{game.score},{agent.epsilon:.4f},{steps}\n"
        with open(LOG_FILE, "a") as f:
            f.write(line)

        print(
            f"Episode {episode:4d} | "
            f"Score: {game.score:6d} | "
            f"Max Tile: {game.grid.max():4d} | "
            f"Epsilon: {agent.epsilon:.3f} | "
            f"Steps: {steps}"
        )

    torch.save(agent.q_network.state_dict(), SAVE_MODEL_PATH)
    print(f"\nModel saved to {SAVE_MODEL_PATH}")


if __name__ == "__main__":
    train()
