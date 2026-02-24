import pickle
import numpy as np
import torch

from dqn_agent import DQNAgent
from game_2048 import Game2048

MODEL_PATH = "dqn_pretrained.pth"
PICKLE_PATH = "all_500_game_results.pkl"

NUM_EVAL_EPISODES = 50
MAX_STEPS = 5000


# -----------------------------
# Preprocessing (must match training)
# -----------------------------
def preprocess_state(grid):
    state = np.zeros_like(grid, dtype=np.float32)
    mask = grid > 0
    state[mask] = np.log2(grid[mask])
    return state.flatten()


# -----------------------------
# Step 1: Imitation Accuracy
# -----------------------------
def compute_imitation_accuracy(agent):
    print("\n=== Step 1: Imitation Accuracy ===")

    with open(PICKLE_PATH, "rb") as f:
        data = pickle.load(f)

    states = []
    actions = []

    for game in data:
        for board, move in game:
            if move is None:
                continue
            states.append(preprocess_state(board))
            actions.append(move)

    states = torch.FloatTensor(np.array(states)).to(agent.device)
    actions = torch.LongTensor(np.array(actions)).to(agent.device)

    agent.q_network.eval()

    with torch.no_grad():
        logits = agent.q_network(states)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == actions).float().mean()

    print(f"Imitation Accuracy: {accuracy.item():.4f}")


# -----------------------------
# Step 2: Gameplay Evaluation
# -----------------------------
def evaluate_agent(agent):
    print("\n=== Step 2: Gameplay Evaluation ===")

    game = Game2048()
    scores = []
    max_tiles = []

    agent.q_network.eval()
    agent.epsilon = 0.0  # fully greedy

    for episode in range(NUM_EVAL_EPISODES):
        grid = game.reset()
        state = preprocess_state(grid)
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:

            with torch.no_grad():
                q_values = agent.q_network(
                    torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                ).squeeze(0)

            # Try highest Q actions first
            actions = torch.argsort(q_values, descending=True).tolist()

            moved = False
            for action in actions:
                next_grid, reward, done, moved = game.step(action)
                if moved:
                    break

            if not moved:
                done = True

            state = preprocess_state(next_grid)
            steps += 1

        scores.append(game.score)
        max_tiles.append(game.grid.max())

        print(
            f"Episode {episode:2d} | "
            f"Score: {game.score:6d} | "
            f"Max Tile: {game.grid.max():4d}"
        )

    print("\n=== Evaluation Summary ===")
    print(f"Average Score: {np.mean(scores):.1f}")
    print(f"Max Score:     {np.max(scores)}")
    print(f"Average Tile:  {np.mean(max_tiles):.1f}")
    print(f"Max Tile Seen: {np.max(max_tiles)}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("Loading pretrained model...")

    agent = DQNAgent()
    agent.q_network.load_state_dict(
        torch.load(MODEL_PATH, map_location=agent.device)
    )

    compute_imitation_accuracy(agent)
    evaluate_agent(agent)