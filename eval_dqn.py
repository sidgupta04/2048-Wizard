import numpy as np
import torch
import pickle
from dqn_agent import DQNAgent
from game_2048 import Game2048

MODEL_PATH = "dqn_2048.pth"
NUM_EVAL_EPISODES = 190
MAX_STEPS = 5000
GRID_SIZE = 4
RESULTS_PATH = "dqn_results_4.pkl"


def preprocess_state(grid):
    """Must match training preprocessing exactly: log2 normalized by 11 (4x4 max tile 2048)."""
    return np.where(grid == 0, 0, np.log2(np.maximum(grid, 1))).flatten().astype(np.float32) / 11.0


def evaluate():
    agent = DQNAgent()
    agent.q_network.load_state_dict(
        torch.load(MODEL_PATH, map_location=agent.device)
    )
    agent.q_network.eval()
    agent.epsilon = 0.0  # no exploration during evaluation

    game = Game2048(grid_size=GRID_SIZE)

    all_games = []   # list of 190 games — each game is a list of (board, action) tuples
    scores = []
    max_tiles = []
    steps_per_game = []

    for episode in range(NUM_EVAL_EPISODES):
        grid = game.reset()
        state = preprocess_state(grid)
        done = False
        steps = 0
        game_history = []

        # First entry: initial board state, no action yet (matches expectimax format)
        game_history.append((grid.copy(), None))

        while not done and steps < MAX_STEPS:
            with torch.no_grad():
                q_values = agent.q_network(
                    torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                ).squeeze(0)

            # Try actions in descending Q-value order, skip invalid moves
            actions = torch.argsort(q_values, descending=True).tolist()
            moved = False
            chosen_action = None
            for action in actions:
                next_grid, reward, done, moved = game.step(action)
                if moved:
                    chosen_action = action
                    break

            if not moved:
                done = True
                break

            # Record (resulting board, action that was taken)
            game_history.append((next_grid.copy(), chosen_action))

            state = preprocess_state(next_grid)
            steps += 1

        all_games.append(game_history)
        scores.append(game.score)
        max_tiles.append(int(game.grid.max()))
        steps_per_game.append(steps)

        print(
            f"Eval Episode {episode:3d} | "
            f"Score: {game.score:6d} | "
            f"Max Tile: {game.grid.max():4d} | "
            f"Steps: {steps}"
        )

    # ---- Summary ----
    print("\n=== Evaluation Summary ===")
    print(f"Average Score: {np.mean(scores):.1f}")
    print(f"Max Score:     {np.max(scores)}")
    print(f"Average Tile:  {np.mean(max_tiles):.1f}")
    print(f"Max Tile Seen: {np.max(max_tiles)}")

    # ---- Save to pickle in same format as expectimax ----
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(all_games, f)
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Type: {type(all_games)}")
    print(f"Length: {len(all_games)}")
    print(f"First step of first game: {all_games[0][0]}")


if __name__ == "__main__":
    evaluate()
