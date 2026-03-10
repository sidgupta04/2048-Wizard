"""
Compute average score (and basic stats) across all Expectimax game results
stored in all_500_game_results.pkl. Score is obtained by replaying each
game's (state, action) trajectory: we compute merge score for each step and
advance to the stored next state (no random spawn), so no pygame dependency.
"""

import pickle
import numpy as np

PKL_PATH = "all_500_game_results.pkl"
GRID_SIZE = 4

# Same as game_2048: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT -> rot90 counts
ROTATIONS = {0: 1, 1: 3, 2: 0, 3: 2}


def _compress(mat):
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for i in range(GRID_SIZE):
        pos = 0
        for j in range(GRID_SIZE):
            if mat[i, j] != 0:
                out[i, pos] = mat[i, j]
                pos += 1
    return out


def _merge(mat):
    score_inc = 0
    mat = mat.copy()
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE - 1):
            if mat[i, j] != 0 and mat[i, j] == mat[i, j + 1]:
                mat[i, j] *= 2
                score_inc += mat[i, j]
                mat[i, j + 1] = 0
    return mat, score_inc


def merge_score_for_move(grid: np.ndarray, action: int) -> int:
    """Apply action to grid (no spawn) and return merge score only."""
    k = ROTATIONS[action]
    g = np.rot90(np.asarray(grid, dtype=int).copy(), k)
    g = _compress(g)
    g, score = _merge(g)
    return score


def score_from_trajectory(trajectory: list) -> int:
    """
    Replay a single game from (state, action) pairs and return final score.
    trajectory: list of (grid, action), with first entry (initial_grid, None).
    We use stored next states so no random spawn is needed.
    """
    if not trajectory or len(trajectory) < 2:
        return 0
    total = 0
    grid = np.asarray(trajectory[0][0], dtype=int).copy()
    for state, action in trajectory[1:]:
        if action is None:
            continue
        total += merge_score_for_move(grid, action)
        grid = np.asarray(state, dtype=int).copy()
    return total


def main():
    with open(PKL_PATH, "rb") as f:
        all_games = pickle.load(f)

    if not isinstance(all_games, list) or len(all_games) == 0:
        raise ValueError(f"Expected non-empty list in {PKL_PATH}")

    scores = []
    for trajectory in all_games:
        s = score_from_trajectory(trajectory)
        scores.append(s)

    scores = np.array(scores, dtype=np.float64)
    n = len(scores)
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    min_score = int(np.min(scores))
    max_score = int(np.max(scores))
    median_score = float(np.median(scores))

    print(f"Games loaded: {n}")
    print(f"Average score: {mean_score:.2f}")
    print(f"Std deviation: {std_score:.2f}")
    print(f"Median score:  {median_score:.2f}")
    print(f"Min score:     {min_score}")
    print(f"Max score:     {max_score}")

    return mean_score


if __name__ == "__main__":
    main()
