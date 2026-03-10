"""
Graph metrics from Expectimax game results (all_500_game_results.pkl).
Plots:
  - Running maximum highest tile vs game index (assumes pkl order = play order).
  - Per-game highest tile scatter.
  - Order-independent: counts and cumulative % that reached at least X.
  - Game length (moves) vs highest tile reached.
  - Max tile over move index (average progression).
"""

import pickle
import numpy as np

# Use non-interactive backend so saving to file works without a display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKL_PATH = "all_500_game_results.pkl"

# 2048 tile values for axis labels
TILE_LABELS = [128, 256, 512, 1024, 2048, 4096]


def highest_tile_in_trajectory(trajectory):
    """Return the maximum tile value seen in any state of this game."""
    if not trajectory:
        return 0
    return max(np.asarray(grid).max() for grid, _ in trajectory)


def main():
    with open(PKL_PATH, "rb") as f:
        all_games = pickle.load(f)

    if not isinstance(all_games, list) or len(all_games) == 0:
        raise ValueError(f"Expected non-empty list in {PKL_PATH}")

    n_games = len(all_games)
    # Per-game highest tile and game length (moves = trajectory length - 1; first entry is (state, None))
    per_game_max_tile = np.array([highest_tile_in_trajectory(t) for t in all_games])
    game_lengths = np.array([max(0, len(t) - 1) for t in all_games], dtype=int)
    running_max_tile = np.maximum.accumulate(per_game_max_tile)
    games_played = np.arange(1, n_games + 1, dtype=int)

    # --- Figure 1: Order-dependent (running max + scatter) ---
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.step(games_played, running_max_tile, where="post", color="steelblue", linewidth=2)
    ax1.set_ylabel("Highest tile reached (so far)")
    ax1.set_title(
        "Running maximum by game index (assumes pkl order = play order; "
        "if order is arbitrary, use the order-independent plot below)"
    )
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Game index (1 to N)")

    ax2.scatter(games_played, per_game_max_tile, alpha=0.5, s=12, color="coral", label="Per-game max tile")
    ax2.plot(games_played, running_max_tile, color="steelblue", linewidth=1.5, label="Running max")
    ax2.set_ylabel("Highest tile in game")
    ax2.set_xlabel("Game index")
    ax2.set_title("Per-game highest tile vs game index")
    ax2.legend(loc="lower right")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("metrics_highest_tile_vs_games.png", dpi=150, bbox_inches="tight")
    print("Saved metrics_highest_tile_vs_games.png")
    plt.close()

    # --- Figure 2: Order-independent (counts and cumulative %) ---
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart: count of games whose max tile equals each value
    counts = np.array([(per_game_max_tile == t).sum() for t in TILE_LABELS], dtype=int)
    ax3.bar([str(t) for t in TILE_LABELS], counts, color="steelblue", edgecolor="white")
    ax3.set_xlabel("Highest tile reached in game")
    ax3.set_ylabel("Number of games (out of N)")
    ax3.set_title("Counts by highest tile reached (order-independent)")
    for i, c in enumerate(counts):
        ax3.text(i, c, str(int(c)), ha="center", va="bottom", fontsize=9)

    # Cumulative % of games that reached at least X (sorted ascending by max tile)
    # For each threshold, fraction of games with max_tile >= threshold
    thresholds = np.array(TILE_LABELS, dtype=float)
    pct_at_least = [100 * np.sum(per_game_max_tile >= t) / n_games for t in thresholds]
    ax4.bar(range(len(thresholds)), pct_at_least, color="coral", edgecolor="white")
    ax4.set_xticks(range(len(thresholds)))
    ax4.set_xticklabels([str(int(t)) for t in thresholds])
    ax4.set_xlabel("At least tile")
    ax4.set_ylabel("% of games")
    ax4.set_title("% of games that reached at least this tile")

    plt.tight_layout()
    plt.savefig("metrics_highest_tile_distribution.png", dpi=150, bbox_inches="tight")
    print("Saved metrics_highest_tile_distribution.png")
    plt.close()

    # --- Figure 3: Game length vs max tile ---
    fig3, ax5 = plt.subplots(figsize=(8, 6))
    ax5.scatter(game_lengths, per_game_max_tile, alpha=0.5, s=20, color="steelblue", edgecolors="white", linewidths=0.3)
    ax5.set_xlabel("Game length (number of moves)")
    ax5.set_ylabel("Highest tile reached")
    ax5.set_yticks(TILE_LABELS)
    ax5.set_yticklabels([str(t) for t in TILE_LABELS])
    ax5.set_title("Game length vs highest tile reached")
    ax5.set_ylim(bottom=0)
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("metrics_game_length_vs_max_tile.png", dpi=150, bbox_inches="tight")
    print("Saved metrics_game_length_vs_max_tile.png")
    plt.close()

    # --- Figure 4: Max tile over move index (average progression) ---
    max_moves = max(len(t) - 1 for t in all_games)  # longest game length
    move_indices = np.arange(max_moves + 1, dtype=int)  # 0 = initial state, 1 = after 1 move, ...
    avg_max_tile = np.zeros(max_moves + 1)
    for t in move_indices:
        max_tiles_at_t = [np.asarray(all_games[i][t][0]).max() for i in range(n_games) if len(all_games[i]) > t]
        avg_max_tile[t] = np.mean(max_tiles_at_t) if max_tiles_at_t else 0
    fig4, ax6 = plt.subplots(figsize=(10, 6))
    ax6.plot(move_indices, avg_max_tile, color="steelblue", linewidth=2)
    ax6.set_xlabel("Move index (0 = initial state)")
    ax6.set_ylabel("Average highest tile on board")
    ax6.set_title("Average max tile over move index (over games that reached that move)")
    ax6.set_ylim(bottom=0)
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("metrics_max_tile_over_moves.png", dpi=150, bbox_inches="tight")
    print("Saved metrics_max_tile_over_moves.png")
    plt.close()


if __name__ == "__main__":
    main()
