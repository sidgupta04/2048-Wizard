import argparse
import pickle
import time
import numpy as np

from game_2048 import Game2048
from expectimax import expectimax_decision, shutdown_executor


def run_single_game(game_index, total_games):
    game = Game2048()
    game.reset()

    history = []

    history.append((game.grid.copy(), None))

    move_count = 0
    while not game.game_over:
        best_move = expectimax_decision(
            game.grid, score=game.score, parallel=True
        )

        if best_move is None:
            break

        history.append((game.grid.copy(), best_move))

        _, _, done, moved = game.step(best_move)
        move_count += 1

        if done:
            break

    history.append((game.grid.copy(), None))

    max_tile = int(np.max(game.grid))
    print(
        f"Game {game_index + 1}/{total_games}  |  "
        f"Score: {game.score:>6}  |  Max tile: {max_tile:>5}  |  "
        f"Moves: {move_count}"
    )

    return history, game.score, max_tile


def main():
    parser = argparse.ArgumentParser(
        description="Run the expectimax AI on 2048 and save game histories."
    )
    parser.add_argument(
        "--num-games", type=int, default=500,
        help="Number of games to play (default: 500)"
    )
    parser.add_argument(
        "--output", type=str, default="game_results.pkl",
        help="Output pickle file path (default: game_results.pkl)"
    )
    args = parser.parse_args()

    num_games = args.num_games
    output_path = args.output

    print(f"Running {num_games} games, saving to {output_path}\n")

    all_games = []
    scores = []
    max_tiles = []
    overall_start = time.time()

    try:
        for i in range(num_games):
            history, score, max_tile = run_single_game(i, num_games)
            all_games.append(history)
            scores.append(score)
            max_tiles.append(max_tile)
    except KeyboardInterrupt:
        print(f"\nInterrupted after {len(all_games)} games. Saving partial results...")
    finally:
        shutdown_executor()

    elapsed = time.time() - overall_start

    with open(output_path, "wb") as f:
        pickle.dump(all_games, f)

    if scores:
        print(f"\n{'=' * 55}")
        print(f"  Completed {len(all_games)} games in {elapsed:.1f}s")
        print(f"  Avg score : {np.mean(scores):.0f}")
        print(f"  Max score : {np.max(scores)}")
        print(f"  Avg max tile: {np.mean(max_tiles):.0f}")
        print(f"  Max tile seen: {np.max(max_tiles)}")
        print(f"  Saved to: {output_path}")
        print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
