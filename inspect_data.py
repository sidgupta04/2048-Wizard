import pickle
import numpy as np

with open("all_500_game_results.pkl", "rb") as f:
    data = pickle.load(f)

with open("output.txt", "w") as f:
    for game_index, game in enumerate(data):

        f.write(f"\n================ GAME {game_index} ================\n\n")

        for step_index, (board, move) in enumerate(game):

            f.write(f"Step {step_index}\n")
            f.write(f"Move: {move}\n")
            f.write("Board:\n")

            board_array = np.array(board)

            for row in board_array:
                f.write(" ".join(f"{int(num):5}" for num in row) + "\n")

            f.write("\n")

print("Saved all games nicely to output.txt")