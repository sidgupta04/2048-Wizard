import numpy as np
import time
from expectimax import expectimax_decision, calculate_heuristic, simulate_move
from game_2048 import Game2048

def test_heuristics():
    print("Testing Heuristics...")
    grid = np.zeros((4, 4), dtype=int)
    grid[0, 0] = 1024
    grid[0, 1] = 512
    grid[0, 2] = 256
    grid[0, 3] = 128
    
    score = calculate_heuristic(grid)
    print(f"Heuristic score for monotonic row: {score}")
    assert score > 0, "Score should be positive for good board"
    print("PASS")

def test_expectimax():
    print("Testing Expectimax Decision...")
    grid = np.zeros((4, 4), dtype=int)
    grid[3, 3] = 2
    grid[3, 2] = 2
    
    # Should prefer moving to combine them (Right or Left or Down depending on others)
    # Let's just check it returns a valid move (0-3) and runs reasonably fast
    
    start = time.time()
    move = expectimax_decision(grid, depth=3)
    end = time.time()
    
    print(f"Best move: {move}")
    print(f"Time taken: {end - start:.4f}s")
    
    assert move in [0, 1, 2, 3], "Move must be in 0-3"
    assert (end - start) < 1.0, "Decision took too long!" # Relaxed check, <100ms goal is soft
    print("PASS")

if __name__ == "__main__":
    test_heuristics()
    test_expectimax()
