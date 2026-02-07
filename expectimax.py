import random
import numpy as np
import logging
import time

logger = logging.getLogger('expectimax')
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler('expectimax.log')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

# --- Expectimax AI Solver ---

def get_empty_cells(grid):
    return list(zip(*np.where(grid == 0)))

def calculate_heuristic(grid):
    """
    Combined heuristic:
    score = w1 * H_empty + w2 * H_mono + w3 * H_smooth + w4 * H_max + w5 * H_corner + w6 * H_merge
    """
    w1 = 2.7
    w2 = 1.0
    w3 = 0.1
    w4 = 1.0  # Max Tile
    w5 = 2.0  # Corner Bias
    w6 = 0.8  # Merge Potential
    
    return (
        w1 * heuristic_empty(grid) + 
        w2 * heuristic_monotonicity(grid) + 
        w3 * heuristic_smoothness(grid) +
        w4 * heuristic_max_tile(grid) +
        w5 * heuristic_corner_bias(grid) +
        w6 * heuristic_merge_potential(grid)
    )

def heuristic_empty(grid):
    return len(get_empty_cells(grid))

def heuristic_max_tile(grid):
    """Encourages building large tiles."""
    max_val = np.max(grid)
    if max_val == 0:
        return 0
    return np.log2(max_val)

def heuristic_corner_bias(grid):
    """Encourages keeping the largest tile in a corner."""
    max_val = np.max(grid)
    if max_val == 0:
        return 0
    
    # Check 4 corners
    rows, cols = grid.shape
    corners = [
        grid[0, 0], grid[0, cols-1],
        grid[rows-1, 0], grid[rows-1, cols-1]
    ]
    
    if max_val in corners:
        return 2.0 * np.log2(max_val) # Bonus scaling with value
    else:
        return -2.0 * np.log2(max_val) # Penalty

def heuristic_merge_potential(grid):
    """Counts number of possible merges."""
    merge_count = 0
    # Horizontal
    for r in range(4):
        for c in range(3):
            if grid[r, c] != 0 and grid[r, c] == grid[r, c+1]:
                merge_count += 1
    # Vertical
    for c in range(4):
        for r in range(3):
            if grid[r, c] != 0 and grid[r, c] == grid[r+1, c]:
                merge_count += 1
    return merge_count

def heuristic_monotonicity(grid):
    # Monotonicity mask (snake pattern)
    # Higher weights in top-left corner
    W = np.array([
        [16, 15, 14, 13],
        [ 9, 10, 11, 12],
        [ 8,  7,  6,  5],
        [ 1,  2,  3,  4]
    ])
    
    score = 0
    for r in range(4):
        for c in range(4):
            if grid[r, c] > 0:
                score += W[r, c] * np.log2(grid[r, c])
    return score

def heuristic_smoothness(grid):
    score = 0
    # Horizontal
    for r in range(4):
        for c in range(3):
            if grid[r, c] > 0 and grid[r, c+1] > 0:
                v1 = np.log2(grid[r, c])
                v2 = np.log2(grid[r, c+1])
                score -= abs(v1 - v2)
    # Vertical
    for c in range(4):
        for r in range(3):
            if grid[r, c] > 0 and grid[r+1, c] > 0:
                v1 = np.log2(grid[r, c])
                v2 = np.log2(grid[r+1, c])
                score -= abs(v1 - v2)
    return score

def simulate_move(grid, action):
    """
    Simulation of a move on a grid copy.
    Replicates the logic in Game2048.step but without side effects.
    """
    grid_copy = grid.copy()
    
    # Rotation logic from Game2048.step
    rotations = {0: 1, 1: 3, 2: 0, 3: 2}
    k = rotations[action]
    
    grid_copy = np.rot90(grid_copy, k)
    
    # Compress, Merge, Compress
    # We duplicate the simple logic here to avoid class dependency or invasive refactoring
    def compress(mat):
        new_mat = np.zeros((4, 4), dtype=int)
        for i in range(4):
            pos = 0
            for j in range(4):
                if mat[i][j] != 0:
                    new_mat[i][pos] = mat[i][j]
                    pos += 1
        return new_mat

    def merge(mat):
        for i in range(4):
            for j in range(3):
                if mat[i][j] != 0 and mat[i][j] == mat[i][j + 1]:
                    mat[i][j] *= 2
                    mat[i][j + 1] = 0
        return mat

    grid_copy = compress(grid_copy)
    grid_copy = merge(grid_copy)
    grid_copy = compress(grid_copy)
    
    grid_copy = np.rot90(grid_copy, -k)
    
    moved = not np.array_equal(grid, grid_copy)
    return grid_copy, moved

def place_tile(grid, pos, value):
    new_grid = grid.copy()
    new_grid[pos] = value
    return new_grid

def expectimax_decision(grid, depth=3, samples=6, score=0):
    start_time = time.time()
    
    best_move = None
    best_val = float('-inf')
    
    # try each move, and find the one with the highest value
    for action in range(4):
        new_grid, moved = simulate_move(grid, action)
        if not moved:
            continue
            
        val = expectimax_value(new_grid, depth, is_player=False, samples=samples)
        if val > best_val:
            best_val = val
            best_move = action
    
    elapsed_ms = (time.time() - start_time) * 1000
    empty_tiles = len(get_empty_cells(grid))
    logger.info(f"Move calc: {elapsed_ms:.1f}ms | Empty tiles: {empty_tiles} | Score: {score}")
            
    return best_move

def expectimax_value(grid, depth, is_player, samples, cache=None):
    if depth == 0:
        return calculate_heuristic(grid)
        
    if cache is None:
        cache = {}
        
    # Simple caching based on bytes
    # Note: For a proper implementation we might want to pass cache around more effectively
    # or make this a class. For now, we'll skip global caching or keep it local to the decision recursion if passed.
    # But since cache defaults to None and isn't returned, it's effectively local to one call if not improved.
    # Let's trust the logic works for now without persistent caching across decisions.
    
    if is_player:
        best = float('-inf')
        move_possible = False
        for action in range(4):
            new_grid, moved = simulate_move(grid, action)
            if moved:
                move_possible = True
                val = expectimax_value(new_grid, depth, False, samples, cache)
                best = max(best, val)
        
        if not move_possible: # Game Over or stuck
            return -1e6 # Big penalty
        return best
    else:
        # Chance Node
        empty_cells = get_empty_cells(grid)
        if not empty_cells:
            return calculate_heuristic(grid)
            
        # Sampling
        if len(empty_cells) <= samples:
            # Check all possibilities if small number
            outcomes = []
            for cell in empty_cells:
                 # 2 (0.9 prob), 4 (0.1 prob)
                 outcomes.append((cell, 2, 0.9))
                 outcomes.append((cell, 4, 0.1))
                 
            weighted_sum = 0
            total_prob = 0 # Should sum to len(empty_cells) technically if we sum probs, but here we average expected values
            
            # Correct logic: Expected value = sum( prob(outcome) * value(outcome) )
            # Here prob of (cell, 2) is (1/N) * 0.9
            
            # Let's follow the plan: "Average sampled values to estimate expected value"
            # If we enumerate ALL, we do exact calculation.
            running_sum = 0
            for cell, val, prob in outcomes:
                nb = place_tile(grid, cell, val)
                # Probability of this specific tile occurrence is prob
                # Probability of this specific CELL is 1/len(empty_cells)
                # So weight is prob * (1/len(empty_cells)) NOT handled by simple average if we mix 2/4.
                
                # EASIER IMPLEMENTATION based on plan "enumerate all (cell, tile) pairs exactly"
                # If we average, we need to treat them as samples.
                
                v = expectimax_value(nb, depth - 1, True, samples, cache)
                running_sum += v * prob
            
            return running_sum / len(empty_cells)

        else:
            # Random sampling
            running_sum = 0
            for _ in range(samples):
                cell_idx = random.randrange(len(empty_cells))
                cell = empty_cells[cell_idx]
                val = 2 if random.random() < 0.9 else 4
                
                nb = place_tile(grid, cell, val)
                running_sum += expectimax_value(nb, depth - 1, True, samples, cache)
                
            return running_sum / samples
