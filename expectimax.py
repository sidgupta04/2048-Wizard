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

def get_empty_cells(grid):
    return list(zip(*np.where(grid == 0)))

def calculate_heuristic(grid):
    w1 = 2.7
    w2 = 1.5
    w3 = 1.0
    w4 = 2.0
    w5 = 1.0
    
    return (
        w1 * heuristic_empty(grid) + 
        w2 * heuristic_monotonicity(grid) + 
        w3 * heuristic_smoothness(grid) +
        w4 * heuristic_corner_bias(grid) +
        w5 * heuristic_merge_potential(grid)
    )

def heuristic_empty(grid):
    n = grid.shape[0]
    return len(get_empty_cells(grid)) / float(n * n)

def heuristic_corner_bias(grid):
    max_val = np.max(grid)
    if max_val == 0:
        return 0
    
    corners = [grid[0, 0]]
    log_max = np.log2(max_val)
    MAX_LOG = 11.0 
    
    if max_val in corners:
        return log_max / MAX_LOG
    else:
        return -log_max / MAX_LOG

def heuristic_merge_potential(grid):
    n = grid.shape[0]
    merge_count = 0
    for r in range(n):
        for c in range(n - 1):
            if grid[r, c] != 0 and grid[r, c] == grid[r, c+1]:
                merge_count += 1
    for c in range(n):
        for r in range(n - 1):
            if grid[r, c] != 0 and grid[r, c] == grid[r+1, c]:
                merge_count += 1
    return merge_count / float(2 * n * (n - 1))

_mono_cache = {}

def _get_mono_constants(n):
    if n not in _mono_cache:
        w = np.zeros((n, n), dtype=int)
        val = n * n
        for r in range(n):
            cols = range(n) if r % 2 == 0 else range(n - 1, -1, -1)
            for c in cols:
                w[r, c] = val
                val -= 1
        _mono_cache[n] = (w, float(np.sum(w)) * 11.0)
    return _mono_cache[n]

def heuristic_monotonicity(grid):
    n = grid.shape[0]
    mono_w, mono_max = _get_mono_constants(n)
    score = 0
    for r in range(n):
        for c in range(n):
            if grid[r, c] > 0:
                score += mono_w[r, c] * np.log2(grid[r, c])
    return score / mono_max

_smooth_cache = {}

def _get_smooth_max_penalty(n):
    if n not in _smooth_cache:
        # original value was 44.0 for n=4 (24 adjacent pairs)
        # scale proportionally for other grid sizes
        pairs = 2 * n * (n - 1)
        _smooth_cache[n] = 44.0 * pairs / 24.0
    return _smooth_cache[n]

def heuristic_smoothness(grid):
    n = grid.shape[0]
    score = 0
    for r in range(n):
        for c in range(n - 1):
            if grid[r, c] > 0 and grid[r, c+1] > 0:
                v1 = np.log2(grid[r, c])
                v2 = np.log2(grid[r, c+1])
                score -= abs(v1 - v2)
    for c in range(n):
        for r in range(n - 1):
            if grid[r, c] > 0 and grid[r+1, c] > 0:
                v1 = np.log2(grid[r, c])
                v2 = np.log2(grid[r+1, c])
                score -= abs(v1 - v2)
    return score / _get_smooth_max_penalty(n)

def simulate_move(grid, action):
    n = grid.shape[0]
    grid_copy = grid.copy()
    
    rotations = {0: 1, 1: 3, 2: 0, 3: 2}
    k = rotations[action]
    
    grid_copy = np.rot90(grid_copy, k)
    
    def compress(mat):
        new_mat = np.zeros((n, n), dtype=int)
        for i in range(n):
            pos = 0
            for j in range(n):
                if mat[i][j] != 0:
                    new_mat[i][pos] = mat[i][j]
                    pos += 1
        return new_mat

    def merge(mat):
        for i in range(n):
            for j in range(n - 1):
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

def _evaluate_action(args):
    grid, action, depth, samples = args
    new_grid, moved = simulate_move(grid, action)
    if not moved:
        return action, None
    val = expectimax_value(new_grid, depth, is_player=False, samples=samples)
    return action, val

_executor = None

def _get_executor():
    global _executor
    if _executor is None:
        from concurrent.futures import ProcessPoolExecutor
        _executor = ProcessPoolExecutor(max_workers=4)
    return _executor

def shutdown_executor():
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False)
        _executor = None

def expectimax_decision(grid, depth=3, samples=6, score=0, parallel=True):
    from concurrent.futures import as_completed
    
    start_time = time.time()
    
    best_move = None
    best_val = float('-inf')
    
    if parallel:
        executor = _get_executor()
        args_list = [(grid, action, depth, samples) for action in range(4)]
        
        futures = [executor.submit(_evaluate_action, args) for args in args_list]
        
        for future in as_completed(futures):
            action, val = future.result()
            if val is not None and val > best_val:
                best_val = val
                best_move = action
    else:
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
    mode = "parallel" if parallel else "sequential"
    logger.info(f"Move calc ({mode}): {elapsed_ms:.1f}ms | Empty tiles: {empty_tiles} | Score: {score}")
            
    return best_move

def expectimax_value(grid, depth, is_player, samples, cache=None):
    if depth == 0:
        return calculate_heuristic(grid)
        
    if cache is None:
        cache = {}
            
    if is_player:
        best = float('-inf')
        move_possible = False
        for action in range(4):
            new_grid, moved = simulate_move(grid, action)
            if moved:
                move_possible = True
                val = expectimax_value(new_grid, depth, False, samples, cache)
                best = max(best, val)
        
        if not move_possible:
            return -1e6
        return best
    else:
        empty_cells = get_empty_cells(grid)
        if not empty_cells:
            return calculate_heuristic(grid)
            
        if len(empty_cells) <= samples:
            outcomes = []
            for cell in empty_cells:
                 outcomes.append((cell, 2, 0.9))
                 outcomes.append((cell, 4, 0.1))
                 
            weighted_sum = 0
            total_prob = 0 
            
            running_sum = 0
            for cell, val, prob in outcomes:
                nb = place_tile(grid, cell, val)
                v = expectimax_value(nb, depth - 1, True, samples, cache)
                running_sum += v * prob
            
            return running_sum / len(empty_cells)

        else:
            running_sum = 0
            for _ in range(samples):
                cell_idx = random.randrange(len(empty_cells))
                cell = empty_cells[cell_idx]
                val = 2 if random.random() < 0.9 else 4
                
                nb = place_tile(grid, cell, val)
                running_sum += expectimax_value(nb, depth - 1, True, samples, cache)
                
            return running_sum / samples
