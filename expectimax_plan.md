# Expectimax Integration Spec — 2048 (Python, single-threaded, real-time, depth=3, stochastic sampling)

**Purpose:** concise, implementable one-page spec for an advanced agent to add an Expectimax solver into `game_2048.py`. The solver will play the game automatically: as best moves are calculated and applied the Pygame board updates in real time. Use **depth = 3 (player plies)** and **stochastic sampling** at chance nodes.

---

## High-level behavior

* At each solver step (when UI is idle and **not animating**), compute `best_move = expectimax_decision(grid, depth=3)`.
* Immediately before applying the move:

  * set `pygame_ui.previous_grid = grid.copy()` and `pygame_ui.current_action = best_move`
  * call `game.step(best_move)` (existing API) — this spawns the tile and updates `game.last_spawned_tile`.
  * set `pygame_ui.animating = True` and `pygame_ui.animation_start_time = time.time()` so the board animates the move in real time (existing animation code will take care of visuals).
* Repeat until `game.game_over`.

---

## Three heuristics (use these only — simple, effective)

1. **Empty tiles**
   `H_empty(board) = count_zero_cells(board)`
   Reward more empty cells (favors survival and mobility).

2. **Monotonicity (snake/gradient pattern)**

   * Predefine a **weight matrix** (snake order) `W` (higher weights near preferred corner). Example (corner upper-left):

     ```
     W = [[16,15,14,13],
          [ 9,10,11,12],
          [ 8, 7, 6, 5],
          [ 1, 2, 3, 4]]
     ```
   * `H_mono(board) = Σ_{r,c} W[r,c] * log2(board[r,c])` (treat empty as 0 contribution).
     Encourages large tiles to follow gradient/monotone layout toward the chosen corner.

3. **Smoothness**

   * Convert tiles to `v = log2(value)` (skip zeros).
   * `H_smooth = - Σ_{neighbors (i,j)} |v_i - v_j|` (sum over horizontal + vertical neighbors)
     Penalizes big jumps that block merges.

**Combined heuristic (leaf evaluation):**

```
score(board) = w1 * H_empty + w2 * H_mono + w3 * H_smooth
# Suggested initial weights:
w1 = 2.7, w2 = 1.0, w3 = 0.1  (note H_smooth is negative so add * (+) but formula uses -|diff|)
```

*Tweak weights later if needed.*

---

## Expectimax specifics (concrete rules)

* **Depth convention:** `depth` counts **player moves remaining**.

  * At **player node**: do not decrement depth.
  * At **chance node**: recurse with `depth - 1`.
* **Player node:** return `max` over legal moves (ignore moves that don't change board).
* **Chance node (stochastic sampling):**

  * Let `empty = list(empty_cells)`. If `len(empty) <= 6`, enumerate all `(cell, tile)` pairs exactly; else **sample S pairs** (with replacement) where each sample:

    * pick `cell` uniformly from `empty`
    * pick tile `t = 2` with prob 0.9 else `t = 4`
  * For each sample produce board and call `expectimax_value(new_board, depth-1, is_player=True)`. Average sampled values to estimate expected value.
  * **Suggested samples per chance node:** `S = 6` (adjust for performance).
* **Terminal state:** if no legal moves, return very low value (e.g., `-1e6`) or `heuristic` with strong negative penalty.
* **Caching:** cache evaluated boards with depth key (transposition table) using `board.tobytes()` to speed repeated states. Clear cache between top-level calls or keep with depth param.

---

## Pseudocode (readily translatable to Python)

```text
function get_best_move(grid, depth=3, samples=6):
    best_move, best_val = None, -inf
    for move in legal_moves(grid):
        nb = simulate_move(grid, move)   # use Game2048 move logic but non-destructive copy
        if boards_equal(nb, grid): continue
        val = expectimax_value(nb, depth, is_player=False, samples)
        if val > best_val:
            best_val, best_move = val, move
    return best_move

function expectimax_value(grid, depth, is_player, samples):
    if depth == 0 or terminal(grid):
        return heuristic(grid)
    if is_player:
        best = -inf
        for move in legal_moves(grid):
            nb = simulate_move(grid, move)
            if boards_equal(nb, grid): continue
            best = max(best, expectimax_value(nb, depth, False, samples))
        return best
    else:  # chance node
        empty = empty_cells(grid)
        if len(empty) == 0: return heuristic(grid)
        if len(empty) <= 6:
            outcomes = all (cell,tile) pairs
        else:
            outcomes = sample S pairs (cell uniform, tile random with p=0.9/0.1)
        sum_val = 0
        for (cell, tile) in outcomes:
            nb = place_tile(grid, cell, tile)
            sum_val += expectimax_value(nb, depth-1, True, samples)
        return sum_val / len(outcomes)
```

---

## Implementation notes & integration tips

* **Non-destructive simulation:** Use `grid.copy()` + same rotate/compress/merge logic or call an internal helper `simulate_step(grid, action)` that returns new grid (without spawning the random tile) and `moved` flag. For chance node `place_tile(grid, cell, tile)` must not call `spawn_tile()` randomness — it should set tile deterministically.
* **Legal moves:** reuse `Game2048._compress/_merge` logic to check if a move changes the board; skip moves that don't change the grid.
* **Time budget:** Aim for <100ms per decision. If solver blocks UI too long:

  * reduce `samples` (e.g., to 3)
  * reduce depth to 2 for interactive mode
  * add per-node early cutoff if time exceeds threshold
* **Visualization:** the UI already animates when `previous_grid`, `current_action`, and `animating` are set — follow that API.
* **Auto-play toggle:** add a key (e.g., `A`) to toggle AI on/off in `Pygame2048.run()` so user can switch between manual play and solver.
* **Random seed:** for reproducible behavior during testing, allow seeding `random`/`numpy.random`.

---

## Deliverables (what to add to `game_2048.py`)

1. `expectimax_decision(grid, depth=3, samples=6)` — top-level move chooser.
2. `expectimax_value(grid, depth, is_player, samples, cache)` — recursive evaluator.
3. `heuristic(grid)` — implements the 3 heuristics and weighted sum.
4. `simulate_move(grid, action)` and `place_tile(grid, (r,c), tile)` helpers (non-destructive).
5. Integration in `Pygame2048.run()`:

   * handle `AI_MODE` flag
   * when `not animating and not game_over and AI_MODE`: compute best_move, then set `previous_grid`, `current_action`, call `game.step(best_move)`, set animating/time.

---

## Quick tuning checklist

* Start `depth=3`, `samples=6`, weights `[2.7, 1.0, 0.1]`.
* If decision time > 100ms, drop samples or depth.
* Add caching of `board.tobytes()` keyed by `(depth,is_player)` for speedups.
