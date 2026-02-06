import pygame
import random
import numpy as np
import time

# --- Constants ---
WIDTH, HEIGHT = 450, 500
GRID_SIZE = 4
TILE_SIZE = 100
PADDING = 10
BACKGROUND_COLOR = (187, 173, 160)
EMPTY_TILE_COLOR = (205, 193, 180)
FONT_COLOR = (119, 110, 101)
ANIMATION_DURATION = 0.15  # seconds for sliding animation
POP_ANIMATION_DURATION = 0.2  # seconds for pop-out animation

# Colors for specific tile values
TILE_COLORS = {
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}

# Action mapping for RL
# 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
ACTIONS = {
    0: 'UP',
    1: 'DOWN',
    2: 'LEFT',
    3: 'RIGHT'
}

from expectimax import expectimax_decision

class Game2048:
    """
    The Game Logic Class.
    Designed to be compatible with RL environments (like OpenAI Gym).
    """
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.last_spawned_tile = None  # Track newly spawned tile position
        self.reset()

    def reset(self):
        """Resets the game state and returns the initial grid."""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.score = 0
        self.game_over = False
        self.last_spawned_tile = None
        self.spawn_tile()
        self.spawn_tile()
        return self.grid.copy()

    def spawn_tile(self):
        """Adds a new tile (2 or 4) to a random empty spot."""
        empty_cells = list(zip(*np.where(self.grid == 0)))
        if empty_cells:
            r, c = random.choice(empty_cells)
            self.grid[r, c] = 4 if random.random() > 0.9 else 2
            self.last_spawned_tile = (r, c)

    def step(self, action):
        """
        Executes an action and returns (next_state, reward, done, moved).
        Action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        """
        if self.game_over:
            return self.grid.copy(), 0, True, False

        original_grid = self.grid.copy()
        reward = 0
        self.last_spawned_tile = None

        # Rotate grid to align all moves to "LEFT" logic for simplicity
        # 0: Up -> Rotate 90 deg counter-clockwise (Top becomes Left)
        # 1: Down -> Rotate 90 deg clockwise (Bottom becomes Left)
        # 2: Left -> No rotation
        # 3: Right -> Rotate 180 (Right becomes Left)
        
        rotations = {0: 1, 1: 3, 2: 0, 3: 2} # Number of 90deg rot90 calls
        k = rotations[action]
        
        # Orient grid so we are always moving LEFT
        self.grid = np.rot90(self.grid, k)
        
        # Compress (move left), Merge, Compress again
        self.grid, _ = self._compress(self.grid)
        self.grid, merge_score = self._merge(self.grid)
        self.grid, _ = self._compress(self.grid)
        
        # Restore orientation
        self.grid = np.rot90(self.grid, -k)

        reward = merge_score
        self.score += reward

        # Check if anything moved
        moved = not np.array_equal(original_grid, self.grid)
        if moved:
            self.spawn_tile()
        
        # Check Game Over
        if not self._can_move():
            self.game_over = True

        return self.grid.copy(), reward, self.game_over, moved

    def _compress(self, mat):
        """Slides non-zero elements to the left."""
        new_mat = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for i in range(self.grid_size):
            pos = 0
            for j in range(self.grid_size):
                if mat[i][j] != 0:
                    new_mat[i][pos] = mat[i][j]
                    pos += 1
        return new_mat, 0

    def _merge(self, mat):
        """Merges adjacent same-value tiles."""
        score_inc = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size - 1):
                if mat[i][j] != 0 and mat[i][j] == mat[i][j + 1]:
                    mat[i][j] *= 2
                    score_inc += mat[i][j]
                    mat[i][j + 1] = 0
        return mat, score_inc

    def _can_move(self):
        """Checks if any moves are possible."""
        if 0 in self.grid:
            return True
        # Check horizontal neighbors
        for r in range(self.grid_size):
            for c in range(self.grid_size - 1):
                if self.grid[r][c] == self.grid[r][c+1]:
                    return True
        # Check vertical neighbors
        for r in range(self.grid_size - 1):
            for c in range(self.grid_size):
                if self.grid[r][c] == self.grid[r+1][c]:
                    return True
        return False

class Pygame2048:
    """
    The Visualization Class.
    Handles user input and rendering.
    """
    def __init__(self, game_logic):
        pygame.init()
        self.game = game_logic
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("2048 - RL Friendly")
        self.font = pygame.font.Font(pygame.font.get_default_font(), 36)
        self.score_font = pygame.font.Font(pygame.font.get_default_font(), 24)
        
        # Animation state
        self.animating = False
        self.animation_start_time = 0
        self.previous_grid = None
        self.current_action = None
        self.new_tile_positions = {}  # Track tiles that need pop animation: (r, c) -> start_time
        self.ai_mode = False

    def _get_tile_position(self, r, c):
        """Get the screen position for a grid cell."""
        return (
            c * TILE_SIZE + PADDING * (c + 1),
            r * TILE_SIZE + PADDING * (r + 1)
        )
    
    def _ease_out_cubic(self, t):
        """Easing function for smooth animation."""
        return 1 - (1 - t) ** 3
    
    def _calculate_tile_movement(self, prev_grid, curr_grid, action):
        """Calculate where each tile moved from and to."""
        movements = {}  # (new_r, new_c) -> (old_r, old_c)
        
        if prev_grid is None:
            return movements
        
        # Rotate grids to align with LEFT movement
        rotations = {0: 1, 1: 3, 2: 0, 3: 2}
        k = rotations[action]
        
        prev_rotated = np.rot90(prev_grid, k)
        curr_rotated = np.rot90(curr_grid, k)
        
        # Track movements row by row
        for r in range(GRID_SIZE):
            # Build lists of tile positions and values in this row
            prev_tiles = []  # (col, value)
            curr_tiles = []  # (col, value)
            
            for c in range(GRID_SIZE):
                if prev_rotated[r, c] != 0:
                    prev_tiles.append((c, prev_rotated[r, c]))
                if curr_rotated[r, c] != 0:
                    curr_tiles.append((c, curr_rotated[r, c]))
            
            # Match tiles: work from left to right
            # After compression, tiles maintain relative order
            prev_i = 0
            curr_i = 0
            
            while prev_i < len(prev_tiles) and curr_i < len(curr_tiles):
                prev_col, prev_val = prev_tiles[prev_i]
                curr_col, curr_val = curr_tiles[curr_i]
                
                # Check if this is a merge (two same values become one double value)
                if prev_i + 1 < len(prev_tiles):
                    next_col, next_val = prev_tiles[prev_i + 1]
                    if prev_val == next_val and curr_val == prev_val * 2:
                        # This is a merge - the merged tile comes from the leftmost of the two
                        prev_orig = self._unrotate_position(r, prev_col, -k)
                        curr_orig = self._unrotate_position(r, curr_col, -k)
                        movements[curr_orig] = prev_orig
                        prev_i += 2  # Skip both merged tiles
                        curr_i += 1
                        continue
                
                # Normal movement (no merge)
                if prev_val == curr_val:
                    # Same tile, just moved
                    prev_orig = self._unrotate_position(r, prev_col, -k)
                    curr_orig = self._unrotate_position(r, curr_col, -k)
                    movements[curr_orig] = prev_orig
                    prev_i += 1
                    curr_i += 1
                elif curr_val > prev_val:
                    # This might be a merge from a different pair, or new tile
                    # Try to match with next prev tile
                    if prev_i + 1 < len(prev_tiles):
                        prev_i += 1
                    else:
                        curr_i += 1
                else:
                    # New tile appeared (shouldn't happen in normal gameplay)
                    curr_i += 1
        
        return movements
    
    def _unrotate_position(self, r, c, k):
        """Rotate a position back to original orientation."""
        if k == 0:
            return (r, c)
        elif k == 1:  # 90 deg counter-clockwise
            return (c, GRID_SIZE - 1 - r)
        elif k == 2:  # 180 deg
            return (GRID_SIZE - 1 - r, GRID_SIZE - 1 - c)
        elif k == 3:  # 90 deg clockwise
            return (GRID_SIZE - 1 - c, r)
        return (r, c)
    
    def draw_grid(self):
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw Score
        score_text = self.score_font.render(f"Score: {self.game.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, HEIGHT - 40))

        grid = self.game.grid
        current_time = time.time()
        
        # Calculate animation progress
        if self.animating:
            elapsed = current_time - self.animation_start_time
            if elapsed >= ANIMATION_DURATION:
                self.animating = False
                self.previous_grid = None
                animation_progress = 1.0
                # Start pop animation for new tile when slide completes
                if self.game.last_spawned_tile is not None:
                    if self.game.last_spawned_tile not in self.new_tile_positions:
                        self.new_tile_positions[self.game.last_spawned_tile] = current_time
            else:
                animation_progress = self._ease_out_cubic(elapsed / ANIMATION_DURATION)
        else:
            animation_progress = 1.0
        
        # Calculate tile movements if animating
        movements = {}
        if self.animating and self.previous_grid is not None and self.current_action is not None:
            movements = self._calculate_tile_movement(self.previous_grid, grid, self.current_action)
        
        # Draw empty tiles first
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                rect_x, rect_y = self._get_tile_position(r, c)
                rect = pygame.Rect(rect_x, rect_y, TILE_SIZE - PADDING, TILE_SIZE - PADDING)
                pygame.draw.rect(self.screen, EMPTY_TILE_COLOR, rect, border_radius=5)
        
        # Draw tiles with animation
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                value = grid[r][c]
                if value == 0:
                    continue
                
                color = TILE_COLORS.get(value, (60, 58, 50))
                
                # Calculate position with animation
                target_x, target_y = self._get_tile_position(r, c)
                
                if (r, c) in movements and animation_progress < 1.0:
                    # Tile is moving
                    old_r, old_c = movements[(r, c)]
                    start_x, start_y = self._get_tile_position(old_r, old_c)
                    rect_x = start_x + (target_x - start_x) * animation_progress
                    rect_y = start_y + (target_y - start_y) * animation_progress
                else:
                    rect_x, rect_y = target_x, target_y
                
                # Pop-out animation for new tiles (only once)
                scale = 1.0
                if (r, c) in self.new_tile_positions:
                    pop_start = self.new_tile_positions[(r, c)]
                    pop_elapsed = current_time - pop_start
                    if pop_elapsed < POP_ANIMATION_DURATION:
                        pop_progress = pop_elapsed / POP_ANIMATION_DURATION
                        # Scale from 0 to 1.2 to 1.0 (pop out then settle)
                        if pop_progress < 0.5:
                            scale = pop_progress * 2 * 1.2
                        else:
                            scale = 1.2 - (pop_progress - 0.5) * 2 * 0.2
                    else:
                        # Animation complete, remove from tracking so it stays at normal size
                        del self.new_tile_positions[(r, c)]
                
                # Draw tile with scale
                tile_width = int((TILE_SIZE - PADDING) * scale)
                tile_height = int((TILE_SIZE - PADDING) * scale)
                offset_x = (TILE_SIZE - PADDING - tile_width) // 2
                offset_y = (TILE_SIZE - PADDING - tile_height) // 2
                
                rect = pygame.Rect(
                    rect_x + offset_x,
                    rect_y + offset_y,
                    tile_width,
                    tile_height
                )
                
                pygame.draw.rect(self.screen, color, rect, border_radius=5)

                if value != 0:
                    text_color = FONT_COLOR if value < 8 else (249, 246, 242)
                    text_surface = self.font.render(str(value), True, text_color)
                    text_rect = text_surface.get_rect(center=rect.center)
                    self.screen.blit(text_surface, text_rect)
        
        if self.game.game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 128))
            self.screen.blit(overlay, (0, 0))
            final_score_text = self.score_font.render(f"Final Score: {self.game.score}", True, (0, 0, 0))
            final_score_rect = final_score_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
            self.screen.blit(final_score_text, final_score_rect)
            game_over_text = self.font.render("Game Over!", True, (0, 0, 0))
            text_rect = game_over_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            self.screen.blit(game_over_text, text_rect)
            restart_text = self.score_font.render("Press 'R' to Restart", True, (50, 50, 50))
            restart_rect = restart_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
            self.screen.blit(restart_text, restart_rect)

        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            clock.tick(60) # Higher FPS for smoother animations
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.game.reset()
                        self.animating = False
                        self.previous_grid = None
                        self.new_tile_positions = {}
                    
                    if not self.game.game_over and not self.animating:
                        action = None
                        if event.key == pygame.K_UP:
                            action = 0
                        elif event.key == pygame.K_DOWN:
                            action = 1
                        elif event.key == pygame.K_LEFT:
                            action = 2
                        elif event.key == pygame.K_RIGHT:
                            action = 3
                        
                        if action is not None:
                            # Store previous grid for animation
                            self.previous_grid = self.game.grid.copy()
                            self.current_action = action
                            
                            # Execute Step (RL Friendly Structure)
                            _, _, done, moved = self.game.step(action)
                            
                            if moved:
                                # Start animation
                                self.animating = True
                                self.animation_start_time = time.time()
                                # Don't clear new_tile_positions - let them complete their animations
                            
                            if done:
                                print("Game Over Reached")

                    if event.key == pygame.K_a:
                        self.ai_mode = not self.ai_mode
                        print(f"AI Mode: {'ON' if self.ai_mode else 'OFF'}")

            # AI Logic
            if self.ai_mode and not self.game.game_over and not self.animating:
                # Add a small delay or check allows UI to breathe slightly? 
                # Plan says: "At each solver step (when UI is idle and not animating)"
                # Since we are in the main loop, this runs once per frame if conditions met.
                
                # We need to make sure we don't block the event loop for too long.
                # But implementation plan says "compute best_move = expectimax_decision..."
                
                best_move = expectimax_decision(self.game.grid)
                if best_move is not None:
                     self.previous_grid = self.game.grid.copy()
                     self.current_action = best_move
                     
                     _, _, done, moved = self.game.step(best_move)
                     
                     if moved:
                         self.animating = True
                         self.animation_start_time = time.time()
                else:
                    # No moves possible?
                    pass

            self.draw_grid()

        pygame.quit()

if __name__ == "__main__":
    logic = Game2048()
    ui = Pygame2048(logic)
    ui.run()
