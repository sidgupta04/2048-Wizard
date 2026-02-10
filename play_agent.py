import pygame
import time
import numpy as np
import torch

from dqn_agent import DQNAgent
from game_2048 import Game2048, Pygame2048

MODEL_PATH = "dqn_2048.pth"
MOVE_DELAY = 0.2  

def preprocess_state(grid):
    state = np.zeros_like(grid, dtype=float)
    mask = grid > 0
    state[mask] = np.log2(grid[mask])
    return state.flatten()


def play():
    agent = DQNAgent()
    agent.q_network.load_state_dict(
        torch.load(MODEL_PATH, map_location=agent.device)
    )
    agent.q_network.eval()
    agent.epsilon = 0.0

    game = Game2048()
    ui = Pygame2048(game)

    clock = pygame.time.Clock()
    running = True
    state = preprocess_state(game.grid)

    while running:
        clock.tick(60)

        for event in ui.game_events():
            if event == "QUIT":
                running = False
            elif event == "RESTART":
                game.reset()
                ui.animating = False
                ui.previous_grid = None
                ui.new_tile_positions = {}
                state = preprocess_state(game.grid)

        if ui.animating:
            ui.draw_grid()
            continue

        if game.game_over:
            ui.draw_grid()
            continue

        with torch.no_grad():
            q_values = agent.q_network(
                torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            ).squeeze(0)

        actions = torch.argsort(q_values, descending=True).tolist()

        moved = False
        for action in actions:
            prev_grid = game.grid.copy()
            next_grid, _, _, moved = game.step(action)
            if moved:
                ui.previous_grid = prev_grid
                ui.current_action = action
                ui.animating = True
                ui.animation_start_time = time.time()
                state = preprocess_state(next_grid)
                break

        ui.draw_grid()
        time.sleep(MOVE_DELAY)

    ui.quit()

if __name__ == "__main__":
    play()