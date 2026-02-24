import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dqn_agent import DQNAgent

print("Script started")

MODEL_SAVE_PATH = "dqn_pretrained.pth"
NUM_EPOCHS = 15
BATCH_SIZE = 256
LR = 5e-4


def preprocess_state(grid):
    state = np.zeros_like(grid, dtype=np.float32)
    mask = grid > 0
    state[mask] = np.log2(grid[mask])
    return state.flatten()


def load_dataset():
    with open("all_500_game_results.pkl", "rb") as f:
        data = pickle.load(f)

    states = []
    actions = []

    for game in data:
        for board, move in game:
            if move is None:
                continue

            states.append(preprocess_state(board))
            actions.append(move)

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int64)

    states = torch.from_numpy(states)
    actions = torch.from_numpy(actions)

    return states, actions


def pretrain():
    print("Entered pretrain")

    agent = DQNAgent()
    agent.q_network.train()

    states, actions = load_dataset()
    states = states.to(agent.device)
    actions = actions.to(agent.device)

    dataset_size = states.shape[0]
    print("Total training samples:", dataset_size)

    optimizer = optim.Adam(agent.q_network.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        permutation = torch.randperm(dataset_size)
        total_loss = 0

        for i in range(0, dataset_size, BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_states = states[indices]
            batch_actions = actions[indices]

            logits = agent.q_network(batch_states)
            loss = loss_fn(logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {total_loss:.4f}")

    torch.save(agent.q_network.state_dict(), MODEL_SAVE_PATH)
    print("Pretrained model saved")


if __name__ == "__main__":
    print("Calling pretrain")
    pretrain()