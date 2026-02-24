---
layout: default
title: Status
---

## Summary
We will begin by implementing an Expectimax-based agent for 2048 and use it as an expert to generate high-quality gameplay trajectories. Using these expert state–action pairs, we will train a simple neural network through imitation learning and then refine this model using a Deep Q-Network (DQN) to incorporate long-term reward feedback. This hybrid Expectimax→DQN pipeline allows us to create a fast learned policy that initially imitates expert play but can potentially outperform it through reinforcement learning. Our approach also enables us to study how different game conditions and board structures influence both planning-based and DQN-based decision-making.
