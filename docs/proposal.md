---
layout: default
title: Proposal
---

## Summary
We propose to train a reinforcement learning (RL) AI Agent to play the puzzle game 2048. The environment is a standard 2048 board, which takes place on a 4x4 grid, where tiles of the same value are combined to form a higher value tile. At each step, the agent observes the current grid of the tile as inputs and chooses 4 state spaces (up, down, left, right) as output. Our goal is for the agent to learn a policy that maximizes long term game score and consistently reaches high value tiles. 

## AI/ML Algorithms
We are anticipating using the Expectimax algorithm with different heuristics to create an agent capable of winning 2048.

## Evaluation Plan
Our primary metrics for evaluating the success of our agent will be the score in the game (sum of all tiles in the game) and the highest tile reached. We will start with a baseline of an agent that randomly uses the 4 control options, which in our initial testing, frequently reached a max tile of 128. Our minimum goal for finishing the project would be an agent capable of reaching a max tile of 512, with an ideal goal being an agent capable of winning the game (reaching 2048). Our moonshot goal would be 4096. To truly evaluate the agent, we can run the game multiple times with a particular agent, and average out the max tile reached/max score reached. 

We may experiment with increasing the size of the gameboard to be larger (perhaps 6x6) and evaluating hypotheses about how that would affect the agent. 

## Meeting with Instructor
We met with Dr. Fox on January 21st at 4:00 PM, where he mentioned that a previous team last year had created a 2048 agent, and emphasized that we should focus on differentiating our approach from theirs.

## AI Usage
We utilized ChatGPT to help brainstorm initial algorithms and ideas for this project.
