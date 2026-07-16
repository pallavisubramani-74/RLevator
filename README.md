RLevator
A reinforcement learning agent that learns to control an elevator — dispatching it intelligently to pick up and drop off passengers more efficiently than a standard rule-based controller.
<p align="center">
  <img alt="RLevator" src="images/RLevator.gif" width="40%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Karps" src="images/Karps.gif" width="40%">
</p>
This started as our final project for CS 229 (Fall 2022), built by Tejas Narayanan and Kiran Bhat. Read the paper here.
Why Elevators?
Elevator dispatching sounds simple, but it's actually a surprisingly rich control problem: you're juggling multiple floors, unpredictable passenger arrivals, and competing goals (minimize wait time vs. minimize travel time vs. handle multiple riders at once). Most real-world elevators still run on hand-tuned heuristics. We wanted to see whether a reinforcement learning agent could learn a better policy from scratch — and whether it could generalize as the building got taller and more complex.
How It Works

Environments (envs/): A series of custom Gym-style elevator simulations, iterated from v1 through v7, each adding complexity — more floors, more passengers, more realistic dynamics.
Agents (agents/): Standard, rule-based elevator controllers used as a baseline to benchmark the RL agent against.
Training (train_elevator_agent.py): Trains a Stable-Baselines3 agent on the elevator environment, with optional curriculum learning that gradually increases the number of floors as training progresses.
Benchmarking (benchmark_agents.py): Runs trained agents (and the standard controller) through the same environment and compares their performance using mean and standard deviation of episodic rewards.
Visualization (elevator_animation.py, results_plotter.py): Tools to animate an agent's elevator behavior and plot training/benchmark results.

Installation
RLevator was developed using Python 3.10.
