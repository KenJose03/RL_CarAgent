# 2D RL Car Simulator with LiDAR and Reward Visualization

This project is a reinforcement learning (RL) environment and agent for training a car to stay within road lanes using LiDAR-like sensors. The environment is visualized using Pygame, and the agent is based on Deep Q-Learning (DQN).

## Features
- 2D car simulation with lane and road edge detection
- LiDAR sensor simulation for state input
- Reward visualization (background color changes)
- DQN agent with model saving/loading
- Adjustable simulation speed

## Files
- `main.py`: Main training and simulation loop
- `dqn_agent.py`: DQN agent implementation
- `convert_weights.py`: Utility for model weights
- `dqn_agent_episode_*.pth`, `dqn_agent_final.pth`: Saved model weights

## Requirements
- Python 3.8+
- Pygame
- NumPy
- (Optional) PyTorch (if used in `dqn_agent.py`)

Install dependencies:
```bash
pip install pygame numpy torch
```

## Usage
Run the simulator and training loop:
```bash
python main.py
```

You will be prompted to select the simulation speed mode.

## License
MIT License
