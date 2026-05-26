# Autonomous Drone Racing with RL and Sim-to-Real Transfer

This repository contains the codebase for training and evaluating end-to-end Reinforcement Learning (RL) policies for a quadrotor drone navigating an agile, 7-gate racing track. Built on top of **NVIDIA Isaac Lab** (Isaac Sim), the project incorporates a custom implementation of Proximal Policy Optimization (PPO) tailored to handle complex 3D aerodynamic constraints and gate-traversal local optima.

📺 **[Watch the Autonomous Quadcopter Race Simulation Video here!](https://drive.google.com/file/d/1Do3HnfU8Fb_0Kvx9uo5ykoU6fZubQy55/view)**

---

## Key Accomplishments & Features
* **Custom PPO Architecture:** Built an optimized PPO pipeline within a decoupled `rsl_rl` framework, scaling training workloads to 8,192 parallel environments.
* **Gate-Centric Progress Reward:** Designed a vertical-progress and orientation-aligned reward function to overcome early altitude local optima, enabling robust, continuous multi-gate traversal on tracks with obstacles up to 2.0 meters.
* **Robust Sim-to-Real Pipeline:** Integrated extensive domain randomization over aerodynamic drag coefficients, variable PID gains, and thrust-to-weight ratios to bridge the reality gap.
* **Telemetry Diagnostics:** Analyzed real-world flight ROS2 bag data to isolate and debug perception-control latency mismatches and high-frequency policy jitter.

---

## Repository Structure

├── logs/rsl_rl/quadcopter_direct/   # Saved training checkpoints and tensorboard logs
├── scripts/
│   └── rsl_rl/
│       ├── train_race.py            # Main entry point for parallel policy training
│       └── play_race.py             # Evaluation and video rendering script
├── src/
│   └── rsl_rl_custom/               # Custom core PPO implementation and policy networks
└── README.md

## Training Script
python scripts/rsl_rl/train_race.py \
    --task Isaac-Quadcopter-Race-v0 \
    --num_envs 8192 \
    --max_iterations 5000 \
    --headless
