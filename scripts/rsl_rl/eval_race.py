# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation script: runs policy N trials with domain-randomized dynamics,
reports completion rate, average lap time, and per-trial statistics."""

"""Launch Isaac Sim Simulator first."""

import sys
import os

local_rsl_path = os.path.abspath("src/third_parties/rsl_rl_local")
if os.path.exists(local_rsl_path):
    sys.path.insert(0, local_rsl_path)

import argparse
from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description="Evaluate a trained RL policy under domain randomization.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments (= number of trials).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--max_steps", type=int, default=3000, help="Max steps per trial (default 3000 = 60s at 50Hz).")
parser.add_argument("--num_laps", type=int, default=3, help="Number of laps to complete.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import src.isaac_quad_sim2real.tasks  # noqa: F401


def main():
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True
    )

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    env_cfg.is_train = False
    env_cfg.episode_length_s = args_cli.max_steps * env_cfg.decimation * env_cfg.sim.dt + 1.0
    env_cfg.max_n_laps = args_cli.num_laps
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    raw_env = env.unwrapped
    num_envs = raw_env.num_envs
    n_gates = raw_env._waypoints.shape[0]
    gates_for_laps = n_gates * args_cli.num_laps

    # --- Domain randomization (matching evaluation ranges from handout) ---
    cfg = raw_env.cfg
    strategy = raw_env.strategy
    strategy._randomize_dynamics(torch.arange(num_envs, device=raw_env.device))

    # --- Reposition all drones with TA-spec spawn (ground level, randomized position) ---
    dev = raw_env.device
    all_ids = torch.arange(num_envs, device=dev)

    gate0_pos = raw_env._waypoints[0]           # (6,) — x, y, z, roll, pitch, yaw
    x0_wp = gate0_pos[0]
    y0_wp = gate0_pos[1]
    theta = gate0_pos[-1]                        # gate heading (yaw)

    # TA-spec spawn: x_local in [-3.0, -0.5], y_local in [-1.0, 1.0], z = 0.05
    x_local = torch.empty(num_envs, device=dev).uniform_(-3.0, -0.5)
    y_local = torch.empty(num_envs, device=dev).uniform_(-1.0, 1.0)

    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    x_rot = cos_t * x_local - sin_t * y_local
    y_rot = sin_t * x_local + cos_t * y_local
    initial_x = x0_wp - x_rot
    initial_y = y0_wp - y_rot

    # Point drone toward gate
    initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
    quat = quat_from_euler_xyz(
        torch.zeros(num_envs, device=dev),
        torch.zeros(num_envs, device=dev),
        initial_yaw,
    )

    root_state = raw_env._robot.data.default_root_state[all_ids].clone()
    root_state[:, 0] = initial_x
    root_state[:, 1] = initial_y
    root_state[:, 2] = 0.05  # ground level
    root_state[:, 3:7] = quat
    root_state[:, 7:] = 0.0  # zero velocity

    raw_env._robot.write_root_link_pose_to_sim(root_state[:, :7], all_ids)
    raw_env._robot.write_root_com_velocity_to_sim(root_state[:, 7:], all_ids)

    # Reset env bookkeeping to gate 0
    raw_env._idx_wp[all_ids] = 0
    raw_env._n_gates_passed[all_ids] = 0
    raw_env._actions[all_ids] = 0.0
    raw_env._previous_actions[all_ids] = 0.0
    raw_env.episode_length_buf[all_ids] = 0

    print(f"[EVAL] Spawn: TA-spec, ground level (z=0.05), x_local~U(-3,-0.5), y_local~U(-1,1)")

    # Tracking arrays — each env = one trial, first episode outcome only
    trial_done = torch.zeros(num_envs, dtype=torch.bool, device=dev)  # frozen once first episode ends
    outcome = torch.zeros(num_envs, dtype=torch.long, device=dev)     # 0=pending, 1=completed, 2=crashed, 3=timeout
    finish_step = torch.full((num_envs,), args_cli.max_steps, dtype=torch.long, device=dev)
    gates_passed = torch.zeros(num_envs, dtype=torch.long, device=dev)
    # Track peak gates per env (before any reset wipes the counter)
    peak_gates = torch.zeros(num_envs, dtype=torch.long, device=dev)

    # Get initial observations
    obs = env.get_observations()
    if hasattr(obs, "get"):
        obs = obs["policy"]

    dt_per_step = env_cfg.decimation * env_cfg.sim.dt

    print(f"\n[EVAL] Running {num_envs} trials | {args_cli.num_laps} laps | max {args_cli.max_steps} steps ({args_cli.max_steps * dt_per_step:.1f}s)")
    print(f"[EVAL] Domain randomization: ON (handout ranges)")
    print("-" * 60)

    for step in range(args_cli.max_steps):
        # Snapshot gates BEFORE step (in case step triggers a reset that zeros the counter)
        pre_step_gates = raw_env._n_gates_passed.long().clone()
        active = ~trial_done
        peak_gates[active] = torch.max(peak_gates[active], pre_step_gates[active])

        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)
            if hasattr(obs, "get"):
                obs = obs["policy"]

        # Check for completion BEFORE checking crashes (use pre-step snapshot)
        newly_completed = active & (pre_step_gates >= gates_for_laps)
        if newly_completed.any():
            finish_step[newly_completed] = step
            gates_passed[newly_completed] = pre_step_gates[newly_completed]
            outcome[newly_completed] = 1
            trial_done[newly_completed] = True
            active = ~trial_done

        # Check for env termination (crash or timeout) — only for still-active envs
        dones_bool = dones.bool().squeeze()
        terminated = raw_env.reset_terminated.squeeze()
        timed_out = raw_env.reset_time_outs.squeeze()

        newly_crashed = active & dones_bool & terminated
        if newly_crashed.any():
            gates_passed[newly_crashed] = peak_gates[newly_crashed]
            outcome[newly_crashed] = 2
            trial_done[newly_crashed] = True
            active = ~trial_done

        newly_timed_out = active & dones_bool & timed_out & ~terminated
        if newly_timed_out.any():
            gates_passed[newly_timed_out] = peak_gates[newly_timed_out]
            outcome[newly_timed_out] = 3
            trial_done[newly_timed_out] = True
            active = ~trial_done

        # Early exit if all envs are done
        if trial_done.all():
            print(f"[EVAL] All trials completed at step {step}")
            break

    # Mark any still-active envs as timed out
    still_active = ~trial_done
    if still_active.any():
        gates_passed[still_active] = peak_gates[still_active]
        outcome[still_active] = 3
        trial_done[still_active] = True

    # --- Results ---
    gates_passed_np = gates_passed.cpu().numpy()
    outcome_np = outcome.cpu().numpy()
    finish_step_np = finish_step.cpu().numpy()

    n_completed = (outcome_np == 1).sum()
    n_crashed = (outcome_np == 2).sum()
    n_timeout = (outcome_np == 3).sum()

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Trials:          {num_envs}")
    print(f"Completed:       {n_completed}/{num_envs} ({100*n_completed/num_envs:.1f}%)")
    print(f"Crashed:         {n_crashed}/{num_envs} ({100*n_crashed/num_envs:.1f}%)")
    print(f"Timed out:       {n_timeout}/{num_envs} ({100*n_timeout/num_envs:.1f}%)")
    print(f"Gates/trial avg: {gates_passed_np.mean():.1f} / {gates_for_laps}")
    print("-" * 60)

    completed_mask = (outcome_np == 1)
    if n_completed > 0:
        completion_times = finish_step_np[completed_mask] * dt_per_step
        print(f"Completion time (s):")
        print(f"  Mean:   {completion_times.mean():.2f}")
        print(f"  Median: {np.median(completion_times):.2f}")
        print(f"  Min:    {completion_times.min():.2f}")
        print(f"  Max:    {completion_times.max():.2f}")
        print(f"  Std:    {completion_times.std():.2f}")
    else:
        print("No trials completed 3 laps.")

    crashed_mask = (outcome_np == 2)
    if n_crashed > 0:
        crash_gates = gates_passed_np[crashed_mask]
        print(f"\nCrash statistics:")
        print(f"  Avg gates before crash: {crash_gates.mean():.1f}")
        crash_gate_mod = crash_gates % n_gates
        for g in range(n_gates):
            count = (crash_gate_mod == g).sum()
            if count > 0:
                print(f"  Crashes approaching gate {g}: {count}")

    print("=" * 60)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
