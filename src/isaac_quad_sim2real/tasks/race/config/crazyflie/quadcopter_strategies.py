# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = [key.split("_reward_scale")[0] for key in env.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        # Domain randomization ranges (matching evaluation conditions from handout)
        self._dr_ranges = {
            'twr':          (self.cfg.thrust_to_weight * 0.95, self.cfg.thrust_to_weight * 1.05),
            'k_aero_xy':    (self.cfg.k_aero_xy * 0.5,        self.cfg.k_aero_xy * 2.0),
            'k_aero_z':     (self.cfg.k_aero_z * 0.5,         self.cfg.k_aero_z * 2.0),
            'kp_omega_rp':  (self.cfg.kp_omega_rp * 0.85,     self.cfg.kp_omega_rp * 1.15),
            'ki_omega_rp':  (self.cfg.ki_omega_rp * 0.85,     self.cfg.ki_omega_rp * 1.15),
            'kd_omega_rp':  (self.cfg.kd_omega_rp * 0.7,      self.cfg.kd_omega_rp * 1.3),
            'kp_omega_y':   (self.cfg.kp_omega_y * 0.85,      self.cfg.kp_omega_y * 1.15),
            'ki_omega_y':   (self.cfg.ki_omega_y * 0.85,      self.cfg.ki_omega_y * 1.15),
            'kd_omega_y':   (self.cfg.kd_omega_y * 0.7,       self.cfg.kd_omega_y * 1.3),
        }

        # Apply initial domain randomization across all envs
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._randomize_dynamics(all_ids)

        # Motor time constants (not randomized in evaluation)
        self.env._tau_m[:] = self.env._tau_m_value

    def _randomize_dynamics(self, env_ids: torch.Tensor):
        """Randomize physical parameters for the given environment indices."""
        n = len(env_ids)
        dr = self._dr_ranges

        # Thrust to weight ratio
        self.env._thrust_to_weight[env_ids] = torch.empty(n, device=self.device).uniform_(*dr['twr'])

        # Aerodynamic drag coefficients
        self.env._K_aero[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*dr['k_aero_xy'])
        self.env._K_aero[env_ids, 1] = self.env._K_aero[env_ids, 0]  # same for x and y
        self.env._K_aero[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*dr['k_aero_z'])

        # PID gains — roll and pitch
        self.env._kp_omega[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*dr['kp_omega_rp'])
        self.env._kp_omega[env_ids, 1] = self.env._kp_omega[env_ids, 0]
        self.env._ki_omega[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*dr['ki_omega_rp'])
        self.env._ki_omega[env_ids, 1] = self.env._ki_omega[env_ids, 0]
        self.env._kd_omega[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*dr['kd_omega_rp'])
        self.env._kd_omega[env_ids, 1] = self.env._kd_omega[env_ids, 0]

        # PID gains — yaw
        self.env._kp_omega[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*dr['kp_omega_y'])
        self.env._ki_omega[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*dr['ki_omega_y'])
        self.env._kd_omega[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*dr['kd_omega_y'])

    def get_rewards(self) -> torch.Tensor:
       

        # gate crossing detection logic
        x_gate_now = self.env._pose_drone_wrt_gate[:, 0]
        gate_half = self.env.cfg.gate_model.gate_side / 2.0
        dist_to_gate = torch.linalg.norm(self.env._pose_drone_wrt_gate, dim=1)

        gate_crossed = (self.env._prev_x_drone_wrt_gate > 0) & (x_gate_now <= 0)
        within_gate = (
            (torch.abs(self.env._pose_drone_wrt_gate[:, 1]) < gate_half * 1.2) &
            (torch.abs(self.env._pose_drone_wrt_gate[:, 2]) < gate_half * 1.2)
        )
        near_gate = dist_to_gate < 2.0
        gate_passed = gate_crossed & within_gate & near_gate

        # advance waypoint
        ids_gate_passed = torch.where(gate_passed)[0]
        self.env._n_gates_passed[ids_gate_passed] += 1
        self.env._idx_wp[ids_gate_passed] = (
            self.env._idx_wp[ids_gate_passed] + 1
        ) % self.env._waypoints.shape[0]

        # refresh gate-relative pose for envs that passed
        if len(ids_gate_passed) > 0:
            self.env._desired_pos_w[ids_gate_passed, :3] = self.env._waypoints[
                self.env._idx_wp[ids_gate_passed], :3
            ]
            self.env._pose_drone_wrt_gate[ids_gate_passed], _ = subtract_frame_transforms(
                self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3],
                self.env._waypoints_quat[self.env._idx_wp[ids_gate_passed], :],
                self.env._robot.data.root_link_pos_w[ids_gate_passed],
            )
            self.env._last_distance_to_goal[ids_gate_passed] = torch.linalg.norm(
                self.env._pose_drone_wrt_gate[ids_gate_passed], dim=1
            )

        # Update prev_x
        self.env._prev_x_drone_wrt_gate = x_gate_now.clone()
        if len(ids_gate_passed) > 0:
            self.env._prev_x_drone_wrt_gate[ids_gate_passed] = self.env._pose_drone_wrt_gate[ids_gate_passed, 0]

        # Recompute distance after potential advancement
        dist_to_gate = torch.linalg.norm(self.env._pose_drone_wrt_gate, dim=1)

        # crash detection
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask

        # reward terms
        drone_pos_w = self.env._robot.data.root_link_pos_w
        drone_vel_w = self.env._robot.data.root_com_lin_vel_w
        curr_gate_pos = self.env._waypoints[self.env._idx_wp, :3]

        # 1. GATE PASS (sparse, main objective)
        gate_pass_reward = gate_passed.float()

        # 2. VEL TOWARD GATE (anti-hover, directional)
        dir_to_gate = curr_gate_pos - drone_pos_w
        dir_to_gate_norm = dir_to_gate / (torch.linalg.norm(dir_to_gate, dim=1, keepdim=True) + 1e-6)
        vel_toward = torch.sum(drone_vel_w * dir_to_gate_norm, dim=1)
        vel_toward_reward = torch.tanh(vel_toward)

        # 3. PROGRESS (distance reduction)
        progress_reward = (self.env._last_distance_to_goal - dist_to_gate).clamp(min=-0.1, max=0.5)
        self.env._last_distance_to_goal = dist_to_gate.clone()

        # 4. CENTERING (lateral offset penalty, gated by proximity)
        lateral_offset = torch.linalg.norm(self.env._pose_drone_wrt_gate[:, 1:3], dim=1)
        proximity_weight = torch.exp(-dist_to_gate)
        centering_penalty = -lateral_offset * proximity_weight

        # 5. ESCAPE (post-gate-3: strong push toward gate 4)
        is_post_gate3 = (self.env._idx_wp == 4).float()
        gate4_pos = self.env._waypoints[4, :3]
        dir_to_gate4 = gate4_pos - drone_pos_w
        dir_to_gate4_norm = dir_to_gate4 / (torch.linalg.norm(dir_to_gate4, dim=1, keepdim=True) + 1e-6)
        vel_to_gate4 = torch.sum(drone_vel_w * dir_to_gate4_norm, dim=1)
        escape_reward = torch.tanh(vel_to_gate4) * is_post_gate3

        # 6. CRASH (penalty)
        crash_penalty = crashed.float()

        # aggregate rewards with scales from config
        if self.cfg.is_train:
            rewards = {
                "gate_pass":  gate_pass_reward * self.env.rew['gate_pass_reward_scale'],
                "vel_toward": vel_toward_reward * self.env.rew['vel_toward_reward_scale'],
                "progress":   progress_reward * self.env.rew['progress_reward_scale'],
                "centering":  centering_penalty * self.env.rew['centering_reward_scale'],
                "escape":     escape_reward * self.env.rew['escape_reward_scale'],
                "crash":      crash_penalty * self.env.rew['crash_reward_scale'],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(
                self.env.reset_terminated,
                torch.ones_like(reward) * self.env.rew['death_cost'],
                reward,
            )

            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:
            reward = torch.zeros(self.num_envs, device=self.device)

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations. Read reset_idx() and quadcopter_env.py to see which drone info is extracted from the sim.
        The following code is an example. You should delete it or heavily modify it once you begin the racing task."""

        # Drone state
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b   # (N, 3) velocity in body frame
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b        # (N, 3) body rates
        drone_quat_w    = self.env._robot.data.root_quat_w           # (N, 4) orientation

        # Current gate: drone position in gate frame (already computed in _get_dones)
        drone_pos_curr_gate = self.env._pose_drone_wrt_gate          # (N, 3)

        # Next gate: look-ahead so the policy can plan trajectories through gates
        next_wp_idx = (self.env._idx_wp + 1) % self.env._waypoints.shape[0]
        drone_pos_next_gate, _ = subtract_frame_transforms(
            self.env._waypoints[next_wp_idx, :3],
            self.env._waypoints_quat[next_wp_idx, :],
            self.env._robot.data.root_link_pos_w,
        )                                                             # (N, 3)

        # Previous actions for temporal smoothness awareness
        prev_actions = self.env._previous_actions                    # (N, 4)

        obs = torch.cat(
            [
                drone_lin_vel_b,      # 3  — velocity in body frame
                drone_ang_vel_b,      # 3  — body rates
                drone_quat_w,         # 4  — orientation
                drone_pos_curr_gate,  # 3  — position relative to current gate
                drone_pos_next_gate,  # 3  — position relative to next gate (look-ahead)
                prev_actions,         # 4  — previous actions
            ],                        # total: 20 dims
            dim=-1,
        )
        observations = {"policy": obs}

        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # Curriculum reset: 50% gate 0, 30% gate 2 or 3 (power loop), 20% random other
        if self.cfg.is_train:
            rand = torch.rand(n_reset, device=self.device)
            gate0 = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)
            gate1 = torch.ones(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)
            # Gates 2 and 3 are the power loop — randomly pick one of them
            power_loop_gates = torch.randint(2, 4, (n_reset,), device=self.device, dtype=self.env._idx_wp.dtype)
            # Random gate from the full set for general robustness
            n_gates = self.env._waypoints.shape[0]
            random_gates = torch.randint(0, n_gates, (n_reset,), device=self.device, dtype=self.env._idx_wp.dtype)
            # 40% gate 0, 10% gate 1, 30% power loop (gates 2-3), 20% random
            waypoint_indices = torch.where(rand < 0.4, gate0,
                              torch.where(rand < 0.5, gate1,
                              torch.where(rand < 0.8, power_loop_gates, random_gates)))

            # Domain randomization: re-randomize dynamics for reset envs
            self._randomize_dynamics(env_ids)
        else:
            waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)

        # get starting pose behind gate in approach direction
        x0_wp = self.env._waypoints[waypoint_indices][:, 0]
        y0_wp = self.env._waypoints[waypoint_indices][:, 1]
        theta  = self.env._waypoints[waypoint_indices][:, -1]
        z_wp   = self.env._waypoints[waypoint_indices][:, 2]

        # Gate-0 starts: ground level with TA-spec position uncertainty
        # Mid-track starts: at gate altitude with tighter noise
        is_gate0 = (waypoint_indices == 0)
        x_local = torch.where(is_gate0,
            torch.empty(n_reset, device=self.device).uniform_(-3.0, -0.5),
            -2.0 * torch.ones(n_reset, device=self.device))
        y_local = torch.where(is_gate0,
            torch.empty(n_reset, device=self.device).uniform_(-1.0, 1.0),
            torch.empty(n_reset, device=self.device).uniform_(-0.4, 0.4))

        # rotate local offset into world frame
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local
        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        # Gate-0: ground level (0.05); mid-track: gate altitude ± noise
        z_local = torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.2)
        initial_z = torch.where(is_gate0,
            0.05 * torch.ones(n_reset, device=self.device),
            z_local + z_wp)

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        # point drone towards the zeroth gate with small yaw noise
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        quat = quat_from_euler_xyz(
            torch.zeros(n_reset, device=self.device),
            torch.zeros(n_reset, device=self.device),
            initial_yaw + torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.2),
        )
        default_root_state[:, 3:7] = quat

        # Handle play mode initial position
        if not self.cfg.is_train:
            # x_local and y_local are randomly sampled
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # rotate local pos to global frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            # point drone towards the zeroth gate
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices = self.env._initial_wp

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = waypoint_indices

        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._pose_drone_wrt_gate[env_ids], dim=1
        )
        self.env._n_gates_passed[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        self.env._prev_x_drone_wrt_gate[env_ids] = 1.0

        self.env._crashed[env_ids] = 0