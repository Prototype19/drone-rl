# Copyright (c) 2026, drone-rl project. BSD-3-Clause.
#
# Phase 3 custom hover env. This is a faithful copy of the stock Isaac Lab
# quadcopter env (isaaclab_tasks/direct/quadcopter/quadcopter_env.py) with two
# deliberate changes, both noted inline with `# HOVER:` comments:
#   1. The per-episode RANDOM goal is replaced by a FIXED hover point
#      (env-origin xy, height = cfg.hover_height). This turns the stock
#      "reach-and-hold-a-random-point" task into a stationary hover.
#   2. The GUI debug-window class is dropped -- the Spark runs headless-only, so
#      it was dead code. The goal-marker debug vis is kept (it renders into the
#      recorded play video).
# Observation (12-D ego-centric), action (1 thrust + 3 moments), reward (3 terms)
# and all scales are IDENTICAL to the stock env. Per CLAUDE.md rule #5 the reward
# locks after this phase.
#
# M4 (domain randomization #1) adds per-env, config-gated randomization of init
# state, mass, and "motor" strength (a thrust gain + constant moment bias -- the
# wrench-level analog of uneven motors in this collective-thrust env). All three
# toggles default to False; with them off this env is byte-for-byte the M3 hover
# env. The reward is untouched (still locked). Per SPEC M4 / CLAUDE.md rule #6 the
# toggles are enabled ONE AT A TIME across separate retrains.

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


@configclass
class CrazyflieHoverEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    action_space = 4
    observation_space = 12
    state_space = 0
    debug_vis = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # HOVER: fixed hover target height (m) above each env origin. The stock env
    # sampled goal z in U(0.5, 1.5) and goal xy in U(-2, 2); we hold a fixed point.
    hover_height = 1.0

    # reward scales (identical to stock env)
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0

    # --- M4 domain randomization (#1) ---------------------------------------
    # Enabled ONE AT A TIME across separate retrains (SPEC M4 "add one at a time,
    # retrain after each"; CLAUDE.md rule #6 "one concept per training run").
    # Flip the toggles per run; the final M4 checkpoint has all three True.
    # Run 1: init_state. Run 2: + mass. Run 3: + motor.
    randomize_init_state: bool = True   # M4 run 1: init-state randomization
    randomize_mass: bool = True         # M4 run 2: + mass +/-20%
    randomize_motor: bool = True        # M4 run 3: + motor strength +/-15%

    # init-state ranges, applied at reset when randomize_init_state.
    init_pos_xy_range = 0.5        # m, +/- offset about the env origin (xy)
    init_pos_z_range = (0.5, 1.5)  # m, absolute spawn-height band (inside the
    #                                0.1-2.0 m termination bounds, around hover_height)
    init_tilt_range = 0.2         # rad, +/- roll & pitch (~11 deg); yaw is U(-pi, pi)
    init_lin_vel_range = 0.5      # m/s, +/- per axis
    init_ang_vel_range = 0.5      # rad/s, +/- per axis

    # mass randomization, when randomize_mass: scale total mass by U(1-r, 1+r),
    # inertia scaled by the same ratio.
    mass_scale_range = 0.20       # +/-20% (~27 g +/- 5.4 g)

    # "per-motor strength" in this wrench-controlled env, when randomize_motor:
    # a per-env collective-thrust gain + a constant per-env body-moment bias
    # (uneven motors -> lift loss + a steady trim torque the policy must counter).
    motor_thrust_gain_range = 0.15   # +/-15% on applied collective thrust
    motor_moment_bias_scale = 0.15   # constant bias up to 15% of moment_scale

    # --- M5 domain randomization (#2) ---------------------------------------
    # Cumulative on top of M4 (all three M4 toggles stay True); enabled ONE AT A
    # TIME across separate retrains (CLAUDE.md rule #6). Run 1: force. Run 2: + noise.
    randomize_force: bool = True    # M5 run 1: external force perturbations
    observe_noise: bool = False     # M5 run 2: Gaussian observation noise (off for run 1)

    # force perturbation, when randomize_force: a body-frame disturbance force =
    # sustained per-episode "wind" (resampled each reset) + occasional impulse
    # "gust/touch" kicks (sampled per step). Magnitudes are fractions of the
    # robot weight, applied alongside the control thrust in _pre_physics_step.
    force_wind_scale = 0.15         # sustained wind, U(0, this) x weight, random dir
    force_impulse_scale = 0.30      # impulse kick, U(0, this) x weight, random dir
    force_impulse_prob = 0.01       # per-step probability of an impulse (~5/episode)

    # observation noise, when observe_noise: Gaussian noise added to the obs.
    obs_pos_noise_std = 0.02        # m, on goal-relative position (desired_pos_b)
    obs_grav_noise_std = 0.02       # on the unit gravity direction, then renormalized
    #                                 (~1-2 deg orientation noise)


class CrazyflieHoverEnv(DirectRLEnv):
    cfg: CrazyflieHoverEnvCfg

    def __init__(self, cfg: CrazyflieHoverEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # M4 motor randomization: per-env thrust gain (x collective thrust) and a
        # constant per-env body-moment bias. Defaults (1.0 / 0.0) reproduce the
        # baseline wrench exactly when randomize_motor is False.
        self._thrust_gain = torch.ones(self.num_envs, 1, device=self.device)
        self._moment_bias = torch.zeros(self.num_envs, 3, device=self.device)
        # M5 force perturbation: per-env sustained "wind" force (body frame),
        # resampled each reset. Zero reproduces the baseline when randomize_force
        # is False. Impulse "gust/touch" kicks are sampled per step.
        self._wind_force = torch.zeros(self.num_envs, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = (
            self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        ) * self._thrust_gain[:, 0]
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:] + self._moment_bias
        if self.cfg.randomize_force:
            self._apply_force_perturbation()

    def _apply_action(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        grav_b = self._robot.data.projected_gravity_b
        # M5 observation noise: Gaussian on the position-like (goal-relative) and
        # orientation-like (gravity direction) channels. No-op when observe_noise
        # is False, so obs is byte-for-byte the M4 obs.
        if self.cfg.observe_noise:
            desired_pos_b = desired_pos_b + torch.randn_like(desired_pos_b) * self.cfg.obs_pos_noise_std
            grav_b = grav_b + torch.randn_like(grav_b) * self.cfg.obs_grav_noise_std
            grav_b = grav_b / grav_b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                grav_b,
                desired_pos_b,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # HOVER: fixed goal at the env origin (xy) and cfg.hover_height (z), instead of
        # the stock per-episode random sample. This is the one task-level deviation.
        self._desired_pos_w[env_ids, :2] = self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = self.cfg.hover_height

        # M4: per-env domain randomization, resampled each reset. Each helper is a
        # no-op when its cfg toggle is False (toggles enabled one at a time).
        if self.cfg.randomize_mass:
            self._randomize_mass(env_ids)
        if self.cfg.randomize_motor:
            self._randomize_motor(env_ids)
        if self.cfg.randomize_force:
            self._randomize_force(env_ids)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        if self.cfg.randomize_init_state:
            default_root_state = self._randomize_init_state(default_root_state, env_ids)
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _rand(self, n: int, low: float, high: float) -> torch.Tensor:
        """Sample an ``(n,)`` tensor uniformly from ``[low, high]`` on the sim device."""
        return torch.empty(n, device=self.device).uniform_(low, high)

    def _randomize_init_state(self, root_state: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """Randomize spawn pose and velocity (M4 init-state randomization).

        Overwrites the position, orientation, and velocity columns of ``root_state``
        (shape ``(len(env_ids), 13)``) in place and returns it. Height is sampled as
        an absolute band inside the termination bounds; xy is an offset about the env
        origin.
        """
        n = len(env_ids)
        origins = self._terrain.env_origins[env_ids]
        # position: xy offset about the env origin, absolute z band
        root_state[:, 0] = origins[:, 0] + self._rand(n, -self.cfg.init_pos_xy_range, self.cfg.init_pos_xy_range)
        root_state[:, 1] = origins[:, 1] + self._rand(n, -self.cfg.init_pos_xy_range, self.cfg.init_pos_xy_range)
        root_state[:, 2] = self._rand(n, self.cfg.init_pos_z_range[0], self.cfg.init_pos_z_range[1])
        # orientation: small roll/pitch tilt, full-circle yaw
        roll = self._rand(n, -self.cfg.init_tilt_range, self.cfg.init_tilt_range)
        pitch = self._rand(n, -self.cfg.init_tilt_range, self.cfg.init_tilt_range)
        yaw = self._rand(n, -torch.pi, torch.pi)
        root_state[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
        # velocity: small initial linear (7:10) and angular (10:13), world frame
        root_state[:, 7:10] = self._rand(n * 3, -self.cfg.init_lin_vel_range, self.cfg.init_lin_vel_range).view(n, 3)
        root_state[:, 10:13] = self._rand(n * 3, -self.cfg.init_ang_vel_range, self.cfg.init_ang_vel_range).view(n, 3)
        return root_state

    def _randomize_mass(self, env_ids: torch.Tensor) -> None:
        """Scale each env's total mass by ``U(1-r, 1+r)`` (M4 mass randomization).

        Mirrors Isaac Lab's ``randomize_rigid_body_mass`` event: randomization is
        applied to the *default* mass/inertia so repeated resets don't compound, and
        inertia is scaled by the same ratio. The thrust formula keeps using the
        nominal weight, so a heavier drone has a lower effective thrust-to-weight --
        the mismatch the policy must learn to absorb. PhysX mass/inertia views are
        CPU-side, hence the ``.cpu()`` env ids.
        """
        ids_cpu = env_ids.cpu()
        r = self.cfg.mass_scale_range
        scale = torch.empty(len(ids_cpu), 1).uniform_(1.0 - r, 1.0 + r)
        masses = self._robot.root_physx_view.get_masses()
        masses[ids_cpu] = self._robot.data.default_mass.to("cpu")[ids_cpu] * scale
        self._robot.root_physx_view.set_masses(masses, ids_cpu)
        inertias = self._robot.root_physx_view.get_inertias()
        inertias[ids_cpu] = self._robot.data.default_inertia.to("cpu")[ids_cpu] * scale[..., None]
        self._robot.root_physx_view.set_inertias(inertias, ids_cpu)

    def _randomize_motor(self, env_ids: torch.Tensor) -> None:
        """Resample per-env thrust gain and constant moment bias (M4 motor DR).

        The wrench-level analog of uneven motors in this collective-thrust env: a
        +/-``motor_thrust_gain_range`` gain on applied thrust, plus a constant
        body-moment bias up to ``motor_moment_bias_scale`` of ``moment_scale`` (a
        steady trim torque the policy must counteract). Both are applied in
        ``_pre_physics_step``.
        """
        n = len(env_ids)
        r = self.cfg.motor_thrust_gain_range
        self._thrust_gain[env_ids, 0] = self._rand(n, 1.0 - r, 1.0 + r)
        bias = self.cfg.motor_moment_bias_scale * self.cfg.moment_scale
        self._moment_bias[env_ids] = self._rand(n * 3, -bias, bias).view(n, 3)

    def _sample_force(self, n: int, scale: float) -> torch.Tensor:
        """Sample ``(n, 3)`` body-frame forces: random direction, magnitude
        ``U(0, scale) * robot_weight`` (M5 force perturbation)."""
        direction = torch.randn(n, 3, device=self.device)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        magnitude = self._rand(n, 0.0, scale).unsqueeze(-1) * self._robot_weight
        return direction * magnitude

    def _randomize_force(self, env_ids: torch.Tensor) -> None:
        """Resample the per-episode sustained 'wind' force (M5 force perturbation).

        The impulse 'gust/touch' component is sampled per step in
        ``_apply_force_perturbation``; only the sustained wind is set here.
        """
        self._wind_force[env_ids] = self._sample_force(len(env_ids), self.cfg.force_wind_scale)

    def _apply_force_perturbation(self) -> None:
        """Add a body-frame disturbance force = sustained wind + occasional impulse.

        ``self._wind_force`` is the per-episode sustained component (resampled at
        reset). On top of it, each step has a ``force_impulse_prob`` chance per env
        of a one-step random 'gust/touch' kick. Both are added to the control
        thrust already set in ``_pre_physics_step`` (the wrench composer takes
        body-frame forces); x/y are pure disturbance, z stacks on the control
        thrust. Called only when ``randomize_force`` is True, so the x/y thrust
        components stay 0 in the baseline.
        """
        disturb = self._wind_force.clone()
        hit = torch.rand(self.num_envs, device=self.device) < self.cfg.force_impulse_prob
        if hit.any():
            disturb += self._sample_force(self.num_envs, self.cfg.force_impulse_scale) * hit.unsqueeze(-1)
        self._thrust[:, 0, 0] = disturb[:, 0]
        self._thrust[:, 0, 1] = disturb[:, 1]
        self._thrust[:, 0, 2] += disturb[:, 2]

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
