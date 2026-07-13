# Copyright (c) 2026, drone-rl project. BSD-3-Clause.
#
# M8 obstacle-avoidance env. Subclasses the M7 waypoint env (keeps waypoints,
# success bonus, and all M4-M6 domain randomization) and adds:
#   - K kinematic pillar obstacles per env, positions randomized each reset;
#   - a 5-ray "Multiranger" distance sensor (front/back/left/right/up) via
#     MultiMeshRayCaster -- the RayCaster-family sensor that supports per-env
#     distinct/moving meshes (the base RayCaster bakes ONE static mesh shared by
#     all envs, so it cannot see per-env randomized obstacles);
#   - obs 12 -> 17 (append the 5 clipped ray distances);
#   - a proximity penalty (grows as the nearest ray-distance drops below a safe
#     margin) and collision termination (died when min ray-distance < body radius).
#
# Reward change (locked-reward exception, SPEC Decision Log 2026-07-12, rule #5):
# the M7 waypoint terms are inherited verbatim; only the proximity penalty is
# added and collision is added to `died`. Scoped to this task; the hover/waypoint
# tasks are untouched.

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.sensors import MultiMeshRayCaster, MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils import configclass

from crazyflie_hover.crazyflie_waypoint_env import CrazyflieWaypointEnv, CrazyflieWaypointEnvCfg

NUM_OBSTACLES = 6
OBSTACLE_SIZE = (0.3, 0.3, 3.0)   # square pillars spanning the workspace height
RAY_CLIP = 4.0                    # m, Multiranger max range


def multiranger_pattern(cfg: PatternBaseCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """5-ray Multiranger pattern in the sensor body frame: front/back/left/right/up.

    Returns (ray_starts, ray_directions), each (5, 3). Starts at the sensor origin;
    directions are the +x/-x/+y/-y/+z body axes.
    """
    ray_starts = torch.zeros(5, 3, device=device)
    ray_directions = torch.tensor(
        [
            [1.0, 0.0, 0.0],   # front (+x)
            [-1.0, 0.0, 0.0],  # back  (-x)
            [0.0, 1.0, 0.0],   # left  (+y)
            [0.0, -1.0, 0.0],  # right (-y)
            [0.0, 0.0, 1.0],   # up    (+z)
        ],
        device=device,
    )
    return ray_starts, ray_directions


@configclass
class MultirangerPatternCfg(PatternBaseCfg):
    func = multiranger_pattern


def _obstacle_collection_cfg() -> RigidObjectCollectionCfg:
    """K kinematic pillar obstacles per env (positions randomized at reset).

    Spawned under /World/envs/env_.*/Obstacle_<i> so they clone per env and match
    the sensor's ``Obstacle_.*`` raycast regex. Kinematic: movable by pose writes,
    unaffected by physics. Initial positions are spread on a ring and overwritten
    each reset.
    """
    objs = {}
    for i in range(NUM_OBSTACLES):
        angle = 2.0 * 3.14159265 * i / NUM_OBSTACLES
        px = 1.4 * torch.cos(torch.tensor(angle)).item()
        py = 1.4 * torch.sin(torch.tensor(angle)).item()
        objs[f"obstacle_{i}"] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Obstacle_{i}",
            spawn=sim_utils.CuboidCfg(
                size=OBSTACLE_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.25, 0.2)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, OBSTACLE_SIZE[2] / 2.0)),
        )
    return RigidObjectCollectionCfg(rigid_objects=objs)


@configclass
class CrazyflieObstacleEnvCfg(CrazyflieWaypointEnvCfg):
    # Obs grows by the 5 ray distances.
    observation_space = 17

    # Obstacles + sensor.
    obstacles: RigidObjectCollectionCfg = _obstacle_collection_cfg()
    multiranger: MultiMeshRayCasterCfg = MultiMeshRayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/ground", track_mesh_transforms=False),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/envs/env_.*/Obstacle_.*", track_mesh_transforms=True
            ),
        ],
        pattern_cfg=MultirangerPatternCfg(),
        ray_alignment="base",
        max_distance=RAY_CLIP,
        debug_vis=False,
    )

    ray_clip = RAY_CLIP              # m, distances clipped/normalized to this
    # Obstacle placement: polar ring about the env origin, clear of the spawn area.
    obstacle_radius_range = (0.8, 2.0)   # m, |xy| from env origin
    # Waypoints keep this xy clearance from every obstacle center (rejection-sampled).
    waypoint_clearance = 0.5        # m

    # Reward / termination.
    safe_distance = 0.6             # m, proximity penalty ramps up below this
    proximity_penalty_scale = 3.0   # penalty weight (x step_dt, like the vel terms)
    collision_radius = 0.15         # m, died when nearest ray-distance drops below this


class CrazyflieObstacleEnv(CrazyflieWaypointEnv):
    cfg: CrazyflieObstacleEnvCfg

    def __init__(self, cfg: CrazyflieObstacleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Log the added proximity penalty as its own episode-sum term.
        self._episode_sums["obstacle_proximity"] = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        # Full override of the hover env's _setup_scene: the obstacle collection and
        # the ray-cast sensor must exist BEFORE clone_environments so they replicate
        # into every env. Mirrors the base setup otherwise.
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._obstacles = RigidObjectCollection(self.cfg.obstacles)
        self.scene.rigid_object_collections["obstacles"] = self._obstacles
        self._multiranger = MultiMeshRayCaster(self.cfg.multiranger)
        self.scene.sensors["multiranger"] = self._multiranger

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _ray_distances(self) -> torch.Tensor:
        """(num_envs, 5) Multiranger distances, misses/over-range clipped to ray_clip."""
        hits = self._multiranger.data.ray_hits_w                 # (N, 5, 3), inf on miss
        pos = self._multiranger.data.pos_w                       # (N, 3)
        d = torch.linalg.norm(hits - pos.unsqueeze(1), dim=-1)   # (N, 5)
        return torch.nan_to_num(d, posinf=self.cfg.ray_clip).clamp(max=self.cfg.ray_clip)

    def _get_observations(self) -> dict:
        obs = super()._get_observations()["policy"]              # (N, 12)
        rays = self._ray_distances()                             # (N, 5)
        return {"policy": torch.cat([obs, rays], dim=-1)}        # (N, 17)

    def _get_rewards(self) -> torch.Tensor:
        # Inherited waypoint reward (distance + vel penalties + success bonus, and the
        # waypoint resample), then the added obstacle proximity penalty.
        reward = super()._get_rewards()
        min_ray = self._ray_distances().min(dim=1).values        # (N,)
        proximity = torch.clamp(1.0 - min_ray / self.cfg.safe_distance, min=0.0)  # 0..1
        penalty = proximity * self.cfg.proximity_penalty_scale * self.step_dt
        self._episode_sums["obstacle_proximity"] += -penalty
        return reward - penalty

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        died, time_out = super()._get_dones()
        collided = self._ray_distances().min(dim=1).values < self.cfg.collision_radius
        return torch.logical_or(died, collided), time_out

    def _randomize_obstacles(self, env_ids: torch.Tensor) -> None:
        """Randomize pillar xy on a ring about the env origin (kept clear of spawn)."""
        n = len(env_ids)
        origins = self._terrain.env_origins[env_ids]                       # (n, 3)
        lo, hi = self.cfg.obstacle_radius_range
        r = self._rand(n * NUM_OBSTACLES, lo, hi).view(n, NUM_OBSTACLES)
        theta = self._rand(n * NUM_OBSTACLES, -3.14159265, 3.14159265).view(n, NUM_OBSTACLES)
        pose = self._obstacles.data.object_pose_w[env_ids].clone()        # (n, K, 7)
        pose[:, :, 0] = origins[:, 0:1] + r * torch.cos(theta)
        pose[:, :, 1] = origins[:, 1:2] + r * torch.sin(theta)
        pose[:, :, 2] = OBSTACLE_SIZE[2] / 2.0
        object_ids = torch.arange(NUM_OBSTACLES, device=self.device)
        self._obstacles.write_object_pose_to_sim(pose, env_ids=env_ids, object_ids=object_ids)

    def _sample_waypoint(self, env_ids: torch.Tensor) -> None:
        # Base random workspace sample, then rejection-resample any waypoint that
        # lands within waypoint_clearance (xy) of an obstacle so goals stay reachable.
        super()._sample_waypoint(env_ids)
        obst_xy = self._obstacles.data.object_pos_w[env_ids][:, :, :2]     # (n, K, 2)
        for _ in range(4):
            wp_xy = self._desired_pos_w[env_ids][:, :2]                     # (n, 2)
            dxy = torch.linalg.norm(wp_xy.unsqueeze(1) - obst_xy, dim=-1)   # (n, K)
            too_close = (dxy < self.cfg.waypoint_clearance).any(dim=1)      # (n,)
            if not bool(too_close.any()):
                break
            super()._sample_waypoint(env_ids[too_close])

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        # Randomize obstacles BEFORE the waypoint sample so clearance uses the new
        # pillar positions. super() (waypoint) then samples the waypoint, which our
        # _sample_waypoint override keeps clear of the obstacles.
        self._randomize_obstacles(env_ids)
        super()._reset_idx(env_ids)