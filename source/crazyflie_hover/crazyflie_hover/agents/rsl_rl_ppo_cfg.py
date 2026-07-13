# Copyright (c) 2026, drone-rl project. BSD-3-Clause.
#
# Faithful copy of the stock quadcopter rsl_rl PPO config (Phase 2 inventory),
# renamed for the hover task. Hyperparameters are intentionally identical to the
# stock env so Phase 3 is a clean reproduction; tuning is deferred to Phase 4+.

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class CrazyflieHoverPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200  # stock default; override on the CLI for full runs
    save_interval = 50
    experiment_name = "crazyflie_hover"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class CrazyflieWaypointPPORunnerCfg(CrazyflieHoverPPORunnerCfg):
    # M7 waypoint following: identical PPO hyperparameters to the hover runner
    # (per the locked-hyperparameter convention), only the experiment name differs
    # so waypoint logs/checkpoints land in their own directory.
    experiment_name = "crazyflie_waypoint"
