from dataclasses import dataclass

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class RslRlEncoderActorCfg(RslRlModelCfg):
    """Actor config for EncoderModel (history encoder + decoder + velocity head)."""

    class_name: str = "rsl_rl.models:EncoderModel"
    encoder_hidden_dims: tuple = (256, 128)
    encoder_latent_dim: int = 32
    decoder_hidden_dims: tuple = (128, 256)
    actor_obs_key: str = "actor"
    history_obs_key: str = "history"
    privileged_obs_key: str = "critic"
    velocity_obs_key: str = "velocity"
    decoder_loss_coef: float = 1.0
    velocity_loss_coef: float = 1.0


def make_wf_tron_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for WF-TRON task."""
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=15000,
        save_interval=200,
        wandb_project="mjlab_wf_tron",
        experiment_name="wf_tron",
        obs_groups={"actor": ("actor", "history", "critic", "velocity"), "critic": ("critic",)},
        actor=RslRlEncoderActorCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )
