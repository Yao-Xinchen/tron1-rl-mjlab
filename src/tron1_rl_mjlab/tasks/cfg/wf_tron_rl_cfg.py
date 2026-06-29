from dataclasses import dataclass

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class RslRlEncoderRNNActorCfg(RslRlModelCfg):
    """Actor config for EncoderRNNActorModel.

    Architecture: 2D CNN (current depth frame) + MLP (proprio) → GRU → actor head.
    The CNN runs no_grad for the GRU path; a separate with_grad CNN pass trains
    the CNN decoder heads.
    """

    class_name: str = "rsl_rl.models:EncoderRNNActorModel"

    # Proprioceptive MLP encoder
    actor_obs_group: str = "actor"
    proprio_encoder_hidden_dims: tuple = (128, 64)
    proprio_encoder_output_dim: int = 64

    # 2D CNN for current depth image
    depth_obs_group: str = "depth_camera"
    cnn_output_channels: tuple = (32, 64, 64)
    cnn_kernel_size: int = 3
    cnn_strides: tuple = (2, 2, 2)
    cnn_output_dim: int = 128

    # GRU that fuses proprio + vision encodings
    rnn_hidden_dim: int = 512
    rnn_num_layers: int = 1

    # GRU decoder heads: GRU latent → privileged obs + height map
    privileged_decoder_obs_group: str = "critic"
    height_map_decoder_obs_group: str = "height_map"
    decoder_hidden_dims: tuple = (512, 256)

    # CNN depth autoencoder: CNN features → reconstructed depth image
    depth_decoder_hidden_dims: tuple = (256, 512)


@dataclass
class RslRlPPOWithDecoderAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO with supervised decoder reconstruction loss."""

    class_name: str = "rsl_rl.algorithms:PPOWithDecoder"

    # Obs groups to use as reconstruction targets
    privileged_obs_group: str = "critic"
    height_map_obs_group: str = "height_map"

    # Obs group containing raw depth images (buffered during rollout for CNN update)
    depth_obs_group: str = "depth_camera"

    # CNN autoencoder params (independent of PPO / GRU reconstruction params)
    # Fraction of rollout images used per CNN update  (e.g. 0.25 → 6144 / 24576)
    cnn_image_fraction: float = 0.25
    # Images per CNN gradient step; controls peak CNN activation memory
    cnn_mini_batch_size: int = 768

    # Weight on GRU reconstruction losses relative to PPO loss
    recon_loss_coef: float = 1.0


def make_wf_tron_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for WF-TRON task."""
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=4000,
        save_interval=200,
        wandb_project="mjlab_wf_tron",
        experiment_name="wf_tron",
        obs_groups={
            # Actor receives current proprio + depth image
            "actor": ("actor", "depth_camera"),
            # Critic receives privileged obs + height map (concatenated by MLPModel)
            "critic": ("critic", "height_map"),
        },
        actor=RslRlEncoderRNNActorCfg(
            hidden_dims=(256, 128),
            activation="elu",
            distribution_cfg={
                "class_name": "rsl_rl.modules:GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            class_name="rsl_rl.models:MLPModel",
            hidden_dims=(512, 256, 128),
            activation="elu",
        ),
        algorithm=RslRlPPOWithDecoderAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=32,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            recon_loss_coef=1.0,
        ),
    )
