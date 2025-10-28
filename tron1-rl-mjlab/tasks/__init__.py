import gymnasium as gym

gym.register(
    id="WF_Tron",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.wf_tron_env_cfg:WfTronEnvCfg",
        "rl_cfg_entry_point": f"{__name__}.cfg.wf_tron_env_cfg:RslRlOnPolicyRunnerCfg",
    },
)
