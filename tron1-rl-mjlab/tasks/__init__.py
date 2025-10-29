import gymnasium as gym

gym.register(
    id="Mjlab-WF-Tron",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"tron1-rl-mjlab.tasks.cfg.wf_tron_env_cfg:WfTronEnvCfg",
        "rl_cfg_entry_point": f"tron1-rl-mjlab.tasks.cfg.wf_tron_rl_cfg:WfTronRlCfg",
    },
)
