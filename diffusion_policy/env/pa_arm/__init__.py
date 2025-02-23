import diffusion_policy.env.pa_arm
from gym.envs.registration import register

register(
    id="pa-arm-sim-v0",
    entry_point="envs.pa_arm.pa_arm_sim_env:PAArmSimEnv",
    max_episode_steps=200,
    reward_threshold=1.0,
)

register(
    id="pa-arm-real-v0",
    entry_point="envs.pa_arm.pa_arm_real_env:PAArmRealEnv",
    max_episode_steps=200,
    reward_threshold=1.0,
)
