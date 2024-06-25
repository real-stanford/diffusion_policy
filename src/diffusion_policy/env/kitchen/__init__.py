"""Environments using kitchen and Franka robot."""
from gym.envs.registration import register

register(
    id="kitchen-microwave-kettle-light-slider-v0",
    entry_point="diffusion_policy.env.kitchen.v0:KitchenMicrowaveKettleLightSliderV0",
    max_episode_steps=280,
    reward_threshold=1.0,
)

register(
    id="kitchen-microwave-kettle-burner-light-v0",
    entry_point="diffusion_policy.env.kitchen.v0:KitchenMicrowaveKettleBottomBurnerLightV0",
    max_episode_steps=280,
    reward_threshold=1.0,
)

register(
    id="kitchen-kettle-microwave-light-slider-v0",
    entry_point="diffusion_policy.env.kitchen.v0:KitchenKettleMicrowaveLightSliderV0",
    max_episode_steps=280,
    reward_threshold=1.0,
)

register(
    id="kitchen-all-v0",
    entry_point="diffusion_policy.env.kitchen.v0:KitchenAllV0",
    max_episode_steps=280,
    reward_threshold=1.0,
)
