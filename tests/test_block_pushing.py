import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from diffusion_policy.env.block_pushing.block_pushing_multimodal import BlockPushMultimodal
from gym.wrappers import FlattenObservation
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_wrapper import VideoWrapper

def test():
    env = MultiStepWrapper(
            VideoWrapper(
                FlattenObservation(
                    BlockPushMultimodal()
                ),
                enabled=True,
                steps_per_render=2
            ),
            n_obs_steps=2,
            n_action_steps=8,
            max_episode_steps=16
        )
    env = BlockPushMultimodal()
    obs = env.reset()
    import pdb; pdb.set_trace()

    env = FlattenObservation(BlockPushMultimodal())
    obs = env.reset()
    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)
    print(obs[8:10] + action - next_obs[8:10])
    import pdb; pdb.set_trace()

    for i in range(3):
        obs, reward, done, info = env.step(env.action_space.sample())
    img = env.render()
    import pdb; pdb.set_trace()
    print("Done!", done)

if __name__ == '__main__':
    test()
