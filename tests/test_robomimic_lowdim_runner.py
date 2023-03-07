import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from diffusion_policy.env_runner.robomimic_lowdim_runner import RobomimicLowdimRunner

def test():
    import os
    from omegaconf import OmegaConf
    cfg_path = os.path.expanduser('~/dev/diffusion_policy/diffusion_policy/config/task/lift_lowdim.yaml')
    cfg = OmegaConf.load(cfg_path)
    cfg['n_obs_steps'] = 1
    cfg['n_action_steps'] = 1
    cfg['past_action_visible'] = False
    runner_cfg = cfg['env_runner']
    runner_cfg['n_train'] = 1
    runner_cfg['n_test'] = 0
    del runner_cfg['_target_']
    runner = RobomimicLowdimRunner(
        **runner_cfg, 
        output_dir='/tmp/test')

    # import pdb; pdb.set_trace()

    self = runner
    env = self.env
    env.seed(seeds=self.env_seeds)
    obs = env.reset()

if __name__ == '__main__':
    test()
