from omegaconf import OmegaConf
from robomimic.config import config_factory
import robomimic.scripts.generate_paper_configs as gpc
from robomimic.scripts.generate_paper_configs import (
    modify_config_for_default_image_exp,
    modify_config_for_default_low_dim_exp,
    modify_config_for_dataset,
)

def get_robomimic_config(
        algo_name='bc_rnn', 
        hdf5_type='low_dim', 
        task_name='square', 
        dataset_type='ph'
    ):
    base_dataset_dir = '/tmp/null'
    filter_key = None

    # decide whether to use low-dim or image training defaults
    modifier_for_obs = modify_config_for_default_image_exp
    if hdf5_type in ["low_dim", "low_dim_sparse", "low_dim_dense"]:
        modifier_for_obs = modify_config_for_default_low_dim_exp

    algo_config_name = "bc" if algo_name == "bc_rnn" else algo_name
    config = config_factory(algo_name=algo_config_name)
    # turn into default config for observation modalities (e.g.: low-dim or rgb)
    config = modifier_for_obs(config)
    # add in config based on the dataset
    config = modify_config_for_dataset(
        config=config, 
        task_name=task_name, 
        dataset_type=dataset_type, 
        hdf5_type=hdf5_type, 
        base_dataset_dir=base_dataset_dir,
        filter_key=filter_key,
    )
    # add in algo hypers based on dataset
    algo_config_modifier = getattr(gpc, f'modify_{algo_name}_config_for_dataset')
    config = algo_config_modifier(
        config=config, 
        task_name=task_name, 
        dataset_type=dataset_type, 
        hdf5_type=hdf5_type,
    )
    return config
    

