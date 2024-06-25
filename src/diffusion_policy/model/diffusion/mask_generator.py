from typing import Sequence, Optional
import torch
from torch import nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


def get_intersection_slice_mask(
    shape: tuple, 
    dim_slices: Sequence[slice], 
    device: Optional[torch.device]=None
    ):
    assert(len(shape) == len(dim_slices))
    mask = torch.zeros(size=shape, dtype=torch.bool, device=device)
    mask[dim_slices] = True
    return mask


def get_union_slice_mask(
    shape: tuple, 
    dim_slices: Sequence[slice], 
    device: Optional[torch.device]=None
    ):
    assert(len(shape) == len(dim_slices))
    mask = torch.zeros(size=shape, dtype=torch.bool, device=device)
    for i in range(len(dim_slices)):
        this_slices = [slice(None)] * len(shape)
        this_slices[i] = dim_slices[i]
        mask[this_slices] = True
    return mask


class DummyMaskGenerator(ModuleAttrMixin):
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def forward(self, shape):
        device = self.device
        mask = torch.ones(size=shape, dtype=torch.bool, device=device)
        return mask


class LowdimMaskGenerator(ModuleAttrMixin):
    def __init__(self,
        action_dim, obs_dim,
        # obs mask setup
        max_n_obs_steps=2, 
        fix_obs_steps=True, 
        # action mask
        action_visible=False
        ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible

    @torch.no_grad()
    def forward(self, shape, seed=None):
        device = self.device
        B, T, D = shape
        assert D == (self.action_dim + self.obs_dim)

        # create all tensors on this device
        rng = torch.Generator(device=device)
        if seed is not None:
            rng = rng.manual_seed(seed)

        # generate dim mask
        dim_mask = torch.zeros(size=shape, 
            dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[...,:self.action_dim] = True
        is_obs_dim = ~is_action_dim

        # generate obs mask
        if self.fix_obs_steps:
            obs_steps = torch.full((B,), 
            fill_value=self.max_n_obs_steps, device=device)
        else:
            obs_steps = torch.randint(
                low=1, high=self.max_n_obs_steps+1, 
                size=(B,), generator=rng, device=device)
            
        steps = torch.arange(0, T, device=device).reshape(1,T).expand(B,T)
        obs_mask = (steps.T < obs_steps).T.reshape(B,T,1).expand(B,T,D)
        obs_mask = obs_mask & is_obs_dim

        # generate action mask
        if self.action_visible:
            action_steps = torch.maximum(
                obs_steps - 1, 
                torch.tensor(0,
                    dtype=obs_steps.dtype, 
                    device=obs_steps.device))
            action_mask = (steps.T < action_steps).T.reshape(B,T,1).expand(B,T,D)
            action_mask = action_mask & is_action_dim

        mask = obs_mask
        if self.action_visible:
            mask = mask | action_mask
        
        return mask


class KeypointMaskGenerator(ModuleAttrMixin):
    def __init__(self, 
            # dimensions
            action_dim, keypoint_dim,
            # obs mask setup
            max_n_obs_steps=2, fix_obs_steps=True, 
            # keypoint mask setup
            keypoint_visible_rate=0.7, time_independent=False,
            # action mask
            action_visible=False,
            context_dim=0, # dim for context
            n_context_steps=1
            ):
        super().__init__()
        self.action_dim = action_dim
        self.keypoint_dim = keypoint_dim
        self.context_dim = context_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.keypoint_visible_rate = keypoint_visible_rate
        self.time_independent = time_independent
        self.action_visible = action_visible
        self.n_context_steps = n_context_steps
    
    @torch.no_grad()
    def forward(self, shape, seed=None):
        device = self.device
        B, T, D = shape
        all_keypoint_dims = D - self.action_dim - self.context_dim
        n_keypoints = all_keypoint_dims // self.keypoint_dim
        
        # create all tensors on this device
        rng = torch.Generator(device=device)
        if seed is not None:
            rng = rng.manual_seed(seed)
        
        # generate dim mask
        dim_mask = torch.zeros(size=shape, 
            dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[...,:self.action_dim] = True
        is_context_dim = dim_mask.clone()
        if self.context_dim > 0:
            is_context_dim[...,-self.context_dim:] = True
        is_obs_dim = ~(is_action_dim | is_context_dim)
        # assumption trajectory=cat([action, keypoints, context], dim=-1)

        # generate obs mask
        if self.fix_obs_steps:
            obs_steps = torch.full((B,), 
            fill_value=self.max_n_obs_steps, device=device)
        else:
            obs_steps = torch.randint(
                low=1, high=self.max_n_obs_steps+1, 
                size=(B,), generator=rng, device=device)
            
        steps = torch.arange(0, T, device=device).reshape(1,T).expand(B,T)
        obs_mask = (steps.T < obs_steps).T.reshape(B,T,1).expand(B,T,D)
        obs_mask = obs_mask & is_obs_dim

        # generate action mask
        if self.action_visible:
            action_steps = torch.maximum(
                obs_steps - 1, 
                torch.tensor(0,
                    dtype=obs_steps.dtype, 
                    device=obs_steps.device))
            action_mask = (steps.T < action_steps).T.reshape(B,T,1).expand(B,T,D)
            action_mask = action_mask & is_action_dim

        # generate keypoint mask
        if self.time_independent:
            visible_kps = torch.rand(size=(B, T, n_keypoints), 
                generator=rng, device=device) < self.keypoint_visible_rate
            visible_dims = torch.repeat_interleave(visible_kps, repeats=self.keypoint_dim, dim=-1)
            visible_dims_mask = torch.cat([
                torch.ones((B, T, self.action_dim), 
                    dtype=torch.bool, device=device),
                visible_dims,
                torch.ones((B, T, self.context_dim), 
                    dtype=torch.bool, device=device),
            ], axis=-1)
            keypoint_mask = visible_dims_mask
        else:
            visible_kps = torch.rand(size=(B,n_keypoints), 
                generator=rng, device=device) < self.keypoint_visible_rate
            visible_dims = torch.repeat_interleave(visible_kps, repeats=self.keypoint_dim, dim=-1)
            visible_dims_mask = torch.cat([
                torch.ones((B, self.action_dim), 
                    dtype=torch.bool, device=device),
                visible_dims,
                torch.ones((B, self.context_dim), 
                    dtype=torch.bool, device=device),
            ], axis=-1)
            keypoint_mask = visible_dims_mask.reshape(B,1,D).expand(B,T,D)
        keypoint_mask = keypoint_mask & is_obs_dim

        # generate context mask
        context_mask = is_context_dim.clone()
        context_mask[:,self.n_context_steps:,:] = False

        mask = obs_mask & keypoint_mask 
        if self.action_visible:
            mask = mask | action_mask
        if self.context_dim > 0:
            mask = mask | context_mask

        return mask


def test():
    # kmg = KeypointMaskGenerator(2,2, random_obs_steps=True)
    # self = KeypointMaskGenerator(2,2,context_dim=2, action_visible=True)
    # self = KeypointMaskGenerator(2,2,context_dim=0, action_visible=True)
    self = LowdimMaskGenerator(2,20, max_n_obs_steps=3, action_visible=True)
