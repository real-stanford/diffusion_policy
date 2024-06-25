from typing import Union
import pytorch3d.transforms as pt
import torch
import numpy as np
import functools

class RotationTransformer:
    valid_reps = [
        'axis_angle',
        'euler_angles',
        'quaternion',
        'rotation_6d',
        'matrix'
    ]

    def __init__(self, 
            from_rep='axis_angle', 
            to_rep='rotation_6d', 
            from_convention=None,
            to_convention=None):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == 'euler_angles':
            assert from_convention is not None
        if to_rep == 'euler_angles':
            assert to_convention is not None

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != 'matrix':
            funcs = [
                getattr(pt, f'{from_rep}_to_matrix'),
                getattr(pt, f'matrix_to_{from_rep}')
            ]
            if from_convention is not None:
                funcs = [functools.partial(func, convention=from_convention) 
                    for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != 'matrix':
            funcs = [
                getattr(pt, f'matrix_to_{to_rep}'),
                getattr(pt, f'{to_rep}_to_matrix')
            ]
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention) 
                    for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])
        
        inverse_funcs = inverse_funcs[::-1]
        
        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: list) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.from_numpy(x)
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y
        
    def forward(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.forward_funcs)
    
    def inverse(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.inverse_funcs)


def test():
    tf = RotationTransformer()

    rotvec = np.random.uniform(-2*np.pi,2*np.pi,size=(1000,3))
    rot6d = tf.forward(rotvec)
    new_rotvec = tf.inverse(rot6d)

    from scipy.spatial.transform import Rotation
    diff = Rotation.from_rotvec(rotvec) * Rotation.from_rotvec(new_rotvec).inv()
    dist = diff.magnitude()
    assert dist.max() < 1e-7

    tf = RotationTransformer('rotation_6d', 'matrix')
    rot6d_wrong = rot6d + np.random.normal(scale=0.1, size=rot6d.shape)
    mat = tf.forward(rot6d_wrong)
    mat_det = np.linalg.det(mat)
    assert np.allclose(mat_det, 1)
    # rotaiton_6d will be normalized to rotation matrix
