from typing import List, Optional, Union, Dict, Callable,Any
import numbers
import cv2
import time
import pathlib
from multiprocessing.managers import SharedMemoryManager
from collections.abc import Sequence
import numpy as np
from diffusion_policy.real_world.single_webcam import SingleWebcam
from diffusion_policy.real_world.video_recorder import VideoRecorder


class MultiWebcam:
    def __init__(self,
        device_indices: Optional[List[int]] = None,
        shm_manager: Optional[SharedMemoryManager] = None,
        resolution=(640, 480),
        capture_fps=30,
        put_fps=None,
        put_downsample=True,
        record_fps=None,
        get_max_k=30,
        transform: Optional[Union[Callable[[Dict], Dict], List[Optional[Callable]]]] = None,
        vis_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]] = None,
        recording_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]] = None,
        video_recorder: Optional[Union[VideoRecorder, List[VideoRecorder]]] = None,
        verbose=False
    ):
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        # Get connected webcams if device_indices not provided
        if device_indices is None:
            device_indices = SingleWebcam.get_connected_devices_index()
        n_cameras = len(device_indices)

        transform = repeat_to_list(transform, n_cameras, Callable)
        vis_transform = repeat_to_list(vis_transform, n_cameras, Callable)
        recording_transform = repeat_to_list(recording_transform, n_cameras, Callable)
        video_recorder = repeat_to_list(video_recorder, n_cameras, VideoRecorder)

        cameras = dict()
        for i, device_idx in enumerate(device_indices):
            cameras[device_idx] = SingleWebcam(
                shm_manager=shm_manager,
                device_index=device_idx,
                resolution=resolution,
                capture_fps=capture_fps,
                put_fps=put_fps,
                put_downsample=put_downsample,
                record_fps=record_fps,
                get_max_k=get_max_k,
                transform=transform[i],
                vis_transform=vis_transform[i],
                recording_transform=recording_transform[i],
                video_recorder=video_recorder[i],
                verbose=verbose
            )

        self.cameras = cameras
        self.shm_manager = shm_manager

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def n_cameras(self):
        return len(self.cameras)

    @property
    def is_ready(self):
        return all(camera.is_ready for camera in self.cameras.values())

    def start(self, wait=True, put_start_time=None):
        if put_start_time is None:
            put_start_time = time.time()
        for camera in self.cameras.values():
            camera.start(wait=False, put_start_time=put_start_time)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        for camera in self.cameras.values():
            camera.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        for camera in self.cameras.values():
            camera.start_wait()

    def stop_wait(self):
        for camera in self.cameras.values():
            camera.join()

    def get(self, k=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Return frames from all webcams.
        {
            0: {'color': (T,H,W,C), 'timestamp': (T,)},
            1: ...
        }
        """
        if out is None:
            out = dict()
        for i, camera in enumerate(self.cameras.values()):
            this_out = out.get(i, None)
            this_out = camera.get(k=k, out=this_out)
            out[i] = this_out
        return out

    def get_vis(self, out=None):
        results = []
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if out is not None:
                this_out = {key: v[i:i+1].reshape(v.shape[1:]) for key, v in out.items()}
            this_out = camera.get_vis(out=this_out)
            if out is None:
                results.append(this_out)
        if out is None and results:
            out = {key: np.stack([x[key] for x in results]) for key in results[0]}
        return out

    def set_property(self, prop_id, value):
        n_camera = len(self.cameras)
        value = repeat_to_list(value, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_property(prop_id, value[i])

    def set_exposure(self, exposure=None):
        if exposure is not None:
            self.set_property(cv2.CAP_PROP_EXPOSURE, exposure)

    def set_brightness(self, brightness=None):
        if brightness is not None:
            self.set_property(cv2.CAP_PROP_BRIGHTNESS, brightness)

    def start_recording(self, video_path: Union[str, List[str]], start_time: float):
        if isinstance(video_path, str):
            video_dir = pathlib.Path(video_path)
            assert video_dir.parent.is_dir()
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = [str(video_dir.joinpath(f'{i}.mp4').absolute()) for i in range(self.n_cameras)]
        assert len(video_path) == self.n_cameras

        for i, camera in enumerate(self.cameras.values()):
            camera.start_recording(video_path[i], start_time)

    def stop_recording(self):
        for camera in self.cameras.values():
            camera.stop_recording()

    def restart_put(self, start_time):
        for camera in self.cameras.values():
            camera.restart_put(start_time)

def repeat_to_list(x, n: int, cls) -> List[Any]:
    if x is None:
        return [lambda d: d] * n  # Identity transform
    if not isinstance(x, list):
        return [x] * n
    assert len(x) == n
    return x

