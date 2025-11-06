import pathlib
from typing import List, Optional, Callable, Dict, Union
import cv2
import time
import enum
import numpy as np
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.shared_memory.shared_ndarray import SharedNDArray
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from diffusion_policy.real_world.video_recorder import VideoRecorder
from diffusion_policy.common.timestamp_accumulator import get_accumulate_timestamp_idxs
from threadpoolctl import threadpool_limits


class Command(enum.Enum):
    SET_PROPERTY = 0
    START_RECORDING = 1
    STOP_RECORDING = 2
    RESTART_PUT = 3


class SingleWebcam(mp.Process):
    MAX_PATH_LENGTH = 4096  # Linux path max size

    def __init__(
            self,
            shm_manager: SharedMemoryManager,
            device_index=0,
            resolution=(640, 480),
            capture_fps=30,
            put_fps=None,
            put_downsample=True,
            record_fps=None,
            get_max_k=30,
            transform: Optional[Callable[[Dict], Dict]] = None,
            vis_transform: Optional[Callable[[Dict], Dict]] = None,
            recording_transform: Optional[Callable[[Dict], Dict]] = None,
            video_recorder: Optional[VideoRecorder] = None,
            verbose=False
    ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps
        if record_fps is None:
            record_fps = capture_fps

        resolution = tuple(resolution)
        shape = resolution[::-1]  # (H, W)
        examples = dict()
        examples['color'] = np.empty(shape + (3,), dtype=np.uint8)
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0

        vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if vis_transform is None else vis_transform(dict(examples)),
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=capture_fps
        )

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps
        )

        # Command queue
        cmd_examples = {
            'cmd': Command.SET_PROPERTY.value,
            'prop_id': int(cv2.CAP_PROP_EXPOSURE),
            'prop_value': 0.0,
            'video_path': np.chararray(self.MAX_PATH_LENGTH),
            'recording_start_time': 0.0,
            'put_start_time': 0.0
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=cmd_examples,
            buffer_size=128
        )

        if video_recorder is None:
            video_recorder = VideoRecorder.create_h264(
                fps=record_fps,
                codec='h264',
                input_pix_fmt='bgr24',
                crf=18,
                thread_type='FRAME',
                thread_count=1
            )

        self.device_index = device_index
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.record_fps = record_fps
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder
        self.verbose = verbose
        self.put_start_time = None

        # Shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue


    @staticmethod
    def get_connected_devices_index(max_devices=5):
        device_indexes = []
        for index in range(max_devices):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                device_indexes.append(index)
                cap.release()
        return sorted(device_indexes)

    # ========= Context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= User API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()

    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)

    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)

    def set_property(self, prop_id, value: float):
      self.command_queue.put({
            'cmd': np.array(Command.SET_PROPERTY.value),
            'prop_id': np.array(prop_id),
            'prop_value': np.array(value)
        })

    def set_exposure(self, exposure=None):
        if exposure is not None:
            self.set_property(cv2.CAP_PROP_EXPOSURE, exposure)

    def set_brightness(self, brightness=None):
        if brightness is not None:
            self.set_property(cv2.CAP_PROP_BRIGHTNESS, brightness)
            
    def start_recording(self, video_path: str, start_time: float=-1):
        assert self.enable_color

        path_len = len(video_path.encode('utf-8'))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError('video_path too long.')
        self.command_queue.put({
            'cmd': Command.START_RECORDING.value,
            'video_path': video_path,
            'recording_start_time': start_time
        })
        print("initiated recording for", video_path)
        
    def stop_recording(self):
        self.command_queue.put({
            'cmd': Command.STOP_RECORDING.value
        })
    
    def restart_put(self, start_time):
        for camera in self.cameras.values():
            camera.restart_put(start_time)



    def run(self):
        # Limit threads
        threadpool_limits(1)
        cv2.setNumThreads(1)

        w, h = self.resolution
        fps = self.capture_fps

        cap = cv2.VideoCapture(self.device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)

        if not cap.isOpened():
            raise RuntimeError(f"Camera at index {self.device_index} could not be opened.")

        if self.verbose:
            print(f'[SingleWebcam {self.device_index}] Main loop started.')

        put_idx = None
        put_start_time = self.put_start_time or time.time()
        iter_idx = 0
        t_start = time.time()

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                receive_time = time.time()

                if not ret:
                    if self.verbose:
                        print(f"[SingleWebcam {self.device_index}] Frame grab failed.")
                    continue

                data = {
                    'camera_receive_timestamp': receive_time,
                    'camera_capture_timestamp': receive_time,
                    'color': frame
                }

                put_data = self.transform(dict(data)) if self.transform else data

                if self.put_downsample:
                    local_idxs, global_idxs, put_idx = get_accumulate_timestamp_idxs(
                        timestamps=[receive_time],
                        start_time=put_start_time,
                        dt=1 / self.put_fps,
                        next_global_idx=put_idx,
                        allow_negative=True
                    )
                    for step_idx in global_idxs:
                        put_data['step_idx'] = step_idx
                        put_data['timestamp'] = receive_time
                        self.ring_buffer.put(put_data, wait=False)
                else:
                    step_idx = int((receive_time - put_start_time) * self.put_fps)
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = receive_time
                    self.ring_buffer.put(put_data, wait=False)

                if iter_idx == 0:
                    self.ready_event.set()

                vis_data = self.vis_transform(dict(data)) if self.vis_transform else data
                self.vis_ring_buffer.put(vis_data, wait=False)

                rec_data = self.recording_transform(dict(data)) if self.recording_transform else data

                if self.video_recorder.is_ready():
                    self.video_recorder.write_frame(rec_data['color'], frame_time=receive_time)

                t_end = time.time()
                if self.verbose:
                    freq = np.round(1 / (t_end - t_start), 1)
                    print(f'[SingleWebcam {self.device_index}] FPS {freq}')
                t_start = t_end
                commands = {}
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Exception as e:
                    print("Failed to get commands:", e)
                    n_cmd = 0

                for i in range(n_cmd):
                    command = {key: value[i] for key, value in commands.items()}
                    cmd = command['cmd']
                    if cmd == Command.SET_PROPERTY.value:
                        cap.set(command['prop_id'], command['prop_value'])
                    elif cmd == Command.START_RECORDING.value:
                        video_path = ''.join(command['video_path']).strip('\x00')
                        start_time = command['recording_start_time'] or None
                        self.video_recorder.start(video_path, start_time=start_time)
                        print(f"[SingleWebcam {self.device_index}] Recording started at {video_path} with start time {start_time}")
                    elif cmd == Command.STOP_RECORDING.value:
                        self.video_recorder.stop()
                        put_idx = None
                    elif cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command['put_start_time']

                iter_idx += 1
        finally:
            self.video_recorder.stop()
            cap.release()
            self.ready_event.set()

        if self.verbose:
            print(f'[SingleWebcam {self.device_index}] Exiting worker process.')