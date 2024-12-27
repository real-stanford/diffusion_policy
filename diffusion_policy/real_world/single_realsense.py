from typing import Optional, Callable, Dict
import os
import enum
import time
# import torch
import numpy as np
import pyrealsense2 as rs
import pyzed.sl as sl
import multiprocessing as mp
import cv2
import threading as th
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.common.timestamp_accumulator import get_accumulate_timestamp_idxs
from diffusion_policy.shared_memory.shared_ndarray import SharedNDArray
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from diffusion_policy.real_world.video_recorder import VideoRecorder
from aloha_scripts.constants import *

class Command(enum.Enum):
    SET_COLOR_OPTION = 0
    SET_DEPTH_OPTION = 1
    START_RECORDING = 2
    STOP_RECORDING = 3
    RESTART_PUT = 4

class SingleRealsense(mp.Process):
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes

    def __init__(
            self, 
            shm_manager: SharedMemoryManager,
            serial_number,
            resolution=(1280,720),
            capture_fps=30,
            put_fps=None,
            put_downsample=True,
            record_fps=None,
            enable_color=True,
            enable_depth=False,
            enable_infrared=False,
            get_max_k=30,
            advanced_mode_config=None,
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

        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = dict()
        if enable_color:
            examples['color'] = np.empty(
                shape=shape+(3,), dtype=np.uint8)
        if enable_depth:
            examples['depth'] = np.empty(
                shape=shape, dtype=np.uint16)
        if enable_infrared:
            examples['infrared'] = np.empty(
                shape=shape, dtype=np.uint8)
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0

        vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if vis_transform is None 
                else vis_transform(dict(examples)),
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=capture_fps
        )

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None
                else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps
        )

        # create command queue
        examples = {
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': rs.option.exposure.value,
            'option_value': 0.0,
            'video_path': np.array('a'*self.MAX_PATH_LENGTH),
            'recording_start_time': 0.0,
            'put_start_time': 0.0
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=128
        )

        # create shared array for intrinsics
        intrinsics_array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(7,),
                dtype=np.float64)
        intrinsics_array.get()[:] = 0

        # create video recorder
        if video_recorder is None:
            # realsense uses bgr24 pixel format
            # default thread_type to FRAEM
            # i.e. each frame uses one core
            # instead of all cores working on all frames.
            # this prevents CPU over-subpscription and
            # improves performance significantly
            video_recorder = VideoRecorder.create_h264(
                fps=record_fps, 
                codec='h264',
                input_pix_fmt='bgr24', 
                crf=18,
                thread_type='FRAME',
                thread_count=1)

        # copied variables
        self.serial_number = serial_number
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.record_fps = record_fps
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_infrared = enable_infrared
        self.advanced_mode_config = advanced_mode_config
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder
        self.verbose = verbose
        self.put_start_time = None
        # self.image_recorder = ImageRecorder(camera_names=self.serial_number,init_node=True)

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue
        self.intrinsics_array = intrinsics_array
    
    @staticmethod
    def get_connected_devices_serial():
        serials = list()
        for d in rs.context().devices:
            if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                serial = d.get_info(rs.camera_info.serial_number)
                product_line = d.get_info(rs.camera_info.product_line)
                if product_line == 'D400':
                    # only works with D400 series
                    serials.append(serial)
        serials = sorted(serials)
        #print('serials :', serials)
        return serials

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        super().start()
        if wait:
            self.start_wait()
        # print("camera process is alive:", self.is_ready)
    
    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        # print("in camera start_wait")
        self.ready_event.wait()
        # print("camera process after start_wait:", self.is_ready)
    
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
    
    # ========= user API ===========
    def set_color_option(self, option: rs.option, value: float):
        self.command_queue.put({
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': option.value,
            'option_value': value
        })
    
    def set_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.set_color_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_color_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_color_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_color_option(rs.option.gain, gain)
    
    def set_white_balance(self, white_balance=None):
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            self.set_color_option(rs.option.white_balance, white_balance)

    def get_intrinsics(self):
        assert self.ready_event.is_set()
        fx, fy, ppx, ppy = self.intrinsics_array.get()[:4]
        mat = np.eye(3)
        mat[0,0] = fx
        mat[1,1] = fy
        mat[0,2] = ppx
        mat[1,2] = ppy
        return mat

    def get_depth_scale(self):
        assert self.ready_event.is_set()
        scale = self.intrinsics_array.get()[-1]
        return scale
    
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
        
    def stop_recording(self):
        self.command_queue.put({
            'cmd': Command.STOP_RECORDING.value
        })
    
    def restart_put(self, start_time):
        self.command_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': start_time
        })
     
    # ========= interval API ===========
    def run(self):
        # limit threads
        threadpool_limits(1)
        cv2.setNumThreads(1)
        # print("Inside run")
        ##
        try:
            
            """
            Initialize wrist camera
            Put: config.enable_device(WRIST_CAM_ID) if using UR rbot
            Put: config.enable_device(WRIST_CAM_MASTER_ID) if using replica
            """
            if self.serial_number == 'cam_wrist':
                # Initialize the RealSense pipeline
                config = rs.config()
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.capture_fps)
                config.enable_device(WRIST_CAM_ID)
                pipeline = rs.pipeline()
                pipeline.start(config)
                sensor = pipeline.get_active_profile().get_device().query_sensors()[0]
                sensor.set_option(rs.option.enable_auto_exposure, False)
                sensor.set_option(rs.option.exposure, 10000) ## experiment with exposure value 

            """
            Initialize Zed camera
            """    
            if self.serial_number == 'cam_low':
                # print('Inside ZED init')
                # Create a ZED camera object
                zed = sl.Camera()
                # Set configuration parameters
                init_params = sl.InitParameters()
                init_params.camera_resolution = sl.RESOLUTION.HD1080  # Change the resolution as needed
                init_params.camera_fps = self.capture_fps  # Use the specified FPS
                init_params.depth_mode = sl.DEPTH_MODE.NONE  # Disable depth calculation
                init_params.camera_disable_self_calib = True
                # Initialize the camera
                err = zed.open(init_params)
                # print(err)
                if err != sl.ERROR_CODE.SUCCESS:
                    print(f"Error initializing ZED camera: {err}")
                    zed.close()
                    return
                # Create a runtime parameters object
                runtime_params = sl.RuntimeParameters()


            """
            Initialize Logitec camera
            Put: os.path.realpath("/dev/CAM_HIGH") for UR robot setup
            Put: os.path.realpath("/dev/CAM_HIGH_MASTER") for replica robot setup
            """ 
            # Logitech camera setup
            if self.serial_number == 'cam_high':
                cam_path = os.path.realpath("/dev/CAM_HIGH")
                cam_idx = int(cam_path.split("/dev/video")[-1])
                cap = cv2.VideoCapture(cam_idx)
                cap.set(cv2.CAP_PROP_EXPOSURE, 0)
                cap.set(cv2.CAP_PROP_FPS, self.capture_fps)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

            if self.serial_number == 'cam_front':
                # Initialize the RealSense pipeline
                config = rs.config()
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.capture_fps)
                config.enable_device(FRONT_CAM_ID)
                pipeline = rs.pipeline()
                pipeline.start(config)
                # sensor = pipeline.get_active_profile().get_device().query_sensors()[0]
                # sensor.set_option(rs.option.enable_auto_exposure, False)
                # sensor.set_option(rs.option.exposure, 10000)  

            # report global time
            # https://github.com/IntelRealSense/librealsense/pull/3909

            """
            Can omit this step to incorporate D405 data         ## abhi 3/7/24

            d = pipeline_profile.get_device().first_color_sensor()
            d.set_option(rs.option.global_time_enabled, 1)
            """
            # setup advanced mode
            """if self.advanced_mode_config is not None:
                json_text = json.dumps(self.advanced_mode_config)
                device = pipeline_profile.get_device()
                advanced_mode = rs.rs400_advanced_mode(device)
                advanced_mode.load_json(json_text)

            # get
            color_stream = pipeline_profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            order = ['fx', 'fy', 'ppx', 'ppy', 'height', 'width']
            for i, name in enumerate(order):
                self.intrinsics_array.get()[i] = getattr(intr, name)

            if self.enable_depth:
                depth_sensor = pipeline_profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                self.intrinsics_array.get()[-1] = depth_scale
                """
            
            # one-time setup (intrinsics etc, ignore for now)
            if self.verbose:
                print(f'[SingleRealsense {self.serial_number}] Main loop started.')

            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()
            iter_idx = 0
            t_start = time.time()

            # frame grabbing part
            # print("entering data aquisition")
            while not self.stop_event.is_set():
                black_img = np.zeros((480,640,3), dtype= np.uint8)
                # wait for frames to come in
                if self.serial_number == 'cam_wrist':
                    # Realsense image acquisition
                    rs_frames = pipeline.wait_for_frames()
                    rs_color_frame = rs_frames.get_color_frame()
                    if rs_color_frame:
                        frameset = np.asanyarray(rs_color_frame.get_data())
                        # frameset = cv2.cvtColor(rs_color_image, cv2.COLOR_BGR2RGB)
                        
                if self.serial_number == 'cam_low':
                    # ZED image acquisition
                    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                        zed_image_frame = sl.Mat()
                        zed.retrieve_image(zed_image_frame, sl.VIEW.RIGHT)
                        frameset = cv2.resize(zed_image_frame.get_data()[:, :, :3], (640,480))
                        """
                        Below is cropping logic when performing domain gap
                        comment/ uncomment these lines as needed
                        """
                        # black_img[y1:y2, x1:x2] = frameset[y1:y2, x1:x2]
                        # frameset = black_img.copy()
                        """
                        Cropping logic ends here
                        """
                        # alpha_channel = frameset[:, :, 3]
                        # print(np.min(alpha_channel), np.max(alpha_channel))
                        # frameset = cv2.cvtColor(frameset, cv2.COLOR_BGR2RGB)
                
                if self.serial_number == 'cam_high':
                    _, frameset = cap.read()
                    frameset = cv2.resize(frameset, (640,480))
                    """
                    Below is cropping logic when performing domain gap
                    comment/ uncomment these lines as needed
                    """
                    # black_img[y1:y2, x1:x2] = frameset[y1:y2, x1:x2]
                    # frameset = black_img.copy()

                    """
                    Cropping logic ends here
                    """
                    # frameset = cv2.cvtColor(frameset, cv2.COLOR_BGR2RGB)

                if self.serial_number == 'cam_front':
                    # Realsense image acquisition
                    rs_frames = pipeline.wait_for_frames()
                    rs_color_frame = rs_frames.get_color_frame()
                    if rs_color_frame:
                        frameset = np.asanyarray(rs_color_frame.get_data())
                        black_img[30:240, 200:550] = frameset[30:240, 200:550] 
                        frameset = black_img.copy()

                # frameset = pipeline.wait_for_frames()
                receive_time = time.time()
                # align frames to color
                # frameset = align.process(frameset)

                # grab data
                data = dict()
                data['camera_receive_timestamp'] = receive_time
                # realsense report in ms
                # data['camera_capture_timestamp'] = frameset[f'{self.serial_number}_timestamps']
                data['camera_capture_timestamp'] = time.time()
                # print("camera is:", self.serial_number)
                if self.enable_color:
                    color_frame = frameset
                    # cv2.imwrite(f'diffusion_policy/LHW_images/{self.serial_number}.png', color_frame)
                    # print("camera data from ros:",color_frame.shape, type(color_frame))
                    # data['color'] = np.asarray(color_frame.get_data())
                    data['color'] = color_frame
                    # t = color_frame.get_timestamp() / 1000
                    # t = frameset[f'{self.serial_number}_timestamps']
                    t= time.time()
                    # print("timestamp from ros:",t, type(t))
                    data['camera_capture_timestamp'] = t
                    # print('device', time.time() - t)
                    # print(color_frame.get_frame_timestamp_domain())
                if self.enable_depth:
                    data['depth'] = np.asarray(
                        frameset.get_depth_frame().get_data())
                if self.enable_infrared:
                    data['infrared'] = np.asarray(
                        frameset.get_infrared_frame().get_data())


                # apply transform
                put_data = data
                # print(self.transform)
                if self.transform is not None:
                    put_data = self.transform(dict(data))

                # by default is false; value from real_env.py           # abhi 3/7/24
                if self.put_downsample:
                    # print("inside if stmt:", self.put_downsample)                
                    # put frequency regulation
                    local_idxs, global_idxs, put_idx \
                        = get_accumulate_timestamp_idxs(
                            timestamps=[receive_time],
                            start_time=put_start_time,
                            dt=1/self.put_fps,
                            # this is non in first iteration
                            # and then replaced with a concrete number
                            next_global_idx=put_idx,
                            # continue to pump frames even if not started.
                            # start_time is simply used to align timestamps.
                            allow_negative=True
                        )

                    for step_idx in global_idxs:
                        put_data['step_idx'] = step_idx
                        # put_data['timestamp'] = put_start_time + step_idx / self.put_fps
                        put_data['timestamp'] = receive_time
                        # print(step_idx, data['timestamp'])
                        self.ring_buffer.put(put_data, wait=True) ## observations from camera put in ring buffer
                else:
                    step_idx = int((receive_time - put_start_time) * self.put_fps)
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = receive_time
                    self.ring_buffer.put(put_data)
                # print("************* put data color shapes")
                # print("put data dict:", put_data["color"].shape)
                # signal ready
                if iter_idx == 0:
                    self.ready_event.set()
                
                # put to vis
                vis_data = data
                if self.vis_transform == self.transform:
                    vis_data = put_data
                elif self.vis_transform is not None:
                    vis_data = self.vis_transform(dict(data))
                self.vis_ring_buffer.put(vis_data, wait=True)
                
                # record frame
                rec_data = data
                if self.recording_transform == self.transform:
                    rec_data = put_data
                elif self.recording_transform is not None:
                    rec_data = self.recording_transform(dict(data))

                if self.video_recorder.is_ready():
                    self.video_recorder.write_frame(rec_data['color'], 
                        frame_time=receive_time)

                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                if self.verbose:
                    print(f'[SingleRealsense {self.serial_number}] FPS {frequency}')

                # fetch command from queue
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    """if cmd == Command.SET_COLOR_OPTION.value:
                        sensor = pipeline_profile.get_device().first_color_sensor()
                        option = rs.option(command['option_enum'])
                        value = float(command['option_value'])
                        sensor.set_option(option, value)
                        # print('auto', sensor.get_option(rs.option.enable_auto_exposure))
                        # print('exposure', sensor.get_option(rs.option.exposure))
                        # print('gain', sensor.get_option(rs.option.gain))
                    elif cmd == Command.SET_DEPTH_OPTION.value:
                        sensor = pipeline_profile.get_device().first_depth_sensor()
                        option = rs.option(command['option_enum'])
                        value = float(command['option_value'])
                        sensor.set_option(option, value)"""
                    if cmd == Command.START_RECORDING.value:
                        video_path = str(command['video_path'])
                        start_time = command['recording_start_time']
                        if start_time < 0:
                            start_time = None
                        self.video_recorder.start(video_path, start_time=start_time)
                    elif cmd == Command.STOP_RECORDING.value:
                        self.video_recorder.stop()
                        # stop need to flush all in-flight frames to disk, which might take longer than dt.
                        # soft-reset put to drop frames to prevent ring buffer overflow.
                        put_idx = None
                    elif cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command['put_start_time']
                        # self.ring_buffer.clear()

                iter_idx += 1
        finally:
            self.video_recorder.stop()
            if self.serial_number == 'cam_wrist':
                pipeline.stop()
            if self.serial_number == 'cam_low':
                zed.close()
            if self.serial_number == 'cam_high':
                cap.release()
            # rs_config.disable_all_streams()
            self.ready_event.set()
        
        if self.verbose:
            print(f'[SingleRealsense {self.serial_number}] Exiting worker process.')