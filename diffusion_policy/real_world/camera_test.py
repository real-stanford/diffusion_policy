from diffusion_policy.real_world.multi_realsense import SingleRealsense#, MultiRealsense
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)
import numpy as np
from diffusion_policy.real_world.video_recorder import VideoRecorder

frequency=10
n_obs_steps=2
# obs
obs_image_resolution=(640,480)
max_obs_buffer_size=30
camera_serial_numbers=None
obs_float32=False
# action
max_pos_speed=0.25
max_rot_speed=0.6
# robot
tcp_offset=0.13
init_joints=False
# video capture params
video_capture_fps=30
video_capture_resolution=(1280,720)
# saving params
record_raw_video=True
thread_per_video=2
video_crf=21
# vis params
enable_multi_cam_vis=True
multi_cam_vis_resolution=(1280,720)
# shared memory
shm_manager=None

output_dir = "/home/carrobo2024/diffusion_policy/diffusion_policy/real_world/data"

if shm_manager is None:
    shm_manager = SharedMemoryManager()
    shm_manager.start()
if camera_serial_numbers is None:
    camera_serial_numbers = SingleRealsense.get_connected_devices_serial()

color_tf = get_image_transform(
    input_res=video_capture_resolution,
    output_res=obs_image_resolution, 
    # obs output rgb
    bgr_to_rgb=True)
color_transform = color_tf
if obs_float32:
    color_transform = lambda x: color_tf(x).astype(np.float32) / 255

def transform(data):
    data['color'] = color_transform(data['color'])
    return data

rw, rh, col, row = optimal_row_cols(
    n_cameras=len(camera_serial_numbers),
    in_wh_ratio=obs_image_resolution[0]/obs_image_resolution[1],
    max_resolution=multi_cam_vis_resolution
)
vis_color_transform = get_image_transform(
    input_res=video_capture_resolution,
    output_res=(rw,rh),
    bgr_to_rgb=False
)
def vis_transform(data):
    data['color'] = vis_color_transform(data['color'])
    return data

recording_transfrom = None
recording_fps = video_capture_fps
recording_pix_fmt = 'bgr24'
if not record_raw_video:
    recording_transfrom = transform
    recording_fps = frequency
    recording_pix_fmt = 'rgb24'

video_recorder = VideoRecorder.create_h264(
    fps=recording_fps, 
    codec='h264',
    input_pix_fmt=recording_pix_fmt, 
    crf=video_crf,
    thread_type='FRAME',
    thread_count=thread_per_video)

realsense = SingleRealsense(
    serial_number=camera_serial_numbers,
    shm_manager=shm_manager,
    resolution=video_capture_resolution,
    capture_fps=video_capture_fps,
    put_fps=video_capture_fps,
    # send every frame immediately after arrival
    # ignores put_fps
    put_downsample=False,
    record_fps=recording_fps,
    enable_color=True,
    enable_depth=False,
    enable_infrared=False,
    get_max_k=max_obs_buffer_size,
    transform=transform,
    vis_transform=vis_transform,
    recording_transform=recording_transfrom,
    video_recorder=video_recorder,
    verbose=False
    )