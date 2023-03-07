import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import cv2
import json
import time
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.real_world.single_realsense import SingleRealsense

def test():
    
    serials = SingleRealsense.get_connected_devices_serial()
    # import pdb; pdb.set_trace()
    serial = serials[0]
    config = json.load(open('/home/cchi/dev/diffusion_policy/diffusion_policy/real_world/realsense_config/415_high_accuracy_mode.json', 'r'))

    def transform(data):
        color = data['color']
        h,w,_ = color.shape
        factor = 2
        color = cv2.resize(color, (w//factor,h//factor), interpolation=cv2.INTER_AREA)
        # color = color[:,140:500]
        data['color'] = color
        return data

    # at 960x540 with //3, 60fps and 30fps are indistinguishable

    with SharedMemoryManager() as shm_manager:
        with SingleRealsense(
            shm_manager=shm_manager,
            serial_number=serial,
            resolution=(1280,720),
            # resolution=(960,540),
            # resolution=(640,480),
            capture_fps=30,
            enable_color=True,
            # enable_depth=True,
            # enable_infrared=True,
            # advanced_mode_config=config,
            # transform=transform,
            # recording_transform=transform
            # verbose=True
            ) as realsense:
            cv2.setNumThreads(1) 
            realsense.set_exposure(exposure=150, gain=5)
            intr = realsense.get_intrinsics()
            print(intr)


            video_path = 'data_local/test.mp4'
            rec_start_time = time.time() + 2
            realsense.start_recording(video_path, start_time=rec_start_time)

            data = None
            while True:
                data = realsense.get(out=data)
                t = time.time()
                # print('capture_latency', data['receive_timestamp']-data['capture_timestamp'], 'receive_latency', t - data['receive_timestamp'])
                # print('receive', t - data['receive_timestamp'])

                # dt = time.time() - data['timestamp']
                # print(dt)
                # print(data['capture_timestamp'] - rec_start_time)

                bgr = data['color']
                # print(bgr.shape)
                cv2.imshow('default', bgr)
                key = cv2.pollKey()
                # if key == ord('q'):
                #     break
                # elif key == ord('r'):
                #     video_path = 'data_local/test.mp4'
                #     realsense.start_recording(video_path)
                # elif key == ord('s'):
                #     realsense.stop_recording()
                
                time.sleep(1/60)
                if time.time() > (rec_start_time + 20.0):
                    break


if __name__ == "__main__":
    test()
