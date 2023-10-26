import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import cv2
import json
import time
import numpy as np
from diffusion_policy.real_world.multi_realsense import MultiRealsense
from diffusion_policy.real_world.video_recorder import VideoRecorder

def test():
    config = json.load(open('/home/cchi/dev/diffusion_policy/diffusion_policy/real_world/realsense_config/415_high_accuracy_mode.json', 'r'))

    def transform(data):
        color = data['color']
        h,w,_ = color.shape
        factor = 4
        color = cv2.resize(color, (w//factor,h//factor), interpolation=cv2.INTER_AREA)
        # color = color[:,140:500]
        data['color'] = color
        return data

    from diffusion_policy.common.cv2_util import get_image_transform
    color_transform = get_image_transform(
        input_res=(1280,720),
        output_res=(640,480), 
        bgr_to_rgb=False)
    def transform(data):
        data['color'] = color_transform(data['color'])
        return data

    # one thread per camera
    video_recorder = VideoRecorder.create_h264(
        fps=30,
        codec='h264',
        thread_type='FRAME'
    )

    with MultiRealsense(
            resolution=(1280,720),
            capture_fps=30,
            record_fps=15,
            enable_color=True,
            # advanced_mode_config=config,
            transform=transform,
            # recording_transform=transform,
            # video_recorder=video_recorder,
            verbose=True
        ) as realsense:
        realsense.set_exposure(exposure=150, gain=5)
        intr = realsense.get_intrinsics()
        print(intr)

        video_path = 'data_local/test'
        rec_start_time = time.time() + 1
        realsense.start_recording(video_path, start_time=rec_start_time)
        realsense.restart_put(rec_start_time)

        out = None
        vis_img = None
        while True:
            out = realsense.get(out=out)

            # bgr = out['color']
            # print(bgr.shape)
            # vis_img = np.concatenate(list(bgr), axis=0, out=vis_img)
            # cv2.imshow('default', vis_img)
            # key = cv2.pollKey()
            # if key == ord('q'):
            #     break

            time.sleep(1/60)
            if time.time() > (rec_start_time + 20.0):
                break


if __name__ == "__main__":
    test()
