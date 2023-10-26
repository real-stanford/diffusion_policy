import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
from diffusion_policy.common.cv2_util import get_image_transform


def test():
    tf = get_image_transform((1280,720), (640,480), bgr_to_rgb=False)
    in_img = np.zeros((720,1280,3), dtype=np.uint8)
    out_img = tf(in_img)
    # print(out_img.shape)
    assert out_img.shape == (480,640,3)

    # import pdb; pdb.set_trace()

if __name__ == '__main__':
    test()
