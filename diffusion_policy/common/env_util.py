import cv2
import numpy as np


def render_env_video(env, states, actions=None):
    observations = states
    imgs = list()
    for i in range(len(observations)):
        state = observations[i]
        env.set_state(state)
        if i == 0:
            env.set_state(state)
        img = env.render()
        # draw action
        if actions is not None:
            action = actions[i]
            coord = (action / 512 * 96).astype(np.int32)
            cv2.drawMarker(img, coord, 
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=8, thickness=1)
        imgs.append(img)
    imgs = np.array(imgs)
    return imgs
