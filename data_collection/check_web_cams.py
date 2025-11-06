import cv2
import numpy as np
from cv2_enumerate_cameras import enumerate_cameras
import time
from time import sleep
import argparse
import os 


def get_idx_caps(key_name=""):
    indices = []
    caps = []
    for camera_info in enumerate_cameras(cv2.CAP_GSTREAMER):
        if key_name in camera_info.name:
            indices.append(camera_info.index)
            print(f'{camera_info.index}: {camera_info.name}')

    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Camera {idx} is available.")
            caps.append((idx,cap))
        else:
            print(f"Camera {idx} is not available.")
    return caps

def destroy_all_windows(idx_caps):
    for idx,cap in idx_caps:
        cap.release()
    cv2.destroyAllWindows()

def play_cameras(idx_caps):

    frames = {}
    for idx,cap in idx_caps:
        frames[idx] = []

    frame = {}
    for idx,cap in idx_caps:
        frame[idx] = None

    prev_time = 0.0
    while True:
        ret = True
        for idx,cap in idx_caps:
            r,f = cap.read()
            ret = ret and r
            frame[idx] = f

        if ret:
            for idx,cap in idx_caps:
                #frames[idx].append(frame[idx])
                cv2.imshow(f'Camera {idx}', frame[idx])
        else:
            print(f"Failed to read from camera ")
        c = cv2.waitKey(1)
        if c == 27:
            destroy_all_windows(idx_caps)
            break
    return frames

def write_video(images,output_file_name):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file_name+".mp4", fourcc, 30.0, (120, 120))

    for j in range(images.shape[0]):
        img = images[j].copy()
        img = cv2.resize(img, (120, 120), interpolation=cv2.INTER_AREA)
        video.write(img)

    video.release()
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Play and record from cameras.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()


    idx_caps = get_idx_caps("Brio")
    print(f"Number of cameras: {len(idx_caps)}")
    if len(idx_caps) ==  0:
        print("Not enough cameras found.")
        exit(1)

    frames = play_cameras(idx_caps)


    destroy_all_windows(idx_caps)



    

    
