import argparse
import time
import threading
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import os
from cv2_enumerate_cameras import enumerate_cameras

def parse_arguments():
    parser = argparse.ArgumentParser(description='Threaded camera recorder')
    parser.add_argument('--demo_idx', type=int, default=0, help='Demo index')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    return parser.parse_args()

def write_video(images, output_file_name):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file_name + ".mp4", fourcc, 30.0, (720, 720))
    for img in images:
        img = cv2.resize(img, (720, 720), interpolation=cv2.INTER_AREA)
        video.write(img)
    video.release()

class CameraThread:
    def __init__(self, idx, fps):
        self.idx = idx
        self.fps = fps
        self.cap = cv2.VideoCapture(idx)
        self.frames = []
        self.running = False
        self.thread = threading.Thread(target=self._loop)

    def start(self):
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {self.idx} failed to open.")
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

    def _loop(self):
        interval = 1.0 / self.fps
        while self.running:
            start_time = time.time()
            ret, frame = self.cap.read()
            if ret:
                self.frames.append(frame)
            elapsed = time.time() - start_time
            time.sleep(max(0, interval - elapsed))

class CameraNode(Node):
    def __init__(self, args):
        super().__init__('camera_node')
        self.args = args
        self.cameras = self.get_camera_threads("Brio")
        print(f"Number of cameras: {len(self.cameras)}")
        if not self.cameras:
            print("No cameras found.")
            exit(1)
        for cam in self.cameras:
            cam.start()
        print("started")
        self.tick = time.time()

    def stop_and_save(self):
        for cam in self.cameras:
            cam.stop()

        tock = time.time()
        print("time_taken:", tock - self.tick)

        for cam in self.cameras:
            print("saving", cam.idx)
            images = cam.frames
            images = [cv2.resize(img, (720, 720), interpolation=cv2.INTER_AREA) for img in images]
            images = np.array(images)
            print(images.shape)
            output_path = f"data/videos/{self.args.demo_idx}/{cam.idx}"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            write_video(images, output_path)
            print(f"✅ Video saved to: {os.path.abspath(output_path)}")

    def get_camera_threads(self, key_name):
        indices = []
        for camera_info in enumerate_cameras(cv2.CAP_GSTREAMER):
            if key_name in camera_info.name:
                indices.append(camera_info.index)
                print(f"{camera_info.index}: {camera_info.name}")

        cameras = []
        for idx in indices:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                print(f"Camera {idx} could not be opened.")
                continue

            ret,frame = cap.read()
            if not ret:
                print(f"Camera {idx} opened but cannot read a frame.")
                cap.release()
                continue

            print(f"Camera {idx} is available and working.")
            cap.release()

        # Only now create a camera thread
            try:
                cam = CameraThread(idx, self.args.fps)
                cameras.append(cam)
            except RuntimeError as e:
                print(f"Thread creation failed for camera {idx}: {e}")

        return cameras

def main():
    args = parse_arguments()
    print("FPS:", args.fps)
    rclpy.init()
    camera_node = CameraNode(args)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Saving video...")
    finally:
        camera_node.stop_and_save()
        camera_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()