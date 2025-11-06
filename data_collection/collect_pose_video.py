import argparse
from multiprocessing import get_context
import subprocess
from diffusion_policy.data_collection import human_demo_pose, human_demo_video
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

def run_demo_collect(demo_idx, fps):
    script_path = os.path.join(script_dir, 'human_demo_video.py')
    subprocess.run([sys.executable, script_path,
                    '--demo_idx', str(demo_idx),
                    '--fps', str(fps)])
    
def run_human_data(fps, demo_idx):
    script_path = os.path.join(script_dir, 'human_demo_pose.py')
    subprocess.run([sys.executable, script_path,
                    '--demo_idx', str(demo_idx),
                    '--fps', str(fps)])

def main():
    parser = argparse.ArgumentParser(description='Run both human video capture and pose logger')
    parser.add_argument('--demo_idx', type=int, default=0, help='Demo index')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    args = parser.parse_args()

    ctx = get_context("spawn")  # ✅ use safe context for ROS

    p1 = ctx.Process(target=run_demo_collect, args=(args.demo_idx, args.fps))
    p2 = ctx.Process(target=run_human_data, args=(args.fps, args.demo_idx))

    p1.start()
    p2.start()

    try:
        p1.join()
        p2.join()
    except KeyboardInterrupt:
        print("Interrupted. Terminating processes...")
        p1.terminate()
        p2.terminate()

if __name__ == '__main__':
    main()