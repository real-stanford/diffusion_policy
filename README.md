# Diffusion Policy for Franka using Franka_ros2

The previous repo by real-stanford uses UR5 robot which has RTDE control interface inbuilt But we need to implement on franka which uses ROS2 to communicate.

[[Project page]](https://diffusion-policy.cs.columbia.edu/)
[[Paper]](https://diffusion-policy.cs.columbia.edu/#paper)
[[Data]](https://diffusion-policy.cs.columbia.edu/data/)
[[Colab (state)]](https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B?usp=sharing)
[[Colab (vision)]](https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg?usp=sharing)


[Cheng Chi](http://cheng-chi.github.io/)<sup>1</sup>,
[Siyuan Feng](https://www.cs.cmu.edu/~sfeng/)<sup>2</sup>,
[Yilun Du](https://yilundu.github.io/)<sup>3</sup>,
[Zhenjia Xu](https://www.zhenjiaxu.com/)<sup>1</sup>,
[Eric Cousineau](https://www.eacousineau.com/)<sup>2</sup>,
[Benjamin Burchfiel](http://www.benburchfiel.com/)<sup>2</sup>,
[Shuran Song](https://www.cs.columbia.edu/~shurans/)<sup>1</sup>

<sup>1</sup>Columbia University,
<sup>2</sup>Toyota Research Institute,
<sup>3</sup>MIT

<img src="media/teaser.png" alt="drawing" width="100%"/>
<img src="media/multimodal_sim.png" alt="drawing" width="100%"/>.


## 🛠️ Installation
### 🖥️ Simulation
To reproduce our simulation benchmark results, install our conda environment on a Linux machine with Nvidia GPU. On Ubuntu 20.04 you need to install the following apt packages for mujoco:
```console
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation: 
```console
$ mamba env create -f conda_environment.yaml
```

but you can use conda as well: 
```console
$ conda env create -f conda_environment.yaml
```

The `conda_environment_macos.yaml` file is only for development on MacOS and does not have full support for benchmarks.

### 🦾 Real Robot
Hardware (for Push-T):
* 1x [Franka_Emika]([(https://franka.de/)) or ([Franka Control_Interface]([(https://frankaemika.github.io/docs/overview.html])) is required
* 3x [RealSense D415](https://www.intelrealsense.com/depth-camera-d415/) or normal webcams
* 1x 3D printed [End effector](https://cad.onshape.com/documents/a818888644a15afa6cc68ee5/w/2885b48b018cda84f425beca/e/3e8771c2124cee024edd2fed?renderMode=0&uiState=63ffcba6631ca919895e64e5)
* 1x 3D printed [T-block](https://cad.onshape.com/documents/f1140134e38f6ed6902648d5/w/a78cf81827600e4ff4058d03/e/f35f57fb7589f72e05c76caf?renderMode=0&uiState=63ffcbc9af4a881b344898ee)
* USB-C cables and screws for RealSense

Software:
* Ubuntu 22.04 (tested)
* Mujoco dependencies: 
`sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf`
* [RealSense SDK](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)
* Spacemouse dependencies: 
`sudo apt install libspnav-dev spacenavd; sudo systemctl start spacenavd`
* Conda environment `mamba env create -f conda_environment_real.yaml`


## 🦾 Demo, Training and Eval on a Real Robot
Make sure your Franka robot is running has already setup with franka_ros2 of not follow here
### Download Training Data
Under the repo root, create data subdirectory:
```console
[diffusion_policy]$ mkdir data && cd data
```

Download the corresponding zip file from [(https://storage.googleapis.com/franka_real_data/franka_real.zip)]((https://storage.googleapis.com/franka_real_data/franka_real.zip))
```console
[data]$ wget [[data](https://storage.googleapis.com/franka_real_data/franka_real.zip)](https://storage.googleapis.com/franka_real_data/franka_real.zip)
```

Extract training data:
```console
[data]$ unzip franka_real && rm -f franka_real.zip && cd ..
```

If you need to create your own data or collect new data do this 
```console
[diffusion_policy]$ cd ~/diff_ws/data_collection
[data_collection]$ python3 collect_pose_video.py --fps 10 --demo_idx 0   #give your desired fps and keep increasing demo_idx upto no.of datapoints
```
To train the collected data
```console
[data_collection] cd ~diff_ws/diffusion_policy
[diffusion_policy] python train.py --config-name=train_diffusion_unet_real_image_workspace task.dataset_path=/home/user/diff_ws/diffusion_policy/data
```
Assuming the training has finished and you have a checkpoint at `data/outputs/blah/checkpoints/latest.ckpt`, launch the evaluation script with:
```console
python eval_real_robot.py -i data/outputs/blah/checkpoints/latest.ckpt -o data/eval_pusht_real --robot_ip 192.168.1.11
```
