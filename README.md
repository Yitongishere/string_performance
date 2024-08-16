# String Performace 
Visit our [project page](https://metaverse-ai-lab-thu.github.io/String-Performance-Dataset-SPD/) for more details.

Get the [String Performance Dataset (SPD)](https://forms.gle/oNFFu3NRoVwkV1Xd9). 

[![poster.png](https://github.com/Metaverse-AI-Lab-THU/String-Performance-Dataset-SPD/blob/main/static/images/poster.png)](https://metaverse-ai-lab-thu.github.io/String-Performance-Dataset-SPD/static/images/poster.png)

And if you want to delve into the code or reproduce the final MoCap results from the raw data, please check the following.

## 1. Data Prepration

### 1.1. Get the data
Download the raw data from the dataset. By downloading any piece of data, you will get the RGB videos from various shooting angles (format in `.avi`), the performance audio (format in `.wav`), and the info summary of the corresponding piece (format in `.json`). The `summary.json` include the metadata of the performance itself, the camara parameters, and the frame range of the MoCap results corresponding to the original video.

### 1.2. Get the code
Clone this repo, and install the dependencies.


## 2. MoCap Pipeline: Acquire MoCap Results from Raw Data
You can modify the arguments in the scripts to meet your requirements.

### 2.1. Extracting Frames from Videos
This process is implemented in the `frame_extract_pipeline.py` which is called by the `script_frame_extract.py`. 

### 2.2. Human Keypoints Detection from 2D imagery
This process is implemented in the `infer_pipeline.py` which is called by the `script_infer_humanpose.py`. 

`model.pth` for pose estimator should be downloaded ahead for `infer.py`

Preferred pose estimator model is [this](https://drive.google.com/file/d/1Oy9O18cYk8Dk776DbxpCPWmJtJCl-OCm/view).

### 2.3. Instrument Keypoints Tracking from 2D imagery
This process is implemented in the `TrackKeyPoints_pipeline.py` which is called by the `script_track_key_points.py`. 

#### 2.3.1. Instrument Body
Track the keypoints of instruments using [TAPIR](https://deepmind-tapir.github.io/).


#### 2.3.2. Bow
Use [YOLOv8](https://github.com/ultralytics/ultralytics) for Bow Detection and you can [download](https://huggingface.co/datasets/shiyi098/string_performance_dataset-SPD/resolve/main/cello_kp_2d/yolov8/checkpoints/bow_detection.pt?download=true) our pretrained model. 

To fit the positions of the bow in the graphics furtherly, we use [DeepLSD](https://github.com/cvg/DeepLSD) and [deeplsd_md.tar](https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_md.tar) as the checkpoint.

### 2.4. Triangulation: Get 3D initial pose
With the 2D keypoints obtained from above steps, we apply triangulation for getting the initial 3D pose.

This process is implemented in the `triangulation_pipeline.py` which is called by the `script_triangulation.py`. 

### 2.5. Get Clues from Audio
This process is implemented in the `contact_points_pipeline.py` which is called by the `script_cp_detection.py`.

#### 2.5.1. Pitch Detection
Put the raw audio file (format in `.wav`) in the 'audio/wavs' directory.

Use CREPE for pitch detection to obtain the pitch curve. CREPE is a realiable pitch tracker with more than 99.9% accuracy with 25 cents as the threshold.

#### 2.5.2. String Mapping for Note-Playing Position
Based on the pitch curve and the Pitch-Finger model, we infer the real-world note-playing position and their changes during the performance.

### 2.6. Pose Refinement by HPE and Audio-Guided Approach
#### 2.6.1. Hand Pose from HPE
HPE model is designed for obtaining the 6d rotation representation from 2D imagery. We apply hand estimation on the 2D imagery from various shooting angles before integrating these results. With the integrated rotation, we graft the hand pose onto the whole body skeleton.

HPE model is currently not an open-source algorithm in the near feature as it is now serving as commmercial use. You may use the [EasyMocap]([https:](https://github.com/zju3dv/EasyMocap)) toolbox to obtain the MANO parameters from monocular videos and convert the pose parameters to the 6D representation as its alternative.

This process is implemented in the `integrate_handpose_pipeline.py` which is called by the `script_integrate_ik.py`.

#### 2.6.2. Audio-Guided Inverse Kinematics
This process is implemented in the `inverse_kinematic_pipeline.py` which is called by the `script_integrate_ik.py`.


## 3. Installation

### 3.1. Create Virtual Environment and Install by Pip
We use python=3.8 here. Actually python ~= 3.8 should be fine.
```
git clone https://github.com/Yitongishere/string_performance.git
cd string_performance
conda create -n string_performance python=3.8
conda activate string_performance
pip install -r requirements.txt
```

### 3.2. Pytorch Installation
#### 3.2.1. CUDA version
`pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html`

#### 3.2.2. CPU version (NOT RECOMMEND)
`pip install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html`

You could also follow the [instructions](https://pytorch.org/get-started/previous-versions/) on the PyTorch official site.

### 3.3. MMCV installation (for human keypoints detection from 2D imagery)
You need to follow the instruction on the [MMCV official Installation Guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) depending on the type of system, CUDA version, PyTorch version, and MMCV version(mmcv~=2.0.0 is preferred).

Our Example (Windows or linux, torch==1.9.1+cu111, mmcv=2.0.0)

`pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html`

### 3.4. CREPE installation (for pitch tracking)
You may need to install either Tensorflow or Torch as well.

Tensorflow: Please refer to [CREPE Documentation](https://github.com/marl/crepe)

Torch: Please refer to [TORCHCREPE Documentation](https://github.com/maxrmorrison/torchcrepe)

### 3.5. TAP-Net installation (for instrument keypoints tracking from 2D imagery)

Follow the instructions below or the [official guide](https://github.com/google-deepmind/tapnet#live-demo) provided by google.

Note: For inferring by pytorch models of TAPIR (TAP-Net/BootsTAPIR), you are required to install pytorch>=2.1.0 or you will miss key(s) in state_dict of the checkpoints when loading them and get wrong outputs.

#### 3.5.1. Install requirements for inference:
`pip install -r requirements_inference.txt`

#### 3.5.2. Please first switch to the working directory for tapnet:
`cd cello_kp_2d\tapnet`



If you want to use the GPU/TPU version of Jax:

[Linux System, **Recommended**]

Install Jax referring to the [jax manual](https://github.com/google/jax#installation).

[Windows System] 

You may need to use the [Wheel](https://whls.blob.core.windows.net/unstable/index.html).

We use the jaxlib-0.3.22+cuda11.cudnn82-cp38-cp38-win_amd64.whl with the configuration of Windows 11, CUDA 11.1+CUDNN8.2(NVIDIA RTX3060), Python=3.8.

#### 3.5.3. Download the checkpoint: (Optional, TrackKeypoints.py could automatically download it)
```
mkdir checkpoints
cd checkpoints
wget -P checkpoints https://storage.googleapis.com/dm-tapnet/causal_tapir_checkpoint.npy
```

Here is an effect of presentation.

<!-- <img src="https://metaverse-ai-lab-thu.github.io/String-Performance-Dataset-SPD/static/images/SPD_cello_tapnet_demo_21334181.gif" alt="cello keypoints Example" width="50%" height="50%" /> -->
<img src="https://github.com/Metaverse-AI-Lab-THU/String-Performance-Dataset-SPD/blob/main/static/images/SPD_cello_tapnet_demo_21334181.gif" alt="cello keypoints Example" width="50%" height="50%" />


