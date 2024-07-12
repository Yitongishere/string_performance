# String Performace

This is the repository of Digital Human Instrument Performance Research Group.

We are now concentrating on cello playing.

https://metaverse-ai-lab-thu.github.io/String-Performance-Dataset-SPD/static/videos/1.mp4
https://github.com/Metaverse-AI-Lab-THU/String-Performance-Dataset-SPD/tree/main/static/videos/1.mp4

## Install & Data Prep

Clone this repo, and install the dependencies.

The total installation process includes three steps.

### 1. Create Virtual Environment and Install by Pip

We use python=3.8 here. Actually python ~= 3.8 should be fine.
```
git clone https://github.com/Yitongishere/cello_performance.git
cd cello_performance
conda create -n cello_performance python=3.8
conda activate cello_performance
pip install -r requirements.txt
```

### 2. Pytorch Installation
#### CUDA version
`pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html`

#### CPU version (NOT RECOMMEND)
`pip install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html`

You could also follow the [instructions](https://pytorch.org/get-started/previous-versions/) on the PyTorch official site.

### 3. MMCV installation
You need to follow the instruction on the [MMCV official Installation Guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) depending on the type of system, CUDA version, PyTorch version, and MMCV version(mmcv~=2.0.0 is preferred).

Our Example (Windows or linux, torch==1.9.1+cu111, mmcv=2.0.0)

`pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html`

### 4. CREPE installation
You may need to install either Tensorflow or Torch as well.

Tensorflow: Please refer to [CREPE Documentation](https://github.com/marl/crepe)

Torch: Please refer to [TORCHCREPE Documentation](https://github.com/maxrmorrison/torchcrepe)

### 5. TAP-Net installation
If you want to track the cello key points using [TAPIR](https://deepmind-tapir.github.io/), you can either
follow the instructions below or the [official guide](https://github.com/google-deepmind/tapnet#live-demo) provided by google.

Note: For inferring by pytorch models of TAPIR (TAP-Net/BootsTAPIR), you are required to install pytorch>=2.1.0 or you will miss key(s) in state_dict of the checkpoints when loading them and get wrong outputs.

5.1. Install requirements for inference:
`pip install -r requirements_inference.txt`

5.2. Please first switch to the working directory for tapnet:
`cd cello_kp_2d\tapnet`



If you want to use the GPU/TPU version of Jax:

[Linux System, **Recommended**]

Install Jax referring to the [jax manual](https://github.com/google/jax#installation).

[Windows System] 

You may need to use the [Wheel](https://whls.blob.core.windows.net/unstable/index.html).

We use the jaxlib-0.3.22+cuda11.cudnn82-cp38-cp38-win_amd64.whl with the configuration of Windows 11, CUDA 11.1+CUDNN8.2(NVIDIA RTX3060), Python=3.8.

5.3. Download the checkpoint: (Optional, TrackKeypoints.py could automatically download it)
```
mkdir checkpoints
cd checkpoints
wget -P checkpoints https://storage.googleapis.com/dm-tapnet/causal_tapir_checkpoint.npy
```

Here is an effect of presentation.

<!-- <img src="https://metaverse-ai-lab-thu.github.io/String-Performance-Dataset-SPD/static/images/SPD_cello_tapnet_demo_21334181.gif" alt="cello keypoints Example" width="50%" height="50%" /> -->
<img src="https://github.com/Metaverse-AI-Lab-THU/String-Performance-Dataset-SPD/blob/main/static/images/SPD_cello_tapnet_demo_21334181.gif" alt="cello keypoints Example" width="50%" height="50%" />

## 2d Key Points Detection
`model.pth` for pose estimator should be downloaded ahead for `infer.py`

Preferred pose estimator model is [this](https://drive.google.com/file/d/1Oy9O18cYk8Dk776DbxpCPWmJtJCl-OCm/view).

## 2d Bow Detection
We use [YOLOv8](https://github.com/ultralytics/ultralytics) for Bow Detection and you can [download](https://huggingface.co/datasets/shiyi098/string_performance_dataset-SPD/resolve/main/cello_kp_2d/yolov8/checkpoints/bow_detection.pt?download=true) our pretrained model. 

To fit the positions of the bow in the graphics furtherly, we use [DeepLSD](https://github.com/cvg/DeepLSD) and [deeplsd_md.tar](https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_md.tar) as the checkpoint.

## Triangulation
You need to prepare 2D key points coordinates in order to run our `triangulation.py`

## Pitch Detection
You need to prepare audio file with `.wav` format in 'audio' directory. Run `pitch_detect.py` for audio pitch detection, while `freq_position.py` for locating thoses pitches on cello.
