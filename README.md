# Cello Performace

This is the repository of Digital Human Instrument Performance Research Group.

We are now concentrating on cello playing.

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

You could also follow the instructions on the PyTorch official site: `https://pytorch.org/get-started/previous-versions/`

### 3. MMCV installation
You need to follow the instruction on the [MMCV official Installation Guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) depending on the type of system, CUDA version, PyTorch version, and MMCV version(mmcv~=2.0.0 is preferred).

Our Example (Windows or linux, torch==1.9.1+cu111, mmcv=2.0.0)

`pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html`

### 4. CREPE installation
You may need to install tensorflow as well.
Please refer to [CREPE Documentation](https://github.com/marl/crepe)

## 2d Key Points Detection
`model.pth` for pose estimator should be downloaded ahead for `infer.py`

Preferred pose estimator model is [this](https://drive.google.com/file/d/1Oy9O18cYk8Dk776DbxpCPWmJtJCl-OCm/view).

## Triangulation
You need to prepare 2D key points coordinates in order to run our `triangulation.py`

## Pitch Detection
