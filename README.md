# Cello Performace

This is the repository of Digital Human Instrument Performance Research Group.

We are now concentrating on cello playing.

## Install & Data Prep

Clone this repo, and install the dependencies.

Python >= 3.8 should be fine.

```
git clone https://github.com/Yitongishere/cello_performance.git
cd cello_performance
conda create -n cello_performance python=3.8
conda activate cello_performance
pip install -r requirements.txt
```

## 2d Key Points Detection
`model.pth` for pose estimator should be downloaded ahead for `infer.py`

Preferred pose estimator model link: https://drive.google.com/file/d/1Oy9O18cYk8Dk776DbxpCPWmJtJCl-OCm/view

## Triangulation
You need to prepare 2D key points coordinates in order to run our `triangulation.py`

