VOCE: Variational Optimization with Conservative Estimation for Offline Safe Reinforcement Learning
==================================
This project provides the open source implementation of the VOCE method introduced in the paper: "VOCE: Variational Optimization with Conservative Estimation for Offline Safe Reinforcement Learning" . 

## Installation
#### 1. System requirements
- Tested in Ubuntu 20.04, should be fine with Ubuntu 18.04
- I would recommend to use [Anaconda3](https://docs.anaconda.com/anaconda/install/) for python env management

#### 2. System-wise dependencies installation
Since we will use `mujoco` and `mujocu_py` for the `safety-gym` environment experiments, so some dependencies should be installed with `sudo` permissions. To install the dependencies, run
```
cd envs/safety-gym 
bash setup_dependency.sh
bash setup_mujoco.sh
source ~/.bashrc
```
And enter the sudo password to finish dependencies installation.

#### 3. Anaconda Python env setup
Back to the repo root folder, **activate a python 3.7 virtual anaconda env**, and then run
```
conda create -n voce python=3.7
conda activate voce
cd ../.. && bash install_all.sh
```
It will install the modified `safety_gym` and this repo's python package dependencies that are listed in `requirement.txt`.  Note that we modify the original environment repos to accelerate the training process, so not using our provided envs may require additional hyper-parameters fine-tuning.

#### 4. Install pytorch
To install the pytroch, run 
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```
You can also refer to this tutorial [tutorial here](https://pytorch.org/get-started/locally/) for installation on your platform.

## Training
### How to run a single experiment
Before running, you need to download the dataset. You can download the dataset using this [link](https://drive.google.com/file/d/1Kqz6HJgafcqQ2QtcOQA9xDgYj_-ru_tQ/view?usp=sharing) and move it to the "./buffer" directory.

Simply run
```
python script/voce_main.py -offline -wan
```

### Check the result of the experiment
You can log in to Wandb to view the training results.
