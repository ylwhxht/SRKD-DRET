
# Sunshine to Rainstorm: Cross-Weather Knowledge Distillation for Robust 3D Object Detection (AAAI2024)


This is a official code release of SRKD&DRET (Sunshine to Rainstorm: Cross-Weather Knowledge Distillation for Robust 3D Object Detection). 

## Framework
![image](https://github.com/ylwhxht/SRKD-DRET/blob/main/framework.png?raw=true)

## Getting Started (Installation, Environment)

This code is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), [Fog Simulation work](https://github.com/MartinHahner/LiDAR_fog_sim) and [SPRAY](https://github.com/Wachemanston/Reconstruction-and-Synthesis-of-Lidar-Point-Clouds-of-Spray).

Configuration files and installation can refer to their official website guidelines.

We tested on the following environment:

* Python 3.8.13 (3.10+ does not support open3d.)
* Ubuntu 18.04/20.04
* Torch 1.11.0+cu113
* CUDA 11.3


## DRET 


### First Stage (DRET folder)

#### Installation and Running

**please just refer to [SPRAY](https://github.com/Wachemanston/Reconstruction-and-Synthesis-of-Lidar-Point-Clouds-of-Spray).**

*note: If you have any questions, please feel free to raise issue*

The workload of reproducing SPRAY based on Unity3D virtual engine is relatively high.

As a consequence, we can provide the following rain particles obtained during the DRET process (txt file, Nx3 (x, y, z)). But please note that we cannot provide the final rainy point cloud due to Waymo's policy. Just download the rain particles and run the second stage of scene processing code to obtain the final rain point cloud.


[Rain Particle File Link Here](https://drive.google.com/file/d/1q5HMEo3dayOy1GqRCHmVNiYxCI1jkABd/view?usp=drive_link)


### Second Stage (DRET Rain Simulation folder)
#### Installation

please just refer to [Fog Simulation work](https://github.com/MartinHahner/LiDAR_fog_sim)

#### Running

By running **DERP_rain_simulation.py** and changing it to the path of the rain particle txt file.


## SRKD

### Installation

please just refer to [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).


### Running 
####  train
* **multi-GPUs**

`bash scripts/dist_train.sh num_gpus --cfg_file your_cfg --batch_size your_batch --workers your_worker --epochs your_epoch --extra_tag your_tag`

**for example**

`bash scripts/dist_train.sh 2 --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_KD.yaml --batch_size 4 --workers 2 --epochs 50 --extra_tag "pv_rcnn++_fullKD"`

* **single-GPU**


`python3 train.py --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_KD.yaml --batch_size 4 --workers 2 --epochs 50 --extra_tag "pv_rcnn++_fullKD"`

#### test
* **all_ckpt**


`/usr/bin/python3 test.py --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_KD.yaml --batch_size 2 --extra_tag 'pv_rcnn++_fullKD' --eval_all`

* **single_ckpt**

 
`test.py --cfg_file cfgs/waymo_models/pv_rcnn_plusplus_KD.yaml --batch_size 2 --extra_tag  'pv_rcnn++_fullKD_delincar_4gpus' --ckpt /home/hx/OpenPCDet/output/waymo_models/pv_rcnn_plusplus_KD/pv_rcnn++_fullKD_delincar_4gpus/ckpt/checkpoint_epoch_27.pth`
