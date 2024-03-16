an
# Sunshine to Rainstorm: Cross-Weather Knowledge Distillation for Robust 3D Object Detection (AAAI2024)


This is a official code release of SRKD&DRET (Sunshine to Rainstorm: Cross-Weather Knowledge Distillation for Robust 3D Object Detection). 

## Framework
![image](https://github.com/ylwhxht/SRKD-DRET/blob/main/framework.png?raw=true)

## Getting Started

This code is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and [SPRAY](https://github.com/Wachemanston/Reconstruction-and-Synthesis-of-Lidar-Point-Clouds-of-Spray).

More detailed guidance coming soon...

### DRET

#### config

**please just refer to [SPRAY](https://github.com/Wachemanston/Reconstruction-and-Synthesis-of-Lidar-Point-Clouds-of-Spray), or we will release more detailed guidance later.**

*note：We have only uploaded some of the code that we have modified. If there are too many other libraries and files, please follow the SPRAY official website's instructions for installation. If you have any questions, please feel free to raise them*


Meanwhile, we can provide the rain particles obtained during the DRET process (txt file, Nx3 (x, y, z))

If needed, please contact huangxun@stu.xmu.edu. But please note that we cannot provide the final rainy point cloud due to Waymo's policy.


### SRKD

#### config

please just refer to [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and [SPRAY](https://github.com/Wachemanston/Reconstruction-and-Synthesis-of-Lidar-Point-Clouds-of-Spray), or we will release more detailed guidance later.

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
