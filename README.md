# UniOwner-Comm

## Installation

This project is based on the [Coalign](https://github.com/yifanlu0227/CoAlign) project. Please refer to the original project’s installation guide.

Please visit the feishu docs [CoAlign Installation Guide](https://udtkdfu8mk.feishu.cn/docx/LlMpdu3pNoCS94xxhjMcOWIynie) for details!

Or you can refer to [OpenCOOD data introduction](https://opencood.readthedocs.io/en/latest/md_files/data_intro.html)
and [OpenCOOD installation](https://opencood.readthedocs.io/en/latest/md_files/installation.html) guide to prepare
data and install CoAlign. The installation is totally the same as OpenCOOD, except some dependent packages required by CoAlign.

## Dataset

First, use `mkdir` to create a `dataset` folder in this project directory, then download or link the dataset using `ln`, organizing the dataset according to the format below:

```text
dataset
├── my_dair_v2x 
│   └── v2x_c
├── OPV2V
│   ├── test
│   ├── train
│   └── validate
├── V2X-Sim-2.0
│   ├── v2.0
│   ├── maps
│   └── sweeps
└── v2xsim2_info
    ├── v2xsim_infos_test.pkl
    ├── v2xsim_infos_train.pkl
    └── v2xsim_infos_val.pkl
```

## Train and Inference

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp_syncbn.py \
  -y opencood/hypes_yaml/opv2v/lidar_only/pointpillar_uniowner_comm.yaml \
  -p uoc_bs4_att --run_test
```

## Acknowlege

This project based on the code of [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) and [CoAlign](https://github.com/yifanlu0227/CoAlign), thanks to their great code framework.

