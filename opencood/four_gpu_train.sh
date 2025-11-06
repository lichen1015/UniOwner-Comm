# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp_syncbn.py \
#   -y opencood/hypes_yaml/v2xset/pointpillar_cosdh.yaml \
#   -p baseline/cosdh_bs2 --run_test &&
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp_syncbn.py \
#   -y opencood/hypes_yaml/v2xset/pointpillar_uoc.yaml \
#   -p baseline/uoc_bs2 --run_test 
# # Coalign 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp_syncbn.py \
#   -y opencood/hypes_yaml/opv2v/lidar_only/pointpillar_coalign.yaml \
#   -p baseline/coalign_bs2 --run_test &&
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp_syncbn.py \
#   -y opencood/hypes_yaml/opv2v/lidar_only/pointpillar_disconet.yaml \
#   -p baseline/disconet_bs2 --run_test &&
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp_syncbn.py \
#   -y opencood/hypes_yaml/opv2v/lidar_only/pointpillar_v2vnet.yaml \
#   -p baseline/v2vnet_bs2 --run_test &&
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp_syncbn.py \
#   -y opencood/hypes_yaml/opv2v/lidar_only/pointpillar_v2xvit.yaml \
#   -p baseline/v2xvit_bs2 --run_test
# DAIR-V2X
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp_syncbn.py \
#   -y opencood/hypes_yaml/dairv2x/lidar_only/pointpillar_coalign.yaml \
#   -p baseline/coalign_bs2 --run_test &&
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp_syncbn.py \
#   -y opencood/hypes_yaml/dairv2x/lidar_only/pointpillar_disconet.yaml \
#   -p baseline/disconet_bs2 --run_test &&
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp_syncbn.py \
#   -y opencood/hypes_yaml/dairv2x/lidar_only/pointpillar_v2vnet.yaml \
#   -p baseline/v2vnet_bs2 --run_test &&
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp_syncbn.py \
#   -y opencood/hypes_yaml/dairv2x/lidar_only/pointpillar_v2xvit.yaml \
#   -p baseline/v2xvit_bs1 --run_test &&
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp_syncbn.py \
#   -y opencood/hypes_yaml/dairv2x/lidar_only/pointpillar_where2comm.yaml \
#   -p baseline/where2comm_bs4 --run_test &&
#   # v2x-sim
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp_syncbn.py \
#   -y opencood/hypes_yaml/v2xsim/lidar_only/pointpillar_coalign.yaml \
#   -p baseline/coalign_bs4 --run_test &&
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp_syncbn.py \
#   -y opencood/hypes_yaml/v2xsim/lidar_only/pointpillar_disconet.yaml \
#   -p baseline/disconet_bs2 --run_test &&
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp_syncbn.py \
#   -y opencood/hypes_yaml/v2xsim/lidar_only/pointpillar_v2vnet.yaml \
#   -p baseline/v2vnet_bs4 --run_test &&
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp_syncbn.py \
#   -y opencood/hypes_yaml/v2xsim/lidar_only/pointpillar_v2xvit.yaml \
#   -p baseline/v2xvit_bs1 --run_test &&

  # v2xset
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp_syncbn.py \
  -y opencood/hypes_yaml/v2xset/lidar_only/pointpillar_coalign.yaml \
  -p baseline/coalign_bs4 --run_test &&
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp_syncbn.py \
  -y opencood/hypes_yaml/v2xset/lidar_only/pointpillar_disconet.yaml \
  -p baseline/disconet_bs2 --run_test &&
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp_syncbn.py \
  -y opencood/hypes_yaml/v2xset/lidar_only/pointpillar_v2vnet.yaml \
  -p baseline/v2vnet_bs4 --run_test &&
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node=4 --use_env opencood/tools/train_ddp_syncbn.py \
  -y opencood/hypes_yaml/v2xset/lidar_only/pointpillar_v2xvit.yaml \
  -p baseline/v2xvit_bs1 --run_test
# 
# # heuristic
# python script_dist_train.py \
#   -y opencood/hypes_yaml/opv2v/lidar_only/pointpillar_selective_comm.yaml \
#   --mode heuristic --topk 1 \
#   --project vis_noise_byadd/heuristic_k1_bs4 \
#   --gpus 0,1,2,3 --nproc 4 --workdir . &&
#   python script_dist_train.py \
#   -y opencood/hypes_yaml/opv2v/lidar_only/pointpillar_selective_comm.yaml \
#   --mode heuristic --topk 2 \
#   --project vis_noise_byadd/heuristic_k2_bs4 \
#   --gpus 0,1,2,3 --nproc 4 --workdir . &&
#   python script_dist_train.py \
#   -y opencood/hypes_yaml/opv2v/lidar_only/pointpillar_selective_comm.yaml \
#   --mode heuristic --topk 3 \
#   --project vis_noise_byadd/heuristic_k3_bs4 \
#   --gpus 0,1,2,3 --nproc 4 --workdir . &&
#   python script_dist_train.py \
#   -y opencood/hypes_yaml/opv2v/lidar_only/pointpillar_selective_comm.yaml \
#   --mode heuristic --topk 4 \
#   --project vis_noise_byadd/heuristic_k4_bs4 \
#   --gpus 0,1,2,3 --nproc 4 --workdir . &&
# python script_dist_train.py \
#   -y opencood/hypes_yaml/opv2v/lidar_only/pointpillar_selective_comm.yaml \
#   --mode heuristic --topk 5 \
#   --project vis_noise_byadd/heuristic_k5_bs4 \
#   --gpus 0,1,2,3 --nproc 4 --workdir . &&
# heuristic_roi
# python script_dist_train.py \
#   -y opencood/hypes_yaml/opv2v/lidar_only/pointpillar_selective_comm.yaml \
#   --mode heuristic_roi --topk 1 \
#   --project vis_noise_byadd/heuristic_roi_k1_bs4 \
#   --gpus 0,1,2,3 --nproc 4 --workdir . &&
# python script_dist_train.py \
#   -y opencood/hypes_yaml/opv2v/lidar_only/pointpillar_selective_comm.yaml \
#   --mode heuristic_roi --topk 2 \
#   --project vis_noise_byadd/heuristic_roi_k2_bs4 \
#   --gpus 0,1,2,3 --nproc 4 --workdir . &&
# python script_dist_train.py \
#   -y opencood/hypes_yaml/opv2v/lidar_only/pointpillar_selective_comm.yaml \
#   --mode heuristic_roi --topk 3 \
#   --project vis_noise_byadd/heuristic_roi_k3_bs4 \
#   --gpus 0,1,2,3 --nproc 4 --workdir . &&
# python script_dist_train.py \
#   -y opencood/hypes_yaml/opv2v/lidar_only/pointpillar_selective_comm.yaml \
#   --mode heuristic_roi --topk 4 \
#   --project vis_noise_byadd/heuristic_roi_k4_bs4 \
#   --gpus 0,1,2,3 --nproc 4 --workdir . &&
# python script_dist_train.py \
#   -y opencood/hypes_yaml/opv2v/lidar_only/pointpillar_selective_comm.yaml \
#   --mode heuristic_roi --topk 5 \
#   --project vis_noise_byadd/heuristic_roi_k5_bs4 \
#   --gpus 0,1,2,3 --nproc 4 --workdir . 

# heuristic_noise
# python script_dist_train.py \
#   -y opencood/hypes_yaml/opv2v/lidar_only/pointpillar_selective_comm.yaml \
#   --mode heuristic_noise --topk 1 \
#   --project vis_noise_byadd/heuristic_noise_roi_k1_bs4 \
#   --gpus 0,1,2,3 --nproc 4 --workdir . &&
# python script_dist_train.py \
#   -y opencood/hypes_yaml/opv2v/lidar_only/pointpillar_selective_comm.yaml \
#   --mode heuristic_noise --topk 2 \
#   --project vis_noise_byadd/heuristic_noise_roi_k2_bs4 \
#   --gpus 0,1,2,3 --nproc 4 --workdir . &&
# python script_dist_train.py \
#   -y opencood/hypes_yaml/opv2v/lidar_only/pointpillar_selective_comm.yaml \
#   --mode heuristic_noise --topk 3 \
#   --project vis_noise_byadd/heuristic_noise_roi_k3_bs4 \
#   --gpus 0,1,2,3 --nproc 4 --workdir . &&
# python script_dist_train.py \
#   -y opencood/hypes_yaml/opv2v/lidar_only/pointpillar_selective_comm.yaml \
#   --mode heuristic_noise --topk 4 \
#   --project vis_noise_byadd/heuristic_noise_roi_k4_bs4 \
#   --gpus 0,1,2,3 --nproc 4 --workdir . &&
# python script_dist_train.py \
#   -y opencood/hypes_yaml/opv2v/lidar_only/pointpillar_selective_comm.yaml \
#   --mode heuristic_noise --topk 5 \
#   --project vis_noise_byadd/heuristic_noise_roi_k5_bs4 \
#   --gpus 0,1,2,3 --nproc 4 --workdir .
