# CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir opencood/logs/v2xset/baseline/where2comm_bs2_2025_11_01_12_38 # --no_score #  [--fusion_method intermediate]
# CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_bandwidth.py --model_dir opencood/logs/opv2v/baseline/MonoSender_version2_bs2_2025_10_29_12_07

# CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_bandwidth.py --model_dir opencood/logs/opv2v/baseline/MonoSender_version2_bs2_2025_10_29_12_07 \
#  --k_ratio 0.9 &&
#   CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_bandwidth.py --model_dir opencood/logs/opv2v/baseline/MonoSender_version2_bs2_2025_10_29_12_07 \
#  --k_ratio 0.8  &&
#   CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_bandwidth.py --model_dir opencood/logs/opv2v/baseline/MonoSender_version2_bs2_2025_10_29_12_07 \
#  --k_ratio 0.7  &&
#   CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_bandwidth.py --model_dir opencood/logs/opv2v/baseline/MonoSender_version2_bs2_2025_10_29_12_07 \
#  --k_ratio 0.6  &&
#  CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_bandwidth.py --model_dir opencood/logs/opv2v/baseline/MonoSender_version2_bs2_2025_10_29_12_07 \
#  --k_ratio 0.5  &&
#  CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_bandwidth.py --model_dir opencood/logs/opv2v/baseline/MonoSender_version2_bs2_2025_10_29_12_07 \
#  --k_ratio 0.4  &&
#  CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_bandwidth.py --model_dir opencood/logs/opv2v/baseline/MonoSender_version2_bs2_2025_10_29_12_07 \
#  --k_ratio 0.3  &&
#  CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_bandwidth.py --model_dir opencood/logs/opv2v/baseline/MonoSender_version2_bs2_2025_10_29_12_07 \
#  --k_ratio 0.2  &&
#  CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_bandwidth.py --model_dir opencood/logs/opv2v/baseline/MonoSender_version2_bs2_2025_10_29_12_07 \
#  --k_ratio 0.1 &&
# # Where2comm
# CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_bandwidth.py --model_dir opencood/logs/opv2v/baseline/where2comm_bs2_2025_10_28_15_23 \
#  --threshold 0.0 &&
# CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_bandwidth.py --model_dir opencood/logs/opv2v/baseline/where2comm_bs2_2025_10_28_15_23 \
#  --threshold 0.1 &&
# CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_bandwidth.py --model_dir opencood/logs/opv2v/baseline/where2comm_bs2_2025_10_28_15_23 \
#  --threshold 0.4 &&
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_bandwidth.py --model_dir opencood/logs/opv2v/baseline/where2comm_bs2_2025_10_28_15_23 \
 --k_ratio 0.01 &&
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_bandwidth.py --model_dir opencood/logs/opv2v/baseline/where2comm_bs2_2025_10_28_15_23 \
 --k_ratio 0.001