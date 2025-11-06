# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
import argparse
from torch.utils.data import DataLoader

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.visualization import vis_utils
from opencood.data_utils.datasets import build_dataset
from torch.utils.data import DataLoader, Subset

# ======== 配置 ========
SAVE_VIS_INTERVAL = 40                # 和 inference 一样：每 40 帧保存一次
OUTPUT_ROOT = "./visualization_results/opv2v"
DRAW_BOXES = True                    # False=无框；如需带框可设 True
NUM_WORKERS = 0                       # 调试稳妥，OK 后可改 2
SAMPLED_RANGE = None                  # 例如 range(1330,1360)；None 表示整套测试集

def vis_parser():
    parser = argparse.ArgumentParser(description="data visualization")
    parser.add_argument('--color_mode', type=str, default="intensity",
                        help='lidar color rendering mode, e.g. intensity,'
                             'z-value or constant.')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    current_path = os.path.dirname(os.path.realpath(__file__))
    hypes = load_yaml(os.path.join(current_path, '../hypes_yaml/visualization_opv2v.yaml'))

    # 和 inference.py 保持**相同**的数据顺序设置（很重要：shuffle=False）
    dataset = build_dataset(hypes, visualize=True, train=False)
    subset = dataset if SAMPLED_RANGE is None else Subset(dataset, SAMPLED_RANGE)

    loader = DataLoader(
        subset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=dataset.collate_batch_test, pin_memory=False
    )

    opt = vis_parser()
    vis_utils.visualize_sequence_real_dataloader(loader,
                                                 save_rgb_dir='visualization_results/panorama/',
                                            # params['postprocess']['order'],
                                            # color_mode=opt.color_mode,
                                            # debug_keys=True
                                            )
