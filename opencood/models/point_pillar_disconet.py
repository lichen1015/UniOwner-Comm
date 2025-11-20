# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.data_utils.post_processor import UncertaintyVoxelPostprocessor
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.utils.transformation_utils import normalize_pairwise_tfm, regroup
from opencood.models.fuse_modules.fusion_in_one import DiscoFusion
from opencood.models.point_pillar_comm_multiscale import _elem_bits

class PointPillarDiscoNet(nn.Module):
    def __init__(self, args):
        super(PointPillarDiscoNet, self).__init__()
        self.discrete_ratio = args['voxel_size'][0]
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])
        self.voxel_size = args['voxel_size']
        
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]
        
        self.compression = False
        self.compression_ratio = 1
        if "compression" in args:
            self.compression = True
            self.naive_compressor = NaiveCompressor(self.out_channel, args['compression'])
            self.compression_ratio = args['compression']
            print(f"Compress ratio: {args['compression']}")

        self.fusion_net = DiscoFusion(self.out_channel)

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
            
        self.k_ratio = args.get('k_ratio', 0)
        if 'where2comm' in args:
            self.k_ratio =  args['where2comm']['communication']['k_ratio']
            print(f"Coalign modify K ratio={self.k_ratio}")

    def forward(self, data_dict):

            voxel_features = data_dict['processed_lidar']['voxel_features']
            voxel_coords = data_dict['processed_lidar']['voxel_coords']
            voxel_num_points = data_dict['processed_lidar']['voxel_num_points']

            # --- (DiscoNet 特有：教师分支，与带宽计算无关) ---
            teacher_voxel_features = data_dict['teacher_processed_lidar']['voxel_features']
            teacher_voxel_coords = data_dict['teacher_processed_lidar']['voxel_coords']
            teacher_voxel_num_points = data_dict['teacher_processed_lidar']['voxel_num_points']
            # --- (DiscoNet 特有结束) ---

            record_len = data_dict['record_len']
            lidar_pose = data_dict['lidar_pose']
            pairwise_t_matrix = data_dict['pairwise_t_matrix']

            batch_dict = {'voxel_features': voxel_features,
                        'voxel_coords': voxel_coords,
                        'voxel_num_points': voxel_num_points,
                        'record_len': record_len,
                        'pairwise_t_matrix': pairwise_t_matrix}

            batch_dict = self.pillar_vfe(batch_dict)
            batch_dict = self.scatter(batch_dict)

            _, _, H0, W0 = batch_dict['spatial_features'].shape
            normalized_affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])

            batch_dict = self.backbone(batch_dict)

            spatial_features_2d = batch_dict['spatial_features_2d']
            if self.shrink_flag:
                spatial_features_2d = self.shrink_conv(spatial_features_2d)
            
            if self.compression:
                spatial_features_2d = self.naive_compressor(spatial_features_2d, not self.training)

            # --- (新增) 带宽限制与计算 ---
            N, Ci, Hi, Wi = spatial_features_2d.shape

            # 1. 获取带宽计算的必要参数
            avg_collaborators = (sum(record_len) / (len(record_len) + 1e-5)) - 1.0
            fps = getattr(self, "eval_fps", 10.0)
            
            # 假定在 __init__ 中定义了 self.k_ratio
            K_RATIO = getattr(self, "k_ratio", 1.0) 
            use_top_k = (K_RATIO > 0.0 and K_RATIO < 1.0) and (not self.training)
            
            comm_rate_effective = 1.0 # 默认通信率
            feature_to_fuse = spatial_features_2d # 默认发送的特征

            # 2. 如果 k_ratio 生效 (例如 0.3) 且不在训练中
            if use_top_k:
                # 2.1. 计算特征“能量”作为置信图 (N, 1, Hi, Wi)
                confidence_map = torch.norm(spatial_features_2d, dim=1, keepdim=True)
                
                # 2.2. 计算 K
                K = int(Hi * Wi * K_RATIO)

                # 2.3. 展平并找到 Top-K 索引
                flat_confidence = confidence_map.reshape(N, Hi * Wi)
                _, indices = torch.topk(flat_confidence, k=K, dim=-1)

                # 2.4. 创建掩码 (N, H*W)
                communication_mask_flat = torch.zeros_like(flat_confidence)
                communication_mask_flat.scatter_(
                    dim=-1, 
                    index=indices, 
                    src=torch.ones_like(indices, dtype=spatial_features_2d.dtype)
                )

                # 2.5. 恢复形状 (N, 1, Hi, Wi)
                communication_mask = communication_mask_flat.reshape(N, 1, Hi, Wi)
                
                # 2.6. (关键) 应用掩码
                feature_to_fuse = spatial_features_2d * communication_mask
                
                # 2.7. (关键) 更新真实的通信率
                comm_rate_effective = K_RATIO
            
            # 3. 计算带宽 (无论是否 Top-K 都要计算)
            
            # 3.1. 计算单车发送满配特征的比特数
            bits_per_agent = Hi * Wi * Ci * _elem_bits(spatial_features_2d.dtype)
            
            # 3.2. 计算单车实际发送的比特数
            actual_bits_per_agent = bits_per_agent * comm_rate_effective
            
            # 3.3. 计算当前批次(场景)的“平均总带宽”(乘以协作车数)
            total_bits_per_scene = actual_bits_per_agent * avg_collaborators
            
            # 3.4. 转换为 Kbps
            kbps_per_frame = (total_bits_per_scene * fps) / 1000.0
            # --- (新增结束) ---

            # (修改) 传入被掩码 (或未被掩码) 的特征
            spatial_features_2d = self.fusion_net(feature_to_fuse, record_len, normalized_affine_matrix)

            psm = self.cls_head(spatial_features_2d)
            rm = self.reg_head(spatial_features_2d)

            # (修改) 更新 output_dict
            output_dict = {'feature': spatial_features_2d, # DiscoNet 保留了 feature
                        'cls_preds': psm,
                        'reg_preds': rm,
                        'comm_rate': comm_rate_effective,
                        'comm_kbps_per_frame': kbps_per_frame,
                        }
                        
            if self.use_dir:
                output_dict.update({'dir_preds': self.dir_head(spatial_features_2d)})

            return output_dict