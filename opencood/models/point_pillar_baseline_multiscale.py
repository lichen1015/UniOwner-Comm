# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
# Support F-Cooper, Self-Att, DiscoNet(wo KD), V2VNet, V2XViT, When2comm
import torch
import torch.nn as nn
from icecream import ic
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone 
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor, ImprovedCompressor
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, \
        DiscoFusion, V2VNetFusion, V2XViTFusion, When2commFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.models.point_pillar_comm_multiscale import _elem_bits


class PointPillarBaselineMultiscale(nn.Module):
    """
    F-Cooper implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(PointPillarBaselineMultiscale, self).__init__()

        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        is_resnet = args['base_bev_backbone'].get("resnet", False)
        if is_resnet:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        self.voxel_size = args['voxel_size']

        self.fusion_net = nn.ModuleList()
        for i in range(len(args['base_bev_backbone']['layer_nums'])):
            if args['fusion_method'] == "max":
                self.fusion_net.append(MaxFusion())
            if args['fusion_method'] == "att":
                self.fusion_net.append(AttFusion(args['att']['feat_dim'][i]))
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.compression = False
        self.compression_ratio = 1
        self.naive_compressor = None
        if "compression_dim" in args:
            self.compression_stride = args['compression_stride'] if 'compression_stride' in args else 1
            self.compression = True
            self.compression_ratio = args['compression_dim']
            self.compression_ratio *= (self.compression_stride ** 2)
            self.naive_compressor = ImprovedCompressor(64,
                                                       args['compression_dim'], self.compression_stride)
            print(f"Compress ratio: {self.compression_ratio}, stride: {self.compression_stride}")

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
 
        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix()
        
        self.threshold = args.get('threshold', 0)
        if 'where2comm' in args:
            self.threshold =  args['where2comm']['communication']['threshold']
            print(f"Coalign modify threshold={self.threshold}")

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
            voxel_features = data_dict['processed_lidar']['voxel_features']
            voxel_coords = data_dict['processed_lidar']['voxel_coords']
            voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
            record_len = data_dict['record_len']

            batch_dict = {'voxel_features': voxel_features,
                        'voxel_coords': voxel_coords,
                        'voxel_num_points': voxel_num_points,
                        'record_len': record_len}
            # n, 4 -> n, c
            batch_dict = self.pillar_vfe(batch_dict)
            # n, c -> N, C, H, W
            batch_dict = self.scatter(batch_dict)
            # calculate pairwise affine transformation matrix
            _, _, H0, W0 = batch_dict['spatial_features'].shape # original feature map shape H0, W0
            normalized_affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])

            spatial_features = batch_dict['spatial_features']

            if self.compression:
                # only use fp16 during inference to save bandwidth
                spatial_features = self.naive_compressor(spatial_features, use_fp16=not self.training)

            # --- (新增) 带宽累加器 ---
            bits_per_agent_total = 0.0
            bits_denominator_total = 0.0
            # 如果 self 中未定义 eval_fps，则默认为 10.0 Hz
            fps = getattr(self, "eval_fps", 10.0)
            avg_collaborators = (sum(record_len) / (len(record_len) + 1e-5)) - 1.0
            
            # --- (修改) ---
            # 假定在 __init__ 中定义了 self.threshold，例如 self.threshold = 0.5
            THRESHOLD = getattr(self, "threshold", 0.0) 
            use_threshold = (THRESHOLD > 0.0) and (not self.training)
            # --- (修改结束) ---
            
            # multiscale fusion
            feature_list = self.backbone.get_multiscale_feature(spatial_features)
            fused_feature_list = []
            for i, fuse_module in enumerate(self.fusion_net):
                feature_layer_i = feature_list[i]
                # 默认通信率
                comm_rate_i_f = 1.0
                N, Ci, Hi, Wi = feature_layer_i.shape
                
                # --- (修改) ---
                # 如果 threshold 生效 (例如 0.5) 且不在训练中
                if use_threshold:
                    # 1. 计算特征“能量”作为置信图 (N, 1, Hi, Wi)
                    #    我们使用 L2 范数来代表这个 Ci 维向量的重要性
                    confidence_map = torch.norm(feature_layer_i, dim=1, keepdim=True)

                    # 2. (关键) 应用阈值
                    #    (confidence_map > THRESHOLD) 会产生 (N, 1, Hi, Wi) 的 bool 张量
                    #    .to(feature_layer_i.dtype) 将其转换为 1.0 和 0.0
                    communication_mask = (confidence_map > THRESHOLD).to(feature_layer_i.dtype)
                    
                    # 3. (关键) 应用掩码
                    feature_layer_i = feature_layer_i * communication_mask
                    
                    # 4. (关键) 计算 *动态* 的通信率
                    #    (N * Hi * Wi) 是这一层总的像素数, .sum() 是被选中的像素数
                    comm_rate_i_f = communication_mask.sum() / (N * Hi * Wi + 1e-5)
                    comm_rate_i_f = comm_rate_i_f.item() # 转换为 python 标量
                # --- (修改结束) -----------------------------------
                
                # 计算这一层发送的 *总比特数* (Ci * Hi * Wi * bits_per_element)
                bits_for_this_layer = Hi * Wi * Ci * _elem_bits(feature_layer_i.dtype)
                
                # 累加总比特
                bits_per_agent_total += bits_for_this_layer * comm_rate_i_f * avg_collaborators
                bits_denominator_total += bits_for_this_layer
                
                fused_feature_list.append(fuse_module(feature_layer_i, record_len, normalized_affine_matrix))
            fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list)

            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)

            psm = self.cls_head(fused_feature)
            rm = self.reg_head(fused_feature)

            # --- (修改) -----------------------------------
            if bits_denominator_total > 0:
                comm_rate_effective = (bits_per_agent_total / (avg_collaborators + 1e-5)) / (bits_denominator_total + 1e-5)
            else:
                # (修改) 检查 use_threshold
                comm_rate_effective = 0.0 if use_threshold else 1.0

            kbps_per_frame = (bits_per_agent_total * fps) / 1000.0

            output_dict = {'cls_preds': psm,
                        'reg_preds': rm,
                        'comm_rate': comm_rate_effective,
                        'comm_kbps_per_frame': kbps_per_frame,
                        }
            # --- (修改结束) -----------------------------------

            if self.use_dir:
                output_dict.update({'dir_preds': self.dir_head(fused_feature)})

            return output_dict
