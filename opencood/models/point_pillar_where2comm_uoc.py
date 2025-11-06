import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter, SimplePointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone 
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
# from opencood.models.fuse_modules.dynamic_select import Dynamic2Select
from opencood.models.fuse_modules.submodular2comm import Mono2comm
from opencood.models.fuse_modules.gnn_selector import RelationalGNNSelector
from opencood.utils.transformation_utils import normalize_pairwise_tfm
import math

def _elem_bits(dtype: torch.dtype) -> int:
    # 如有量化/自定义位宽，改这里
    return {
        torch.float32: 32, torch.float: 32,
        torch.float16: 16, torch.half: 16,
        torch.bfloat16: 16,
        torch.int64: 64, torch.long: 64,
        torch.int32: 32, torch.int: 32,
        torch.int16: 16, torch.short: 16,
        torch.int8: 8,  torch.uint8: 8,
        torch.bool: 1,  # 若按字节计改成 8
    }.get(dtype, 32)

# ------------------------------
# 轻量通信置信图头：3x3上下文 + 1x1输出 → sigmoid
# 输出 conf_map ∈ (0,1)，形状 (sumN,1,H0,W0)
# ------------------------------
def _gn_groups(c):
    # 选一个能整除通道数的 GN 组数（优先 8/4/2/1）
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1

class ConvGNAct(nn.Sequential):
    def __init__(self, c_in, c_out, k=3, s=1, d=1, groups=1, act='silu'):
        padding = ((k - 1) // 2) * d
        g = _gn_groups(c_out)
        super().__init__(
            nn.Conv2d(c_in, c_out, k, s, padding, dilation=d, groups=groups, bias=False),
            nn.GroupNorm(g, c_out),
            nn.SiLU(inplace=True) if act == 'silu' else nn.ReLU(inplace=True),
        )

class ObjHead(nn.Module):
    def __init__(self, in_channels: int, hidden_ratio: float = 0.5):
        super().__init__()
        h = max(8, int(in_channels * hidden_ratio))
        self.trunk = nn.Sequential(
            ConvGNAct(in_channels, h, k=3, d=2),
            ConvGNAct(h, h, k=3, d=1),
            ConvGNAct(h, 1, k=1),
        )

    def forward(self, x):
        return torch.sigmoid(self.trunk(x))  # (sumN, 1, H, W)


class PointPillarWhere2commUoc(nn.Module):
    """
    Where2comm implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(PointPillarWhere2commUoc, self).__init__()

        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        is_resnet = args['base_bev_backbone'].get("resnet", False)
        if is_resnet:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        self.voxel_size = args['voxel_size']

        self.fusion_net = Mono2comm(args['where2comm'], dim=args['feat_dim'])
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.compression = False
        if "compression" in args:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
            
            
        self.obj_head = ObjHead(in_channels=self.out_channel, hidden_ratio=0.5)
 
        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix()
        
        self.comm_sum = 0.0
        self.comm_n = 0

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
        
        if self.use_dir:
            for p in self.dir_head.parameters():
                p.requires_grad = False
        print("Backbone fixed.")

    def forward(self, data_dict, fusion_flag=True):
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
        
        batch_dict_no_fusion = self.backbone(batch_dict)
        
        # calculate pairwise affine transformation matrix
        _, _, H0, W0 = batch_dict['spatial_features'].shape # original feature map shape H0, W0
        normalized_affine_matrix = normalize_pairwise_tfm(
            data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])

        spatial_features_2d = batch_dict_no_fusion['spatial_features_2d']
        
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        
        psm_single = self.obj_head(spatial_features_2d)
        
        
        fused_feature, comm_rate = self.fusion_net(spatial_features_2d, psm_single,
                                                    record_len, normalized_affine_matrix)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        
        # === 正确/稳定的外层带宽近似（按“比特/帧”计算）===
        _, Cc, Hh, Ww = spatial_features_2d.shape
        if torch.is_tensor(comm_rate):
            comm_rate = float(comm_rate.detach().float().mean().item())
        else:
            comm_rate = float(comm_rate if comm_rate is not None else 1.0)

        # === 关键公式 ===
        # 每个被选中的cell发送 Cc 个元素 → 每cell的载荷是 Cc * bits_per_elem
        bits_i = (Hh * Ww) * Cc * _elem_bits(spatial_features_2d.dtype)
        comm_bits_frame_per_agent = comm_rate * bits_i

        # （可选）把比特/帧折算成 Kbps 以及论文横轴的 log2(Kbps)
        fps = getattr(self, "eval_fps", 10.0)  # 你的评测FPS，默认10Hz；按需改
        kbps_per_agent = comm_bits_frame_per_agent * fps / 1000.0
        # kbps_total     = comm_bits_frame_total  * fps / 1000.0
        # log2_kbps_per_agent = math.log2(max(kbps_per_agent, 1e-9))

        output_dict = {'cls_preds': psm,
                       'reg_preds': rm,
                       'comm_rate': comm_rate,
                       'comm_kbps_per_agent': kbps_per_agent,
        }

        if self.use_dir:
            output_dict.update({'dir_preds': self.dir_head(fused_feature)})
        
        return output_dict






