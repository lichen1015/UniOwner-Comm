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
from .point_pillar_comm_multiscale import _elem_bits

class OAOHead(nn.Module):
    def __init__(self, in_channels: int, hidden_ratio: float = 0.5):
        super().__init__()
        h = max(8, int(in_channels * hidden_ratio))
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, h, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(h, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, 1, kernel_size=1, bias=True),
        )

    def forward(self, x):
        # x: (sumN, C, H, W)
        return torch.sigmoid(self.trunk(x))  # (sumN, 1, H, W)


class PointPillarUniownerComm(nn.Module):
    """
    PointPillar + Multi-scale fusion (Submodular2Comm).
    关键改动：
      1) 新增 obj_head 直接产出每车通信置信图 conf_map: (sumN,1,H0,W0)
      2) forward 不再用 cls_head logits 当 psm_single；改为 conf_map
      3) 推理阶段“非融合直出”的逻辑抽出到 inference_single_no_fusion()
    """
    def __init__(self, args):
        super(PointPillarUniownerComm, self).__init__()

        # === VFE & Scatter ===
        self.pillar_vfe = PillarVFE(
            args['pillar_vfe'],
            num_point_features=4,
            voxel_size=args['voxel_size'],
            point_cloud_range=args['lidar_range']
        )
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])

        self.simple_scatter = SimplePointPillarScatter(
            feature_dim=1,
            grid_size=args['point_pillar_scatter']['grid_size']
        )

        self.req_points_threshold = int(args.get('req_points_threshold', -1))
        if self.req_points_threshold > 0:
            print(f"[Info] req_points_threshold = {self.req_points_threshold}")

        # === Backbone ===
        is_resnet = args['base_bev_backbone'].get("resnet", False)
        if is_resnet:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        self.voxel_size = args['voxel_size']

        # === Multi-scale fusion modules ===
        self.fusion_net = nn.ModuleList()
        for i in range(len(args['base_bev_backbone']['layer_nums'])):
            self.fusion_net.append(Mono2comm(args['where2comm'], dim=args['feat_dim'][i]))

        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_flag = 'shrink_header' in args
        if self.shrink_flag:
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.compression = ("compression" in args) and bool(args['compression'])
        if self.compression:
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'], kernel_size=1)

        self.use_dir = 'dir_args' in args
        if self.use_dir:
            self.dir_head = nn.Conv2d(
                self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'], kernel_size=1
            )

        self.obj_head = OAOHead(in_channels=self.out_channel, hidden_ratio=0.5)

        # === 可选冻结 ===
        if args.get('backbone_fix', False):
            self.backbone_fix()

    def backbone_fix(self):
        for p in self.pillar_vfe.parameters(): p.requires_grad = False
        for p in self.scatter.parameters(): p.requires_grad = False
        for p in self.backbone.parameters(): p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters(): p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters(): p.requires_grad = False

        for p in self.cls_head.parameters(): p.requires_grad = False
        for p in self.reg_head.parameters(): p.requires_grad = False
        if self.use_dir:
            for p in self.dir_head.parameters(): p.requires_grad = False


        print("[Info] Backbone fixed (obj_head kept trainable).")

    def _build_req_mask(self, voxel_num_points, voxel_coords, record_len, like_tensor):
        if not (not self.training and self.req_points_threshold > 0):
            return None
        points_map = self.simple_scatter(voxel_num_points.unsqueeze(1), voxel_coords).float()  # (sumN,1,H0,W0)
        req_mask = (points_map < self.req_points_threshold).float()
        return req_mask.to(dtype=like_tensor.dtype, device=like_tensor.device)
    
    def _bits_per_pixel_for_layer(self, layer_idx: int) -> float:
        bpv = getattr(self, "payload_bits_per_value", 8)      # 默认按 8bit
        vpp = getattr(self, "values_per_pixel", 1)
        if isinstance(bpv, (list, tuple)):
            bpv = bpv[layer_idx]
        if isinstance(vpp, (list, tuple)):
            vpp = vpp[layer_idx]
        return float(bpv) * float(vpp)

    @torch.no_grad()
    def inference_single_no_fusion(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']

        record_len = data_dict['record_len']
        batch_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'record_len': record_len
        }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        feat_2d = batch_dict['spatial_features_2d']
        if self.shrink_flag:
            feat_2d = self.shrink_conv(feat_2d)
        if self.compression:
            feat_2d = self.naive_compressor(feat_2d)
        conf_map = self.obj_head(feat_2d)  # (sumN,1,H0,W0)

        cls_preds = self.cls_head(feat_2d)
        reg_preds = self.reg_head(feat_2d)
        out = {'cls_preds': cls_preds, 'reg_preds': reg_preds, 'conf_map': conf_map}
        if self.use_dir:
            out['dir_preds'] = self.dir_head(feat_2d)
        return out

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'record_len': record_len
        }
        
        ego_flag = True
        if '_ego_flag' in data_dict:
            # ego_flag indicates whether the data is from ego vehicle
            # in reference, if not ego_flag, no need to fuse feature, just use the single detection results
            ego_flag = data_dict['_ego_flag']

        # === Encode to BEV ===
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        
        # for intermediate-late fusion, get the single detection results
        if not ego_flag and not self.training:
            return self.inference_single_no_fusion(batch_dict)

        req_mask = self._build_req_mask(
            voxel_num_points, voxel_coords, record_len, like_tensor=batch_dict['spatial_features_2d']
        )

        feat_2d_single = batch_dict['spatial_features_2d']          # (sumN,C,H0,W0)
        if self.shrink_flag:
            feat_2d_single = self.shrink_conv(feat_2d_single)
        if self.compression:
            feat_2d_single = self.naive_compressor(feat_2d_single)

        psm_single = self.obj_head(feat_2d_single)

        _, _, H0, W0 = batch_dict['spatial_features'].shape
        normalized_affine_matrix = normalize_pairwise_tfm(
            data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0]
        )

        feature_list = self.backbone.get_multiscale_feature(batch_dict['spatial_features'])
        fused_feature_list = []

        bits_numer_total = 0.0       # Σ_i  (comm_rate_i_mean * Hi*Wi * Ci * bits_i)
        bits_denom_total = 0.0       # Σ_i  (Hi*Wi * Ci * bits_i)
        avg_collaborators = (sum(record_len) / (len(record_len) + 1e-5)) - 1.0
        
        for i, fuse_module in enumerate(self.fusion_net):
            x_i = feature_list[i]  # (sumN, C_i, H_i, W_i)
            x_out, comm_rate_i = fuse_module(
                x_i,
                psm_single,
                record_len,
                normalized_affine_matrix,
                req_mask
            )
            fused_feature_list.append(x_out)

            if torch.is_tensor(comm_rate_i):
                comm_rate_i_f = float(comm_rate_i.detach().float().mean().item())
            else:
                comm_rate_i_f = float(comm_rate_i if comm_rate_i is not None else 1.0)

            _, Ci, Hi, Wi = x_i.shape
            # num_senders = sum(max(int(n) - 1, 0) for n in record_len)
            bits_i = Hi * Wi * Ci * _elem_bits(x_i.dtype)

            bits_numer_total += comm_rate_i_f * bits_i * avg_collaborators  
            bits_denom_total += bits_i

        fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list)  # (sumN, C_out, H0, W0)
        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)
        if self.compression:
            fused_feature = self.naive_compressor(fused_feature)

        psm = self.cls_head(fused_feature)
        rm  = self.reg_head(fused_feature)

        if bits_denom_total > 0:
            comm_rate_effective = bits_numer_total / bits_denom_total   # 0~1
        else:
            comm_rate_effective = 1.0
            
        kbps_per_frame = (bits_numer_total * 10) / 1000.0  # kbps
            
        output_dict = {
            'cls_preds': psm,
            'reg_preds': rm,
            'comm_rate': comm_rate_effective,
            'comm_kbps_per_frame': kbps_per_frame,
        }
        
        if self.use_dir:
            output_dict['dir_preds'] = self.dir_head(fused_feature)

        return output_dict 






