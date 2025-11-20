import numpy as np
import torch
from torch import nn
from icecream import ic
from opencood.models.fuse_modules.att_fuse import ScaledDotProductAttention
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.models.fuse_modules.fuse_utils import regroup as Regroup
from opencood.models.fuse_modules.att_fuse import ScaledDotProductAttention
from opencood.models.comm_modules.uniowner_comm import MonoCommunication
from opencood.models.fuse_modules.where2comm_attn import TransformerFusion
from opencood.models.fuse_modules.when2com_fuse import policy_net4, km_generator_v2, MIMOGeneralDotProductAttention, AdditiveAttentin
import torch.nn.functional as F
from .fusion_in_one import AttFusion, MaxFusion, Where2comm

class Mono2comm(Where2comm):
    def __init__(self, args, dim):
        super(Mono2comm, self).__init__(args, dim=dim)
        
        self.naive_communication = MonoCommunication(args['communication'])


    def forward(self, x, psm_single, record_len, normalized_affine_matrix, req_mask=None):
        """
        Fusion forwarding.

        Parameters:
            x: Input data, (sum(n_cav), C, H, W).
            record_len: List, (B).
            normalized_affine_matrix : torch.Tensor
                The normalized affine transformation matrix from each cav to ego, 
                shape: (B, L, L, 2, 3) 

        Returns:
            Fused feature.
        """

        _, C, H, W = x.shape
        B, L = normalized_affine_matrix.shape[:2]
        
        # warp the confidence map
        batch_node_features = self.regroup(x, record_len)
        batch_confidence_maps = self.regroup(psm_single, record_len)
        batch_warp_x = []
        batch_warp_confidence_maps = []
        batch_warp_maks_list = []
        
        for b in range(B):
            N = record_len[b]
            t_matrix = normalized_affine_matrix[b][:N, :N, :, :]
            i = 0
            confidence_map_L, _, confidence_map_H, confidence_map_W = batch_confidence_maps[b].shape
            warp_mask = torch.ones((confidence_map_L, 1, confidence_map_H, confidence_map_W)).to(x.device)
            warp_mask = warp_affine_simple(warp_mask, t_matrix[i, :, :, :], (H, W))
            confidence_map = warp_affine_simple(batch_confidence_maps[b],
                                                t_matrix[i, :, :, :], (H, W))
            warp_x = warp_affine_simple(batch_node_features[b],
                                        t_matrix[i, :, :, :], (H, W))
            batch_warp_confidence_maps.append(confidence_map)
            batch_warp_x.append(warp_x)
            batch_warp_maks_list.append(warp_mask)
        warp_x = torch.cat(batch_warp_x, dim=0)
        
        # Prune
        communication_masks, \
            communication_rates = self.naive_communication(batch_warp_confidence_maps, B,
                                                            batch_warp_maks_list,
                                                            record_len=record_len,
                                                            warp_x_list=batch_warp_x,
                                                            warp_conf_list=batch_warp_confidence_maps,
                                                            warp_vis_list=batch_warp_maks_list,
                                                            req_mask=req_mask)
            
        # mask the features
        if self.fully:
            communication_masks = torch.tensor(1).to(warp_x.device)
        else:
            if warp_x.shape[-1] != communication_masks.shape[-1]:
                communication_masks = F.interpolate(
                    communication_masks, size=(warp_x.shape[-2], warp_x.shape[-1]),
                    mode='bilinear', align_corners=False)
            warp_x = warp_x * communication_masks
        
        x_out = self.fuse_modules(warp_x, record_len,
                                  normalized_affine_matrix, use_warp_feature=False)
        
        return x_out, communication_rates