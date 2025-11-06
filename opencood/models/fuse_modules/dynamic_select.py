import numpy as np
import torch
from torch import nn
from icecream import ic
from opencood.models.fuse_modules.att_fuse import ScaledDotProductAttention
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.models.fuse_modules.fuse_utils import regroup as Regroup
from opencood.models.fuse_modules.att_fuse import ScaledDotProductAttention
from opencood.models.comm_modules.where2comm import Communication
from opencood.models.fuse_modules.where2comm_attn import TransformerFusion
from opencood.models.fuse_modules.when2com_fuse import policy_net4, km_generator_v2, MIMOGeneralDotProductAttention, AdditiveAttentin
import torch.nn.functional as F
from .fusion_in_one import AttFusion, MaxFusion
import math


class Dynamic2Select(nn.Module):
    def __init__(self, args, dim):
        super().__init__()
        self.fully = args['fully']

        # ---- fusion head ----
        if args['fusion'] == 'att':
            self.fuse_modules = AttFusion(dim)
        elif args['fusion'] == 'max':
            self.fuse_modules = MaxFusion()
        else:
            raise ValueError(f"[Where2commSelective] Unknown fusion mode: {args['fusion']}")
        
        self.naive_communication = Communication(args['communication'])

        # 这里开始是我的参数
        sel = args.get('selector', {})
        self.sel_enable       = bool(sel.get('enable', False))
        # self.sel_mode         = sel.get('mode', 'global')     # 'global' | 'tile'
        self.sel_topk         = int(sel.get('topk', 2))       # basic Top-K (K=1/2)
        self.sel_soft_tau     = float(sel.get('soft_tau', 0.0))  # >0: train with soft weights
        self.sel_supply_from  = sel.get('supply', 'cls')      # 'cls' | 'energy'

    # 这个原方法就带了
    @staticmethod
    def _regroup(x, record_len):
        cum = torch.cumsum(record_len, dim=0)
        return torch.tensor_split(x, cum[:-1].cpu())
    
    # 下面三个是我的方法
    @staticmethod
    def _supply_from_energy(feats: torch.Tensor):
        # feats: (N,C,H,W) -> (N,1,H,W) in [0,1]
        hm = torch.clamp(feats.pow(2).mean(dim=1, keepdim=True), min=0)
        return hm / (hm.amax(dim=[2, 3], keepdim=True) + 1e-6)

    @staticmethod
    def _supply_from_cls(cmaps: torch.Tensor):
        # cmaps: (N,A,H,W) -> (N,1,H,W)
        return torch.sigmoid(cmaps).amax(dim=1, keepdim=True)

    @staticmethod
    def _dilate_mask(mask: torch.Tensor, k: int):
        if k <= 1: return mask
        pad = k // 2
        return F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)

    # ---------------- forward ----------------
    def forward(self, x, psm_single, record_len, normalized_affine_matrix, req_mask=None):
        """
        x: (sumN, C, H, W)
        psm_single: (sumN, A, Hc, Wc) or (sumN, 1, Hc, Wc)
        """
        assert x.dim() == 4, f"x must be (sumN,C,H,W), got {tuple(x.shape)}"
        _, C, H, W = x.shape
        B = int(record_len.shape[0])

        # ---- if psm_single came as (B, A, Hc, Wc), replicate per agent ----
        if psm_single.shape[0] == B:
            per_agent = []
            for b in range(B):
                N = int(record_len[b])
                per_agent.append(psm_single[b:b+1].repeat(N, 1, 1, 1))
            psm_single = torch.cat(per_agent, dim=0)  # (sumN, A, Hc, Wc)

        # ---- regroup per batch ----
        feats_list = self._regroup(x, record_len)          # [(N_b,C,H,W), ...]
        conf_list  = self._regroup(psm_single, record_len) # [(N_b,A,Hc,Wc), ...]

        warp_x_list, warp_conf_list, warp_vis_list = [], [], []

        # ---- warp everything to ego coords ----
        for b in range(B):
            N = int(record_len[b])
            # 这一步和原本的Where2comm一样的
            T = normalized_affine_matrix[b, :N, :N].to(dtype=x.dtype, device=x.device).contiguous()
            ego_idx = 0

            cmaps = conf_list[b]                 # (N, A, Hc, Wc)
            cm_L, _, cm_H, cm_W = cmaps.shape

            vis = torch.ones((cm_L, 1, cm_H, cm_W), device=x.device, dtype=x.dtype)
            vis = warp_affine_simple(vis, T[ego_idx], (H, W), mode='nearest')        # (N,1,H,W)

            conf_warp  = warp_affine_simple(cmaps,         T[ego_idx], (H, W))       # (N,A,H,W)
            feats_warp = warp_affine_simple(feats_list[b], T[ego_idx], (H, W))       # (N,C,H,W)

            warp_vis_list.append(vis)
            warp_conf_list.append(conf_warp)
            warp_x_list.append(feats_warp)

        # concat across agents
        warp_x  = torch.cat(warp_x_list,  dim=0)  # (sumN,C,H,W)
        vis_all = torch.cat(warp_vis_list, dim=0) # (sumN,1,H,W)

        # ---- S∧D pruning (original Where2comm) ----
        # req_mask is passed through (dtype/device harmonized)
        if req_mask is not None:
            if isinstance(req_mask, (list, tuple)):
                req_mask = [rm.to(dtype=x.dtype, device=x.device) for rm in req_mask]
            else:
                req_mask = req_mask.to(dtype=x.dtype, device=x.device)

        comm_masks, _ = self.naive_communication(warp_conf_list, B, warp_vis_list, req_mask)
        if isinstance(comm_masks, (list, tuple)):
            comm_masks = torch.cat(comm_masks, dim=0)               # (sumN,1,Hc',Wc')
        if comm_masks.shape[-2:] != (H, W):
            comm_masks = F.interpolate(comm_masks, size=(H, W), mode='bilinear', align_corners=False)

        # ---- selection O (optional) ---- 这里开始是选车代码
        owner_masks = None
        if self.sel_enable:
            owner_masks_list = []
            for b in range(B):
                N = int(record_len[b])
                conf_b = warp_conf_list[b]   # (N,A,H,W)
                feats_b = warp_x_list[b]     # (N,C,H,W)
                vis_b   = warp_vis_list[b]   # (N,1,H,W)

                # Supply S_j
                if self.sel_supply_from == 'cls' and conf_b.shape[1] > 1:
                    S_full = self._supply_from_cls(conf_b)     # (N,1,H,W)
                else:
                    S_full = self._supply_from_energy(feats_b) # (N,1,H,W)

                # 取该 batch 的配准矩阵，构造 T_ego: (N,2,3)
                T_full = normalized_affine_matrix[b, :N, :N].to(dtype=x.dtype, device=x.device).contiguous()
                ego_idx = 0
                T_ego = T_full[ego_idx]   # (N,2,3) agent->ego
                owner_up_b = self._select_global_heuristic_nearest(
                    T_ego, H, W,
                    topk=self.sel_topk,
                    soft_tau=self.sel_soft_tau if self.training else 0.0,
                    device=S_full.device, dtype=S_full.dtype
                )

                # Ensure ego always passes (non-inplace)
                ego_mask = torch.zeros_like(owner_up_b); ego_mask[0:1] = 1.0
                owner_up_b = torch.maximum(owner_up_b, ego_mask)

                owner_masks_list.append(owner_up_b)

            owner_masks = torch.cat(owner_masks_list, dim=0)  # (sumN,1,H,W)

        # ---- final mask and effective rate ----
        if self.fully:
            final_masks = torch.ones_like(warp_x[:, :1, :, :])
        else:
            final_masks = comm_masks
            if owner_masks is not None:
                final_masks = final_masks * owner_masks  # S ∧ D ∧ O

        # prune and fuse
        warp_x = warp_x * final_masks

        # effective communication rate (robust, per-agent average inside visible region)
        num = (final_masks * vis_all).flatten(1).sum(dim=1)    # (sumN,)
        den = vis_all.flatten(1).sum(dim=1).clamp_min(1.0)     # (sumN,)
        effective_rate = (num / den).mean()

        x_out = self.fuse_modules(warp_x, record_len, normalized_affine_matrix, use_warp_feature=False)
        return x_out, effective_rate

    def _select_global_heuristic_nearest(
        self,
        T_ego: torch.Tensor,  # (N, 2, 3) 仿射矩阵：agent->ego（已是归一化坐标系）
        H: int,
        W: int,
        topk: int = 2,
        soft_tau: float = 0.0,
        device=None,
        dtype=None,
    ):
        """
        距离最近的启发式选择（不看S/D）：从非ego的 {1..N-1} 中选择与ego平面位移最近的 K 个。
        这里用 T_ego[..., :2, 2] 的平移量近似"距离"（归一化像素坐标系）。
        返回: owner_up_b (N,1,H,W)
        """
        device = device if device is not None else next(self.parameters()).device
        dtype  = dtype if dtype is not None else torch.float32

        N = T_ego.shape[0]
        owner_up_b = torch.zeros((N, 1, H, W), device=device, dtype=dtype)
        if N <= 1:
            return owner_up_b

        # 提取平移分量 (tx, ty) 作为与ego的平面位移（归一化）
        t = T_ego[..., :2, 2]                          # (N, 2)
        dist2 = (t ** 2).sum(dim=-1)                   # (N,)
        dist2 = dist2.clone()
        dist2[0] = float('inf')                        # 屏蔽 ego 本身（索引0）

        # 选出 topk 个最近（dist 最小）
        k = min(max(int(topk), 0), N - 1)
        if k == 0:
            return owner_up_b

        # torch.topk 默认返回最大的；这里取负号拿最小的
        _, idx = torch.topk(-dist2, k=k, dim=0)
        idx = idx.to(device=device, dtype=torch.long)

        if soft_tau > 0:
            # soft：按 1/(dist+eps) 做归一化权重（仅在选中的 K 个里分配）
            eps = 1e-6
            invd = 1.0 / (torch.sqrt(dist2[idx]) + eps)   # (K,)
            w = invd / invd.sum().clamp_min(eps)
            owner_up_b[idx, :, :, :] = w.view(-1, 1, 1, 1)
        else:
            # 硬选择：选中的车全幅 1
            owner_up_b[idx, :, :, :] = 1.0

        return owner_up_b
