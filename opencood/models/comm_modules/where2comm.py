# -*- coding: utf-8 -*-
# Author: Yue Hu <phyllis1sjtu@outlook.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math


class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()
        # Threshold of objectiveness
        self.k_ratio = 0
        self.threshold = 0
        if 'k_ratio' in args:
            self.k_ratio = args['k_ratio']
        if 'threshold' in args:
            self.threshold = args['threshold']
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1,
                                             padding=(kernel_size - 1) // 2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        else:
            self.smooth = False


    def init_gaussian_filter(self, k_size=5, sigma=1.0):
        center = k_size // 2
        x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) +
                                                             np.square(y)) / (2 * np.square(sigma)))

        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(
            self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, batch_confidence_maps, B, batch_warp_maks_list, req_mask=None):
        """
        Args:
            batch_confidence_maps: [(L1, H, W), (L2, H, W), ...]
            batch_warp_maks_list: [(1, H, W), (1, H, W), ...], used to mask padding areas
        """

        _, _, H, W = batch_confidence_maps[0].shape
        if req_mask is not None:
            _, H_points_mask, W_points_mask = req_mask[0].shape
            if H_points_mask != H or W_points_mask != W:
                req_mask = F.interpolate(req_mask, size=(H, W), mode='nearest')

        communication_masks = []
        communication_rates = []
        for b in range(B):
            ori_communication_maps, _ = batch_confidence_maps[b].sigmoid().max(dim=1, keepdim=True)
            # Note: If there is no warp mask, the padding value will be 0.5
            # and it will affect the selection of communication mask!
            ori_communication_maps = ori_communication_maps * batch_warp_maks_list[b]
            
            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps)
            else:
                communication_maps = ori_communication_maps

            L = communication_maps.shape[0]
            if self.training:
                # Official training proxy objective
                K = int(H * W * random.uniform(0, 1))
                communication_maps = communication_maps.reshape(L, H * W)
                _, indices = torch.topk(communication_maps, k=K, sorted=False)
                communication_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                ones_fill = torch.ones(L, K, dtype=communication_maps.dtype, device=communication_maps.device)
                communication_mask = torch.scatter(communication_mask, -1, indices, ones_fill).reshape(L, 1, H, W)
            elif self.k_ratio:
                K = int(H * W * self.k_ratio)
                communication_maps = communication_maps.reshape(L, H * W)
                _, indices = torch.topk(communication_maps, k=K, sorted=False)
                communication_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                ones_fill = torch.ones(L, K, dtype=communication_maps.dtype, device=communication_maps.device)
                communication_mask = torch.scatter(communication_mask, -1, indices, ones_fill).reshape(L, 1, H, W)
            elif self.threshold:
                ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
                zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                communication_mask = torch.where(communication_maps > self.threshold, ones_mask, zeros_mask)
            else:
                communication_mask = torch.ones_like(communication_maps).to(communication_maps.device)

            if req_mask is not None:
                communication_mask = req_mask[b] * communication_mask
            
            if L > 1:
                communication_rate = communication_mask[1:].sum() / ((L - 1) * H * W)
            else:
                communication_rate = 0.0
            # Ego
            communication_mask[0] = 1

            communication_masks.append(communication_mask)
            communication_rates.append(communication_rate)
        communication_rates = sum(communication_rates) / B
        communication_masks = torch.cat(communication_masks, dim=0)
        return communication_masks, communication_rates


class MonoCommunication(Communication):
    def __init__(self, args):
        super(MonoCommunication, self).__init__(args)
        
        self.supply_norm     = str(args.get('supply_norm', 'none'))          # 'none' | 'qnorm'
        self.supply_smooth   = int(args.get('supply_smooth_k', 0))           # 0/3/5...
        self.P_bias          = float(args.get('P_bias', 0.1))
        self.P_weight        = float(args.get('P_weight', 0.9))
        self.sel_supply_from = str(args.get('supply_from', 'cls'))  # 'cls' | 'energy'
        self.soft_tau        = float(args.get('soft_tau', 0.0))
        self.dual_ratio      = float(args.get('dual_ratio', 0.0))
        self.dilate_k        = int(args.get('dilate', 3))
        
        if args.get("total_kbps", 0) != 0:
            self.total_kbps = args.get("total_kbps", 0)
            self.fps = int(args.get("fps", 10))
            self.payload_bits_per_value = int(args.get("payload_bits_per_value", 32))
            self.values_per_pixel = int(args.get("values_per_pixel", 1))
            
    @staticmethod
    def calculate_k(per_sender_kbps, H, W, bits_per_value, values_per_pixel=1, frequency_hz=10):
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be > 0")
        bits_per_frame = per_sender_kbps * 1000.0 / float(frequency_hz)  # 6700,000 bps /10 = 670,000 bit
        bits_per_pixel = float(bits_per_value) * float(values_per_pixel)
        if bits_per_pixel <= 0:
            return 0
        max_pixels_per_sender = int(bits_per_frame // bits_per_pixel) # 670,000 / 32
        k = max(0, min(max_pixels_per_sender, int(H) * int(W)))
        
        # print(f"[Bandwidth] Per Sender={per_sender_kbps}")
        # print(f"All Pixel={int(H) * int(W)} -> k = {k}")
        return k

    def forward(
        self,
        batch_confidence_maps,
        B,
        batch_warp_maks_list,
        record_len,
        warp_vis_list,
        warp_conf_list,
        warp_x_list,
        req_mask=None
    ):
        """
        Args:
            batch_confidence_maps: list[B] of (L, A, H, W) logits
            batch_warp_maks_list : list[B] of (L, 1, H, W)  # warp 可见/有效区域
            record_len           : (B,)
            warp_vis_list        : list[B] of (L, 1, H, W)
            warp_conf_list       : list[B] of (L, A|1, H, W)
            warp_x_list          : list[B] of (L, C, H, W)
            req_mask             : None | list[B] of (L,1,H,W) | (sumN,1,H,W)
        Returns:
            communication_masks  : (sumN,1,H,W)  # 已包含 owner
            communication_rates  : scalar, 平均到 batch 的“非 ego 有效发送占比”
        """
        _, _, H, W = batch_confidence_maps[0].shape

        req_mask_list = None
        if req_mask is not None:
            if isinstance(req_mask, (list, tuple)):
                req_mask_list = []
                for bi in range(B):
                    r = req_mask[bi]
                    if r.shape[-2:] != (H, W):
                        r = F.interpolate(r, size=(H, W), mode='nearest')
                    req_mask_list.append(r)
            else:
                # tensor of (sumN,1,H,W) → 拆成 [ (L1,1,H,W), ... ]
                req_splits = self._regroup(req_mask, record_len)
                req_mask_list = []
                for bi in range(B):
                    r = req_splits[bi]
                    if r.shape[-2:] != (H, W):
                        r = F.interpolate(r, size=(H, W), mode='nearest')
                    req_mask_list.append(r)

        final_masks_batches = []
        effective_rates = []

        for bi in range(B):
            ori_maps, _ = batch_confidence_maps[bi].sigmoid().max(dim=1, keepdim=True)   # (L,1,H,W)
            ori_maps = ori_maps * batch_warp_maks_list[bi]                                # 避免 padding 干扰
            comm_maps = self.gaussian_filter(ori_maps) if getattr(self, "smooth", False) else ori_maps

            L = int(record_len[bi])
            flat = comm_maps.reshape(L, -1)

            if self.training:
                K = int(H * W * random.uniform(0, 1))
                _, idx = torch.topk(flat, k=K, sorted=False)
                comm_mask = torch.zeros_like(flat)
                comm_mask.scatter_(-1, idx, 1.0)
                comm_mask = comm_mask.reshape(L, 1, H, W)

            elif getattr(self, 'total_kbps', 0) > 0:
                # per_sender_kbps = self.total_kbps / float(L - 1) if  L > 1 else self.total_kbps # true bandwidth
                per_sender_kbps = self.total_kbps / 4

                K = self.calculate_k(
                    per_sender_kbps=per_sender_kbps,
                    H=H, W=W,
                    bits_per_value=self.payload_bits_per_value,
                    values_per_pixel=self.values_per_pixel,
                    frequency_hz=self.fps,
                )
                comm_mask = torch.zeros_like(flat)
                K = min(K, H * W)
                _, idx = torch.topk(flat, k=K, dim=-1, sorted=False)
                comm_mask.scatter_(-1, idx, 1.0)
                comm_mask = comm_mask.reshape(L, 1, H, W)

            elif getattr(self, "k_ratio", 0):
                K = int(H * W * self.k_ratio)
                _, idx = torch.topk(flat, k=K, sorted=False)
                comm_mask = torch.zeros_like(flat)
                comm_mask.scatter_(-1, idx, 1.0)
                comm_mask = comm_mask.reshape(L, 1, H, W)

            elif getattr(self, "threshold", 0):
                comm_mask = (comm_maps > self.threshold).to(comm_maps.dtype)

            else:
                comm_mask = torch.ones_like(comm_maps)

            if req_mask_list is not None:
                comm_mask = comm_mask * req_mask_list[bi]

            comm_mask[0:1] = 1.0

            vis_b   = warp_vis_list[bi]    # (L,1,H,W)
            conf_b  = warp_conf_list[bi]   # (L,A|1,H,W)
            feats_b = warp_x_list[bi]      # (L,C,H,W)

            if getattr(self, "sel_supply_from", "cls") == 'cls':
                S_full = conf_b.clamp(0, 1)
                if S_full.shape[1] > 1:
                    S_full = S_full.mean(dim=1, keepdim=True)
            else:
                energy = feats_b.pow(2).mean(dim=1, keepdim=True)
                S_full = self._qnorm01_masked(energy, vis_b)  # 你已有实现

            S_full = S_full * (vis_b > 0).float()

            if getattr(self, "supply_smooth", 0) > 1:
                S_full = F.avg_pool2d(
                    S_full,
                    kernel_size=self.supply_smooth,
                    stride=1,
                    padding=self.supply_smooth // 2
                )

            if getattr(self, "supply_norm", "qnorm") == 'qnorm':
                S_full = self._qnorm01_masked(S_full, vis_b)

            P = self._build_pattern_gate(conf_b=conf_b, feats_b=feats_b, vis_b=vis_b)
            S_full_biased = S_full * (self.P_bias + self.P_weight * P).repeat(L, 1, 1, 1)

            base_b = (comm_mask > 0).to(comm_mask.dtype) * (vis_b > 0).to(comm_mask.dtype)

            owner_b = self._assign_owner(
                base_mask_b = base_b,        # (L,1,H,W)
                S_full      = S_full_biased, # (L,1,H,W)
                vis_b       = vis_b,         # (L,1,H,W)
                soft_tau    = self.soft_tau,
                dual_ratio  = self.dual_ratio,
                dilate_k    = self.dilate_k,
                # smooth_k  = getattr(self, "sel_smooth_k", 0),
            )                                # -> (L,1,H,W) ∈ [0,1]，候选外为0

            final_b = comm_mask * owner_b
            final_b[0:1] = 1.0  # ego 保底

            if L > 1:
                eff_rate_b = final_b[1:].sum() / ((L - 1) * H * W)
            else:
                eff_rate_b = torch.zeros((), device=comm_mask.device, dtype=comm_mask.dtype)

            final_masks_batches.append(final_b)
            effective_rates.append(eff_rate_b)

        communication_rates = torch.stack(effective_rates).mean()
        communication_masks = torch.cat(final_masks_batches, dim=0)  # (sumN,1,H,W)
        return communication_masks, communication_rates

    
    @staticmethod
    def _regroup(x, record_len):
        cum = torch.cumsum(record_len, dim=0)
        return torch.tensor_split(x, cum[:-1].cpu())
    
    @staticmethod
    def _dilate_mask(mask: torch.Tensor, k: int):
        if k <= 1:
            return mask
        pad = k // 2
        return F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
    
    
    def _assign_owner(self, base_mask_b: torch.Tensor,  # (N,1,H,W) ROI 候选（comm_masks 已决定）
                    S_full: torch.Tensor,             # (N,1,H,W) 供给分数（cls/energy），暂不含 P 偏置
                    vis_b: torch.Tensor,              # (N,1,H,W) 可见性硬门
                    soft_tau: float,
                    dual_ratio: float,
                    dilate_k: int,
                    margin_thr: float = 0.10,         # 判别间隙阈： (t1 - t2)/(t1+eps) < margin_thr
                    smooth_k: int = 0):               # 区域一致性平滑核（0 关闭；3/5 开）
        """
        目标：
        - 在 ROI 内对所有车辆（含 ego）进行像素级所有权分配；
        - 训练期支持软所有权（softmax）；推理期 winner-take-all；
        - 边界歧义处允许“少量双拥有”（可选）；
        - 可选膨胀/平滑以消除棋盘格。
        返回：
        owner_masks_b: (N,1,H,W) ∈ {0,1}（或训练期为软值），在 ROI∧可见域之外恒为0。
        """
        N, _, H, W = base_mask_b.shape
        device, dtype = base_mask_b.device, base_mask_b.dtype

        cand = (base_mask_b > 0) & (vis_b > 0)                # (N,1,H,W)
        if N == 1:
            return cand.to(dtype)

        logits = S_full.squeeze(1)                            # (N,H,W)
        neg_inf = torch.finfo(logits.dtype).min if logits.dtype.is_floating_point else -1e9
        logits = torch.where(cand.squeeze(1), logits, torch.full_like(logits, neg_inf))  # 

        ego_bias = getattr(self, "ego_bias", 0.0)             # 可在 __init__ 中设定，默认 0
        if ego_bias != 0.0:
            logits[0] = logits[0] + ego_bias

        if self.training and soft_tau > 0:
            weights = F.softmax(logits / max(soft_tau, 1e-3), dim=0)  # (N,H,W)
            owner = weights.unsqueeze(1) * cand.float()               # (N,1,H,W)，候选外仍为0
            if smooth_k and smooth_k > 1:
                owner = F.avg_pool2d(owner, kernel_size=smooth_k, stride=1, padding=smooth_k // 2)
                owner = owner * cand.float()
            if dilate_k and dilate_k > 1:
                owner = self._dilate_mask(owner, k=dilate_k) * cand.float()
            return owner.to(dtype)

        k = min(2, N)
        top_vals, top_idx = torch.topk(logits, k=k, dim=0)    # (k,H,W)
        t1_val, t1_idx = top_vals[0], top_idx[0]              # (H,W)

        # top-1
        owner = torch.zeros_like(S_full)                      # (N,1,H,W)
        owner.scatter_(0, t1_idx.unsqueeze(0).unsqueeze(0), 1.0)
        owner = owner * cand.float()

        if dual_ratio > 0 and N >= 2:
            t2_val, t2_idx = top_vals[1], top_idx[1]
            eps = 1e-6
            rel_margin = (t1_val - t2_val) / (t1_val + eps)           # (H,W)
            ambiguous = (rel_margin < margin_thr) & (cand.any(dim=0).squeeze(0))  # (H,W)
            strong2 = (t2_val >= dual_ratio * (t1_val + eps))
            keep2 = (ambiguous & strong2)

            second = torch.zeros_like(owner)
            second.scatter_(0, t2_idx.unsqueeze(0).unsqueeze(0), 1.0)
            second = second * cand.float()
            second = second * keep2.unsqueeze(0).unsqueeze(0).float()
            owner = torch.clamp(owner + second, 0, 1)

        if smooth_k and smooth_k > 1:
            prob_s = F.avg_pool2d(owner, kernel_size=smooth_k, stride=1, padding=smooth_k // 2)
            if dual_ratio <= 0:
                idx = prob_s.squeeze(1).argmax(dim=0)             # (H,W)
                hard = torch.zeros_like(owner)
                hard.scatter_(0, idx.unsqueeze(0).unsqueeze(0), 1.0)
                owner = hard * cand.float()
            else:
                owner = prob_s * cand.float()

        if dilate_k and dilate_k > 1:
            owner = self._dilate_mask(owner, k=dilate_k) * cand.float()

        return owner.to(dtype)

    def _build_pattern_gate(self, conf_b, feats_b, vis_b):
        """
        基于模式的预门控：返回 P (1,1,H,W)，只在 ego 平面上，供后续“供给偏置/选车/Owner”使用
        conf_b: (N,1,H,W) ← 新：每车单通道 conf_map∈[0,1]；兼容老格式 (N,A,H,W)
        feats_b: (N,C,H,W)
        vis_b : (N,1,H,W)  可见性/落点有效掩码（warp 后）
        """
        N, C, H, W = feats_b.shape
        device, dtype = feats_b.device, feats_b.dtype

        energy = feats_b.pow(2).mean(dim=1, keepdim=True)     # (N,1,H,W)
        S_e = self._qnorm01_masked(energy, vis_b)             # 仅在可见域做分位数归一
        S_e = S_e * (vis_b > 0).float()

        if conf_b is None:
            S_c_raw = torch.zeros_like(S_e)
        else:
            if conf_b.shape[1] == 1:                          # 新格式：单通道 conf∈[0,1]
                S_c_raw = conf_b.clamp(0, 1)
            else:                                             # 兼容老格式：多锚 logits
                S_c_raw = torch.sigmoid(conf_b).amax(dim=1, keepdim=True)
        S_c = self._qnorm01_masked(S_c_raw, vis_b)
        S_c = S_c * (vis_b > 0).float()

        alpha = getattr(self, 'alpha', 0.8)
        S_mix = alpha * S_e + (1 - alpha) * S_c               # (N,1,H,W)

        D_e = self._qnorm01_masked(1.0 - S_e[0:1], vis_b[0:1])         # (1,1,H,W)
        D_c = self._qnorm01_masked(1.0 - S_c[0:1], vis_b[0:1])
        w_e, w_c = 0.7, 0.3                                            # 经验权重，可调
        D = (w_e * D_e + w_c * D_c) * (vis_b[0:1] > 0).float()         # (1,1,H,W)

        q20_e = self._quantile_masked(S_e[0:1], vis_b[0:1], 0.2)
        q20_c = self._quantile_masked(S_c[0:1], vis_b[0:1], 0.2)
        Ke = (S_e[0:1] <= q20_e) & (vis_b[0:1] > 0)
        Kc = (S_c[0:1] <= q20_c) & (vis_b[0:1] > 0)
        K_empty = (Ke & Kc).float()                                   # (1,1,H,W)

        if N > 1:
            S_nei = S_mix[1:] * (vis_b[1:] > 0).float()   # (N-1,1,H,W)
            Sup_nei = S_nei.mean(dim=0, keepdim=True)     # (1,1,H,W)
        else:
            Sup_nei = torch.zeros_like(D)                  # (1,1,H,W)

        P = (D * Sup_nei) * (1.0 - K_empty)                            # (1,1,H,W)
        P = self._qnorm01_masked(P, vis_b[0:1]) * (vis_b[0:1] > 0).float()
        return P

    def _qnorm01_masked(self, x, mask, q_low=0.2, q_high=0.8, eps=1e-6):
        """
        x, mask: (N,1,H,W). 仅在 mask>0 的位置做分位数归一。
        """
        N, _, H, W = x.shape
        x = x.clone()
        y = torch.zeros_like(x)
        for i in range(N):
            m = (mask[i] > 0)
            if m.any():
                xi = x[i][m]
                lo = torch.quantile(xi, q_low)
                hi = torch.quantile(xi, q_high)
                yi = ((x[i] - lo) / (hi - lo + eps)).clamp(0, 1)
                yi = yi * m.float()
                y[i] = yi
        return y

    def _quantile_masked(self, x, mask, q, eps=1e-6):
        """
        返回每张图（批内）在 mask>0 范围内的标量分位数阈 x_q（形状对齐 x）
        """
        N, _, H, W = x.shape
        out = torch.zeros_like(x)
        for i in range(N):
            m = (mask[i] > 0)
            if m.any():
                xi = x[i][m]
                thr = torch.quantile(xi, q)
                out[i] = thr.expand_as(x[i])
        return out
