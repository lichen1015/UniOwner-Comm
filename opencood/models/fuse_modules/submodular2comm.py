import numpy as np
import torch
from torch import nn
from icecream import ic
from opencood.models.fuse_modules.att_fuse import ScaledDotProductAttention
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.models.fuse_modules.fuse_utils import regroup as Regroup
from opencood.models.fuse_modules.att_fuse import ScaledDotProductAttention
from opencood.models.comm_modules.where2comm import MonoCommunication
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

class Submodular2Comm(nn.Module):
    def __init__(self, args, dim):
        super().__init__()
        self.fully = args['fully']

        if args['fusion'] == 'att':
            self.fuse_modules = AttFusion(dim)
        elif args['fusion'] == 'max':
            self.fuse_modules = MaxFusion()
        else:
            raise ValueError(f"[Where2commSelective] Unknown fusion mode: {args['fusion']}")

        self.naive_communication = Communication(args['communication'])

        # ---------- selector 超参（新） ----------
        sel = args.get('selector', {})
        self.sel_soft_tau = float(sel.get('soft_tau', 0.0))  # 训练期可用 soft ownership
        self.sel_supply_from = sel.get('supply', 'cls')  # 'cls' | 'energy'
        self.sel_dual_ratio = float(sel.get('dual_ratio', 0.0))  # 0 关闭；如 0.85 开启少数双拥有者
        self.sel_dilate_k = int(sel.get('dilate', 3))  # 形态学膨胀核（3/5）

    # ---------------- utils ----------------
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


    # ---------------- forward（where2comm 结构 + owner_mask） ----------------
    def forward(self, x, psm_single, record_len, normalized_affine_matrix, req_mask=None):
        """
        Fusion forwarding.

        Parameters:
            x: Input data, (sum(n_cav), C, H, W).
            record_len: List/Tensor, (B).
            normalized_affine_matrix : torch.Tensor
                The normalized affine transformation matrix from each cav to ego,
                shape: (B, L, L, 2, 3)

        Returns:
            Fused feature.
        """
        assert x.dim() == 4, f"x must be (sumN,C,H,W), got {tuple(x.shape)}"
        _, C, H, W = x.shape
        B, L = normalized_affine_matrix.shape[:2]

        # 1) regroup（保持 where2comm 接口）
        batch_node_features     = self._regroup(x,           record_len)
        batch_confidence_maps   = self._regroup(psm_single,  record_len)

        # 2) warp 到 ego（保持 where2comm 写法/变量名）
        batch_warp_x = []
        batch_warp_confidence_maps = []
        batch_warp_maks_list = []

        for b in range(B):
            N = int(record_len[b])
            t_matrix = normalized_affine_matrix[b][:N, :N, :, :]
            i = 0
            confidence_map_L, _, confidence_map_H, confidence_map_W = batch_confidence_maps[b].shape

            warp_mask = torch.ones((confidence_map_L, 1, confidence_map_H, confidence_map_W),
                                device=x.device, dtype=x.dtype)
            warp_mask = warp_affine_simple(warp_mask, t_matrix[i, :, :, :], (H, W))

            confidence_map = warp_affine_simple(batch_confidence_maps[b],
                                                t_matrix[i, :, :, :], (H, W))
            warp_x = warp_affine_simple(batch_node_features[b],
                                        t_matrix[i, :, :, :], (H, W))

            batch_warp_confidence_maps.append(confidence_map)
            batch_warp_x.append(warp_x)
            batch_warp_maks_list.append(warp_mask)

        warp_x = torch.cat(batch_warp_x, dim=0)

        # 3) Prune（保持 where2comm）
        communication_masks, communication_rates = self.naive_communication(
            batch_warp_confidence_maps, B, batch_warp_maks_list, req_mask
        )

        # 4) ★ MonoSender 专属：基于通信 ROI 计算 owner_mask（抽取为单独方法）
        owner_masks = self._compute_owner_masks_from_comm(
            record_len=record_len,
            warp_x_list=batch_warp_x,
            warp_conf_list=batch_warp_confidence_maps,
            warp_vis_list=batch_warp_maks_list,
            comm_masks=communication_masks,
        )

        # ---------- final mask ----------
        if self.fully:
            final_masks = torch.ones_like(warp_x[:, :1, :, :])
        else:
            final_masks = communication_masks

        if owner_masks is not None:
            final_masks = final_masks * owner_masks  # 先掩码，再幅度缩放（顺序等价，但这样读起来更直观）
        warp_x = warp_x * final_masks
        
        # === Effective communication rate (non-ego average) ===
        comm_list  = self._regroup(communication_masks,  record_len)                          # [(L,1,H,W)] * B
        owner_list = self._regroup(owner_masks, record_len) if owner_masks is not None else None

        effective_rates = []
        for b in range(B):
            comm_b = comm_list[b]                      # (L,1,Hb,Wb)
            L, _, Hb, Wb = comm_b.shape
            if L <= 1:
                effective_rates.append(torch.zeros((), device=x.device, dtype=x.dtype))
                continue

            if owner_list is None:
                # 无 owner_mask 时，effective 等于 planned
                eff_b = comm_b[1:].sum() / ((L - 1) * Hb * Wb)
            else:
                owner_b = owner_list[b]                # (L,1,*,*)
                # 尺度对齐（稳妥处理）
                if owner_b.shape[-2:] != (Hb, Wb):
                    owner_b = F.interpolate(owner_b, size=(Hb, Wb), mode='nearest')

                final_tx_b = comm_b * owner_b          # intersection：预算 ∧ 拥有者
                eff_b = final_tx_b[1:].sum() / ((L - 1) * Hb * Wb)

            effective_rates.append(eff_b)

        effective_rate = torch.stack(effective_rates).mean()  # 标量，batch 平均

        x_out = self.fuse_modules(warp_x, record_len, normalized_affine_matrix, use_warp_feature=False)
        return x_out, effective_rate


    # ---------------- MonoSender 定制：从通信 ROI 计算 owner masks（抽取后的唯一方法） ----------------
    def _compute_owner_masks_from_comm(self,
                                    record_len,
                                    warp_x_list,
                                    warp_conf_list,
                                    warp_vis_list,
                                    comm_masks):
        """
        输入均为 warp 到 ego 平面的张量/列表：
        - record_len: (B,)
        - warp_x_list[b]      : (N,C,H,W)
        - warp_conf_list[b]   : (N,1 or A,H,W)  # 外层已聚合为通信用的单/多通道置信图
        - warp_vis_list[b]    : (N,1,H,W)      # 可见域/warp mask
        - comm_masks          : (sumN,1,H,W)   # 通信预算 ROI（naive_communication 输出）

        输出：
        - owner_masks         : (sumN,1,H,W)   # 每像素单/少量双拥有者的软掩码
        """
        if self.fully or comm_masks is None:
            return None

        base_masks_list = self._regroup(comm_masks, record_len)
        owner_masks_list = []

        # 可调参数（若不存在则给默认）
        supply_norm   = getattr(self, "supply_norm", "none")        # 'none' | 'qnorm'
        supply_smooth = int(getattr(self, "supply_smooth_k", 0))    # 0/3/5...
        P_bias        = float(getattr(self, "P_bias", 0.1))
        P_weight      = float(getattr(self, "P_weight", 0.9))
        sel_supply_from = getattr(self, "sel_supply_from", "cls")   # 'cls' | 'energy'

        for b in range(len(record_len)):
            N       = int(record_len[b])
            base_b  = base_masks_list[b]     # (N,1,H,W)  —— 通信 ROI（已预算）
            vis_b   = warp_vis_list[b]       # (N,1,H,W)
            conf_b  = warp_conf_list[b]      # (N,1(or A),H,W)
            feats_b = warp_x_list[b]         # (N,C,H,W)

            # 1) 供给 S_full ∈ [0,1]
            if sel_supply_from == 'cls':
                S_full = conf_b.clamp(0, 1)
                if S_full.shape[1] > 1:
                    # 若为多通道，取均值或 max，按你习惯（这里取均值）
                    S_full = S_full.mean(dim=1, keepdim=True)
            else:
                energy = feats_b.pow(2).mean(dim=1, keepdim=True)   # (N,1,H,W)
                S_full = self._qnorm01_masked(energy, vis_b)        # 需已有实现

            S_full = S_full * (vis_b > 0).float()

            if supply_smooth > 1:
                S_full = F.avg_pool2d(S_full, kernel_size=supply_smooth, stride=1,
                                    padding=supply_smooth // 2)

            if supply_norm == 'qnorm':
                S_full = self._qnorm01_masked(S_full, vis_b)

            # 2) P：需求/模式偏置，范围 [0,1]，位于 ego 平面
            P = self._build_pattern_gate(conf_b=conf_b, feats_b=feats_b, vis_b=vis_b)  # 需已有实现
            # 3) 用 P 对供给加性偏置（更柔和，避免硬裁剪）
            S_full_biased = S_full * (P_bias + P_weight * P).repeat(N, 1, 1, 1)

            # 4) 在 ROI∧可见域内进行像素级拥有者分配（已有实现）
            owner_b = self._assign_owner(
                base_mask_b = base_b,          # (N,1,H,W)
                S_full      = S_full_biased,   # (N,1,H,W)
                vis_b       = vis_b,           # (N,1,H,W)
                soft_tau    = getattr(self, "sel_soft_tau", 0.0) if self.training else 0.0,
                dual_ratio  = getattr(self, "sel_dual_ratio", 0.0),
                dilate_k    = getattr(self, "sel_dilate_k", 0),
                # smooth_k  = getattr(self, "sel_smooth_k", 0),
            )
            owner_masks_list.append(owner_b)

        owner_masks = torch.cat(owner_masks_list, dim=0)  # (sumN,1,H,W)
        return owner_masks

    def _build_pattern_gate(self, conf_b, feats_b, vis_b):
        """
        基于模式的预门控：返回 P (1,1,H,W)，只在 ego 平面上，供后续“供给偏置/选车/Owner”使用
        conf_b: (N,1,H,W) ← 新：每车单通道 conf_map∈[0,1]；兼容老格式 (N,A,H,W)
        feats_b: (N,C,H,W)
        vis_b : (N,1,H,W)  可见性/落点有效掩码（warp 后）
        """
        N, C, H, W = feats_b.shape
        device, dtype = feats_b.device, feats_b.dtype

        # ---------- 1) 供给图 S_e / S_c：先 masked-quantile 再乘可见 ----------
        # 能量供给（通道能量）
        energy = feats_b.pow(2).mean(dim=1, keepdim=True)     # (N,1,H,W)
        S_e = self._qnorm01_masked(energy, vis_b)             # 仅在可见域做分位数归一
        S_e = S_e * (vis_b > 0).float()

        # 语义供给（conf）
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

        # ---------- 2) 需求 D：自车“差”的地方（能量与语义都可反向）
        # 注意：也用 masked-quantile，避免不可见区污染分位数
        D_e = self._qnorm01_masked(1.0 - S_e[0:1], vis_b[0:1])         # (1,1,H,W)
        D_c = self._qnorm01_masked(1.0 - S_c[0:1], vis_b[0:1])
        w_e, w_c = 0.7, 0.3                                            # 经验权重，可调
        D = (w_e * D_e + w_c * D_c) * (vis_b[0:1] > 0).float()         # (1,1,H,W)

        # ---------- 3) 已知空 K_empty：用分位数自适应而不是常数阈 ----------
        # 在 ego 可见域内，按20%分位当作“很低”
        q20_e = self._quantile_masked(S_e[0:1], vis_b[0:1], 0.2)
        q20_c = self._quantile_masked(S_c[0:1], vis_b[0:1], 0.2)
        Ke = (S_e[0:1] <= q20_e) & (vis_b[0:1] > 0)
        Kc = (S_c[0:1] <= q20_c) & (vis_b[0:1] > 0)
        K_empty = (Ke & Kc).float()                                   # (1,1,H,W)

        # ---------- 4) 邻车支持 Sup_nei：用全部车辆 ----------
        if N > 1:
            # 每个邻车的供给：可见域内的混合供给 S_mix，保持你前面的定义
            S_nei = S_mix[1:] * (vis_b[1:] > 0).float()   # (N-1,1,H,W)
            # 方式 A（推荐，默认）：简单均值（对邻车数不敏感）
            Sup_nei = S_nei.mean(dim=0, keepdim=True)     # (1,1,H,W)
        else:
            Sup_nei = torch.zeros_like(D)                  # (1,1,H,W)

        # ---------- 5) 模式门控：需求×支持，抑制已知空 ----------
        P = (D * Sup_nei) * (1.0 - K_empty)                            # (1,1,H,W)
        P = self._qnorm01_masked(P, vis_b[0:1]) * (vis_b[0:1] > 0).float()  # 规范到[0,1]
        return P  # 连续值，供后续“供给偏置/选车/Owner”使用

    # ==== 辅助：带可见掩码的分位归一 & 分位取值 ====
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

        # 1) 统一候选掩码：ROI ∧ 可见
        cand = (base_mask_b > 0) & (vis_b > 0)                # (N,1,H,W)
        if N == 1:
            # 只有 ego：cand 即 owner
            return cand.to(dtype)

        # 2) 构造被 mask 的 logits（所有车一起竞争）
        logits = S_full.squeeze(1)                            # (N,H,W)
        # 不在候选内的像素屏蔽为 -inf，避免被选中
        neg_inf = torch.finfo(logits.dtype).min if logits.dtype.is_floating_point else -1e9
        logits = torch.where(cand.squeeze(1), logits, torch.full_like(logits, neg_inf))

        # （可选）轻微“自车偏置”以做稳定 tie-breaker（默认关闭）
        ego_bias = getattr(self, "ego_bias", 0.0)             # 可在 __init__ 中设定，默认 0
        if ego_bias != 0.0:
            logits[0] = logits[0] + ego_bias

        # 3) 训练期：soft ownership（更平滑的梯度）
        if self.training and soft_tau > 0:
            weights = F.softmax(logits / max(soft_tau, 1e-3), dim=0)  # (N,H,W)
            owner = weights.unsqueeze(1) * cand.float()               # (N,1,H,W)，候选外仍为0
            # 可选轻度平滑
            if smooth_k and smooth_k > 1:
                owner = F.avg_pool2d(owner, kernel_size=smooth_k, stride=1, padding=smooth_k // 2)
                owner = owner * cand.float()
            # 可选膨胀（一般不用在训练期）
            if dilate_k and dilate_k > 1:
                owner = self._dilate_mask(owner, k=dilate_k) * cand.float()
            return owner.to(dtype)

        # 4) 推理期：hard winner-take-all（可选双拥有）
        #   先取像素级 top-2
        k = min(2, N)
        top_vals, top_idx = torch.topk(logits, k=k, dim=0)    # (k,H,W)
        t1_val, t1_idx = top_vals[0], top_idx[0]              # (H,W)

        # 4.1) 置 one-hot 的 Top-1
        owner = torch.zeros_like(S_full)                      # (N,1,H,W)
        owner.scatter_(0, t1_idx.unsqueeze(0).unsqueeze(0), 1.0)   # 沿 dim=0 按索引置1
        owner = owner * cand.float()

        # 4.2) 双拥有（边界歧义 + 次强足够强）
        if dual_ratio > 0 and N >= 2:
            t2_val, t2_idx = top_vals[1], top_idx[1]
            # 仅在候选内计算相对间隙
            eps = 1e-6
            rel_margin = (t1_val - t2_val) / (t1_val + eps)           # (H,W)
            ambiguous = (rel_margin < margin_thr) & (cand.any(dim=0).squeeze(0))  # (H,W)
            strong2 = (t2_val >= dual_ratio * (t1_val + eps))
            keep2 = (ambiguous & strong2)

            second = torch.zeros_like(owner)
            second.scatter_(0, t2_idx.unsqueeze(0).unsqueeze(0), 1.0)
            second = second * cand.float()
            # 只在 keep2 像素上保留第二拥有者
            second = second * keep2.unsqueeze(0).unsqueeze(0).float()
            owner = torch.clamp(owner + second, 0, 1)

        # 4.3) 区域一致性与边界修补（推理期）
        if smooth_k and smooth_k > 1:
            # 先对多通道概率做平均，再重新 hard 化（保持单拥有/少量双拥有）
            prob_s = F.avg_pool2d(owner, kernel_size=smooth_k, stride=1, padding=smooth_k // 2)
            # 仅当未启用双拥有时，重新 hard 化为单拥有；若启用双拥有，可保留 prob_s 直接用于融合权重
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





