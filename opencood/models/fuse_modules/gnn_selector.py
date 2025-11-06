import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationalGNNSelector(nn.Module):
    """
    学习式选车器：为每个 scene 的 ego(i=0) 输出每个发送者 j 的连续权重 veh_weight∈[0,1]
    只用连续量（不做阈值/TopK）：supply=σ(psm_j), demand= req_mask_i 或 (1-σ(psm_i))
    边特征核心：供需重叠 mean(D_i * warp(S_j->i)) + 相对位姿(dx,dy,cosΔyaw,sinΔyaw,dist)
    """
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.in_dim = 8  # [mean_Sj, overlap, dx, dy, cos(dyaw), sin(dyaw), dist, demand_mass]
        self.q_mlp = nn.Sequential(nn.Linear(self.in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.k_mlp = nn.Sequential(nn.Linear(self.in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.e_mlp = nn.Sequential(nn.Linear(2*hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    @staticmethod
    def _yaw_from_rot2x2(R):  # R: (2,2)
        return math.atan2(R[1,0].item(), R[0,0].item())

    def forward(self, psm_single, req_mask, norm_affine, record_len, warp_fn):
        """
        psm_single: (sumN, A, H, W)
        req_mask:   (sumN, 1, H, W) 或 None（为 None 时自动用 1-σ(psm_i)）
        norm_affine:(B,L,L,2,3)  j->i
        warp_fn:    callable(map(1,1,H,W), A(2,3), (H,W)) -> (1,1,H,W)
        return:
            veh_weight: (sumN,) 连续边权；对自环 i==j 置 0
        """
        device = psm_single.device
        sumN = psm_single.size(0)
        H, W = psm_single.shape[-2:]
        veh_weight = torch.zeros((sumN,), device=device)

        conf = torch.sigmoid(psm_single).amax(dim=1, keepdim=True)  # (sumN,1,H,W)
        mean_conf = conf.mean(dim=(-1,-2))                          # (sumN,1)

        ofs = 0
        for b, n in enumerate(record_len.tolist()):
            n = int(n)
            idx = torch.arange(ofs, ofs+n, device=device)
            i = 0  # 与现有 Dynamic2Select 对齐：每个 scene 以 i=0 为 ego
            A_sub = norm_affine[b, :n, :n]              # (n,n,2,3)  j->i
            Di = req_mask[idx][i:i+1] if req_mask is not None else (1.0 - conf[idx][i:i+1])  # (1,1,H,W)
            demand_mass = Di.mean()

            # 构造 ego 的“query特征”（与 sender 维度一致，便于 q×k）
            ego_feat = torch.tensor([
                mean_conf[idx][i,0].item(),  # 用 ego 自身 mean_S 充数
                demand_mass.item(),          # demand_mass
                0.0, 0.0, 1.0, 0.0,          # dx=dy=0, cosΔyaw=1, sinΔyaw=0
                0.0,                         # dist
                demand_mass.item()           # 再次放一份 demand_mass（占位凑维度）
            ], device=device)
            q = self.q_mlp(ego_feat.unsqueeze(0))  # (1,hidden)

            logits = torch.full((n,), -1e9, device=device)  # 先 ban 自环
            for j in range(n):
                Aij = A_sub[i, j]           # (2,3)
                dx, dy = Aij[0,2].item(), Aij[1,2].item()
                dist = math.hypot(dx, dy)

                # yaw 差
                Rii = A_sub[i, i, :2, :2]; yaw_i = self._yaw_from_rot2x2(Rii)
                Rjj = A_sub[j, j, :2, :2]; yaw_j = self._yaw_from_rot2x2(Rjj)
                dyaw = math.atan2(math.sin(yaw_i - yaw_j), math.cos(yaw_i - yaw_j))
                cosd, sind = math.cos(dyaw), math.sin(dyaw)

                # 连续供需重叠（不做阈值）：S_j=σ(psm_j)
                Sj = conf[idx][j:j+1]                 # (1,1,H,W)
                Sj_w = warp_fn(Sj, Aij, (H, W))       # (1,1,H,W)
                overlap = (Di * Sj_w).mean()

                feat_j = torch.tensor([
                    mean_conf[idx][j,0].item(),       # mean_Sj
                    overlap.item(),                   # overlap
                    dx, dy, cosd, sind, dist,
                    demand_mass.item()
                ], device=device)
                k = self.k_mlp(feat_j.unsqueeze(0))   # (1,hidden)

                logit = self.e_mlp(torch.cat([q, k], dim=1)).squeeze(-1)  # (1,)
                logits[j] = logit

            logits[i] = -1e9                     # 自环不通信
            veh_weight[idx] = torch.sigmoid(logits)  # 连续权重
            ofs += n

        return veh_weight.clamp_(0, 1)
