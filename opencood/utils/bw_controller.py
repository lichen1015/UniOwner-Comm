# utils/bw_controller.py
import math
import torch
import torch.nn.functional as F

class BandwidthLimiter:
    """
    统一的带宽控制器：支持 kbps/帧限额、log2(熵)或qbits计费、像素/块索引开销、EMA阈值平滑、令牌桶。
    使用：
        bw = BandwidthLimiter(fps=10, total_kbps=27000, per_agent_budget=False, alloc_mode='equal',
                              qbits=8, channels=64, use_entropy=False, index_mode='pixel',
                              block_size=8, overhead_ratio=0.10, ema_alpha=0.8,
                              token_bucket_capacity_frames=2.0)
        comm_mask, stats = bw.make_mask(comm_maps, feat=feat_or_None)
    """
    def __init__(
        self,
        fps: float,
        total_kbps: float = None,
        per_agent_kbps: float = None,
        per_agent_budget: bool = True,         # True: 每车固定；False: 总额共享
        alloc_mode: str = 'equal',             # 'equal' | 'score'
        qbits: int = 8,
        channels: int = 64,
        use_entropy: bool = False,             # True 启用 log2 熵估计
        entropy_bins: int = 64,                # 直方图bin数（2**qbits上限）
        index_mode: str = 'pixel',             # 'pixel' | 'block'
        block_size: int = 8,                   # 块模式时的 tile 大小
        overhead_ratio: float = 0.10,          # 协议冗余比例（预扣）
        bits_header: int = 128,                # 每agent每帧额外头部位
        ema_alpha: float = 0.8,                # 阈值EMA平滑参数
        token_bucket_capacity_frames: float = 2.0,  # 令牌桶容量 = capacity_frames * budget_per_frame
        device: torch.device = None
    ):
        assert (total_kbps is not None) ^ (per_agent_kbps is not None), \
            "total_kbps 与 per_agent_kbps 二选一"
        self.fps = float(fps)
        self.total_kbps = total_kbps
        self.per_agent_kbps = per_agent_kbps
        self.per_agent_budget = per_agent_budget
        self.alloc_mode = alloc_mode
        self.qbits = int(qbits)
        self.channels = int(channels)
        self.use_entropy = bool(use_entropy)
        self.entropy_bins = int(entropy_bins)
        self.index_mode = index_mode
        self.block_size = int(block_size)
        self.overhead_ratio = float(overhead_ratio)
        self.bits_header = int(bits_header)
        self.ema_alpha = float(ema_alpha)
        self.token_bucket_capacity_frames = float(token_bucket_capacity_frames)
        self.device = device

        self._tau_ema = None     # (L,1,1,1) 每车阈值EMA
        self._bucket_bits = None # (L,) 令牌桶位数
        self._frame_idx = 0

    @torch.no_grad()
    def make_mask(self, comm_maps: torch.Tensor, feat: torch.Tensor = None):
        L, _, H, W = comm_maps.shape
        dev = comm_maps.device if self.device is None else self.device

        bits_budget = self._compute_budget_bits(comm_maps, mode=self.alloc_mode)  # (L,)

        bits_per_scalar = self._estimate_bits_per_scalar(feat, dev)
        payload_bits_per_pixel = bits_per_scalar * self.channels

        index_bits_per_pixel = float(math.ceil(math.log2(max(1, H * W))))  # 像素模式用

        K_vec = self._solve_K(bits_budget, payload_bits_per_pixel, index_bits_per_pixel, H, W)

        if self.index_mode == 'pixel':
            comm_mask, tau, n_blocks = self._mask_pixel_topk(comm_maps, K_vec)
        else:
            comm_mask, tau, n_blocks = self._mask_block_topk(comm_maps, K_vec, self.block_size)

        used_bits = self._estimate_bits_used(
            comm_mask, payload_bits_per_pixel, index_bits_per_pixel,
            n_blocks=n_blocks, H=H, W=W
        )
        self._update_bucket(bits_budget, used_bits)

        stats = {
            "bits_budget": bits_budget,
            "used_bits": used_bits,
            "payload_bits_per_cell": float(payload_bits_per_pixel),  # 这里是“每像素”
            "index_bits_per_cell": float(index_bits_per_pixel),      # 像素模式意义更直观
            "K": comm_mask.view(L, -1).sum(dim=1).long(),            # 实际像素数（块模式=像素总数）
            "K_blocks": (n_blocks if n_blocks is not None else None),
            "tau": tau.squeeze().float(),
            "frame": self._frame_idx
        }
        self._frame_idx += 1
        return comm_mask.to(comm_maps.dtype), stats


    def _compute_budget_bits(self, comm_maps, mode='equal'):
        L, _, H, W = comm_maps.shape
        dev = comm_maps.device if self.device is None else self.device

        if self.per_agent_budget:
            kbps_per_agent = self.per_agent_kbps if (self.per_agent_kbps is not None) \
                            else float(self.total_kbps) / max(L, 1)
            bits_each = kbps_per_agent * 1000.0 / self.fps
            bits_each = bits_each * (1.0 - self.overhead_ratio)      # 不要在这里减 header
            bits_vec = torch.full((L,), max(0.0, bits_each), device=dev)
        else:
            total_bits = self.total_kbps * 1000.0 / self.fps
            total_bits = total_bits * (1.0 - self.overhead_ratio)
            if mode == 'equal':
                bits_vec = torch.full((L,), max(0.0, total_bits / max(L, 1)), device=dev)
            else:
                score = comm_maps.view(L, -1).sum(dim=1) + 1e-6
                w = score / score.sum()
                bits_vec = w * max(0.0, total_bits)

        # --- 令牌桶：每帧刷新容量，并且把旧余量clamp到新容量 ---
        cap = bits_vec * self.token_bucket_capacity_frames
        if (self._bucket_bits is None) or (self._bucket_bits.numel() != L):
            self._bucket_bits = cap.clone()
        else:
            # clamp旧余量到新上限
            self._bucket_bits = torch.minimum(self._bucket_bits, cap)
        self._bucket_cap = cap

        # 每帧“可用” = 分配 + 桶余量（不超过容量）
        allow = torch.minimum(bits_vec + self._bucket_bits, self._bucket_cap)
        return allow


    # ---------- bits per scalar ----------
    def _estimate_bits_per_scalar(self, feat, dev):
        if not self.use_entropy or (feat is None):
            return float(self.qbits)
        # 经验：把特征clamp到[-1,1]再量化
        L, C, H, W = feat.shape
        levels = min(2 ** self.qbits, self.entropy_bins)
        x = torch.clamp(feat.detach(), -1, 1)
        x = (x + 1) * 0.5  # [0,1]
        idx = torch.clamp((x * (levels - 1)).round(), 0, levels - 1)
        # 直方图（跨 L,C,H,W 聚合，估计全局平均码率）
        hist = torch.bincount(idx.view(-1).to(torch.int64), minlength=levels).float()
        p = hist / (hist.sum() + 1e-6)
        nz = p > 0
        H_bits = -(p[nz] * torch.log2(p[nz])).sum().item()  # Shannon
        # +1 的冗余margin（接近实际熵编/算子开销）
        return float(min(self.qbits, H_bits + 1.0))

    # ---------- index bits ----------
    def _index_bits_per_cell(self, H, W):
        if self.index_mode == 'pixel':
            return float(math.ceil(math.log2(max(1, H * W))))
        else:
            Hs, Ws = H // self.block_size, W // self.block_size
            Hs, Ws = max(1, Hs), max(1, Ws)
            # 一个“cell”=一个块
            return float(math.ceil(math.log2(Hs * Ws)))

    # ---------- 求 K ----------
    def _solve_K(self, bits_budget, payload_bits_per_pixel, index_bits_per_pixel, H, W):
        bits_usable = torch.clamp(bits_budget - self.bits_header, min=0.0)

        if self.index_mode == 'pixel':
            denom = max(1e-6, payload_bits_per_pixel + index_bits_per_pixel)
            capacity = H * W
            K = torch.floor(torch.clamp(bits_usable / denom, min=0.0)).to(torch.int64)
            return torch.clamp(K, 0, capacity)

        # block mode
        block = self.block_size
        Hs, Ws = max(1, H // block), max(1, W // block)
        block_area = block * block
        payload_per_block = block_area * payload_bits_per_pixel
        index_per_block = float(math.ceil(math.log2(Hs * Ws)))
        denom_blk = max(1e-6, payload_per_block + index_per_block)
        cap_blk = Hs * Ws
        K_blk = torch.floor(torch.clamp(bits_usable / denom_blk, min=0.0)).to(torch.int64)
        return torch.clamp(K_blk, 0, cap_blk)

    def _mask_pixel_topk(self, comm_maps, K_vec):
        L, _, H, W = comm_maps.shape
        if (K_vec.max().item() == 0):
            zero = torch.zeros_like(comm_maps)
            tau = comm_maps.new_full((L,1,1,1), float('inf'))
            return zero, tau, None

        flat = comm_maps.view(L, -1)
        vals, _ = torch.sort(flat, dim=1, descending=True)
        idxs = torch.clamp(K_vec - 1, min=0)
        tau = torch.gather(vals, 1, idxs.view(-1,1)).view(L,1,1,1)

        if self._tau_ema is None or self._tau_ema.shape[0] != L:
            self._tau_ema = tau.clone()
        else:
            self._tau_ema = self.ema_alpha * self._tau_ema + (1.0 - self.ema_alpha) * tau

        mask = (comm_maps >= self._tau_ema).to(comm_maps.dtype)
        # 精裁：确保每行恰好 K
        need = K_vec.view(-1,1) - mask.view(L,-1).sum(dim=1).to(torch.int64)
        if (need != 0).any():
            strict = torch.zeros_like(flat)
            kmax = int(K_vec.max().item())
            v, id_top = torch.topk(flat, k=kmax, dim=1, sorted=False)
            for i in range(L):
                ki = int(K_vec[i].item())
                if ki > 0:
                    strict[i].scatter_(0, id_top[i, :ki], 1.0)
            mask = strict.view(L,1,H,W).to(comm_maps.dtype)
        return mask, tau, None   # 第3个返回值：块模式才用


    def _mask_block_topk(self, comm_maps, K_blk_vec, block):
        L, _, H, W = comm_maps.shape
        Hs, Ws = max(1, H // block), max(1, W // block)
        if (K_blk_vec.max().item() == 0) or (Hs*Ws == 0):
            full = torch.zeros_like(comm_maps)
            tau = comm_maps.new_full((L,1,1,1), float('inf'))
            n_blocks = torch.zeros((L,), device=comm_maps.device, dtype=torch.int64)
            return full, tau, n_blocks

        x = comm_maps[..., :Hs*block, :Ws*block]
        patches = x.unfold(2, block, block).unfold(3, block, block)  # (L,1,Hs,Ws,block,block)
        scores = patches.mean(dim=(-1,-2))                            # (L,1,Hs,Ws)
        flat = scores.flatten(2)                                      # (L,1,Hs*Ws)
        vals, _ = torch.sort(flat, dim=2, descending=True)
        idxs = torch.clamp(K_blk_vec.view(-1,1) - 1, min=0).long()
        tau = torch.gather(vals, 2, idxs.unsqueeze(-1)).view(L,1,1,1)

        if self._tau_ema is None or self._tau_ema.shape[0] != L:
            self._tau_ema = tau.clone()
        else:
            self._tau_ema = self.ema_alpha * self._tau_ema + (1.0 - self.ema_alpha) * tau

        keep_small = torch.zeros_like(flat)
        for i in range(L):
            ki = int(K_blk_vec[i].item())
            if ki > 0:
                v, id_top = torch.topk(flat[i,0], k=ki, dim=0, sorted=False)
                keep_small[i,0].scatter_(0, id_top, 1.0)
        keep = keep_small.view(L,1,Hs,Ws)
        n_blocks = keep_small.sum(dim=2).squeeze(1).to(torch.int64)  # (L,)

        mask = keep.repeat_interleave(block, 2).repeat_interleave(block, 3)
        full = torch.zeros_like(comm_maps)
        full[..., :Hs*block, :Ws*block] = mask
        return full.to(comm_maps.dtype), tau, n_blocks

    def _estimate_bits_used(self, mask, payload_bits_per_pixel, index_bits_per_pixel, n_blocks=None, H=None, W=None):
        L = mask.shape[0]
        K_pix = mask.view(L, -1).sum(dim=1).float()  # 被选像素数
        if self.index_mode == 'pixel':
            used = K_pix * (payload_bits_per_pixel + index_bits_per_pixel) + self.bits_header
        else:
            assert n_blocks is not None
            block = self.block_size
            Hs, Ws = max(1, H // block), max(1, W // block)
            index_per_block = float(math.ceil(math.log2(Hs * Ws)))
            used = K_pix * payload_bits_per_pixel + n_blocks.float() * index_per_block + self.bits_header
        return used

    # ---------- 令牌桶 ----------
    def _apply_token_bucket(self, bits_vec):
        L = bits_vec.numel()
        if self._bucket_bits is None or self._bucket_bits.numel() != L:
            # 桶容量 = capacity_frames × 每帧预算（按当前bits_vec初始化）
            capacity = bits_vec * self.token_bucket_capacity_frames
            self._bucket_bits = capacity.clone()
            self._bucket_cap = capacity.clone()
        # 每帧“可用” = 分配 + 当前桶余量（不超过容量）
        allow = torch.minimum(bits_vec + self._bucket_bits, self._bucket_cap)
        return allow

    def _update_bucket(self, bits_budget, used_bits):
        # 桶剩余 = clamp( 旧余量 + 分配 - 实际消耗 , [0, 容量] )
        new_left = torch.clamp(self._bucket_bits + bits_budget - used_bits, min=0.0)
        self._bucket_bits = torch.minimum(new_left, self._bucket_cap)
