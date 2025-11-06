# vis_data_sequence_opv2v.py

import os, numpy as np, torch, imageio
from torch.utils.data import DataLoader, Subset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import simple_vis
from opencood.utils.camera_utils import denormalize_img

# ======== 配置 ========
SAVE_VIS_INTERVAL = 40                # 和 inference 一样：每 40 帧保存一次
OUTPUT_ROOT = "/mnt/workspace/collabor/CoSDH-main/visualization_results/opv2v"
DRAW_BOXES = True                    # False=无框；如需带框可设 True
NUM_WORKERS = 0                       # 调试稳妥，OK 后可改 2
SAMPLED_RANGE = None

def _build_fused_pcd_with_agent_id(batch):
    """
    返回 shape [N, 5]: x, y, z, intensity, agent_id
    ego 的 agent_id = 0；其它 CAV = 1..K
    若某些点云没有第4列 intensity，则置为0
    """
    def to_4col(pc):
        # 保证至少 [x,y,z,intensity]
        if not torch.is_tensor(pc):
            pc = torch.as_tensor(pc)
        if pc.shape[1] == 3:
            z = pc.new_zeros((pc.shape[0], 1))
            pc = torch.cat([pc, z], dim=1)
        elif pc.shape[1] > 4:
            pc = pc[:, :4]
        return pc

    pcs = []

    ego = to_4col(batch['ego']['origin_lidar'][0])        # [N0,4]
    ego_id = torch.zeros((ego.shape[0], 1), dtype=ego.dtype, device=ego.device)
    pcs.append(torch.cat([ego, ego_id], dim=1))           # [N0,5]

    proj_list = batch['ego'].get('projected_lidar_list', [])
    for k, lidar_np in enumerate(proj_list, start=1):
        pc = to_4col(lidar_np)
        aid = torch.full((pc.shape[0], 1), float(k), dtype=pc.dtype, device=pc.device)
        pcs.append(torch.cat([pc, aid], dim=1))           # [Nk,5]

    fused = torch.cat(pcs, dim=0)                         # [N,5]
    return fused

import numpy as np
import imageio
from matplotlib import cm

def _render_bev_colored(pcd_np, pc_range, out_path,
                        mode='intensity', cmap='turbo',
                        px_per_meter=10, bg='black'):
    """
    pcd_np: [N,4] 或 [N,5] (x,y,z,intensity[,agent_id])
    pc_range: [xmin, ymin, zmin, xmax, ymax, zmax]
    """
    xmin, ymin, zmin, xmax, ymax, zmax = pc_range
    W = int((xmax - xmin) * px_per_meter)
    H = int((ymax - ymin) * px_per_meter)

    # 选择着色值
    if mode == 'intensity' and pcd_np.shape[1] >= 4:
        val = pcd_np[:, 3]
    elif mode == 'z-value':
        val = pcd_np[:, 2]
    elif mode == 'agent' and pcd_np.shape[1] >= 5:
        val = pcd_np[:, 4]
    elif mode == 'density':
        val = None  # 特殊：走直方图分支
    else:
        # 默认退回到强度
        val = pcd_np[:, 3] if pcd_np.shape[1] >= 4 else np.zeros((pcd_np.shape[0],), dtype=float)

    x = pcd_np[:, 0]; y = pcd_np[:, 1]

    m = (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)
    x = x[m]; y = y[m]
    val = val[m] if val is not None else None

    u = ((x - xmin) * px_per_meter).astype(np.int32)
    v = (H - 1 - (y - ymin) * px_per_meter).astype(np.int32)

    # 背景
    if bg == 'white':
        canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    else:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

    if mode == 'density':
        hist, _, _ = np.histogram2d(v, u, bins=[H, W], range=[[0, H], [0, W]])
        hist = np.clip(hist, 0, np.percentile(hist, 99.5))
        hist = hist / (hist.max() + 1e-6)
        rgb = cm.get_cmap(cmap)(hist)[:, :, :3] * 255.0
        canvas = rgb.astype(np.uint8)
    else:
        lo, hi = np.percentile(val, 1), np.percentile(val, 99)
        if hi <= lo:
            lo, hi = val.min(), val.max() + 1e-6
        val_n = np.clip((val - lo) / (hi - lo), 0.0, 1.0)
        col = cm.get_cmap(cmap)(val_n)[:, :3]  # [N,3], 0..1

        # 覆盖到画布（同一像素多点时用“最大值”策略更显眼）
        # 先把 (v,u) 映射到扁平索引
        idx = v * W + u
        # 取每个像素的最大 val_n
        order = np.argsort(val_n)
        idx_sorted = idx[order]
        col_sorted = col[order]
        # 保留“最后一次写”的颜色（对应更大的 val_n）
        canvas = canvas.reshape(-1, 3)
        canvas[idx_sorted] = (col_sorted * 255).astype(np.uint8)
        canvas = canvas.reshape(H, W, 3)

    imageio.imwrite(out_path, canvas)

def save_fused_bev_colored(batch, pc_range, out_dir, frame_idx,
                           modes=('intensity','z-value','agent','density'),
                           cmap='turbo', px_per_meter=10):
    fused = _build_fused_pcd_with_agent_id(batch).cpu().numpy()  # [N,5]
    for m in modes:
        out = os.path.join(out_dir, f'bev_fused_{m}_{frame_idx:05d}.png')
        _render_bev_colored(fused, pc_range, out, mode=m, cmap=cmap, px_per_meter=px_per_meter, bg='black')

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def _to_hwc(img):
    # (C,H,W)->(H,W,C)；(H,W,C) 原样
    if img.ndim == 3 and img.shape[0] in (1,3):
        return np.transpose(img, (1,2,0))
    return img

def _visualize_nobox(pcd, pc_range, save_path, method='bev', left_hand=False, **kwargs):
    if not torch.is_tensor(pcd):
        pcd = torch.as_tensor(pcd)
    simple_vis.visualize(
        {}, pcd, pc_range, save_path,
        method=method, left_hand=left_hand, **kwargs
    )

def _save_all_agent_cams(batch, out_dir, frame_idx):
    # ego 相机
    if 'ego' in batch and 'image_inputs' in batch['ego']:
        imgs = batch['ego']['image_inputs']['imgs'][0]  # (N_cam,C,H,W)
        for cam_id in range(imgs.shape[0]):
            img = _to_hwc(np.array(denormalize_img(imgs[cam_id])))
            imageio.imwrite(os.path.join(out_dir, f'cam_ego_c{cam_id}_{frame_idx:05d}.png'), img)
    # 其它 CAV 相机（如果数据里提供）
    if 'image_inputs_all' in batch:
        for cav_key, v in batch['image_inputs_all'].items():
            imgs = v['imgs'][0]
            for cam_id in range(imgs.shape[0]):
                img = _to_hwc(np.array(denormalize_img(imgs[cam_id])))
                imageio.imwrite(os.path.join(out_dir, f'cam_{cav_key}_c{cam_id}_{frame_idx:05d}.png'), img)
    elif 'cav_image_inputs' in batch:
        for cav_key, v in batch['cav_image_inputs'].items():
            imgs = v['imgs'][0]
            for cam_id in range(imgs.shape[0]):
                img = _to_hwc(np.array(denormalize_img(imgs[cam_id])))
                imageio.imwrite(os.path.join(out_dir, f'cam_{cav_key}_c{cam_id}_{frame_idx:05d}.png'), img)

def main():
    current_path = os.path.dirname(os.path.realpath(__file__))
    hypes = load_yaml(os.path.join(current_path, '../hypes_yaml/visualization_opv2v.yaml'))

    dataset = build_dataset(hypes, visualize=True, train=False)
    subset = dataset if SAMPLED_RANGE is None else Subset(dataset, SAMPLED_RANGE)

    loader = DataLoader(
        subset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=dataset.collate_batch_test, pin_memory=False
    )

    pc_range = hypes['postprocess']['gt_range']
    LEFT_HAND = False  # 和你的 inference 保持一致（Carla 左手系等）

    for i, batch in enumerate(loader):
        # 和 inference 同步：仅当 i 命中间隔时保存
        if i % SAVE_VIS_INTERVAL != 0:
            continue

        # 输出目录：.../opv2v/BEV_00xxx/
        frame_dir = _ensure_dir(os.path.join(OUTPUT_ROOT, f"BEV_{i:05d}"))

        # === 1) Ego 的 BEV & 3D（无框）===
        ego_pcd = batch['ego']['origin_lidar'][0]   # torch.Tensor
        _visualize_nobox(ego_pcd, pc_range, os.path.join(frame_dir, f'ego_bev_nobox_{i:05d}.png'),
                        method='bev', left_hand=LEFT_HAND,
                        point_color_mode='radial', point_cmap='viridis', point_radius=1
                         )
        _visualize_nobox(ego_pcd, pc_range, os.path.join(frame_dir, f'ego_3d_nobox_{i:05d}.png'),
                        method='3d', left_hand=LEFT_HAND,
                        point_color_mode='radial', point_cmap='viridis', point_radius=1 # viridis
                        )

        if DRAW_BOXES:  # 带框
            gt_box_tensor = dataset.post_processor.generate_gt_bbx(batch)
            infer_result = {"gt_box_tensor": gt_box_tensor}
            infer_result["image_inputs"] = batch['ego']['image_inputs']
            simple_vis.visualize(infer_result, ego_pcd, pc_range,
                                 os.path.join(frame_dir, f'ego_bev_{i:05d}.png'),
                                 method='bev', left_hand=LEFT_HAND,
                                 point_color_mode='radial', point_cmap='viridis', point_radius=1)
            simple_vis.visualize(infer_result, ego_pcd, pc_range,
                                 os.path.join(frame_dir, f'ego_3d_{i:05d}.png'),
                                 method='3d', left_hand=LEFT_HAND,
                                 point_color_mode='radial', point_cmap='viridis', point_radius=1)

        # === 2) 各 CAV 的 BEV & 3D（无框，projected 到 ego）===
        proj_list = batch['ego'].get('projected_lidar_list', [])
        for cav_idx, lidar_np in enumerate(proj_list):
            pcd = torch.from_numpy(lidar_np) if not torch.is_tensor(lidar_np) else lidar_np
            _visualize_nobox(pcd, pc_range,
                             os.path.join(frame_dir, f'bev_cav{cav_idx:02d}_nobox_{i:05d}.png'),
                             method='bev', left_hand=True)
            _visualize_nobox(pcd, pc_range,
                             os.path.join(frame_dir, f'3d_cav{cav_idx:02d}_nobox_{i:05d}.png'),
                             method='3d', left_hand=LEFT_HAND)

        # === 3) 各车相机原图（不叠加任何框/轨迹）===
        _save_all_agent_cams(batch, frame_dir, i)
        print(f"[OK] Saved all views (interval={SAVE_VIS_INTERVAL}) for frame i={i:05d} → {os.path.relpath(frame_dir, OUTPUT_ROOT)}")

if __name__ == "__main__":
    main()
