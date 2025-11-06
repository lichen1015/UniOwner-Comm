# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


from matplotlib import pyplot as plt
import numpy as np
import copy
import torch

from opencood.tools.inference_utils import get_cav_box
import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev
from matplotlib import cm
import os, cv2
try:
    import torch
except Exception:
    torch = None

import numpy as np

def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=np.float32)

def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float32)

def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float32)

def apply_view_transform(xyz, center=None, yaw_deg=-25, pitch_deg=35, roll_deg=0, zoom=1.0, left_hand=False):
    """
    xyz: (N,3) 点云
    center: 旋转中心；None=以(0,0,0)为中心（一般就是雷达中心）
    yaw/pitch/roll: 角度，单位°。
       - yaw 围绕 +Z 旋转（水平转向）
       - pitch 围绕 +X 旋转（俯仰：正值=向下看）
       - roll 围绕 +Y 旋转（横滚）
    zoom: >1 更近、<1 更远（近似“相机变焦”）
    left_hand: 你的管线若是左手系，可把 yaw 取反以对齐直觉
    """
    if center is None:
        center = np.zeros((1, 3), dtype=np.float32)
    else:
        center = np.asarray(center, dtype=np.float32).reshape(1, 3)

    # 左手系下常见做法：yaw 取反，其他保持
    if left_hand:
        yaw_deg = -yaw_deg

    ya, pa, ra = np.deg2rad([yaw_deg, pitch_deg, roll_deg]).astype(np.float32)

    # 组合顺序：先 yaw (Z)，再 pitch (X)，再 roll (Y) —— 够用也直观
    R = _rot_z(ya) @ _rot_x(pa) @ _rot_y(ra)

    xyz_c = xyz - center                       # 平移到旋转中心
    xyz_r = (xyz_c @ R.T) * float(zoom)        # 旋转 + 缩放
    xyz_v = xyz_r + center                     # 平移回去
    return xyz_v




# =====
def _to_numpy(x):
    if x is None: return None
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _pick_operand_for_colors(pcd_np, canvas, canvas_xy, mode):
    """
    返回 1D 数组作为 colors_operand；必要时回退，保证不为 None
    """
    C = pcd_np.shape[1]
    has4 = C >= 4
    has5 = C >= 5

    if mode in ('intensity', 'auto'):
        if has4:
            op = pcd_np[:, 3]
            if mode == 'auto':
                uniq = np.unique(op)
                # 少量离散值当作 agent id；否则当 intensity
                if (len(uniq) <= 32) and np.allclose(uniq, np.round(uniq), atol=1e-3):
                    return op
            return op
        # 没有强度列 → 用 z
        return pcd_np[:, 2]

    if mode in ('z', 'z-value'):
        return pcd_np[:, 2]

    if mode == 'agent':
        if has5:
            return pcd_np[:, 4]
        if has4:
            return pcd_np[:, 3]
        # 没有 id → 回退 radial
        mode = 'radial'

    if mode == 'radial':
        # 以世界坐标 (0,0) 映射到画布的位置为“中心”，做径向距离
        origin_center = canvas.get_canvas_coords(np.zeros((1, 2)))[0][0]
        diff = canvas_xy - origin_center
        return np.sqrt((diff ** 2).sum(axis=1))

    if mode == 'constant':
        # constant 不需要 operand，但 3D 画布会 assert，因此给个常量即可
        return np.zeros((pcd_np.shape[0],), dtype=float)

    # 兜底：用 z
    return pcd_np[:, 2]


def visualize(infer_result, pcd, pc_range, save_path,
              method='3d', left_hand=False,
              point_color_mode='auto',
              point_cmap='turbo',
              point_radius=-1,):    
        """
        Visualize the prediction, ground truth with point cloud together.
        They may be flipped in y axis. Since carla is left hand coordinate, while kitti is right hand.

        Parameters
        ----------
        infer_result:
            pred_box_tensor : torch.Tensor
                (N, 8, 3) prediction.

            gt_tensor : torch.Tensor
                (N, 8, 3) groundtruth bbx
            
            uncertainty_tensor : optional, torch.Tensor
                (N, ?)

            lidar_agent_record: optional, torch.Tensor
                (N_agnet, )


        pcd : torch.Tensor
            PointCloud, (N, 4).

        pc_range : list
            [xmin, ymin, zmin, xmax, ymax, zmax]

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        method: str, 'bev' or '3d'

        """
        plt.figure(figsize=[(pc_range[3]-pc_range[0])/40, (pc_range[4]-pc_range[1])/40])
        pc_range = [int(i) for i in pc_range]
        pcd_np = pcd.cpu().numpy()

        pred_box_tensor = infer_result.get("pred_box_tensor", None)
        gt_box_tensor = infer_result.get("gt_box_tensor", None)

        if pred_box_tensor is not None:
            pred_box_np = pred_box_tensor.cpu().numpy()
            pred_name = ['pred'] * pred_box_np.shape[0]

            score = infer_result.get("score_tensor", None)
            if score is not None:
                score_np = score.cpu().numpy()
                pred_name = [f'score:{score_np[i]:.3f}' for i in range(score_np.shape[0])]

            uncertainty = infer_result.get("uncertainty_tensor", None)
            if uncertainty is not None:
                uncertainty_np = uncertainty.cpu().numpy()
                uncertainty_np = np.exp(uncertainty_np)
                d_a_square = 1.6**2 + 3.9**2
                
                if uncertainty_np.shape[1] == 3:
                    uncertainty_np[:,:2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np) 
                    # yaw angle is in radian, it's the same in g2o SE2's setting.

                    pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:.3f} a_u:{uncertainty_np[i,2]:.3f}' \
                                    for i in range(uncertainty_np.shape[0])]

                elif uncertainty_np.shape[1] == 2:
                    uncertainty_np[:,:2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian

                    pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:3f}' \
                                    for i in range(uncertainty_np.shape[0])]

                elif uncertainty_np.shape[1] == 7:
                    uncertainty_np[:,:2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian

                    pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:3f} a_u:{uncertainty_np[i,6]:3f}' \
                                    for i in range(uncertainty_np.shape[0])]                    

        if gt_box_tensor is not None:
            gt_box_np = _to_numpy(gt_box_tensor)
            gt_name = ['gt'] * gt_box_np.shape[0]

        if method == 'bev':
            canvas = canvas_bev.Canvas_BEV_heading_right(
                canvas_shape=((pc_range[4]-pc_range[1])*10, 
                              (pc_range[3]-pc_range[0])*10),
                canvas_x_range=(pc_range[0], pc_range[3]),
                canvas_y_range=(pc_range[1], pc_range[4]), left_hand=left_hand)
            canvas.canvas[...] = 255 if canvas.canvas.dtype == np.uint8 else 1.0
            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)

            operand = _pick_operand_for_colors(pcd_np, canvas, canvas_xy, point_color_mode)
            canvas.draw_canvas_points(
                canvas_xy[valid_mask],
                radius=point_radius,
                colors=point_cmap if isinstance(point_cmap, str) else (255,255,255),
                colors_operand=operand[valid_mask]
            )

            # 下面保持原逻辑：画框（若需要）
            if gt_box_tensor is not None:
                canvas.draw_boxes(gt_box_np, colors=(0,255,0), texts=gt_name)
            if pred_box_tensor is not None:
                canvas.draw_boxes(pred_box_np, colors=(255,0,0), ) # texts=pred_name

            # heterogeneous
            lidar_agent_record = infer_result.get("lidar_agent_record", None)
            cav_box_np = infer_result.get("cav_box_np", None)
            if lidar_agent_record is not None:
                cav_box_np = copy.deepcopy(cav_box_np)
                for i, islidar in enumerate(lidar_agent_record):
                    text = ['lidar'] if islidar else ['camera']
                    color = (0,191,255) if islidar else (255,185,15)
                    canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=text)
        # elif method == '3d':
        #     canvas = canvas_3d.Canvas_3D(left_hand=left_hand)
        #     canvas.canvas[...] = 255 if canvas.canvas.dtype == np.uint8 else 1.0

        #     pcd_np = pcd.cpu().numpy()
        #     pcd3   = pcd_np[:, :3]

        #     # >>> 关键：应用视角（可调这四个量） <<<
        #     VIEW_CFG = {
        #         "yaw_deg":   0,   # 左右转；负值≈向右看一点
        #         "pitch_deg":  0,   # 往下俯视（20~45°都好看）
        #         "roll_deg":    0,   # 基本不用
        #         "zoom":       0.50, # 轻微放大；<1 退远，>1 拉近
        #     }
        #     pcd3_view = apply_view_transform(
        #         pcd3,
        #         center=np.array([0, 0, 0], dtype=np.float32),  # 以雷达原点为“相机绕转中心”
        #         left_hand=left_hand,
        #         **VIEW_CFG
        #     )

        #     # 用“变更视角后的点云”去投影
        #     canvas_xy, valid_mask = canvas.get_canvas_coords(pcd3_view)

        #     # ======= 下面保持你的着色/绘制逻辑不变 =======
        #     if point_color_mode == 'radial':
        #         origin_center = canvas.get_canvas_coords(np.zeros((1, 3), dtype=pcd3_view.dtype))[0][0]
        #         colors_operand = np.linalg.norm(canvas_xy - origin_center, axis=1)
        #     elif point_color_mode in ('intensity', 'auto'):
        #         colors_operand = pcd_np[:, 3] if pcd_np.shape[1] >= 4 else pcd_np[:, 2]
        #     elif point_color_mode in ('z', 'z-value'):
        #         colors_operand = pcd_np[:, 2]
        #     elif point_color_mode == 'agent':
        #         if pcd_np.shape[1] >= 5:
        #             colors_operand = pcd_np[:, 4]
        #         elif pcd_np.shape[1] >= 4:
        #             colors_operand = pcd_np[:, 3]
        #         else:
        #             origin_center = canvas.get_canvas_coords(np.zeros((1, 3), dtype=pcd3_view.dtype))[0][0]
        #             colors_operand = np.linalg.norm(canvas_xy - origin_center, axis=1)
        #     else:
        #         colors_operand = np.zeros((pcd_np.shape[0],), dtype=float)

        #     canvas.draw_canvas_points(
        #         canvas_xy[valid_mask],
        #         radius=point_radius,
        #         colors=point_cmap,                 # 如 'viridis'/'turbo'
        #         colors_operand=colors_operand[valid_mask]
        #     )

        #     if gt_box_tensor is not None:
        #         canvas.draw_boxes(gt_box_np, colors=(0, 255, 0), texts=gt_name)
        #     if pred_box_tensor is not None:
        #         canvas.draw_boxes(pred_box_np, colors=(255, 0, 0))

        #     lidar_agent_record = infer_result.get("lidar_agent_record", None)
        #     cav_box_np = infer_result.get("cav_box_np", None)
        #     if lidar_agent_record is not None:
        #         cav_box_np = copy.deepcopy(cav_box_np)
        #         for i, islidar in enumerate(lidar_agent_record):
        #             text = ['lidar'] if islidar else ['camera']
        #             color = (0, 191, 255) if islidar else (255, 185, 15)
        #             canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=text)

        elif method == '3d':
            canvas = canvas_3d.Canvas_3D(left_hand=left_hand)
            canvas.canvas[...] = 255 if canvas.canvas.dtype == np.uint8 else 1.0  # 关键：3D 只用 xyz
            pcd3 = pcd_np[:, :3]
            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd3)  # 计算 colors_operand（仅对 radial 特殊处理）

            if point_color_mode == 'radial':
                # ⚠️ 这里一定要传 (1,3) 的零点
                origin_center = canvas.get_canvas_coords(np.zeros((1, 3), dtype=pcd3.dtype))[0][0]
                colors_operand = np.linalg.norm(canvas_xy - origin_center, axis=1)
            elif point_color_mode in ('intensity', 'auto'):
                if pcd_np.shape[1] >= 4:  # 第4列当强度；auto 也走这里
                    colors_operand = pcd_np[:, 3]
                else:
                    colors_operand = pcd_np[:, 2]  # 没强度就用 z
            elif point_color_mode in ('z', 'z-value'):
                colors_operand = pcd_np[:, 2]
            elif point_color_mode == 'agent':
                if pcd_np.shape[1] >= 5:
                    colors_operand = pcd_np[:, 4]  # 你融合时放在第5列
                elif pcd_np.shape[1] >= 4:
                    colors_operand = pcd_np[:, 3]  # 或者第4列
                else:
                    # 回退 radial：以画布中心距离
                    origin_center = canvas.get_canvas_coords(np.zeros((1, 3), dtype=pcd3.dtype))[0][0]
                    colors_operand = np.linalg.norm(canvas_xy - origin_center, axis=1)
            else:  # 'constant' 或其他
                colors_operand = np.zeros((pcd_np.shape[0],), dtype=float)

            # 传 cmap 名称 + operand；半径可选让点更饱满
            canvas.draw_canvas_points(
                canvas_xy[valid_mask],
                radius=point_radius,
                colors=point_cmap,  # 例如 'Spectral'/'viridis'/'turbo'
                colors_operand=colors_operand[valid_mask]
            )

            # 下面画框逻辑保持不变
            if gt_box_tensor is not None:
                canvas.draw_boxes(gt_box_np, colors=(0, 255, 0), texts=gt_name)
            if pred_box_tensor is not None:
                canvas.draw_boxes(pred_box_np, colors=(255, 0, 0), )  # texts=pred_name

            # heterogeneous 
            lidar_agent_record = infer_result.get("lidar_agent_record", None)
            cav_box_np = infer_result.get("cav_box_np", None)
            if lidar_agent_record is not None:
                cav_box_np = copy.deepcopy(cav_box_np)
                for i, islidar in enumerate(lidar_agent_record):
                    text = ['lidar'] if islidar else ['camera']
                    color = (0, 191, 255) if islidar else (255, 185, 15)
                    canvas.draw_boxes(cav_box_np[i:i + 1], colors=color, texts=text)

        else:
            raise(f"Not Completed for f{method} visualization.")

        plt.axis("off")

        plt.imshow(canvas.canvas)
        plt.tight_layout()
        plt.savefig(save_path, transparent=False, dpi=500)
        plt.clf()
        plt.close()

