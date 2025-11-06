# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

import time

import cv2
import numpy as np
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import cm

from opencood.utils import box_utils
from opencood.utils import common_utils

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


def bbx2linset(bbx_corner, order='hwl', color=(0, 1, 0)):
    """
    Convert the torch tensor bounding box to o3d lineset for visualization.

    Parameters
    ----------
    bbx_corner : torch.Tensor
        shape: (n, 8, 3).

    order : str
        The order of the bounding box if shape is (n, 7)

    color : tuple
        The bounding box color.

    Returns
    -------
    line_set : list
        The list containing linsets.
    """
    if not isinstance(bbx_corner, np.ndarray):
        bbx_corner = common_utils.torch_tensor_to_numpy(bbx_corner)

    if len(bbx_corner.shape) == 2:
        bbx_corner = box_utils.boxes_to_corners_3d(bbx_corner,
                                                   order)

    # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # Use the same color for all lines
    colors = [list(color) for _ in range(len(lines))]
    bbx_linset = []

    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        bbx[:, :1] = - bbx[:, :1]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbx)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        bbx_linset.append(line_set)

    return bbx_linset


def bbx2oabb(bbx_corner, order='hwl', color=(0, 0, 1)):
    """
    Convert the torch tensor bounding box to o3d oabb for visualization.

    Parameters
    ----------
    bbx_corner : torch.Tensor
        shape: (n, 8, 3).

    order : str
        The order of the bounding box if shape is (n, 7)

    color : tuple
        The bounding box color.

    Returns
    -------
    oabbs : list
        The list containing all oriented bounding boxes.
    """
    if not isinstance(bbx_corner, np.ndarray):
        bbx_corner = common_utils.torch_tensor_to_numpy(bbx_corner)

    if len(bbx_corner.shape) == 2:
        bbx_corner = box_utils.boxes_to_corners_3d(bbx_corner,
                                                   order)
    oabbs = []

    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        bbx[:, :1] = - bbx[:, :1]

        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(bbx)

        oabb = tmp_pcd.get_oriented_bounding_box()
        oabb.color = color
        oabbs.append(oabb)

    return oabbs


def bbx2aabb(bbx_center, order):
    """
    Convert the torch tensor bounding box to o3d aabb for visualization.

    Parameters
    ----------
    bbx_center : torch.Tensor
        shape: (n, 7).

    order: str
        hwl or lwh.

    Returns
    -------
    aabbs : list
        The list containing all o3d.aabb
    """
    if not isinstance(bbx_center, np.ndarray):
        bbx_center = common_utils.torch_tensor_to_numpy(bbx_center)
    bbx_corner = box_utils.boxes_to_corners_3d(bbx_center, order)

    aabbs = []

    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        bbx[:, :1] = - bbx[:, :1]

        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(bbx)

        aabb = tmp_pcd.get_axis_aligned_bounding_box()
        aabb.color = (0, 0, 1)
        aabbs.append(aabb)

    return aabbs


def linset_assign_list(vis,
                       lineset_list1,
                       lineset_list2,
                       update_mode='update'):
    """
    Associate two lists of lineset.

    Parameters
    ----------
    vis : open3d.Visualizer
    lineset_list1 : list
    lineset_list2 : list
    update_mode : str
        Add or update the geometry.
    """
    for j in range(len(lineset_list1)):
        index = j if j < len(lineset_list2) else -1
        lineset_list1[j] = \
            lineset_assign(lineset_list1[j],
                                     lineset_list2[index])
        if update_mode == 'add':
            vis.add_geometry(lineset_list1[j])
        else:
            vis.update_geometry(lineset_list1[j])


def lineset_assign(lineset1, lineset2):
    """
    Assign the attributes of lineset2 to lineset1.

    Parameters
    ----------
    lineset1 : open3d.LineSet
    lineset2 : open3d.LineSet

    Returns
    -------
    The lineset1 object with 2's attributes.
    """

    lineset1.points = lineset2.points
    lineset1.lines = lineset2.lines
    lineset1.colors = lineset2.colors

    return lineset1


def color_encoding(intensity, mode='intensity'):
    """
    Encode the single-channel intensity to 3 channels rgb color.

    Parameters
    ----------
    intensity : np.ndarray
        Lidar intensity, shape (n,)

    mode : str
        The color rendering mode. intensity, z-value and constant are
        supported.

    Returns
    -------
    color : np.ndarray
        Encoded Lidar color, shape (n, 3)
    """
    assert mode in ['intensity', 'z-value', 'constant']

    if mode == 'intensity':
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        int_color = np.c_[
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    elif mode == 'z-value':
        min_value = -1.5
        max_value = 0.5
        norm = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)
        cmap = cm.jet
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        colors = m.to_rgba(intensity)
        colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]
        colors[:, 3] = 0.5
        int_color = colors[:, :3]

    elif mode == 'constant':
        # regard all point cloud the same color
        int_color = np.ones((intensity.shape[0], 3))
        int_color[:, 0] *= 247 / 255
        int_color[:, 1] *= 244 / 255
        int_color[:, 2] *= 237 / 255

    return int_color


def visualize_single_sample_output_gt(pred_tensor,
                                      gt_tensor,
                                      pcd,
                                      show_vis=True,
                                      save_path='',
                                      mode='constant'):
    """
    Visualize the prediction, groundtruth with point cloud together.

    Parameters
    ----------
    pred_tensor : torch.Tensor
        (N, 8, 3) prediction.

    gt_tensor : torch.Tensor
        (N, 8, 3) groundtruth bbx

    pcd : torch.Tensor
        PointCloud, (N, 4).

    show_vis : bool
        Whether to show visualization.

    save_path : str
        Save the visualization results to given path.

    mode : str
        Color rendering mode.
    """

    def custom_draw_geometry(pcd, pred, gt):
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_size = 1.0

        vis.add_geometry(pcd)
        for ele in pred:
            vis.add_geometry(ele)
        for ele in gt:
            vis.add_geometry(ele)

        vis.run()
        vis.destroy_window()

    origin_lidar = pcd
    if not isinstance(pcd, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(pcd)

    origin_lidar_intcolor = \
        color_encoding(origin_lidar[:, -1] if mode == 'intensity'
                       else origin_lidar[:, 2], mode=mode)
    # left -> right hand
    origin_lidar[:, :1] = -origin_lidar[:, :1]

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

    oabbs_pred = bbx2oabb(pred_tensor, color=(1, 0, 0))
    oabbs_gt = bbx2oabb(gt_tensor, color=(0, 1, 0))

    visualize_elements = [o3d_pcd] + oabbs_pred + oabbs_gt
    if show_vis:
        custom_draw_geometry(o3d_pcd, oabbs_pred, oabbs_gt)
    if save_path:
        save_o3d_visualization(visualize_elements, save_path)


def visualize_single_sample_output_bev(pred_box, gt_box, pcd, dataset,
                                       show_vis=True,
                                       save_path=''):
    """
    Visualize the prediction, groundtruth with point cloud together in
    a bev format.

    Parameters
    ----------
    pred_box : torch.Tensor
        (N, 4, 2) prediction.

    gt_box : torch.Tensor
        (N, 4, 2) groundtruth bbx

    pcd : torch.Tensor
        PointCloud, (N, 4).

    show_vis : bool
        Whether to show visualization.

    save_path : str
        Save the visualization results to given path.
    """

    if not isinstance(pcd, np.ndarray):
        pcd = common_utils.torch_tensor_to_numpy(pcd)
    if pred_box is not None and not isinstance(pred_box, np.ndarray):
        pred_box = common_utils.torch_tensor_to_numpy(pred_box)
    if gt_box is not None and not isinstance(gt_box, np.ndarray):
        gt_box = common_utils.torch_tensor_to_numpy(gt_box)

    ratio = dataset.params["preprocess"]["args"]["res"]
    L1, W1, H1, L2, W2, H2 = dataset.params["preprocess"]["cav_lidar_range"]
    bev_origin = np.array([L1, W1]).reshape(1, -1)
    # (img_row, img_col)
    bev_map = dataset.project_points_to_bev_map(pcd, ratio)
    # (img_row, img_col, 3)
    bev_map = \
        np.repeat(bev_map[:, :, np.newaxis], 3, axis=-1).astype(np.float32)
    bev_map = bev_map * 255

    if pred_box is not None:
        num_bbx = pred_box.shape[0]
        for i in range(num_bbx):
            bbx = pred_box[i]

            bbx = ((bbx - bev_origin) / ratio).astype(int)
            bbx = bbx[:, ::-1]
            cv2.polylines(bev_map, [bbx], True, (0, 0, 255), 1)

    if gt_box is not None and len(gt_box):
        for i in range(gt_box.shape[0]):
            bbx = gt_box[i][:4, :2]
            bbx = (((bbx - bev_origin)) / ratio).astype(int)
            bbx = bbx[:, ::-1]
            cv2.polylines(bev_map, [bbx], True, (255, 0, 0), 1)

    if show_vis:
        plt.axis("off")
        plt.imshow(bev_map)
        plt.show()
    if save_path:
        plt.axis("off")
        plt.imshow(bev_map)
        plt.savefig(save_path)


def visualize_single_sample_dataloader(batch_data,
                                       o3d_pcd,
                                       order,
                                       key='origin_lidar',
                                       visualize=False,
                                       save_path='',
                                       oabb=False,
                                       mode='constant'):
    """
    Visualize a single frame of a single CAV for validation of data pipeline.

    Parameters
    ----------
    o3d_pcd : o3d.PointCloud
        Open3d PointCloud.

    order : str
        The bounding box order.

    key : str
        origin_lidar for late fusion and stacked_lidar for early fusion.

    visualize : bool
        Whether to visualize the sample.

    batch_data : dict
        The dictionary that contains current timestamp's data.

    save_path : str
        If set, save the visualization image to the path.

    oabb : bool
        If oriented bounding box is used.
    """

    origin_lidar = batch_data[key]
    if not isinstance(origin_lidar, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(origin_lidar)
    # we only visualize the first cav for single sample
    if len(origin_lidar.shape) > 2:
        origin_lidar = origin_lidar[0]
    origin_lidar_intcolor = \
        color_encoding(origin_lidar[:, -1] if mode == 'intensity'
                       else origin_lidar[:, 2], mode=mode)

    # left -> right hand
    origin_lidar[:, :1] = -origin_lidar[:, :1]

    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

    object_bbx_center = batch_data['object_bbx_center']
    object_bbx_mask = batch_data['object_bbx_mask']
    object_bbx_center = object_bbx_center[object_bbx_mask == 1]

    aabbs = bbx2linset(object_bbx_center, order) if not oabb else \
        bbx2oabb(object_bbx_center, order)
    visualize_elements = [o3d_pcd] + aabbs
    if visualize:
        o3d.visualization.draw_geometries(visualize_elements)

    if save_path:
        save_o3d_visualization(visualize_elements, save_path)

    return o3d_pcd, aabbs


def visualize_inference_sample_dataloader(pred_box_tensor,
                                          gt_box_tensor,
                                          origin_lidar,
                                          o3d_pcd,
                                          mode='constant'):
    """
    Visualize a frame during inference for video stream.

    Parameters
    ----------
    pred_box_tensor : torch.Tensor
        (N, 8, 3) prediction.

    gt_box_tensor : torch.Tensor
        (N, 8, 3) groundtruth bbx

    origin_lidar : torch.Tensor
        PointCloud, (N, 4).

    o3d_pcd : open3d.PointCloud
        Used to visualize the pcd.

    mode : str
        lidar point rendering mode.
    """

    if not isinstance(origin_lidar, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(origin_lidar)
    # we only visualize the first cav for single sample
    if len(origin_lidar.shape) > 2:
        origin_lidar = origin_lidar[0]
    origin_lidar_intcolor = \
        color_encoding(origin_lidar[:, -1] if mode == 'intensity'
                       else origin_lidar[:, 2], mode=mode)

    if not isinstance(pred_box_tensor, np.ndarray):
        pred_box_tensor = common_utils.torch_tensor_to_numpy(pred_box_tensor)
    if not isinstance(gt_box_tensor, np.ndarray):
        gt_box_tensor = common_utils.torch_tensor_to_numpy(gt_box_tensor)

    # left -> right hand
    origin_lidar[:, :1] = -origin_lidar[:, :1]

    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

    gt_o3d_box = bbx2linset(gt_box_tensor, order='hwl', color=(0, 1, 0))
    pred_o3d_box = bbx2linset(pred_box_tensor, color=(1, 0, 0))

    return o3d_pcd, pred_o3d_box, gt_o3d_box


def visualize_sequence_dataloader(dataloader, order, color_mode='constant'):
    """
    Visualize the batch data in animation.

    Parameters
    ----------
    dataloader : torch.Dataloader
        Pytorch dataloader

    order : str
        Bounding box order(N, 7).

    color_mode : str
        Color rendering mode.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().show_coordinate_frame = True

    # used to visualize lidar points
    vis_pcd = o3d.geometry.PointCloud()
    # used to visualize object bounding box, maximum 50
    vis_aabbs = []
    for _ in range(50):
        vis_aabbs.append(o3d.geometry.LineSet())

    while True:
        for i_batch, sample_batched in enumerate(dataloader):
            print(i_batch)
            pcd, aabbs = \
                visualize_single_sample_dataloader(sample_batched['ego'],
                                                   vis_pcd,
                                                   order,
                                                   mode=color_mode)
            if i_batch == 0:
                vis.add_geometry(pcd)
                for i in range(len(vis_aabbs)):
                    index = i if i < len(aabbs) else -1
                    vis_aabbs[i] = lineset_assign(vis_aabbs[i], aabbs[index])
                    vis.add_geometry(vis_aabbs[i])

            for i in range(len(vis_aabbs)):
                index = i if i < len(aabbs) else -1
                vis_aabbs[i] = lineset_assign(vis_aabbs[i], aabbs[index])
                vis.update_geometry(vis_aabbs[i])

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.001)

    vis.destroy_window()


def save_o3d_visualization(element, save_path):
    """
    Save the open3d drawing to folder.

    Parameters
    ----------
    element : list
        List of o3d.geometry objects.

    save_path : str
        The save path.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for i in range(len(element)):
        vis.add_geometry(element[i])
        vis.update_geometry(element[i])

    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(save_path)
    vis.destroy_window()


def visualize_bev(batch_data):
    bev_input = batch_data["processed_lidar"]["bev_input"]
    label_map = batch_data["label_dict"]["label_map"]
    if not isinstance(bev_input, np.ndarray):
        bev_input = common_utils.torch_tensor_to_numpy(bev_input)

    if not isinstance(label_map, np.ndarray):
        label_map = label_map[0].numpy() if not label_map[0].is_cuda else \
            label_map[0].cpu().detach().numpy()

    if len(bev_input.shape) > 3:
        bev_input = bev_input[0, ...]

    plt.matshow(np.sum(bev_input, axis=0))
    plt.axis("off")
    plt.matshow(label_map[0, :, :])
    plt.axis("off")
    plt.show()

def draw_box_plt(boxes_dec, ax, color=None, linewidth_scale=1.0):
    """
    draw boxes in a given plt ax
    :param boxes_dec: (N, 5) or (N, 7) in metric
    :param ax:
    :return: ax with drawn boxes
    """
    if not len(boxes_dec)>0:
        return ax
    boxes_np= boxes_dec
    if not isinstance(boxes_np, np.ndarray):
        boxes_np = boxes_np.cpu().detach().numpy()
    if boxes_np.shape[-1]>5:
        boxes_np = boxes_np[:, [0, 1, 3, 4, 6]]
    x = boxes_np[:, 0]
    y = boxes_np[:, 1]
    dx = boxes_np[:, 2]
    dy = boxes_np[:, 3]

    x1 = x - dx / 2
    y1 = y - dy / 2
    x2 = x + dx / 2
    y2 = y + dy / 2
    theta = boxes_np[:, 4:5]
    # bl, fl, fr, br
    corners = np.array([[x1, y1],[x1,y2], [x2,y2], [x2, y1]]).transpose(2, 0, 1)
    new_x = (corners[:, :, 0] - x[:, None]) * np.cos(theta) + (corners[:, :, 1]
              - y[:, None]) * (-np.sin(theta)) + x[:, None]
    new_y = (corners[:, :, 0] - x[:, None]) * np.sin(theta) + (corners[:, :, 1]
              - y[:, None]) * (np.cos(theta)) + y[:, None]
    corners = np.stack([new_x, new_y], axis=2)
    for corner in corners:
        ax.plot(corner[[0,1,2,3,0], 0], corner[[0,1,2,3,0], 1], color=color, linewidth=0.5*linewidth_scale)
        # draw front line (
        ax.plot(corner[[2, 3], 0], corner[[2, 3], 1], color=color, linewidth=2*linewidth_scale)
    return ax


def draw_points_boxes_plt(pc_range, points=None, boxes_pred=None, boxes_gt=None, save_path=None,
                          points_c='y.', bbox_gt_c='green', bbox_pred_c='red', return_ax=False, ax=None):
    if ax is None:
        ax = plt.figure(figsize=(15, 6)).add_subplot(1, 1, 1)
        ax.set_aspect('equal', 'box')
        ax.set(xlim=(pc_range[0], pc_range[3]),
               ylim=(pc_range[1], pc_range[4]))
    if points is not None:
        ax.plot(points[:, 0], points[:, 1], points_c, markersize=0.1)
    if (boxes_gt is not None) and len(boxes_gt)>0:
        ax = draw_box_plt(boxes_gt, ax, color=bbox_gt_c)
    if (boxes_pred is not None) and len(boxes_pred)>0:
        ax = draw_box_plt(boxes_pred, ax, color=bbox_pred_c)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.savefig(save_path)
    if return_ax:
        return ax
    plt.close() 
    

import numpy as np
try:
    import torch
    _is_tensor = torch.is_tensor
except Exception:
    torch = None
    def _is_tensor(x): return False


def _summ(v):
    """返回简短摘要，而不是打印具体内容"""
    try:
        import torch
        if _is_tensor(v):
            return f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device})"
    except Exception:
        pass
    if isinstance(v, np.ndarray):
        return f"ndarray(shape={v.shape}, dtype={v.dtype})"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}(len={len(v)})"
    if isinstance(v, dict):
        return f"dict(len={len(v)})"
    if isinstance(v, (str, bytes)):
        s = v if isinstance(v, str) else v.decode('utf-8', 'ignore')
        s = s if len(s) <= 60 else s[:57] + "..."
        return f"{type(v).__name__}({s})"
    return type(v).__name__


def print_tree(obj, prefix="ego", indent=0, max_list_probe=1, max_depth=10):
    """
    递归打印键结构，不打印大对象内容。
    - max_list_probe: 对 list/tuple 只探测前几个元素的结构
    - max_depth: 最深递归层数，避免极端嵌套
    """
    if indent > max_depth:
        print("  " * indent + f"{prefix}: … (max_depth reached)")
        return

    pad = "  " * indent
    if isinstance(obj, dict):
        print(f"{pad}{prefix}: dict(len={len(obj)})")
        for k in obj:
            v = obj[k]
            key_path = f"{prefix}.{k}"
            if isinstance(v, dict) or isinstance(v, (list, tuple)):
                print_tree(v, key_path, indent + 1, max_list_probe, max_depth)
            else:
                print(f"{pad}  {key_path}: {_summ(v)}")
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{prefix}: {type(obj).__name__}(len={len(obj)})")
        n = min(len(obj), max_list_probe)
        for i in range(n):
            vi = obj[i]
            sub = f"{prefix}[{i}]"
            if isinstance(vi, dict) or isinstance(vi, (list, tuple)):
                print_tree(vi, sub, indent + 1, max_list_probe, max_depth)
            else:
                print(f"{pad}  {sub}: {_summ(vi)}")
        if len(obj) > n:
            print(f"{pad}  {prefix}[{n}:]: … ({len(obj)-n} more)")
    else:
        print(f"{pad}{prefix}: {_summ(obj)}")


def print_ego(ego_dict):
    """入口：打印 sample_batched['ego'] 的结构"""
    print_tree(ego_dict, prefix="ego", indent=0, max_list_probe=1, max_depth=10)



def _score_img_uint8(hwc):
    # 评分：动态范围 + 中间灰度占比，避免全黑/全白
    x = hwc.astype(np.float32)
    rng = float(x.max() - x.min())
    hist, _ = np.histogram(x, bins=16, range=(0, 255))
    mid = hist[3:13].sum() / (hist.sum() + 1e-6)
    return rng + 1000 * mid  # mid 权重大些，越大越好

def _denorm_try_candidates(chw, mean, std, to_rgb=True, force_auto_contrast=False):
    """
    尝试多套常见配置，返回最合理的 HWC uint8。
    优先级：显式 mean/std > 猜 ImageNet(0~1) > 猜 [-1,1]/[0,1] > 自适应拉伸。
    """
    arr = _to_numpy(chw).astype(np.float32)

    cands = []
    # (A) 显式 mean/std（若有）
    if mean is not None and std is not None:
        m = np.asarray(mean, np.float32); s = np.asarray(std, np.float32)
        img = arr * s[:,None,None] + m[:,None,None]
        scale255 = (img.max() <= 1.0 + 1e-3)  # 如果是 0~1 标度，则乘 255
        if scale255: img = img * 255.0
        if not to_rgb: img = img[[2,1,0], ...]
        candA = np.transpose(np.clip(img, 0, 255).astype(np.uint8), (1,2,0))
        cands.append(("explicit", candA))

    # (B) ImageNet 风格 (0~1 标度)： (x/255 - mean01)/std01
    mean01 = np.array([0.485, 0.456, 0.406], np.float32)
    std01  = np.array([0.229, 0.224, 0.225], np.float32)
    imgB = (arr * std01[:,None,None] + mean01[:,None,None]) * 255.0
    if not to_rgb: imgB = imgB[[2,1,0], ...]
    candB = np.transpose(np.clip(imgB, 0, 255).astype(np.uint8), (1,2,0))
    cands.append(("imagenet01", candB))

    # (C) [-1,1] → [0,1] → 255
    imgC = (arr * 0.5 + 0.5) * 255.0
    if not to_rgb: imgC = imgC[[2,1,0], ...]
    candC = np.transpose(np.clip(imgC, 0, 255).astype(np.uint8), (1,2,0))
    cands.append(("minus1to1", candC))

    # (D) 认为已经是 0~1：×255
    imgD = arr * 255.0
    if not to_rgb: imgD = imgD[[2,1,0], ...]
    candD = np.transpose(np.clip(imgD, 0, 255).astype(np.uint8), (1,2,0))
    cands.append(("zero2one", candD))

    # 选评分最高的候选
    best_name, best_img, best_score = None, None, -1
    for name, img in cands:
        sc = _score_img_uint8(img)
        if sc > best_score:
            best_name, best_img, best_score = name, img, sc
    # 若还是很差，做一次自适应拉伸（只为看清楚）
    if force_auto_contrast or best_score < 500:  # 这个阈值是经验值
        x = np.transpose(arr, (1,2,0))
        p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
        x = (x - p1) / max(p99 - p1, 1e-6)
        x = np.clip(x, 0, 1)
        x = (x * 255.0).astype(np.uint8)
        best_img = x
        best_name = (best_name or "auto") + "+autocontrast"

    return best_img, best_name


def _to_numpy(x):
    import torch
    if isinstance(x, np.ndarray): return x
    if hasattr(torch, "is_tensor") and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _get_cam_pack(ego, key='image_inputs'):
    p = ego[key]
    imgs   = _to_numpy(p['imgs'])          # (B,N,C,H,W) or (N,C,H,W)
    K      = _to_numpy(p['intrins'])       # (B,N,3,3)
    pR     = _to_numpy(p['post_rots'])     # (B,N,3,3)
    pT     = _to_numpy(p['post_trans'])    # (B,N,3)
    if 'rots' in p and 'trans' in p:
        Rce = _to_numpy(p['rots'])         # (B,N,3,3)  cam->ego
        tce = _to_numpy(p['trans'])        # (B,N,3)
        EX = None
    else:
        Rce, tce = None, None
        EX = _to_numpy(p['extrinsics']) if 'extrinsics' in p else None  # (B,N,4,4)

    # strip batch
    if imgs.ndim==5: imgs = imgs[0]
    K, pR, pT = K[0], pR[0], pT[0]
    if Rce is not None: Rce, tce = Rce[0], tce[0]
    if EX is not None:  EX = EX[0]

    mean = p.get('mean', None); std = p.get('std', None); to_rgb = p.get('to_rgb', True)
    return imgs, K, pR, pT, Rce, tce, EX, mean, std, to_rgb

def _denorm_chw(chw, mean, std, to_rgb=True):
    arr = _to_numpy(chw).astype(np.float32)
    if mean is None or std is None:
        mn, mx = float(arr.min()), float(arr.max())
        if mn>=-1.2 and mx<=1.2:
            if mn<0: arr = arr*0.5+0.5  # [-1,1]→[0,1]
            arr *= 255.0
    else:
        mean = np.asarray(mean, np.float32); std = np.asarray(std, np.float32)
        arr = arr*std[:,None,None] + mean[:,None,None]
    if not to_rgb:
        arr = arr[[2,1,0], ...]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.transpose(arr, (1,2,0))  # HWC

def _bilinear_sample(img, x, y):
    H, W = img.shape[:2]
    x0 = np.floor(x).astype(np.int32); x1 = x0+1
    y0 = np.floor(y).astype(np.int32); y1 = y0+1
    x0 = np.clip(x0,0,W-1); x1 = np.clip(x1,0,W-1)
    y0 = np.clip(y0,0,H-1); y1 = np.clip(y1,0,H-1)
    Ia = img[y0,x0]; Ib = img[y1,x0]; Ic = img[y0,x1]; Id = img[y1,x1]
    wa = (x1-x)*(y1-y); wb = (x1-x)*(y-y0)
    wc = (x-x0)*(y1-y); wd = (x-x0)*(y-y0)
    out = Ia*wa[...,None] + Ib*wb[...,None] + Ic*wc[...,None] + Id*wd[...,None]
    return out.astype(np.uint8)

def _smooth1d(x, k=9):
    if k <= 1: return x
    k = int(k)
    w = np.ones(k, dtype=np.float32) / k
    return np.convolve(x.astype(np.float32), w, mode='same')

def crop_pano_to_strip(pano, valid_mask=None, *, 
                       row_cover_thr=0.98,   # 每行有效像素比例阈值，越大越细
                       smooth_k=11,          # 对覆盖率做一点平滑，避免锯齿
                       pad=2):               # 上下各保留几行做缓冲
    """
    把 equirectangular 全景裁成没有黑边的一条长条（仅裁上下）。
    - pano: (H,W,3) uint8
    - valid_mask: (H,W) bool，如果没有就用 pano 非零近似
    """
    H, W = pano.shape[:2]
    if valid_mask is None:
        # 兜底：把全黑视为无效
        valid_mask = (pano.sum(axis=2) > 0)

    # 每一行的覆盖比例
    row_cover = valid_mask.mean(axis=1)           # (H,)
    row_cover_s = _smooth1d(row_cover, smooth_k)  # 平滑一下

    # 选出覆盖率高的行
    keep_rows = row_cover_s >= row_cover_thr

    if not keep_rows.any():
        # 如果阈值过高导致一个也没选上，就退一步取中间 60% 高度
        top = int(H * 0.2)
        bot = int(H * 0.8)
    else:
        # 找到最长的连续 True 段（一般就是中间那一条）
        idx = np.where(keep_rows)[0]
        splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        seg = max(splits, key=len)
        top, bot = seg[0], seg[-1]

    top = max(0, top - pad)
    bot = min(H - 1, bot + pad)
    strip = pano[top:bot + 1, :, :].copy()
    return strip, (top, bot)


def build_equirectangular_panorama_safe(
    ego, image_key='image_inputs',
    pano_w=2048, pano_h=512, v_fov_deg=(-45,45),
    debug=False, return_mask=False
):
    imgs, K, pR, pT, Rce, tce, EX, mean, std, to_rgb = _get_cam_pack(ego, image_key)
    N, C, H, W = imgs.shape
    imgs_rgb, choice_names = [], []
    for i in range(N):
        img_i, how = _denorm_try_candidates(imgs[i], mean, std, to_rgb, force_auto_contrast=False)
        imgs_rgb.append(img_i)
        choice_names.append(how)
    print("[PANO] denorm choices:", choice_names)

    # cam->ego → ego->cam
    if Rce is not None:
        R_ec = np.transpose(Rce, (0,2,1))         # (N,3,3)
    elif EX is not None:
        R_ce = EX[:, :3, :3]
        R_ec = np.transpose(R_ce, (0,2,1))
    else:
        raise ValueError("no extrinsics")

    # 全景经纬
    th = np.linspace(-np.pi, np.pi, pano_w, endpoint=False)
    phi_min = min(np.deg2rad(v_fov_deg[0]), np.deg2rad(v_fov_deg[1]))
    phi_max = max(np.deg2rad(v_fov_deg[0]), np.deg2rad(v_fov_deg[1]))
    phi = np.linspace(phi_max, phi_min, pano_h)   # 关键：反向！
    TH, PH = np.meshgrid(th, phi)
    # ph = np.linspace(np.deg2rad(v_fov_deg[0]), np.deg2rad(v_fov_deg[1]), pano_h)
    # TH, PH = np.meshgrid(th, ph)
    d_ego = np.stack([np.cos(PH)*np.cos(TH),
                      np.cos(PH)*np.sin(TH),
                      np.sin(PH)], axis=-1)  # (H,W,3)

    pano = np.zeros((pano_h, pano_w, 3), np.uint8)
    best = np.full((pano_h, pano_w), -1e9, np.float32)

    def project_one(i, use_T=False, use_post=True):
        # 方向：行向量右乘矩阵，不再转置
        R = R_ec[i].T if use_T else R_ec[i]
        d_cam = d_ego @ R
        z = d_cam[...,2]
        vis = z > 1e-6
        xn = d_cam[...,0]/(z+1e-12); yn = d_cam[...,1]/(z+1e-12)
        xyz1 = np.stack([xn, yn, np.ones_like(xn)], -1)
        uv = xyz1 @ K[i].T
        u, v = uv[...,0], uv[...,1]
        if use_post:
            uv1 = np.stack([u, v, np.ones_like(u)], -1)
            uv1 = uv1 @ pR[i].T + pT[i]   # 行向量右乘
            u, v = uv1[...,0], uv1[...,1]
        Hi, Wi = imgs_rgb[i].shape[:2]
        valid = vis & (u>=0.5)&(u<=Wi-1.5)&(v>=0.5)&(v<=Hi-1.5)
        return u, v, valid, z

    # 尝试 4 组合：是否再转置、是否应用 post
    combos = [(False,True),(False,False),(True,True),(True,False)]
    scores = []
    cache = {}
    for useT, useP in combos:
        tot = 0
        data = []
        for i in range(N):
            u,v,val,z = project_one(i, useT, useP)
            tot += int(val.sum()); data.append((u,v,val,z))
        scores.append(tot); cache[(useT,useP)] = data
    pick = combos[int(np.argmax(scores))]
    if debug:
        print(f"[PANO] valid pixels per combo {dict(zip(combos,scores))}, pick {pick}")

    useT, useP = pick
    for i in range(N):
        u,v,valid,z = cache[(useT,useP)][i]
        if not np.any(valid): continue
        samp = _bilinear_sample(imgs_rgb[i], u[valid], v[valid])
        take = valid & (z > best)
        pano[take] = samp[(z[valid] > best[valid])]
        best[take] = z[take]
        
    valid_mask = best > -1e8   # 哪些像素至少被某路相机覆盖过

    if return_mask:
        return pano, valid_mask
    return pano

def visualize_sequence_real_dataloader(
    dataloader,
    # order,                      # 兼容旧签名（此函数不再使用）
    # color_mode='constant',      # 兼容旧签名（此函数不再使用）
    *,
    image_key='auto',           # 'auto' 或明确键名：image_inputs/camera_imgs/images/rgb/img/cams
    cam_idxs='all',             # 'all' 或 '0,1,3'
    grid_cols=3,                # 多相机并排显示的列数
    save_rgb_dir='',            # 可选：保存每一帧到该目录
    save_video='',              # 可选：保存成 mp4/avi
    fps=10,                     # 保存视频时的帧率
    debug_keys=False            # 打印找到的键与 shape
):
    """
    Visualize dataset images in a loop (RGB only, no point cloud).

    This is a drop-in replacement of the original function:
    - keeps (dataloader, order, color_mode) for compatibility
    - shows camera images from the dataloader batches
    """

    import numpy as np, time, os, cv2
    import matplotlib.pyplot as plt
    try:
        import torch
    except Exception:
        torch = None

    # ---------- helpers ----------
    def _to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.array(x)

    def _normalize_uint8(img):
        img = np.ascontiguousarray(img)
        if img.dtype in (np.float32, np.float64):
            if img.max() <= 1.0:
                img = (img * 255.0).clip(0, 255)
        return img.astype(np.uint8)

    def _reshape_to_hwc(arr):
        """统一成 (H,W,3) 列表；自动剥离多余外层维度（如 batch/cav）"""
        a = arr
        while a.ndim >= 5:
            a = a[0]
        imgs = []
        cam_dim = None
        if a.ndim == 4:
            cam_dim = 0  # (N_cam, ...)
            if a.shape[-1] == 3:     # (N_cam, H, W, 3)
                for k in range(a.shape[0]):
                    imgs.append(_normalize_uint8(a[k]))
            else:                    # (N_cam, 3, H, W)
                for k in range(a.shape[0]):
                    imgs.append(_normalize_uint8(np.transpose(a[k], (1, 2, 0))))
        elif a.ndim == 3:
            if a.shape[0] in (1, 3):  # (C,H,W)
                imgs.append(_normalize_uint8(np.transpose(a, (1, 2, 0))))
            else:                     # (H,W,3)
                imgs.append(_normalize_uint8(a))
        elif a.ndim == 2:             # (H,W) 灰度
            imgs.append(_normalize_uint8(np.stack([a, a, a], axis=-1)))
        return imgs, cam_dim

    def _select_idxs(n, spec):
        if isinstance(spec, str):
            spec = spec.strip().lower()
            if spec == 'all':
                return list(range(n))
            return [int(x) for x in spec.split(',') if x.strip().isdigit()]
        if isinstance(spec, (list, tuple)):
            return [int(x) for x in spec]
        return [0]

    def _tile(images, cols=3):
        if not images: return None
        H = max(im.shape[0] for im in images)
        W = max(im.shape[1] for im in images)
        pads = []
        for im in images:
            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            canvas[:im.shape[0], :im.shape[1]] = im
            pads.append(canvas)
        cols = max(1, int(cols))
        rows = int(np.ceil(len(pads) / cols))
        grid = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)
        for i, im in enumerate(pads):
            r, c = divmod(i, cols)
            grid[r*H:(r+1)*H, c*W:(c+1)*W] = im
        return grid

    # ---------- viewer ----------
    if save_rgb_dir:
        os.makedirs(save_rgb_dir, exist_ok=True)
    writer = None

    import matplotlib
    matplotlib.use('TkAgg' if matplotlib.get_backend() == 'agg' else matplotlib.get_backend())
    matplotlib.rcParams['toolbar'] = 'None'  # 加在 plt.subplots 之前
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_axis_off()
    im_artist = False
    frame_id = 0

    while True:
        for i_batch, sample_batched in enumerate(dataloader):
            ego = sample_batched['ego']

            pano, mask = build_equirectangular_panorama_safe(
                ego, image_key='image_inputs',
                pano_w=2048, pano_h=512, v_fov_deg=(-45, 45),
                debug=True, return_mask=True
            )

            strip, (t, b) = crop_pano_to_strip(
                pano, mask, row_cover_thr=0.985, smooth_k=15, pad=2
            )

            # 显示 / 保存的就是 strip
            if im_artist is None:
                im_artist = ax.imshow(strip)
            else:
                # im_artist.set_data(strip)
                pass
            fig.canvas.draw(); fig.canvas.flush_events()

            if save_rgb_dir:
                cv2.imwrite(os.path.join(save_rgb_dir, f"pano_{frame_id:06d}.png"),
                            cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
            if save_video:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*('mp4v' if save_video.lower().endswith('.mp4') else 'XVID'))
                    H, W = pano.shape[:2]
                    writer = cv2.VideoWriter(save_video, fourcc, fps, (W, H))
                writer.write(cv2.cvtColor(pano, cv2.COLOR_RGB2BGR))
            frame_id += 1
    # 不会走到，但保持完整性
    if writer is not None:
        writer.release()
    plt.ioff()