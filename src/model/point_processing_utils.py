"""
This module contains utility functions for processing point data, including filtering valid targets,
merging person data, smoothing keypoints, and updating keypoint buffers.
"""

import torch
import polars as pl
import numpy as np
from scipy.signal import savgol_filter
from .one_euro_filter import OneEuroFilterTorch
from .keypoint_info import halpe26_keypoint_info


def filter_valid_targets(online_targets, select_id: int = None):
    """
    Filter out valid tracking targets.

    Args:
        online_targets (List): All online tracking targets.
        select_id (int, optional): Select a specific tracking ID.

    Returns:
        Tuple: Valid bounding boxes (Tensor) and tracking IDs (Tensor).
    """
    if not online_targets:
        return torch.empty((0, 4), device='cuda'), torch.empty((0,), dtype=torch.int32, device='cuda')
    tlwhs = []
    for target in online_targets:
        tlwhs.append(target.tlwh)
    # 直接生成張量以減少資料轉換
    tlwhs = torch.tensor(np.array(tlwhs), device='cuda')  # (n, 4)
    track_ids = torch.tensor([target.track_id for target in online_targets], dtype=torch.int32, device='cuda')  # (n,)

    # 計算面積 (w * h)
    areas = tlwhs[:, 2] * tlwhs[:, 3]

    # 過濾面積大於 10 的目標
    valid_mask = areas > 10

    # 如果指定了 select_id，則進一步過濾
    if select_id is not None:
        valid_mask &= (track_ids == select_id)

    # 根據過濾條件提取有效的邊界框和追蹤ID
    valid_tlwhs = tlwhs[valid_mask]
    valid_track_ids = track_ids[valid_mask]

    # 將 (x1, y1, w, h) 轉換為 (x1, y1, x2, y2)
    valid_bbox = torch.cat([valid_tlwhs[:, :2], valid_tlwhs[:, :2] + valid_tlwhs[:, 2:4]], dim=1)

    return valid_bbox.cpu().tolist(), valid_track_ids.cpu().tolist()

def merge_person_data(pred_instances, track_ids: list, frame_num: int = None) -> pl.DataFrame:
    """
    Merge person data from prediction instances and tracking IDs.

    Args:
        pred_instances (dict): Prediction instances containing bounding boxes and keypoints.
        track_ids (list): List of tracking IDs.
        frame_num (int, optional): Frame number.

    Returns:
        pl.DataFrame: DataFrame containing merged person data.
    """
    person_bboxes = pred_instances['bboxes']

    # 優化：提前創建列表，避免多次 append 操作
    new_person_data = []

    # 預先準備一些常用資料結構，減少重複創建
    halpe26_shape = len(halpe26_keypoint_info['keypoints'])

    if any([len(person_bboxes) == 0, len(track_ids) == 0, len(pred_instances) == 0]):
        new_person_data = [{
            'track_id': None,
            'bbox': [],
            'area': None,
            'keypoints': [],
            'frame_number': frame_num
        }]
    else:
        for person, pid, bbox in zip(pred_instances, track_ids, person_bboxes):
            keypoints_data = np.hstack((
                np.round(person['keypoints'][0], 2),
                np.round(person['keypoint_scores'][0], 2).reshape(-1, 1),
                np.full((len(person['keypoints'][0]), 1), False, dtype=bool)
            ))

            new_kpts = np.full((halpe26_shape, keypoints_data.shape[1]), 0.9)
            new_kpts[:26] = keypoints_data

            new_kpts = new_kpts.tolist()
            # 轉換 bbox 為列表
            bbox = bbox.tolist()

            # 優化：將字典構建過程集中處理，減少冗余運算
            person_info = {
                'track_id': np.float64(pid),
                'bbox': bbox,
                'area': np.round(bbox[2] * bbox[3], 2),
                'keypoints': new_kpts
            }
            if frame_num is not None:
                person_info['frame_number'] = frame_num

            new_person_data.append(person_info)

    # 使用 PyArrow 加速 DataFrame 構建
    new_person_df = pl.DataFrame(new_person_data)

    return new_person_df


def smooth_keypoints(person_df: pl.DataFrame, new_person_df: pl.DataFrame, track_ids: list) -> pl.DataFrame:
    """
    Smooth 2D keypoint data.

    Args:
        person_df (pl.DataFrame): DataFrame containing data from the previous frame.
        new_person_df (pl.DataFrame): DataFrame containing data from the current frame.
        track_ids (list): List of track IDs to process.

    Returns:
        pl.DataFrame: Smoothed DataFrame.
    """
    smooth_filter_dict = {}

    # 當前幀無數據時，返回原始 new_person_df
    if person_df.is_empty():
        return new_person_df

    # 獲取上一幀的 frame_number
    last_frame_number = person_df.select("frame_number").tail(1).item()

    for track_id in track_ids:
        # 選擇上一幀和當前幀的數據
        pre_person_data = person_df.filter(
            (person_df['frame_number'] == last_frame_number) &
            (person_df['track_id'] == track_id)
        )
        curr_person_data = new_person_df.filter(new_person_df['track_id'] == track_id)

        # 如果當前幀或前幀沒有該 track_id 的數據，跳過
        if pre_person_data.is_empty() or curr_person_data.is_empty():
            continue

        # 初始化濾波器字典（如果不存在）
        if track_id not in smooth_filter_dict:
            keypoints_len = len(pre_person_data.select("keypoints").row(0)[0])
            smooth_filter_dict[track_id] = {joint: OneEuroFilterTorch() for joint in range(keypoints_len)}

        # 獲取上一幀和當前幀的關鍵點數據
        pre_kpts = torch.tensor(pre_person_data.select("keypoints").row(0)[0], device='cuda')
        curr_kpts = torch.tensor(curr_person_data.select("keypoints").row(0)[0], device='cuda')
        smoothed_kpts = []

        # 使用濾波器平滑每個關節點
        for joint_idx, (pre_kpt, curr_kpt) in enumerate(zip(pre_kpts, curr_kpts)):
            pre_kptx, pre_kpty = pre_kpt[0], pre_kpt[1]
            curr_kptx, curr_kpty, curr_conf, curr_label = curr_kpt[0], curr_kpt[1], curr_kpt[2], curr_kpt[3]

            if all([pre_kptx.item() != 0,
                    pre_kpty.item() != 0,
                    curr_kptx.item() != 0,
                    curr_kpty.item() != 0]):
                # 為每個關節應用單獨的濾波器
                curr_kptx = smooth_filter_dict[track_id][joint_idx](curr_kptx, pre_kptx)
                curr_kpty = smooth_filter_dict[track_id][joint_idx](curr_kpty, pre_kpty)
            smoothed_kpts.append([curr_kptx.cpu().item(), curr_kpty.cpu().item(), curr_conf.item(), curr_label.item()])

        # 更新當前幀的數據
        new_person_df = new_person_df.with_columns(
            pl.when(new_person_df['track_id'] == track_id)
            .then(pl.Series("keypoints", [smoothed_kpts]))
            .otherwise(new_person_df["keypoints"])
            .alias("keypoints")
        )

    return new_person_df

def update_keypoint_buffer(
        person_df:pl.DataFrame,
        track_id:int, kpt_id: int, frame_num:int,
        window_length=5, polyorder=2
        )->list:
    """
    Update the keypoint buffer and apply Savgol filter for smoothing.

    Args:
        person_df (pl.DataFrame): DataFrame containing person data.
        track_id (int): Tracking ID.
        kpt_id (int): Keypoint ID.
        frame_num (int): Frame number.
        window_length (int, optional): Window length for Savgol filter. Default is 5.
        polyorder (int, optional): Polynomial order for Savgol filter. Default is 2.

    Returns:
        list: Smoothed keypoints.
    """
    filtered_df = person_df.filter(
        (person_df['track_id'] == track_id) &
        (person_df['frame_number'] < frame_num)
    ).sort('frame_number')

    if filtered_df.is_empty():
        return None
    filtered_df = filtered_df.sort("frame_number")

    kpt_buffer = []
    for kpts in filtered_df['keypoints']:
        kpt = kpts[kpt_id]
        if kpt is not None and len(kpt) >= 2:
            kpt_buffer.append((kpt[0], kpt[1]))

    # 如果緩衝區長度大於等於窗口長度，則應用Savgol濾波器進行平滑
    if len(kpt_buffer) >= window_length:
        # 確保窗口長度為奇數且不超過緩衝區長度
        if window_length > len(kpt_buffer):
            window_length = len(kpt_buffer) if len(kpt_buffer) % 2 == 1 else len(kpt_buffer) - 1
        # 確保多項式階數小於窗口長度
        current_polyorder = min(polyorder, window_length - 1)

        # 分別提取x和y座標
        x = np.array([point[0] for point in kpt_buffer])
        y = np.array([point[1] for point in kpt_buffer])

        # 應用Savgol濾波器
        x_smooth = savgol_filter(x, window_length=window_length, polyorder=current_polyorder)
        y_smooth = savgol_filter(y, window_length=window_length, polyorder=current_polyorder)

        # 將平滑後的座標重新打包
        smoothed_points = list(zip(x_smooth, y_smooth))
    else:
        # 緩衝區長度不足，直接使用原始座標
        smoothed_points = kpt_buffer

    return smoothed_points
