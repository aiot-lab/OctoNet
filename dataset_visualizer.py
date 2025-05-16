import os
import time
import struct
import threading
import re
import pickle
from datetime import datetime, timedelta
from math import floor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import librosa
import librosa.display

from dataset_loader import get_dataset, get_dataloader
from func_utils import SubpageInterpolating
from shape_converter import DataConverter

VIZ_OUTPUT_ROOT = 'viz_output'

def visualizer_decorator(func):
    """
    Helper decorator to visualizers.
    
    Args:
        batch: The batch to visualize.
        output_dir: The output directory, removing the 'visualize_' prefix and inline/suffix '_batch'.
        fps_out: The output FPS.
        **kwargs: Additional keyword arguments.
    
    Returns:
        Decorated function.
    """
    def wrapper(batch, output_dir: str = func.__name__.replace('visualize_', '').replace('_batch', ''), fps_out: float = 30.0, **kwargs):
        output_dir = os.path.join(VIZ_OUTPUT_ROOT, output_dir)
        os.makedirs(output_dir, exist_ok=True)
        return func(batch, output_dir, fps_out, **kwargs)
    return wrapper

def step_ira_frame(
    ira_tensor: torch.Tensor,  # shape [T_ira, H, W]
    float_idx: float
) -> np.ndarray:
    """
    'Step' interpolation: pick floor(float_idx).
    We clamp if float_idx < 0 or > (T_ira-1).
    """
    T_ira = ira_tensor.shape[0]
    if T_ira == 1:
        # Only one IRA frame total
        return ira_tensor[0].cpu().numpy()

    if float_idx <= 0.0:
        return ira_tensor[0].cpu().numpy()
    if float_idx >= (T_ira - 1):
        return ira_tensor[-1].cpu().numpy()

    i0 = int(np.floor(float_idx))
    return ira_tensor[i0].cpu().numpy()

def step_camera_frame(
    cam_tensor: torch.Tensor,  # shape [T_cam, H, W, 3]
    float_idx: float
) -> np.ndarray:
    """
    Simple step interpolation for the camera frames:
    pick floor(float_idx), clamp out-of-range.
    """
    T_cam_ = cam_tensor.shape[0]
    if T_cam_ == 1:
        return cam_tensor[0].cpu().numpy()

    if float_idx <= 0.0:
        return cam_tensor[0].cpu().numpy()
    if float_idx >= (T_cam_ - 1):
        return cam_tensor[-1].cpu().numpy()

    i0 = int(np.floor(float_idx))
    return cam_tensor[i0].cpu().numpy()

@visualizer_decorator
def visualize_ira_and_rgb_mosaic_batch(
    batch,
    output_dir: str = 'ira_rgb_mosaic',
    fps_out: float = 30.0
    ):
    """
    Create a 2x3 mosaic video for each sample in the batch, at a final output FPS of 30,
    using "step" interpolation for the IRA frames.

    We assume:
    - IRA is ~6.9 FPS, shape [B, 5, T_ira, H, W].
    - Camera is 30 FPS, shape [B, N, T_cam, H, W, 3].
    - Both share the same start/end time, so T_ira frames and T_cam frames 
      cover the same total duration.
    """
    user_ids = batch['user_id']    # list of length B
    activities = batch['activity'] # list of length B
    modality_data = batch['modality_data']

    # 1) Retrieve IRA data
    if 'IRA' not in modality_data or modality_data['IRA'] is None:
        print("No IRA data found in this batch.")
        return
    
    ira_data = modality_data['IRA']   # shape: [B, 5, T_ira, H, W]
    if not isinstance(ira_data, torch.Tensor):
        print("Expected `ira_data` to be a torch.Tensor.")
        return

    B, ira_nodes, T_ira, ira_H, ira_W = ira_data.shape
    if ira_nodes < 5:
        print(f"Warning: This function expects 5 IRA nodes, but found {ira_nodes}.")

    # 2) Retrieve Depth camera RGB frames
    depthcam_data = modality_data.get('depthCamera', None)
    if depthcam_data is None or 'rgb_frames' not in depthcam_data:
        print("No `depthCamera` / `rgb_frames` found in this batch.")
        return

    rgb_frames = depthcam_data['rgb_frames']  # shape: [B, N, T_cam, H, W, 3]
    if not isinstance(rgb_frames, torch.Tensor):
        print("Expected `rgb_frames` to be a torch.Tensor.")
        return

    B2, N_rgb, T_cam, rgb_H, rgb_W, rgb_C = rgb_frames.shape
    if B != B2:
        print("Mismatch in batch dimension for IRA vs DepthCamera RGB.")
        return

    # We'll produce exactly T_cam frames in the final video (30 FPS).
    # For each camera frame i, we figure out which IRA frame to pick.

    for b_idx in range(B):
        user_id = user_ids[b_idx]
        activity = activities[b_idx]
        act_str = activity[0] if isinstance(activity, list) else activity

        frames_list = []
        out_name = f"ira_rgb_{act_str}_user_{user_id}.mp4"
        out_path = os.path.join(output_dir, out_name)

        for i in range(T_cam):
            fig, axes = plt.subplots(
                2, 3, figsize=(12, 8),
                gridspec_kw={'wspace': 0.05, 'hspace': 0.2}
            )
            fig.suptitle(
                f"User {user_id}, Activity: {act_str}, Frame {i+1}/{T_cam}",
                fontsize=16
            )

            # fraction in [0..1], but we only use it to find floor() index
            # for the IRA frames. That is "step" interpolation.
            alpha_time = i / (T_cam - 1) if T_cam > 1 else 0.0
            ira_float_idx = alpha_time * (T_ira - 1)

            node_positions = [
                (0, 0), (0, 1), (0, 2),  # row=0, col=0..2
                (1, 0), (1, 1)           # row=1, col=0..1
            ]
            for node_idx in range(min(ira_nodes, 5)):
                r, c = node_positions[node_idx]
                ax = axes[r, c]

                # -- get the IRA frame using step approach
                node_tensor = ira_data[b_idx, node_idx]  # shape [T_ira, H, W]
                frame_ira = step_ira_frame(node_tensor, ira_float_idx)

                # optional zero-value fill:
                frame_ira = SubpageInterpolating(frame_ira)

                # same 0..37°C normalization from your original code
                min_val = np.min(frame_ira)
                denom = 37.0 - min_val
                if denom < 1e-6:
                    denom = 1e-6
                norm = (frame_ira - min_val) / denom
                norm = np.clip(norm, 0.0, 1.0) * 255.0

                # upsample for better visibility
                up = 20
                large_ira = np.repeat(norm, up, axis=0)
                large_ira = np.repeat(large_ira, up, axis=1)

                # colormap (JET)
                ira_colormap = cv2.applyColorMap(
                    large_ira.astype(np.uint8), cv2.COLORMAP_JET
                )

                ax.imshow(ira_colormap)
                ax.set_title(f"IRA Node {node_idx+1}")
                ax.axis('off')

            # -- Camera (node=0, frame i) --
            ax_rgb = axes[1, 2]
            if N_rgb > 0:
                rgb_frame = rgb_frames[b_idx, 0, i].cpu().numpy()
                rgb_uint8 = rgb_frame.astype(np.uint8)
                ax_rgb.imshow(rgb_uint8)
                ax_rgb.set_title("RGB Node 1")
                ax_rgb.axis('off')
            else:
                ax_rgb.axis('off')

            fig.canvas.draw()
            w_fig, h_fig = fig.canvas.get_width_height()
            img_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img_argb = img_argb.reshape((h_fig, w_fig, 4))

            # ARGB -> RGBA
            img_rgba = img_argb[..., [1,2,3,0]]
            frames_list.append(img_rgba.copy())
            plt.close()

        # Save frames_list to MP4 at 30 FPS
        if len(frames_list) == 0:
            print(f"No frames produced for user={user_id}, activity={act_str}.")
            continue

        height, width = frames_list[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_path, fourcc, fps_out, (width, height))

        for frame_img in frames_list:
            frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGBA2BGR)
            out_writer.write(frame_bgr)
        out_writer.release()

        print(f"[visualize_ira_and_rgb_mosaic_batch] Saved => {out_path}")

@visualizer_decorator
def visualize_seekthermal_and_rgb_mosaic_batch_discard_excess(
    batch,
    output_dir: str = 'seek_rgb_mosaic_discard',
    fps_out: float = 8.8
):
    """
    Discard any extra camera frames if T_cam > T_st, so that
    T_min = min(T_st, T_cam). Then for each i in [0..T_min-1], 
    we directly index SeekThermal[i] and camera[i]. 
    This yields a perfectly aligned 1-to-1 approach 
    (but ignores the subtle difference in sampling times).
    
    Layout per frame:
       [SeekThermal Node 0] | [SeekThermal Node 1] | [RGB]
    Writes T_min frames total at fps_out to .mp4.
    """
    user_ids = batch['user_id']    # list of length B
    activities = batch['activity'] # list of length B
    modality_data = batch['modality_data']

    # 1) Pull out SeekThermal data
    seek_data = modality_data.get('seekThermal', None)
    if seek_data is None:
        print("No SeekThermal data found in this batch.")
        return
    
    if not isinstance(seek_data, torch.Tensor):
        print("Expected `seekThermal` to be a torch.Tensor.")
        return

    B, ST_nodes, T_st, st_H, st_W = seek_data.shape
    if ST_nodes < 1:
        print(f"Warning: expecting >=1 SeekThermal node, found {ST_nodes}.")

    # 2) Pull out RGB data from the depth camera
    depthcam_data = modality_data.get('depthCamera', {})
    if 'rgb_frames' not in depthcam_data or depthcam_data['rgb_frames'] is None:
        print("No `depthCamera.rgb_frames` found in this batch.")
        return

    rgb_frames = depthcam_data['rgb_frames']  # shape: [B, N_rgb, T_cam, H, W, 3]
    if not isinstance(rgb_frames, torch.Tensor):
        print("Expected `rgb_frames` to be a torch.Tensor.")
        return

    B2, N_rgb, T_cam, rgb_H, rgb_W, rgb_C = rgb_frames.shape
    if B != B2:
        print("Mismatch in batch dimension: SeekThermal vs DepthCamera.")
        return

    # 3) Generate a video for each sample in the batch
    for b_idx in range(B):
        user_id = user_ids[b_idx]
        activity = activities[b_idx]
        # If activity is a list, grab the first
        act_str = activity[0] if isinstance(activity, list) else activity

        # We'll keep only min(T_st, T_cam) frames
        T_min = min(T_st, T_cam)

        frames_list = []
        out_name = f"seek_rgb_{act_str}_user_{user_id}_discarded.mp4"
        out_path = os.path.join(output_dir, out_name)

        print(f"[INFO] T_st={T_st}, T_cam={T_cam}, using T_min={T_min} frames...")

        for i in range(T_min):
            # Create a 1 x 3 mosaic (up to 2 SeekThermal nodes + 1 RGB)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle(
                f"User {user_id}, Activity: {act_str}, Frame {i+1}/{T_min}",
                fontsize=15
            )

            # For each of the (up to) 2 SeekThermal nodes
            for node_idx in range(min(ST_nodes, 2)):
                ax = axes[node_idx]
                # Directly index the i-th SeekThermal frame
                node_tensor = seek_data[b_idx, node_idx]  # shape [T_st, st_H, st_W]
                frame_st = node_tensor[i].cpu().numpy()

                # Basic min–max normalization
                st_min, st_max = frame_st.min(), frame_st.max()
                denom = (st_max - st_min) if (st_max > st_min) else 1e-6
                norm = (frame_st - st_min) / denom
                norm = np.clip(norm, 0.0, 1.0) * 255.0

                # Convert to color map
                st_colormap = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)

                ax.imshow(st_colormap)
                ax.set_title(f"SeekThermal Node {node_idx+1}")
                ax.axis('off')

            # Third column: take the i-th camera frame
            ax_rgb = axes[2]
            if N_rgb > 0:
                rgb_frame = rgb_frames[b_idx, 0, i].cpu().numpy()  # [H, W, 3]
                rgb_uint8 = rgb_frame.astype(np.uint8)
                ax_rgb.imshow(rgb_uint8)
                ax_rgb.set_title("RGB Node 1")
                ax_rgb.axis('off')
            else:
                ax_rgb.axis('off')

            # Convert the Matplotlib figure into an RGBA image
            fig.canvas.draw()
            w_fig, h_fig = fig.canvas.get_width_height()
            img_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img_argb = img_argb.reshape((h_fig, w_fig, 4))

            # ARGB -> RGBA
            img_rgba = img_argb[..., [1,2,3,0]]
            frames_list.append(img_rgba.copy())
            plt.close()

        # 4) Save T_min frames to a .mp4 at fps_out
        if len(frames_list) == 0:
            print(f"No frames produced for user={user_id}, activity={act_str}.")
            continue

        height, width = frames_list[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_path, fourcc, fps_out, (width, height))

        for frame_img in frames_list:
            frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGBA2BGR)
            out_writer.write(frame_bgr)

        out_writer.release()
        print(f"[visualize_seekthermal_and_rgb_mosaic_batch_discard_excess] Saved => {out_path}")

@visualizer_decorator
def dump_seekthermal_frames_as_png(batch, output_dir="validation_seekthermal", fps_out=None):
    """
    Save each SeekThermal frame as a separate .png (with a color map)
    so you can manually inspect how the frames progress over time.

    Folder structure:
      validation_seekthermal/
        user_{UID}_act_{ACTIVITY}/
          node_1/
            frame_0000.png
            frame_0001.png
            ...
          node_2/
            frame_0000.png
            ...
    """
    if "seekThermal" not in batch["modality_data"]:
        print("No SeekThermal data found in this batch.")
        return

    seek_data = batch["modality_data"]["seekThermal"]  # shape [B, ST_nodes, T_st, H, W]
    if not isinstance(seek_data, torch.Tensor):
        print("seekThermal is not a torch.Tensor. Unexpected data type.")
        return

    B, ST_nodes, T_st, st_H, st_W = seek_data.shape
    user_ids = batch['user_id']
    activities = batch['activity']

    # Loop over each sample in the batch
    for b_idx in range(B):
        user_id = user_ids[b_idx]
        activity = activities[b_idx]
        # If activity is a list, pick the first
        act_str = activity[0] if isinstance(activity, list) else activity

        # Create a folder for this user/activity
        sample_dir = os.path.join(output_dir, f"user_{user_id}_act_{act_str}")
        os.makedirs(sample_dir, exist_ok=True)

        # For each SeekThermal node
        for node_idx in range(ST_nodes):
            node_tensor = seek_data[b_idx, node_idx]  # shape [T_st, st_H, st_W]
            node_dir = os.path.join(sample_dir, f"node_{node_idx+1}")
            os.makedirs(node_dir, exist_ok=True)

            # Save each frame as a PNG
            for i in range(T_st):
                frame_st = node_tensor[i].cpu().numpy()  # shape [st_H, st_W]

                # Basic min-max normalization -> [0..255]
                st_min, st_max = frame_st.min(), frame_st.max()
                denom = (st_max - st_min) if (st_max > st_min) else 1e-6
                norm = (frame_st - st_min) / denom
                norm_8u = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)

                # Apply a color map (JET)
                st_colormap = cv2.applyColorMap(norm_8u, cv2.COLORMAP_JET)
                # Convert OpenCV's BGR -> RGB so it looks correct in typical viewers
                st_rgb = cv2.cvtColor(st_colormap, cv2.COLOR_BGR2RGB)

                # Save as "frame_0000.png", "frame_0001.png", etc.
                out_fname = f"frame_{i:04d}.png"
                out_path = os.path.join(node_dir, out_fname)
                Image.fromarray(st_rgb).save(out_path)

    print(f"[dump_seekthermal_frames_as_png] Done! Frames saved under '{output_dir}' folder.")

@visualizer_decorator
def visualize_3_depth_3_rgb_mosaic_batch_discard_excess(
    batch,
    output_dir='depth_rgb_3nodes_mosaic',
    fps_out=10.0
):
    """
    For DepthCamera => shape [B, 3, T_depth, H, W]
    and the matching RGB => shape [B, 3, T_cam, H, W, 3],
    discard any extra frames so T_min = min(T_depth, T_cam).

    Then produce a 2x3 mosaic for each i in [0..T_min-1]:
       (top row)    Depth Node 1, Depth Node 3, Depth Node 5
       (bottom row) RGB   Node 1, RGB   Node 3, RGB   Node 5

    Writes out T_min frames total at fps_out to an MP4 for each sample in the batch.
    """
    user_ids = batch['user_id']       # list of length B
    activities = batch['activity']    # list of length B
    modality_data = batch['modality_data']

    # Check for depthCamera data
    depthcam_data = modality_data.get('depthCamera', None)
    if depthcam_data is None:
        print("No `depthCamera` data found in this batch.")
        return

    depth_images = depthcam_data.get('depth_images', None)  # [B, 3, T_depth, H, W]
    rgb_frames  = depthcam_data.get('rgb_frames', None)     # [B, 3, T_cam, H, W, 3]

    if depth_images is None or rgb_frames is None:
        print("No `depth_images` or `rgb_frames` found in depthCamera.")
        return

    if not isinstance(depth_images, torch.Tensor):
        print("Expected depth_images to be a torch.Tensor.")
        return
    if not isinstance(rgb_frames, torch.Tensor):
        print("Expected rgb_frames to be a torch.Tensor.")
        return

    B, N_depth, T_depth, dH, dW = depth_images.shape
    B2, N_rgb, T_cam, rH, rW, rC = rgb_frames.shape

    if B != B2:
        print("Mismatch in batch dimension for depth vs. RGB.")
        return

    # We assume exactly 3 nodes for depth and 3 for RGB
    if N_depth != 3 or N_rgb != 3:
        print(f"Warning: This function expects exactly 3 depth nodes & 3 RGB nodes, but got N_depth={N_depth}, N_rgb={N_rgb}.")

    # Hardcode node labels in the order [1, 3, 5]
    node_labels = [1, 3, 5]

    for b_idx in range(B):
        user_id = user_ids[b_idx]
        activity = activities[b_idx]
        # If activity is a list, pick first
        act_str = activity[0] if isinstance(activity, list) else activity

        # Discard extras
        T_min = min(T_depth, T_cam)
        frames_list = []
        out_name = f"depth_rgb_3nodes_{act_str}_user_{user_id}.mp4"
        out_path = os.path.join(output_dir, out_name)

        print(f"[info] T_depth={T_depth}, T_cam={T_cam}, using T_min={T_min} frames.")

        for i in range(T_min):
            # Make a 2x3 figure
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
            fig.suptitle(
                f"User {user_id}, Activity={act_str}, Frame {i+1}/{T_min}",
                fontsize=14
            )

            # --- Top row: Depth nodes 1,3,5 ---
            for col in range(3):
                ax = axes[0, col]
                # depth_images[b_idx, col, i, ...]
                depth_frame = depth_images[b_idx, col, i].cpu().numpy()  # shape [H, W]

                # Basic min–max norm for visualization
                dmin, dmax = depth_frame.min(), depth_frame.max()
                denom = (dmax - dmin) if (dmax > dmin) else 1e-6
                norm = (depth_frame - dmin) / denom
                ax.imshow(norm, cmap='viridis')
                ax.set_title(f"Depth Node {node_labels[col]}")
                ax.axis('off')

            # --- Bottom row: RGB nodes 1,3,5 ---
            for col in range(3):
                ax = axes[1, col]
                rgb_frame = rgb_frames[b_idx, col, i].cpu().numpy()  # shape [H, W, 3]
                rgb_uint8 = rgb_frame.astype(np.uint8)
                ax.imshow(rgb_uint8)
                ax.set_title(f"RGB Node {node_labels[col]}")
                ax.axis('off')

            fig.canvas.draw()
            w_fig, h_fig = fig.canvas.get_width_height()
            img_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img_argb = img_argb.reshape((h_fig, w_fig, 4))

            # ARGB -> RGBA
            img_rgba = img_argb[..., [1,2,3,0]]
            frames_list.append(img_rgba.copy())
            plt.close()

        if not frames_list:
            print(f"[info] No frames produced for user={user_id}, activity={act_str}.")
            continue

        # Write frames to MP4
        height, width = frames_list[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_vid = cv2.VideoWriter(out_path, fourcc, fps_out, (width, height))

        for frame_img in frames_list:
            frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGBA2BGR)
            out_vid.write(frame_bgr)

        out_vid.release()
        print(f"[visualize_3_depth_3_rgb_mosaic_batch_discard_excess] Saved => {out_path}")

@visualizer_decorator
def visualize_uwb_and_rgb_in_same_row_with_box(
    batch,
    output_dir="uwb_rgb_same_row_dB",
    fps_out=10.0,
    save_pdf_frames=True,
    axis_label_fontsize=14,  # new argument
    tick_label_fontsize=12   # new argument
):
    """
    Modified to:
      1) No titles (no suptitle, no subplot titles).
      2) axis_label_fontsize=14, tick_label_fontsize=12.
      3) Transparent PDF background, size=12x8 inches.

    Otherwise the same logic applies:
      - We'll produce an MP4 where each row is [UWB dB | RGB].
      - Then optionally save a few frames as PDF for UWB node=1 only.
    """
    user_ids   = batch["user_id"]
    activities = batch["activity"]
    modality   = batch["modality_data"]

    # 1) UWB data
    uwb_data = modality.get("uwb", None)
    if uwb_data is None or not isinstance(uwb_data, torch.Tensor):
        print("No 'uwb' data found or not a torch.Tensor.")
        return

    B, N_uwb, T_uwb, cir_len = uwb_data.shape

    # 2) Camera data
    depthcam_data = modality.get("depthCamera", {})
    rgb_data = depthcam_data.get("rgb_frames", None)
    if rgb_data is None or not isinstance(rgb_data, torch.Tensor):
        print("No 'rgb_frames' in 'depthCamera'.")
        return

    B2, N_rgb, T_cam, camH, camW, camC = rgb_data.shape
    if B != B2:
        print("Mismatch in batch dimension for UWB vs Camera.")
        return
    if N_uwb != N_rgb:
        print(f"Warning: N_uwb={N_uwb}, N_rgb={N_rgb}. Matching up to smaller count.")
        N_uwb = min(N_uwb, N_rgb)

    for b_idx in range(B):
        user_id  = user_ids[b_idx]
        activity = activities[b_idx]
        act_str  = activity[0] if isinstance(activity, list) else activity

        out_mp4_name = f"uwb_dB_{act_str}_user_{user_id}.mp4"
        out_mp4_path = os.path.join(output_dir, out_mp4_name)
        
        frames_list = []
        
        # Produce T_uwb frames total, stepping camera in sync
        for i_uwb in range(T_uwb):
            alpha_time = i_uwb / (T_uwb - 1) if (T_uwb > 1) else 0.0
            cam_float_idx = alpha_time * (T_cam - 1) if (T_cam > 1) else 0.0
            cam_idx = int(floor(cam_float_idx))
            cam_idx = max(0, min(cam_idx, T_cam - 1))

            # 1 row => 2*N_uwb columns
            fig, axes = plt.subplots(
                1, 2*N_uwb,
                figsize=(5 * 2 * N_uwb, 4)
            )

            # If N_uwb=1 => axes might not be an array of arrays
            if 2*N_uwb == 1:
                axes = [axes]

            for node_i in range(N_uwb):
                # UWB subplot
                ax_uwb = axes[2 * node_i]
                cir_vector = uwb_data[b_idx, node_i, i_uwb].cpu().numpy()  # shape [cir_len]

                # Convert to dB
                cir_abs = np.abs(cir_vector) + 1e-6
                cir_dB  = 20.0 * np.log10(cir_abs)

                x_vals = np.arange(cir_len)
                ax_uwb.plot(
                    x_vals, cir_dB,
                    marker='o', markersize=3,
                    linewidth=1, color='blue'
                )

                ax_uwb.set_xlabel("Sample Index", fontsize=axis_label_fontsize)
                ax_uwb.set_ylabel("CIR (dB)",     fontsize=axis_label_fontsize)

                # Tick label size
                ax_uwb.tick_params(axis='both', labelsize=tick_label_fontsize)

                # Make bounding box visible
                for spine in ax_uwb.spines.values():
                    spine.set_visible(True)

                # RGB subplot
                ax_cam = axes[2 * node_i + 1]
                if cam_idx < rgb_data.shape[2]:
                    rgb_frame = rgb_data[b_idx, node_i, cam_idx].cpu().numpy().astype(np.uint8)
                    ax_cam.imshow(rgb_frame)
                    # *** no subplot title => removed
                    ax_cam.axis('off')
                else:
                    ax_cam.axis('off')

            # Convert figure => RGBA
            fig.canvas.draw()
            w_fig, h_fig = fig.canvas.get_width_height()
            img_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img_argb = img_argb.reshape((h_fig, w_fig, 4))
            # ARGB => RGBA
            img_rgba = img_argb[..., [1,2,3,0]]
            frames_list.append(img_rgba.copy())
            plt.close(fig)

        # Write frames to mp4
        if frames_list:
            height, width = frames_list[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_mp4_path, fourcc, fps_out, (width, height))
            for frame_img in frames_list:
                frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGBA2BGR)
                writer.write(frame_bgr)
            writer.release()
            print(f"[visualize_uwb_and_rgb_in_same_row_with_box_dB] => {out_mp4_path}")
        else:
            print(f"No frames for user={user_id}, activity={act_str}.")
            continue

        # (C) Optionally save a few UWB-only frames as PDFs
        if save_pdf_frames and T_uwb > 0:
            pdf_indices = [0, (T_uwb // 2), (T_uwb - 1)] if T_uwb > 2 else [0]
            for idx in set(pdf_indices):
                if idx < 0 or idx >= T_uwb:
                    continue

                cir_vector = uwb_data[b_idx, 0, idx].cpu().numpy()
                cir_abs = np.abs(cir_vector) + 1e-6
                cir_dB  = 20.0 * np.log10(cir_abs)
                x_vals = np.arange(cir_len)

                # Create the figure for PDF
                fig_pdf, ax_pdf = plt.subplots()
                # Force 12×8
                fig_pdf.set_size_inches(12, 8)

                # *** no suptitle, no title *** 

                ax_pdf.plot(x_vals, cir_dB, color='#4682B4')
                ax_pdf.set_xlabel("Range Bin", fontsize=axis_label_fontsize)
                ax_pdf.set_ylabel("CIR (dB)", fontsize=axis_label_fontsize)
                ax_pdf.tick_params(axis='both', labelsize=tick_label_fontsize)

                # Example axis limits
                ax_pdf.set_xlim([0, cir_len - 1])
                ax_pdf.set_ylim([-120, -20])

                # Transparent background
                fig_pdf.patch.set_facecolor("none")
                ax_pdf.set_facecolor("none")

                pdf_name = f"uwb_node1_frame_{idx:03d}_user_{user_id}.pdf"
                pdf_path = os.path.join(output_dir, pdf_name)
                plt.tight_layout()
                fig_pdf.savefig(pdf_path, transparent=True, bbox_inches='tight')
                plt.close(fig_pdf)

                print(f"Saved PDF frame => {pdf_path}")

@visualizer_decorator
def visualize_ira_and_rgb_mosaic_batch_downsample_cam(
    batch,
    output_dir: str = 'ira_rgb_mosaic',
    fps_out: float = 30.0
):
    """
    Similar to the original IRA + RGB mosaic (2x3 layout),
    but we produce T_ira frames total, downsampling the 
    higher-fps camera to sync with IRA's timeline.

    Layout (2 rows, 3 columns):
      row=0, col=0..2 => IRA nodes 1,2,3
      row=1, col=0..1 => IRA nodes 4,5
      row=1, col=2 => downsampled camera

    Steps:
      1) We loop i in [0..T_ira-1].
      2) For each i, pick IRA frame i (no interpolation needed).
      3) Also compute cam_float_idx = i/(T_ira-1)*(T_cam-1),
         then floor => step-downsample camera frame.
      4) Build a mosaic with 5 IRA subplots and 1 camera subplot (2x3).
      5) Save T_ira frames to an MP4 at fps_out.
    """
    user_ids = batch['user_id']       # list of length B
    activities = batch['activity']    # list of length B
    modality_data = batch['modality_data']

    # 1) Retrieve IRA data
    if 'IRA' not in modality_data or modality_data['IRA'] is None:
        print("No IRA data found in this batch.")
        return
    
    ira_data = modality_data['IRA']  # [B, 5, T_ira, H, W]
    if not isinstance(ira_data, torch.Tensor):
        print("Expected `ira_data` to be a torch.Tensor.")
        return

    B, ira_nodes, T_ira, ira_H, ira_W = ira_data.shape
    if ira_nodes < 5:
        print(f"Warning: This function expects 5 IRA nodes, found {ira_nodes}.")

    # 2) Retrieve camera frames
    depthcam_data = modality_data.get('depthCamera', None)
    if not depthcam_data or 'rgb_frames' not in depthcam_data:
        print("No `depthCamera.rgb_frames` found in this batch.")
        return

    rgb_frames = depthcam_data['rgb_frames']  # [B, N_rgb, T_cam, H, W, 3]
    if not isinstance(rgb_frames, torch.Tensor):
        print("Expected `rgb_frames` to be a torch.Tensor.")
        return

    B2, N_rgb, T_cam, rgb_H, rgb_W, rgb_C = rgb_frames.shape
    if B != B2:
        print("Mismatch in batch dimension for IRA vs camera RGB.")
        return

    # SubpageInterpolating is the same
    def SubpageInterpolating(subpage: np.ndarray) -> np.ndarray:
        mat = subpage.copy()
        rows, cols = mat.shape
        for rr in range(rows):
            for cc in range(cols):
                if mat[rr, cc] > 0.0:
                    continue
                neighbors = []
                if rr-1 >= 0:
                    neighbors.append(mat[rr-1, cc])
                if rr+1 < rows:
                    neighbors.append(mat[rr+1, cc])
                if cc-1 >= 0:
                    neighbors.append(mat[rr, cc-1])
                if cc+1 < cols:
                    neighbors.append(mat[rr, cc+1])
                if len(neighbors) > 0:
                    mat[rr, cc] = sum(neighbors)/len(neighbors)
                else:
                    mat[rr, cc] = 0.0
        return mat

    # We'll produce T_ira frames total
    for b_idx in range(B):
        user_id = user_ids[b_idx]
        activity = activities[b_idx]
        act_str = activity[0] if isinstance(activity, list) else activity

        frames_list = []
        out_name = f"ira_rgb_{act_str}_user_{user_id}.mp4"
        out_path = os.path.join(output_dir, out_name)

        # For each IRA index i => pick that IRA frame, 
        # and downsample the camera frames
        for i in range(T_ira):
            # fraction in [0..1]
            alpha_time = i / (T_ira - 1) if (T_ira > 1) else 0.0
            # step interpolation => floor
            cam_float_idx = alpha_time * (T_cam - 1)
            cam_idx = int(floor(cam_float_idx)) if T_cam > 1 else 0

            fig, axes = plt.subplots(
                2, 3, figsize=(12, 8),
                gridspec_kw={'wspace': 0.05, 'hspace': 0.2}
            )
            fig.suptitle(
                f"User {user_id}, Activity: {act_str}, IRA Frame {i+1}/{T_ira}",
                fontsize=16
            )

            # positions
            node_positions = [
                (0,0), (0,1), (0,2),
                (1,0), (1,1)
            ]
            # Plot the 5 IRA nodes
            for node_idx in range(min(ira_nodes, 5)):
                r, c = node_positions[node_idx]
                ax_ira = axes[r, c]

                # IRA frame i, node_idx
                frame_ira = ira_data[b_idx, node_idx, i].cpu().numpy()  # shape [H,W]

                # Subpage interpolate
                frame_ira = SubpageInterpolating(frame_ira)

                # 0..37°C normalization
                min_val = np.min(frame_ira)
                denom = 37.0 - min_val
                if denom < 1e-6:
                    denom = 1e-6
                norm = (frame_ira - min_val)/denom
                norm = np.clip(norm, 0.0, 1.0)*255.0

                # upsample for visibility
                up = 20
                large_ira = np.repeat(norm, up, axis=0)
                large_ira = np.repeat(large_ira, up, axis=1)

                ira_colormap = cv2.applyColorMap(
                    large_ira.astype(np.uint8), cv2.COLORMAP_JET
                )

                ax_ira.imshow(ira_colormap)
                ax_ira.set_title(f"IRA Node {node_idx+1}")
                ax_ira.axis('off')

            # bottom-right => camera
            ax_rgb = axes[1, 2]
            if N_rgb > 0 and cam_idx < T_cam:
                rgb_frame = rgb_frames[b_idx, 0, cam_idx].cpu().numpy() # shape [H,W,3]
                rgb_uint8 = rgb_frame.astype(np.uint8)
                ax_rgb.imshow(rgb_uint8)
                ax_rgb.set_title("RGB Node 1")
                ax_rgb.axis('off')
            else:
                ax_rgb.axis('off')

            fig.canvas.draw()
            w_fig, h_fig = fig.canvas.get_width_height()
            img_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img_argb = img_argb.reshape((h_fig, w_fig, 4))

            # ARGB -> RGBA
            img_rgba = img_argb[..., [1,2,3,0]]
            frames_list.append(img_rgba.copy())
            plt.close()

        # Write frames_list to MP4
        if not frames_list:
            print(f"No IRA frames for user={user_id}, activity={act_str}.")
            continue

        height, width = frames_list[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_path, fourcc, fps_out, (width, height))

        for frame_img in frames_list:
            frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGBA2BGR)
            out_writer.write(frame_bgr)
        out_writer.release()

        print(f"[visualize_ira_and_rgb_mosaic_batch_downsample_cam] Saved => {out_path}")

@visualizer_decorator
def visualize_4wifi_time_subcarrier_with_camera(
    batch,
    output_dir="wifi_4nodes_plus_rgb_no_trim_concatTX_80Hz",
    fps_out=10.0,
    BW="40MHz",
    save_pdf=True,
    wifi_sampling_rate=90.0,  # frames per second
    axis_label_fontsize=14,   # new argument
    tick_label_fontsize=12    # new argument
):
    """
    Modified to:
      1) No titles (no suptitle or subplot titles).
      2) axis_label_fontsize=14, tick_label_fontsize=12 for axes/ticks.
      3) produce a single new PDF for Node 2 (index=1) only, with tight_layout().
      4) transparent background for that Node2 PDF.

    Otherwise, it still:
      - Processes WiFi data => amplitude => (2*N_subc, T).
      - Builds an MP4 with up to 4 WiFi nodes + camera on the right.
      - Node indexing is [0..3].
    """
    user_ids   = batch['user_id']
    activities = batch['activity']
    modality_data = batch['modality_data']

    # 1) Pull out WiFi data
    wifi_data = modality_data.get("wifi", None)
    if wifi_data is None or not isinstance(wifi_data, torch.Tensor):
        print("No 'wifi' data found in this batch or not a torch.Tensor.")
        return

    B, wifi_nodes, T_wifi, txrx_dim, subc_dim = wifi_data.shape
    if wifi_nodes < 1:
        print("No WiFi nodes. Exiting.")
        return
    if wifi_nodes > 4:
        print(f"Warning: found {wifi_nodes} WiFi nodes; only showing up to 4.")
        wifi_nodes = 4

    # 2) Pull out Camera data
    depthcam_data = modality_data.get("depthCamera", {})
    if "rgb_frames" not in depthcam_data or depthcam_data["rgb_frames"] is None:
        print("No 'rgb_frames' found in 'depthCamera'.")
        return

    rgb_frames = depthcam_data["rgb_frames"]  # shape => [B, N_rgb, T_cam, H, W, 3]
    if not isinstance(rgb_frames, torch.Tensor):
        print("Expected `rgb_frames` to be a torch.Tensor.")
        return

    B2, N_rgb, T_cam, camH, camW, camC = rgb_frames.shape
    if B != B2:
        print("Mismatch in batch dimension for WiFi vs. camera.")
        return

    # Optional subcarrier deletion
    if BW == "160MHz":
        del_subc = [
            19,47,83,111,130,158,194,222,275,303,339,367,386,414,450,478,
            242,243,244,245,246,247,248,249,250,251,252,253,254,255
        ]
    elif BW == "40MHz":
        del_subc = [5,33,47,66,80,108]
    elif BW == "20MHz":
        del_subc = [7,21,34,48]
    else:
        del_subc = []

    def process_wifi_node(wifi_node_tensor: torch.Tensor) -> np.ndarray:
        """
        For each time t => shape [2, subc_dim_original],
         1) Possibly remove subcarriers
         2) Convert real+imag => amplitude => shape (2, new_subc)
         3) Concatenate => shape (2*new_subc)
        => final shape => [T_w_node, 2*new_subc], then transpose => (2*new_subc, T_w_node).
        """
        wifi_np = wifi_node_tensor.cpu().numpy()  # => shape [T_wifi, 2, subc_dim]
        orig_subc_dim = wifi_np.shape[-1]

        # Subcarrier deletion
        if len(del_subc) > 0 and max(del_subc) < orig_subc_dim:
            keep_mask = np.ones(orig_subc_dim, dtype=bool)
            for sc in del_subc:
                keep_mask[sc] = False
            wifi_np = wifi_np[..., keep_mask]  # => shape (T_wifi, 2, new_subc)

        T_w_node = wifi_np.shape[0]
        new_subc = wifi_np.shape[-1]  # after deletion

        # amplitude => shape [2, new_subc], then concat => [2*new_subc]
        amp_rows = []
        for t in range(T_w_node):
            real_part = np.real(wifi_np[t])
            imag_part = np.imag(wifi_np[t])
            amplitude_2xN = np.sqrt(real_part**2 + imag_part**2)  # => (2, new_subc)
            row_concat = np.concatenate([amplitude_2xN[0], amplitude_2xN[1]], axis=0)
            amp_rows.append(row_concat)

        amplitude_2d = np.array(amp_rows, dtype=np.float32)  # => [T_w_node, 2*new_subc]
        return amplitude_2d.T  # => shape => (2*new_subc, T_w_node)

    for b_idx in range(B):
        user_id = user_ids[b_idx]
        activity = activities[b_idx]
        act_str  = activity[0] if isinstance(activity, list) else activity

        wifi_sample = wifi_data[b_idx]  # shape => [wifi_nodes, T_wifi, 2, subc_dim]
        rgb_sample = rgb_frames[b_idx, 0]  # shape => [T_cam, H, W, 3]

        # (A) Process each WiFi node => amplitude heatmap
        node_plots = []
        for node_i in range(wifi_nodes):
            wifi_node_tensor = wifi_sample[node_i]
            amp_plot = process_wifi_node(wifi_node_tensor)  # => (2*new_subc, T_w_node)
            node_plots.append(amp_plot)

        # (B) Global min..max across all nodes
        all_vals = np.concatenate([p.ravel() for p in node_plots])
        a_min, a_max = all_vals.min(), all_vals.max()
        denom = (a_max - a_min) if a_max > a_min else 1e-6

        # # (C) Build the MP4 with up to wifi_nodes + 1 subplots (the last one is camera)
        # frames_list = []
        # out_name = f"wifi_4nodes_{act_str}_user_{user_id}_80Hz.mp4"
        # out_path = os.path.join(output_dir, out_name)

        # # *** No suptitle => remove fig.suptitle(...) ***

        # for i_cam in range(T_cam):
        #     alpha = i_cam/(T_cam - 1) if T_cam > 1 else 0.0

        #     fig, axes = plt.subplots(
        #         1, wifi_nodes + 1,
        #         figsize=(5 * (wifi_nodes + 1), 4)
        #     )
        #     # *** No suptitle, no subplot titles ***

        #     for node_i in range(wifi_nodes):
        #         ax_wifi = axes[node_i]
        #         data_2d = node_plots[node_i]  # => (2*subc, t_len)
        #         subc_count, t_len = data_2d.shape

        #         # normalize
        #         normed = (data_2d - a_min)/denom

        #         # x-range => [0..(t_len-1)/wifi_sampling_rate], y => [0..subc_count]
        #         x_right = (t_len - 1)/wifi_sampling_rate
        #         extent = [0, x_right, 0, subc_count]

        #         im = ax_wifi.imshow(
        #             normed,
        #             aspect='auto',
        #             origin='lower',
        #             cmap='jet',
        #             extent=extent
        #         )

        #         # step-based index => node_idx in [0..(t_len-1)]
        #         if t_len > 1:
        #             node_float_idx = alpha*(t_len - 1)
        #             node_idx = floor(node_float_idx)
        #         else:
        #             node_idx = 0
        #         x_cur = node_idx / wifi_sampling_rate
        #         ax_wifi.axvline(x=x_cur, color='white', linestyle='--', linewidth=1.5)

        #         # *** No subplot title => removed
        #         # Axis labels with user-defined font size
        #         ax_wifi.set_xlabel("Time (second)", fontsize=axis_label_fontsize)
        #         ax_wifi.set_ylabel("Concatenated Subcarrier Index", fontsize=axis_label_fontsize)

        #         # Tick label size
        #         ax_wifi.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)

        #     # Right => camera
        #     ax_cam = axes[wifi_nodes]
        #     if i_cam < rgb_sample.shape[0]:
        #         rgb_frame = rgb_sample[i_cam].cpu().numpy().astype(np.uint8)
        #         ax_cam.imshow(rgb_frame)
        #         ax_cam.axis('off')
        #     else:
        #         ax_cam.axis('off')

        #     # Use tight_layout to reduce whitespace
        #     fig.tight_layout()

        #     fig.canvas.draw()
        #     w_fig, h_fig = fig.canvas.get_width_height()
        #     img_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        #     img_argb = img_argb.reshape((h_fig, w_fig, 4))

        #     # ARGB => RGBA
        #     img_rgba = img_argb[..., [1,2,3,0]]
        #     frames_list.append(img_rgba.copy())
        #     plt.close(fig)

        # # Save MP4
        # if not frames_list:
        #     print(f"No frames for user={user_id}, activity={act_str}.")
        #     continue

        # height, width = frames_list[0].shape[:2]
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out_writer = cv2.VideoWriter(out_path, fourcc, fps_out, (width, height))

        # for frame_img in frames_list:
        #     frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGBA2BGR)
        #     out_writer.write(frame_bgr)
        # out_writer.release()
        # print(f"[visualize_4wifi_time_subcarrier_with_camera_80Hz] => {out_path}")

        # (D) Optionally save a new PDF for Node2 (index=1) only
        if save_pdf:
            node2_idx = 1  # Node2 in 0-based indexing
            if node2_idx >= wifi_nodes:
                print(f"No Node2 found (wifi_nodes={wifi_nodes}). Skipping node2 PDF.")
                continue

            # Prepare single heatmap
            node2_data_2d = node_plots[node2_idx]  # => shape (2*subc, t_len)
            subc_count, t_len = node2_data_2d.shape
            normed = (node2_data_2d - a_min)/denom

            x_right = (t_len - 1)/wifi_sampling_rate
            extent = [0, x_right, 0, subc_count]

            pdf_name = f"wifi_node2_{act_str}_user_{user_id}_80Hz.pdf"
            pdf_path = os.path.join(output_dir, pdf_name)

            fig_node2, ax_node2 = plt.subplots()
            fig_node2.set_size_inches(12,8)

            im2 = ax_node2.imshow(
                normed,
                aspect='auto',
                origin='lower',
                cmap='jet',
                extent=extent
            )

            # *** No title ***
            ax_node2.set_xlabel("Time (second)", fontsize=axis_label_fontsize)
            ax_node2.set_ylabel("Concatenated Subcarrier Index", fontsize=axis_label_fontsize)
            ax_node2.tick_params(axis='both', labelsize=tick_label_fontsize)

            # If you like, draw a final vertical line at last index or something. 
            # But we'll skip it here.

            plt.tight_layout()

            # Transparent background
            fig_node2.patch.set_facecolor("none")
            ax_node2.set_facecolor("none")

            fig_node2.savefig(pdf_path, transparent=True, bbox_inches="tight")
            plt.close(fig_node2)

            print(f"[visualize_4wifi_time_subcarrier_with_camera_80Hz] => Node2 PDF saved {pdf_path}")

@visualizer_decorator
def visualize_mocap_and_rgb_mosaic_batch_downsample_mocap(
    batch,
    output_dir='mocap_rgb_mosaic_default3d',
    fps_out=10.0,
    axis_label_fontsize=14,
    tick_label_fontsize=12
):
    """
    Modified to:
      1) Only use the first mocap node.
      2) No titles at all (no suptitle, no subplot titles).
      3) Axis label font size=14, tick label font size=12.
      4) Produce 20 PDF snapshots of the skeleton alone.

    The MP4 still shows:
      - 3D skeleton on the left, camera on the right,
      - bounding box + default 3D background.
      - "step" downsampling from T_cam -> T_mocap.

    After finishing the MP4, we produce 20 PDF frames with the skeleton only.
    """
    # Example connections for the skeleton
    connections = [
        (0,1),(1,2),(2,3),
        (4,5),(5,6),(6,7),
        (8,9),(9,10),(10,11),
        (12,13),(13,14),(14,15),
        (16,17),(17,18),(18,19),
        (0,16),(0,12)
    ]
    colormap = plt.get_cmap('hsv')  # distinct colors

    user_ids   = batch['user_id']
    activities = batch['activity']
    modality_data = batch['modality_data']

    # 1) Retrieve mocap data => shape [B, N_mocap, T_mocap, n_joints, 3]
    if 'mocap' not in modality_data or modality_data['mocap'] is None:
        print("No 'mocap' data in this batch.")
        return
    mocap_data = modality_data['mocap']
    B, N_mocap, T_mocap, n_joints, _ = mocap_data.shape
    if N_mocap < 1:
        print("Warning: no mocap node found.")
        return

    # 2) Retrieve camera => shape [B, N_rgb, T_cam, H, W, 3]
    depthcam_data = modality_data.get('depthCamera', {})
    if 'rgb_frames' not in depthcam_data or depthcam_data['rgb_frames'] is None:
        print("No rgb_frames found in depthCamera.")
        return
    rgb_frames = depthcam_data['rgb_frames']
    if not isinstance(rgb_frames, torch.Tensor):
        print("Expected rgb_frames as torch.Tensor.")
        return
    B2, N_rgb, T_cam, camH, camW, camC = rgb_frames.shape
    if B != B2:
        print("Mismatch in batch dimension for mocap vs camera.")
        return

    def get_mocap_index(i_cam, T_cam_, T_mocap_):
        """Step-based index from camera to mocap."""
        if T_cam_ <= 1:
            return 0
        alpha = i_cam / (T_cam_ - 1)
        if T_mocap_ <= 1:
            return 0
        return min(int(floor(alpha * (T_mocap_ - 1))), T_mocap_ - 1)

    for b_idx in range(B):
        user_id  = user_ids[b_idx]
        activity = activities[b_idx]
        act_str  = activity[0] if isinstance(activity, list) else activity

        # We'll just use the first mocap node => shape [T_mocap, n_joints, 3]
        sample_mocap = mocap_data[b_idx, 0].cpu().numpy()
        sample_rgb   = rgb_frames[b_idx, 0]  # shape [T_cam, H, W, 3]

        # Rotate +90° around Z and scale Z by 3
        x_old = sample_mocap[..., 0]
        y_old = sample_mocap[..., 1]
        z_old = sample_mocap[..., 2]
        x_new = -y_old
        y_new =  x_old
        z_new =  z_old * 10.0
        sample_mocap[..., 0] = x_new
        sample_mocap[..., 1] = y_new
        sample_mocap[..., 2] = z_new

        # bounding box
        all_xyz = sample_mocap.reshape(-1, 3)
        min_xyz = all_xyz.min(axis=0) - 50
        max_xyz = all_xyz.max(axis=0) + 50

        # # =============== Build MP4 ===============
        # frames_list = []
        # out_name = f"mocap_rgb_{act_str}_user_{user_id}.mp4"
        # out_path = os.path.join(output_dir, out_name)

        # T_cam_ = sample_rgb.shape[0]
        # for i_cam in range(T_cam_):
        #     mocap_idx = get_mocap_index(i_cam, T_cam_, T_mocap)
        #     skeleton_3d = sample_mocap[mocap_idx]  # shape [n_joints, 3]

        #     fig = plt.figure(figsize=(10, 5))
        #     # *** No suptitle => removed

        #     ax_mocap = fig.add_subplot(1, 2, 1, projection='3d')
        #     ax_rgb   = fig.add_subplot(1, 2, 2)

        #     # *** No subplot titles => removed ***

        #     # scatter joints
        #     ax_mocap.scatter(
        #         skeleton_3d[:, 0], skeleton_3d[:, 1], skeleton_3d[:, 2],
        #         marker='o', s=25, c='blue'
        #     )

        #     # connect lines
        #     n_conn = len(connections)
        #     for c_idx, (start_idx, end_idx) in enumerate(connections):
        #         xs = [skeleton_3d[start_idx, 0], skeleton_3d[end_idx, 0]]
        #         ys = [skeleton_3d[start_idx, 1], skeleton_3d[end_idx, 1]]
        #         zs = [skeleton_3d[start_idx, 2], skeleton_3d[end_idx, 2]]
        #         c_val = colormap(c_idx / n_conn)
        #         ax_mocap.plot(xs, ys, zs, lw=3, c=c_val)

        #     # bounding box
        #     ax_mocap.set_xlim(min_xyz[0], max_xyz[0])
        #     ax_mocap.set_ylim(min_xyz[1], max_xyz[1])
        #     ax_mocap.set_zlim(min_xyz[2], max_xyz[2])

        #     # Axis labels with font size
        #     ax_mocap.set_xlabel("X", fontsize=axis_label_fontsize, labelpad=20)
        #     ax_mocap.set_ylabel("Y", fontsize=axis_label_fontsize, labelpad=20)
        #     ax_mocap.set_zlabel("Z", fontsize=axis_label_fontsize, labelpad=20)

        #     # Tick label size
        #     ax_mocap.tick_params(axis='x', which='major', labelsize=tick_label_fontsize)
        #     ax_mocap.tick_params(axis='y', which='major', labelsize=tick_label_fontsize)
        #     ax_mocap.tick_params(axis='z', which='major', labelsize=tick_label_fontsize)

        #     # Let the default 3D background/panes remain
        #     ax_mocap.view_init(elev=10, azim=90)  # optional viewpoint

        #     # camera
        #     if i_cam < T_cam_:
        #         rgb_frame = sample_rgb[i_cam].cpu().numpy().astype(np.uint8)
        #         ax_rgb.imshow(rgb_frame)
        #         ax_rgb.axis('off')
        #     else:
        #         ax_rgb.axis('off')

        #     fig.canvas.draw()
        #     w_fig, h_fig = fig.canvas.get_width_height()
        #     img_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        #     img_argb = img_argb.reshape((h_fig, w_fig, 4))
        #     img_rgba = img_argb[..., [1,2,3,0]]  # ARGB => RGBA

        #     frames_list.append(img_rgba.copy())
        #     plt.close(fig)

        # # Write frames => MP4
        # if frames_list:
        #     height, width = frames_list[0].shape[:2]
        #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #     out_writer = cv2.VideoWriter(out_path, fourcc, fps_out, (width, height))
        #     for frame_img in frames_list:
        #         frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGBA2BGR)
        #         out_writer.write(frame_bgr)
        #     out_writer.release()
        #     print(f"[visualize_mocap_and_rgb_mosaic_batch_downsample_mocap] => {out_path}")
        # else:
        #     print(f"No frames for user={user_id}, activity={act_str}.")

        # =============== Save 20 PDF snapshots of skeleton only ===============
        if T_mocap > 0:
            pdf_indices = np.linspace(0, T_mocap - 1, 40, dtype=int)
            pdf_indices = np.unique(pdf_indices)  # remove duplicates if T_mocap<20

            

            Z_SCALE = 10.0        # You multiplied z by 10 above
            STEP_MM = 500         # Desired real tick interval 500 mm

            # ---------- Generate PDF ----------
            for idx in pdf_indices:
                fig_pdf = plt.figure()
                fig_pdf.set_size_inches(12, 8)
                ax_pdf  = fig_pdf.add_subplot(111, projection='3d')

                skeleton_3d = sample_mocap[idx]       # Coordinates already multiplied by 10

                # ====== Draw skeleton ======
                ax_pdf.scatter(skeleton_3d[:,0], skeleton_3d[:,1], skeleton_3d[:,2],
                            marker='o', s=25, c='blue')
                for c_idx, (s,e) in enumerate(connections):
                    ax_pdf.plot([skeleton_3d[s,0], skeleton_3d[e,0]],
                                [skeleton_3d[s,1], skeleton_3d[e,1]],
                                [skeleton_3d[s,2], skeleton_3d[e,2]],
                                lw=3, c=colormap(c_idx/len(connections)))

                # ====== 1. Still use the magnified range for zlim ======
                z_min_plot = min_xyz[2]
                z_max_plot = max_xyz[2]
                ax_pdf.set_zlim(z_min_plot, z_max_plot)

                # ====== 2. Calculate "real" ticks and map back to magnified coordinates ======
                real_min = z_min_plot / Z_SCALE
                real_max = z_max_plot / Z_SCALE

                # Take 500 mm as step, round down/up to the nearest multiple of 500
                real_ticks = np.arange(
                    np.floor(real_min / STEP_MM) * STEP_MM,
                    np.ceil (real_max / STEP_MM) * STEP_MM + STEP_MM,
                    STEP_MM
                )
                plot_ticks = real_ticks * Z_SCALE     # Map to magnified coordinates
                ax_pdf.set_zticks(plot_ticks)

                # ====== 3. Labels display real values (integers, no decimals) ======
                ax_pdf.zaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda v, pos: f"{v / Z_SCALE:.0f}")
                )

                # ====== 4. Other appearance settings ======
                ax_pdf.set_xlim(min_xyz[0], max_xyz[0])
                ax_pdf.set_ylim(min_xyz[1], max_xyz[1])

                ax_pdf.set_xlabel("X (mm)", fontsize=axis_label_fontsize)
                ax_pdf.set_ylabel("Y (mm)", fontsize=axis_label_fontsize, labelpad=40)
                ax_pdf.set_zlabel("Z (mm)", fontsize=axis_label_fontsize, labelpad=30)

                ax_pdf.tick_params(axis='x', which='major',
                                labelsize=tick_label_fontsize)
                ax_pdf.tick_params(axis='y', which='major',
                                labelsize=tick_label_fontsize, pad=20)
                ax_pdf.tick_params(axis='z', which='major',
                                labelsize=tick_label_fontsize, pad=20)

                ax_pdf.view_init(elev=10, azim=90)

                pdf_path = os.path.join(
                    output_dir, f"mocap_skeleton_frame_{idx:03d}_user_{user_id}.pdf"
                )
                fig_pdf.tight_layout()
                fig_pdf.savefig(pdf_path, bbox_inches='tight')
                plt.close(fig_pdf)

                print(f"Saved PDF => {pdf_path}")

@visualizer_decorator
def visualize_tof_and_rgb_mosaic_batch_downsample_tof(
    batch,
    output_dir='tof_rgb_mosaic',
    fps_out=7.32 
):
    """
    Visualize ToF Node 4 (8×8) + downsample to match camera timeline.

    We produce T_cam frames total. For each camera index i in [0..T_cam-1],
    we pick ToF => floor(i/(T_cam-1)*(T_tof-1)).

    The final mosaic is 1 row, 2 columns:
      [ ToF heatmap | RGB Node 1 ]

    We ignore the last dimension (18) by either picking channel 0 or 
    averaging across the 18 channels.  Adjust as needed.

    Requirements:
      - 'ToF' => shape [B, 1, T_tof, 8, 8, 18]
      - 'depthCamera.rgb_frames' => shape [B, 1, T_cam, H, W, 3]
    """
    user_ids = batch['user_id']
    activities = batch['activity']
    modality_data = batch['modality_data']

    # 1) Retrieve ToF data
    if 'ToF' not in modality_data or modality_data['ToF'] is None:
        print("No ToF data found in this batch.")
        return

    tof_data = modality_data['ToF']  # shape => [B, 1, T_tof, 8, 8, 18]
    if not isinstance(tof_data, torch.Tensor):
        print("Expected `ToF` to be a torch.Tensor.")
        return

    B, N_tof, T_tof, h_tof, w_tof, c_tof = tof_data.shape
    if N_tof != 1:
        print(f"Warning: This function expects 1 ToF node, but found {N_tof}.")

    # 2) Retrieve RGB data
    depthcam_data = modality_data.get('depthCamera', {})
    if 'rgb_frames' not in depthcam_data or depthcam_data['rgb_frames'] is None:
        print("No `rgb_frames` found under `depthCamera`.")
        return

    rgb_frames = depthcam_data['rgb_frames']  # shape => [B, 1, T_cam, H, W, 3]
    if not isinstance(rgb_frames, torch.Tensor):
        print("Expected `rgb_frames` to be a torch.Tensor.")
        return

    B2, N_rgb, T_cam, cam_H, cam_W, cam_C = rgb_frames.shape
    if B != B2:
        print("Mismatch: ToF batch size != camera batch size.")
        return

    # We'll produce T_cam frames total
    def get_tof_index(i_cam: int, T_cam_: int, T_tof_: int) -> int:
        """
        Step-based downsampling:
          alpha = i_cam/(T_cam_-1)
          tof_float_idx = alpha * (T_tof_-1)
          return floor(tof_float_idx).
        """
        if T_cam_ <= 1:
            return 0
        alpha = i_cam/(T_cam_-1)
        if T_tof_ <= 1:
            return 0
        tof_float_idx = alpha*(T_tof_-1)
        idx = floor(tof_float_idx)
        return max(0, min(idx, T_tof_-1))

    for b_idx in range(B):
        user_id = user_ids[b_idx]
        activity = activities[b_idx]
        act_str = activity[0] if isinstance(activity, list) else activity

        # shape => [1, T_tof, 8, 8, 18]
        sample_tof = tof_data[b_idx, 0]   # => shape [T_tof, 8, 8, 18]
        sample_rgb = rgb_frames[b_idx, 0] # => shape [T_cam, cam_H, cam_W, 3]

        # In case we want to find global min..max for the ToF amplitude, 
        # we can do so here. Or we can compute per-frame. 
        # We'll do per-frame min–max for simpler code.

        frames_list = []
        out_name = f"tof_rgb_{act_str}_user_{user_id}.mp4"
        out_path = os.path.join(output_dir, out_name)

        # Generate T_cam frames
        for i_cam in range(T_cam):
            tof_idx = get_tof_index(i_cam, T_cam, sample_tof.shape[0])

            # single row => 2 columns
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle(
                f"User {user_id}, Activity={act_str}, Frame {i_cam+1}/{T_cam}",
                fontsize=14
            )

            # 1) ToF node
            ax_tof = axes[0]
            # shape => [8,8,18]
            tof_tensor_torch = sample_tof[tof_idx]  # shape [8,8,18]
            tof_tensor = tof_tensor_torch.cpu().numpy()  # convert to numpy

            # Option A: pick the first channel
            # sub_vol = tof_tensor[..., 0]  # shape => [8,8]

            # Option B: average all 18 channels (like a broad amplitude)
            sub_vol = tof_tensor.mean(axis=-1)  # shape => [8,8]

            # min–max normalize => [0..255]
            tmin, tmax = sub_vol.min(), sub_vol.max()
            denom = (tmax - tmin) if (tmax>tmin) else 1e-6
            norm = (sub_vol - tmin)/denom
            norm_255 = (norm*255.0).clip(0,255).astype(np.uint8)

            # apply color map
            tof_colormap = cv2.applyColorMap(norm_255, cv2.COLORMAP_JET)
            # BGR => RGB
            tof_rgb = cv2.cvtColor(tof_colormap, cv2.COLOR_BGR2RGB)

            ax_tof.imshow(tof_rgb)
            ax_tof.set_title("ToF Node 4", fontsize=10)
            ax_tof.axis('off')

            # 2) RGB Node 1
            ax_rgb = axes[1]
            if i_cam < sample_rgb.shape[0]:
                rgb_frame = sample_rgb[i_cam].cpu().numpy()  # shape => [cam_H, cam_W, 3]
                rgb_uint8 = rgb_frame.astype(np.uint8)
                ax_rgb.imshow(rgb_uint8)
                ax_rgb.set_title("RGB Node 1", fontsize=10)
                ax_rgb.axis('off')
            else:
                ax_rgb.axis('off')

            # Convert fig => RGBA
            fig.canvas.draw()
            w_fig, h_fig = fig.canvas.get_width_height()
            img_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img_argb = img_argb.reshape((h_fig, w_fig, 4))

            # ARGB => RGBA
            img_rgba = img_argb[..., [1,2,3,0]]
            frames_list.append(img_rgba.copy())
            plt.close()

        if not frames_list:
            print(f"No frames for user={user_id}, activity={act_str}.")
            continue

        height, width = frames_list[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_path, fourcc, fps_out, (width, height))

        for frame_img in frames_list:
            frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGBA2BGR)
            out_writer.write(frame_bgr)
        out_writer.release()

        print(f"[visualize_tof_and_rgb_mosaic_batch_downsample_tof] Saved => {out_path}")

@visualizer_decorator
def clamp_infinite_fmcw_inplace(data):
    """
    data: Torch tensor of shape [B, 5, T_fmcw, N_points, 4] => (x, y, z, velocity)

    1) Replace inf/NaN with 0
    2) (Optional) clamp to a broad numeric range
    """
    mask_finite = torch.isfinite(data)
    data[~mask_finite] = 0.0

    # If you do want a broad numeric clamp:
    x_min_val, x_max_val = -100.0, 100.0
    y_min_val, y_max_val = -100.0, 100.0
    z_min_val, z_max_val = -100.0, 100.0
    v_min_val, v_max_val = -10.0, 10.0

    data[..., 0].clamp_(min=x_min_val, max=x_max_val)
    data[..., 1].clamp_(min=y_min_val, max=y_max_val)
    data[..., 2].clamp_(min=z_min_val, max=z_max_val)
    data[..., 3].clamp_(min=v_min_val, max=v_max_val)

@visualizer_decorator
def visualize_fmcw_and_rgb_mosaic_batch_raw_fixed_axes(
    batch,
    output_dir='fmcw_rgb_mosaic_raw_fixed_axes',
    fps_out=8.81,
    save_pdf_frames=True,
    axis_label_fontsize=14,   # Axis label font size
    tick_label_fontsize=12    # Tick label font size
):
    """
    Plot FMCW point clouds "as-is," fix axis limits:
       x in [-5, +5], y in [0, +3], z in [-5, +5].

    Now merges each FMCW frame i with [i+1, i+2] so the point cloud is denser.
    Produces:
      1) An .mp4 with T_fmcw frames (2x3 mosaic: up to 5 FMCW nodes + camera).
      2) 10 PDF snapshots (merging frames [idx, idx+1, idx+2]).

    NEW ADDITION:
      - Also saves multiple PDFs for Node 2 only (0-based index=1),
        with no title, 12x8 size, transparent background, merging frames [i..i+2].
      - Let's call them "node2_only_*.pdf".

    The new feature doesn't break or change the original outputs;
    it just adds extra PDFs for Node 2.

    Args:
        batch: The batch from your DataLoader.
        output_dir: Where to save outputs (mp4 + PDFs).
        fps_out: FPS for the final .mp4.
        save_pdf_frames: Whether to save the original 10 PDF snapshots (True/False).
        axis_label_fontsize: Font size used for axis labels (e.g. set_title or set_xlabel).
        tick_label_fontsize: Font size for the x,y,z axis scale numbers.

    Example usage:
        visualize_fmcw_and_rgb_mosaic_batch_raw_fixed_axes(
            batch,
            output_dir="fmcw_viz",
            fps_out=9,
            save_pdf_frames=True,
            axis_label_fontsize=14,
            tick_label_fontsize=12
        )
    """
    user_ids   = batch['user_id']
    activities = batch['activity']
    modality_data = batch['modality_data']

    # 1) FMCW data => shape [B, 5, T_fmcw, N_pts, 4]
    mmwave_data = modality_data.get('mmWave', None)
    if mmwave_data is None or not isinstance(mmwave_data, torch.Tensor):
        print("No mmWave data found or not a torch.Tensor.")
        return
    B, mm_nodes, T_fmcw, N_pts, xyzv_dim = mmwave_data.shape
    if mm_nodes < 1:
        print("No FMCW nodes found. Exiting.")
        return

    # 2) Camera data => shape [B, N_cam, T_cam, H, W, 3]
    depthcam_data = modality_data.get('depthCamera', {})
    if 'rgb_frames' not in depthcam_data or depthcam_data['rgb_frames'] is None:
        print("No rgb_frames found in depthCamera.")
        return
    rgb_frames = depthcam_data['rgb_frames']
    if not isinstance(rgb_frames, torch.Tensor):
        print("'rgb_frames' is not a torch.Tensor.")
        return
    B2, N_cam, T_cam, camH, camW, camC = rgb_frames.shape
    if B != B2:
        print("Mismatch in batch dimension for FMCW vs. Camera.")
        return

    # Optional clamp/NaN fix:
    def clamp_infinite_fmcw_inplace(tensor):
        mask_finite = torch.isfinite(tensor)
        tensor[~mask_finite] = 0.0
    clamp_infinite_fmcw_inplace(mmwave_data)

    # Helper to merge frames [i, i+1, i+2] for denser point clouds
    def gather_3_frames(mm_sample, node_idx, i_fmcw):
        frames_to_merge = []
        T_ = mm_sample.shape[1]  # T_fmcw
        for offset in [0, 1, 2]:
            cur_idx = i_fmcw + offset
            if 0 <= cur_idx < T_:
                frames_to_merge.append(mm_sample[node_idx, cur_idx])  # shape [N_pts, 4]
        if frames_to_merge:
            merged = np.concatenate(frames_to_merge, axis=0)
        else:
            merged = np.zeros((0, 4), dtype=np.float32)
        return merged

    # Subplot positions (up to 5 nodes) => 2×3 grid; bottom-right is camera
    node_positions = [
        (0,0), (0,1), (0,2),
        (1,0), (1,1)
        # (1,2) => camera
    ]

    # Axis limits
    X_MIN, X_MAX = -3, 3
    Y_MIN, Y_MAX =  0, 5
    Z_MIN, Z_MAX = -3, 3

    for b_idx in range(B):
        user_id   = user_ids[b_idx]
        activity  = activities[b_idx]
        act_str   = activity[0] if isinstance(activity, list) else activity

        mm_sample  = mmwave_data[b_idx]  # shape => [mm_nodes, T_fmcw, N_pts, 4]
        rgb_sample = rgb_frames[b_idx, 0] # shape => [T_cam, H, W, 3]

        # ================================
        # (A) Build the MP4
        # ================================
        frames_list = []
        out_mp4_name = f"fmcw_raw_fixed_{act_str}_user_{user_id}.mp4"
        out_mp4_path = os.path.join(output_dir, out_mp4_name)

        for i_fmcw in range(T_fmcw):
            # Step-based camera alignment
            if T_fmcw > 1 and T_cam > 1:
                alpha = i_fmcw / (T_fmcw - 1)
                cam_float_idx = alpha * (T_cam - 1)
                cam_idx = floor(cam_float_idx)
            else:
                cam_idx = 0

            fig, axes = plt.subplots(2, 3, figsize=(14,8), subplot_kw={'projection':'3d'})
            fig.suptitle(
                f"User={user_id}, Activity={act_str}, FMCW Frame {i_fmcw+1}/{T_fmcw}",
                fontsize=16
            )

            # Plot up to 5 nodes
            for node_i in range(min(mm_nodes, 5)):
                r, c = node_positions[node_i]
                ax_3d = axes[r, c]

                # Merge frames [i_fmcw, i_fmcw+1, i_fmcw+2]
                merged_points_4 = gather_3_frames(mm_sample, node_i, i_fmcw)
                x_vals = merged_points_4[:, 0]
                y_vals = merged_points_4[:, 1]
                z_vals = merged_points_4[:, 2]
                v_vals = merged_points_4[:, 3]

                sc = ax_3d.scatter(
                    x_vals, y_vals, z_vals,
                    c=v_vals, cmap='jet', s=15, alpha=0.7
                )
                ax_3d.set_title(f"FMCW Node {node_i+1}", fontsize=axis_label_fontsize)  # *** FONT SIZE OPTION ***
                ax_3d.set_xlim(X_MIN, X_MAX)
                ax_3d.set_ylim(Y_MIN, Y_MAX)
                ax_3d.set_zlim(Z_MIN, Z_MAX)
                # yz-plane => from behind
                ax_3d.view_init(elev=30, azim=-150)

                # *** FONT SIZE OPTION ***: control tick label size
                ax_3d.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
                ax_3d.tick_params(axis='z',    which='major', labelsize=tick_label_fontsize)

            # Camera => bottom-right
            ax_rgb = plt.subplot2grid((2,3), (1,2))
            ax_rgb.set_position(axes[1,2].get_position())
            axes[1,2].remove()

            if cam_idx < rgb_sample.shape[0]:
                rgb_frame = rgb_sample[cam_idx].cpu().numpy().astype(np.uint8)
                ax_rgb.imshow(rgb_frame)
                ax_rgb.set_title("RGB Node 1", fontsize=axis_label_fontsize)  # *** FONT SIZE OPTION ***
                ax_rgb.axis('off')
            else:
                ax_rgb.axis('off')

            # Convert => RGBA
            fig.canvas.draw()
            w_fig, h_fig = fig.canvas.get_width_height()
            img_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img_argb = img_argb.reshape((h_fig, w_fig, 4))

            # ARGB => RGBA
            img_rgba = img_argb[..., [1,2,3,0]]
            frames_list.append(img_rgba.copy())
            plt.close(fig)

        # Write frames => MP4
        if frames_list:
            height, width = frames_list[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(out_mp4_path, fourcc, fps_out, (width, height))
            for frame_img in frames_list:
                frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGBA2BGR)
                out_writer.write(frame_bgr)
            out_writer.release()
            print(f"[visualize_fmcw_and_rgb_mosaic_batch_raw_fixed_axes] => {out_mp4_path}")
        else:
            print(f"No frames for user={user_id}, activity={act_str}. (No video)")

        # ================================
        # (B) Original 10 PDF Snapshots
        # ================================
        if save_pdf_frames and T_fmcw > 0:
            pdf_indices = np.linspace(0, T_fmcw - 1, 10, dtype=int)
            pdf_indices = np.unique(pdf_indices)

            for idx in pdf_indices:
                if idx < 0 or idx >= T_fmcw:
                    continue

                fig_pdf, axes_pdf = plt.subplots(2, 3, figsize=(14,8), subplot_kw={'projection':'3d'})
                fig_pdf.suptitle(
                    f"FMCW Snapshot, user={user_id}, activity={act_str}, frames [{idx},{idx+1},{idx+2}]",
                    fontsize=axis_label_fontsize  # *** FONT SIZE OPTION ***
                )

                if T_fmcw > 1 and T_cam > 1:
                    alpha_pdf = idx / (T_fmcw - 1)
                    cam_float_idx_ = alpha_pdf * (T_cam - 1)
                    cam_idx_ = floor(cam_float_idx_)
                else:
                    cam_idx_ = 0

                for node_i in range(min(mm_nodes, 5)):
                    r, c = node_positions[node_i]
                    ax_3d_pdf = axes_pdf[r, c]

                    # Merge frames [idx, idx+1, idx+2]
                    merged_pts_4 = gather_3_frames(mm_sample, node_i, idx)
                    x_vals = merged_pts_4[:, 0]
                    y_vals = merged_pts_4[:, 1]
                    z_vals = merged_pts_4[:, 2]
                    v_vals = merged_pts_4[:, 3]

                    ax_3d_pdf.scatter(
                        x_vals, y_vals, z_vals,
                        c=v_vals, cmap='jet', s=5, alpha=0.7
                    )
                    ax_3d_pdf.set_title(f"FMCW Node {node_i+1}", fontsize=axis_label_fontsize)  # *** FONT SIZE OPTION ***
                    ax_3d_pdf.set_xlim(X_MIN, X_MAX)
                    ax_3d_pdf.set_ylim(Y_MIN, Y_MAX)
                    ax_3d_pdf.set_zlim(Z_MIN, Z_MAX)
                    ax_3d_pdf.view_init(elev=10, azim=-95)

                    # *** FONT SIZE OPTION ***: control tick label size
                    ax_3d_pdf.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
                    ax_3d_pdf.tick_params(axis='z',    which='major', labelsize=tick_label_fontsize)

                # camera => bottom-right
                ax_cam_pdf = plt.subplot2grid((2,3), (1,2))
                ax_cam_pdf.set_position(axes_pdf[1,2].get_position())
                axes_pdf[1,2].remove()

                if cam_idx_ < rgb_sample.shape[0]:
                    rgb_frame_ = rgb_sample[cam_idx_].cpu().numpy().astype(np.uint8)
                    ax_cam_pdf.imshow(rgb_frame_)
                    ax_cam_pdf.set_title("RGB Node 1", fontsize=axis_label_fontsize)  # *** FONT SIZE OPTION ***
                    ax_cam_pdf.axis('off')
                else:
                    ax_cam_pdf.axis('off')

                # Transparent background
                fig_pdf.patch.set_facecolor("none")
                for ax_ in fig_pdf.axes:
                    ax_.set_facecolor("none")

                pdf_fname = f"fmcw_raw_fixed_{act_str}_user_{user_id}_frame_{idx:03d}.pdf"
                pdf_path = os.path.join(output_dir, pdf_fname)
                # 12×8 is already used above via figsize=(14,8). If you want EXACT 12×8:
                # fig_pdf.set_size_inches(12, 8)

                fig_pdf.savefig(pdf_path, transparent=True, bbox_inches='tight')
                plt.close(fig_pdf)

                print(f"[PDF] => {pdf_path}")

        # ================================
        # (C) NEW OUTPUT: Node 3 Only
        # ================================
        # We'll produce multiple PDFs for Node 3 (0-based index=1),
        # no title, 12×8 inches, transparent background,
        # merging frames [i, i+1, i+2].
        node2_idx = 3  # (Node 3 is index=1 in 0-based indexing)
        if node2_idx < mm_nodes:
            node2_output_dir = os.path.join(output_dir, "node3_only_pdfs")
            os.makedirs(node2_output_dir, exist_ok=True)

            # Let's produce 10 PDFs again, or as many as T_fmcw
            pdf2_indices = np.linspace(0, T_fmcw - 1, 50, dtype=int)
            pdf2_indices = np.unique(pdf2_indices)

            for idx2 in pdf2_indices:
                if idx2 < 0 or idx2 >= T_fmcw:
                    continue

                fig_node2 = plt.figure()
                # Force 12×8
                fig_node2.set_size_inches(12, 8)
                # Transparent figure background
                fig_node2.patch.set_facecolor("none")

                ax_n2 = fig_node2.add_subplot(111, projection='3d')
                ax_n2.set_facecolor("none")

                # Merge frames [idx2, idx2+1, idx2+2]
                merged_n2 = gather_3_frames(mm_sample, node2_idx, idx2)
                x_vals = merged_n2[:, 0]
                y_vals = merged_n2[:, 1]
                z_vals = merged_n2[:, 2]
                v_vals = merged_n2[:, 3]

                ax_n2.scatter(
                    x_vals, y_vals, z_vals,
                    c=v_vals, cmap='jet', s=50, alpha=0.7
                )
                # No title => skip

                ax_n2.set_xlim(X_MIN, X_MAX)
                ax_n2.set_ylim(Y_MAX, Y_MIN)
                ax_n2.set_zlim(Z_MIN, Z_MAX)
                ax_n2.view_init(elev=10, azim=90)

                # *** FONT SIZE OPTION *** 
                # If you want to label axes, do so:
                ax_n2.set_xlabel("X (m)", fontsize=axis_label_fontsize)
                ax_n2.set_ylabel("Y (m)", fontsize=axis_label_fontsize)
                ax_n2.set_zlabel("Z (m)", fontsize=axis_label_fontsize)
                ax_n2.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
                ax_n2.tick_params(axis='z',    which='major', labelsize=tick_label_fontsize)

                pdf2_fname = f"node3_only_{act_str}_user_{user_id}_frame_{idx2:03d}.pdf"
                pdf2_path = os.path.join(node2_output_dir, pdf2_fname)

                fig_node2.savefig(pdf2_path, format="pdf", transparent=True, bbox_inches="tight")
                plt.close(fig_node2)

                print(f"Node3-only PDF => {pdf2_path}")
        else:
            print(f"Node 3 (idx=1) not available (mm_nodes={mm_nodes}). Skipping new PDFs.")

@visualizer_decorator
def visualize_imu_four_rows_no_zscore(
    batch,
    output_dir="imu_4rows_no_zscore",
    fps_out=10.0,
    imu_sampling_rate=60.0,  # Hz
    save_pdf=True,
    axis_label_fontsize=12,  # New arg for axis labels
    tick_label_fontsize=10   # New arg for tick labels
):
    """
    Modified to:
      1) Have NO title (no suptitle, no subplot title).
      2) Use axis_label_fontsize=14 and tick_label_fontsize=12.

    Otherwise, the logic is the same as before:
      - IMU data shape = [B, 1, T_imu, 13, 17].
      - Plot 4 stacked subplots (acc, mag, eul, quat).
      - Camera spans the entire right side.
      - Create an .mp4 and an IMU-only PDF for each sample in the batch.
    """
    # Group definitions: (feature indices) and row label
    group1_indices = [0, 1, 2]       # acc_x, acc_y, acc_z
    group2_indices = [3, 4, 5]       # mag_x, mag_y, mag_z
    group3_indices = [6, 7, 8]       # eul_x, eul_y, eul_z
    group4_indices = [9, 10, 11, 12] # q0, q1, q2, q3

    group_indices = [group1_indices, group2_indices, group3_indices, group4_indices]
    group_labels  = [
        ["acc_x","acc_y","acc_z"],
        ["mag_x","mag_y","mag_z"],
        ["eul_x","eul_y","eul_z"],
        ["q0","q1","q2","q3"]
    ]
    # More academic y-axis labels per group
    group_yaxis_names = [
        "Acceleration (m/s²)",
        "Magnetic Field (-)",
        "Euler Angles (deg)",
        "Quaternions (-)"
    ]

    user_ids     = batch["user_id"]
    activities   = batch["activity"]
    modality     = batch["modality_data"]

    # 1) Pull out IMU data => shape [B, 1, T_imu, 13, 17]
    imu_data = modality.get("imu", None)
    if imu_data is None or not isinstance(imu_data, torch.Tensor):
        print("No valid IMU data found in this batch.")
        return

    B, N_imu, T_imu, N_features, N_sensors = imu_data.shape
    if N_imu < 1:
        print("No IMU node found. Skipping.")
        return

    # 2) Pull out camera frames => shape [B, 1, T_cam, H, W, 3]
    depthcam_data = modality.get("depthCamera", {})
    rgb_frames = depthcam_data.get("rgb_frames", None)
    if rgb_frames is None or not isinstance(rgb_frames, torch.Tensor):
        print("No camera frames found ('depthCamera.rgb_frames').")
        return

    B2, N_rgb, T_cam, camH, camW, camC = rgb_frames.shape
    if B != B2:
        print("Mismatch in batch dimension for IMU vs Camera.")
        return

    for b_idx in range(B):
        user_id  = user_ids[b_idx]
        activity = activities[b_idx]
        act_str  = activity[0] if isinstance(activity, list) else activity

        # (A) Extract sensor=0 => shape [T_imu, 13]
        sample_imu = imu_data[b_idx, 0].cpu().numpy()  # shape [T_imu, 13, 17]
        sample_imu_s0 = sample_imu[..., 0]             # shape [T_imu, 13]

        # (B) Time axis in seconds
        time_axis = np.arange(T_imu) / imu_sampling_rate

        # (C) Precompute y-lims for each group
        group_y_lims = []
        for grp_i, idx_list in enumerate(group_indices):
            sub_data = sample_imu_s0[:, idx_list]  # shape [T_imu, len(idx_list)]
            y_min = sub_data.min()
            y_max = sub_data.max()
            if y_max <= y_min:
                y_max = y_min + 1e-3
            # Add 10% margin:
            padding = 0.1 * (y_max - y_min)
            y_min_plt = y_min - padding
            y_max_plt = y_max + padding
            group_y_lims.append((y_min_plt, y_max_plt))

        # (D) Retrieve camera frames => shape [T_cam, H, W, 3]
        sample_rgb = rgb_frames[b_idx, 0]

        # # --------------- Build the MP4 ---------------
        # frames_list = []
        # out_mp4_name = f"imu_4rows_nozscore_{act_str}_user_{user_id}.mp4"
        # out_mp4_path = os.path.join(output_dir, out_mp4_name)

        # for i_cam in range(T_cam):
        #     fig = plt.figure(figsize=(12, 8))
        #     gs  = GridSpec(nrows=4, ncols=2, width_ratios=[4, 3], figure=fig)

        #     # Left side => 4 subplots stacked vertically
        #     ax_imu = []
        #     for row_idx in range(4):
        #         ax = fig.add_subplot(gs[row_idx, 0])
        #         ax_imu.append(ax)

        #     # Right side => camera spanning all 4 rows
        #     ax_cam = fig.add_subplot(gs[:, 1])

        #     # *** No suptitle here ***  

        #     # ---- Plot the 4 IMU subplots ----
        #     for row_idx, (indices_this, labels_this) in enumerate(zip(group_indices, group_labels)):
        #         ax = ax_imu[row_idx]
        #         y_min_plt, y_max_plt = group_y_lims[row_idx]
        #         y_label_text = group_yaxis_names[row_idx]

        #         # Plot each feature line in this group
        #         for feat_idx, feat_label in zip(indices_this, labels_this):
        #             y_vals = sample_imu_s0[:, feat_idx]
        #             ax.plot(time_axis, y_vals, label=feat_label, lw=1.5)

        #         ax.set_xlim([0, (T_imu - 1)/imu_sampling_rate])
        #         ax.set_ylim([y_min_plt, y_max_plt])
        #         # *** Use axis_label_fontsize ***
        #         ax.set_ylabel(y_label_text, fontsize=axis_label_fontsize)

        #         # *** Tick label size => tick_label_fontsize ***
        #         ax.tick_params(axis='both', labelsize=tick_label_fontsize)

        #         if row_idx == 3:  # bottom row => x-label
        #             ax.set_xlabel("Time (second)", fontsize=axis_label_fontsize)
        #         else:
        #             ax.set_xticks([])

        #         # Legend font size = optional; let's match tick_label_fontsize
        #         ax.legend(fontsize=tick_label_fontsize, loc="upper right")

        #         # *** No subplot title *** 
        #         # ax.set_title(f"Group {row_idx+1}", fontsize=10)

        #     # ---- Camera subplot ----
        #     if i_cam < sample_rgb.shape[0]:
        #         rgb_frame = sample_rgb[i_cam].cpu().numpy().astype(np.uint8)
        #         ax_cam.imshow(rgb_frame)
        #         # *** No title *** 
        #         # ax_cam.set_title("Camera", fontsize=10)
        #         ax_cam.axis('off')
        #     else:
        #         ax_cam.axis('off')

        #     # Convert figure => RGBA
        #     fig.canvas.draw()
        #     w_fig, h_fig = fig.canvas.get_width_height()
        #     img_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        #     img_argb = img_argb.reshape((h_fig, w_fig, 4))
        #     # ARGB => RGBA
        #     img_rgba = img_argb[..., [1,2,3,0]]
        #     frames_list.append(img_rgba.copy())
        #     plt.close(fig)

        # # Write out the .mp4
        # if frames_list:
        #     height, width = frames_list[0].shape[:2]
        #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #     writer = cv2.VideoWriter(out_mp4_path, fourcc, fps_out, (width, height))
        #     for frame_img in frames_list:
        #         frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGBA2BGR)
        #         writer.write(frame_bgr)
        #     writer.release()
        #     print(f"[visualize_imu_four_rows_no_zscore] => MP4 saved: {out_mp4_path}")
        # else:
        #     print(f"No frames for user={user_id}, activity={act_str}. (No video)")

        # --------------- Build the PDF (IMU-only) ---------------
        if save_pdf and T_imu > 0:
            pdf_name = f"imu_4rows_nozscore_{act_str}_user_{user_id}.pdf"
            pdf_path = os.path.join(output_dir, pdf_name)
            fig_pdf = plt.figure(figsize=(12,8))
            gs_pdf  = GridSpec(nrows=4, ncols=1, figure=fig_pdf)

            # *** No title *** 
            # fig_pdf.suptitle(...)

            for row_idx, (indices_this, labels_this) in enumerate(zip(group_indices, group_labels)):
                axp = fig_pdf.add_subplot(gs_pdf[row_idx, 0])
                y_min_plt, y_max_plt = group_y_lims[row_idx]
                y_label_text = group_yaxis_names[row_idx]

                for feat_idx, feat_label in zip(indices_this, labels_this):
                    y_vals = sample_imu_s0[:, feat_idx]
                    axp.plot(time_axis, y_vals, label=feat_label, lw=1.5)

                axp.set_xlim([0, (T_imu - 1)/imu_sampling_rate])
                axp.set_ylim([y_min_plt, y_max_plt])
                axp.set_ylabel(y_label_text, fontsize=axis_label_fontsize)
                axp.tick_params(axis='both', labelsize=tick_label_fontsize)
                axp.legend(fontsize=tick_label_fontsize, loc="upper right")

                if row_idx == 3:
                    axp.set_xlabel("Time (second)", fontsize=axis_label_fontsize)
                else:
                    axp.set_xticks([])

                # *** No subplot title ***
                # axp.set_title(f"Group {row_idx+1}: {labels_this}", fontsize=10)

            plt.tight_layout()
            # Force transparent background
            fig_pdf.patch.set_facecolor("none")
            fig_pdf.align_ylabels()
            for ax_ in fig_pdf.axes:
                ax_.set_facecolor("none")

            fig_pdf.savefig(pdf_path, transparent=True, bbox_inches='tight')
            plt.close(fig_pdf)
            print(f"[visualize_imu_four_rows_no_zscore] => PDF saved: {pdf_path}")

@visualizer_decorator
def visualize_vayyar_txrx_only_and_camera(
    batch,
    output_dir="vayyar_txrx_only_plus_camera",
    fps_out=10.0
):
    """
    Similar to visualize_vayyar_and_camera_mosaic_batch, but ignoring
    the 100 ADC dimension by averaging across it, leaving only
    (time, TX*RX=400).

    Data shape: [B, 1, T_vay, 400, 100], complex
    Steps:
      1) Compute amplitude => shape [T_vay, 400, 100]
      2) Average across the 100 dimension => shape [T_vay, 400]
      3) Transpose => shape [400, T_vay] for imshow
      4) Produce T_cam frames total. For each camera frame i_cam:
         - Step-based alignment => vay_idx in [0..(T_vay-1)]
         - Plot entire 2D heatmap, draw vertical line at x=vay_idx
         - Show camera frame on the right
    """
    user_ids = batch["user_id"]       # [B]
    activities = batch["activity"]    # [B]
    modality_data = batch["modality_data"]

    # 1) Retrieve Vayyar data
    # Expect shape => [B, 1, T_vay, 400, 100], complex
    vayyar_data = modality_data.get("vayyar", None)
    # # Convert to NumPy (handle complex values)
    # vayyar_np = vayyar_data.numpy()  # Ensure it's on CPU

    # # Define output path
    # vayyar_output_path = os.path.join(output_dir, "vayyar_data_for_jtplotting_2.npy")

    # # Save as .npy
    # np.save(vayyar_output_path, vayyar_np)

    # print(f"Vayyar data saved to {vayyar_output_path}")
    if vayyar_data is None:
        print("No 'vayyar' data found in this batch.")
        return
    if not isinstance(vayyar_data, torch.Tensor):
        print("Expected 'vayyar' to be a torch.Tensor.")
        return
    
    B, N_vay, T_vay, txrx_dim, adc_dim = vayyar_data.shape
    if N_vay < 1:
        print("No Vayyar node found. We only handle 1 node for demonstration.")
        return

    # 2) Retrieve camera data
    depthcam_data = modality_data.get("depthCamera", {})
    if "rgb_frames" not in depthcam_data or depthcam_data["rgb_frames"] is None:
        print("No 'rgb_frames' found in 'depthCamera'.")
        return

    rgb_frames = depthcam_data["rgb_frames"]  # shape => [B, 1, T_cam, H, W, 3]
    if not isinstance(rgb_frames, torch.Tensor):
        print("Expected 'rgb_frames' to be a torch.Tensor.")
        return

    B2, N_rgb, T_cam, camH, camW, camC = rgb_frames.shape
    if B != B2:
        print("Mismatch in batch dimension for Vayyar vs Camera.")
        return

    def process_vayyar_txrx_only(vay_node_tensor: torch.Tensor) -> np.ndarray:
        """
        vay_node_tensor => shape [T_vay, 400, 100], complex
        1) amplitude => shape [T_vay, 400, 100]
        2) average across the 100 dimension => shape [T_vay, 400]
        3) transpose => shape [400, T_vay] for imshow
        """
        # shape => (T_vay, 400, 100), complex
        vay_np = vay_node_tensor.cpu().numpy()

        # (A) amplitude => shape [T_vay, 400, 100]
        amplitude_3d = np.abs(vay_np)

        # (B) average across the 100 dimension => [T_vay, 400]
        amp_2d = amplitude_3d.mean(axis=-1)  # => shape [T_vay, 400]

        # (C) transpose => shape => [400, T_vay]
        amplitude_plot = amp_2d.T
        return amplitude_plot

    for b_idx in range(B):
        user_id = user_ids[b_idx]
        activity = activities[b_idx]
        act_str = activity[0] if isinstance(activity, list) else activity

        # shape => [T_vay, 400, 100], complex
        # We'll pick the first node => [b_idx, 0]
        vay_node = vayyar_data[b_idx, 0]  # => shape (T_vay, 400, 100)
        sample_rgb = rgb_frames[b_idx, 0] # => shape [T_cam, H, W, 3]

        # Build the time x (TX*RX=400) plot
        amplitude_plot = process_vayyar_txrx_only(vay_node)  # => shape [400, T_vay]
        subc_dim, T_vay_ = amplitude_plot.shape

        # min..max for color scale
        a_min, a_max = amplitude_plot.min(), amplitude_plot.max()
        denom = (a_max - a_min) if a_max> a_min else 1e-6

        frames_list = []
        out_name = f"vayyar_txrx_only_{act_str}_user_{user_id}.mp4"
        out_path = os.path.join(output_dir, out_name)

        # We'll produce T_cam frames total
        for i_cam in range(T_cam):
            if T_cam>1 and T_vay_>1:
                alpha = i_cam/(T_cam-1)
                vay_float_idx = alpha*(T_vay_-1)
                vay_idx = floor(vay_float_idx)
            else:
                vay_idx = 0

            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14,5))
            fig.suptitle(
                f"User {user_id}, Activity={act_str}, Frame {i_cam+1}/{T_cam}",
                fontsize=15
            )

            # Left: show the amplitude_plot
            normed = (amplitude_plot - a_min)/denom
            ax_left.imshow(normed, aspect='auto', origin='lower', cmap='jet')

            # Draw vertical line => x=vay_idx (time axis => horizontal dim)
            # If you don't want the line, comment out:
            ax_left.axvline(x=vay_idx, color='white', linestyle='--', linewidth=2)
            ax_left.set_title("Vayyar amplitude (avg over ADC)")

            # Hide ticks for a cleaner look
            ax_left.set_xticks([])
            ax_left.set_yticks([])

            # Right: camera
            if i_cam < sample_rgb.shape[0]:
                rgb_frame = sample_rgb[i_cam].cpu().numpy().astype(np.uint8)
                ax_right.imshow(rgb_frame)
                ax_right.set_title("Camera")
                ax_right.axis('off')
            else:
                ax_right.axis('off')

            fig.canvas.draw()
            w_fig, h_fig = fig.canvas.get_width_height()
            img_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img_argb = img_argb.reshape((h_fig, w_fig, 4))

            # ARGB => RGBA
            img_rgba = img_argb[..., [1,2,3,0]]
            frames_list.append(img_rgba.copy())
            plt.close()

        if not frames_list:
            print(f"No frames for user={user_id}, activity={act_str}.")
            continue

        # Write frames => MP4
        height, width = frames_list[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_path, fourcc, fps_out, (width, height))

        for frame_img in frames_list:
            frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGBA2BGR)
            out_writer.write(frame_bgr)
        out_writer.release()

        print(f"[visualize_vayyar_txrx_only_and_camera] => {out_path}")

def _process_mel_spectrogram(raw_mel, smoothing_sigma=(0,1.5)):
    """
    Given a raw mel-spectrogram (shape [n_mels, T_frames]),
    apply the steps:
      1) z-score
      2) power_to_db
      3) percentile clipping (5..95)
      4) Gaussian smoothing
    Returns a 2D array for display.
    """
    # 1) Convert to float32
    arr = raw_mel.astype(np.float32)
    
    # 2) Z-score across the entire array
    arr_mean = np.mean(arr)
    arr_std  = np.std(arr) if np.std(arr)>1e-12 else 1.0
    arr_z = (arr - arr_mean)/arr_std

    # 3) Convert to dB scale
    #    If your data is a "power" or amplitude depends on your pipeline;
    #    here we do power_to_db(arr_z**2, ...) as an example.
    DB = librosa.power_to_db(arr_z**2, ref=np.max)

    # 4) Percentile clip
    p5, p95 = np.percentile(DB, [5, 95])
    DB_clipped = np.clip(DB, p5, p95)

    # 5) Optional Gaussian smoothing
    #    (sigma=(0,1.5)) → stronger smoothing across time dimension
    DB_smoothed = gaussian_filter(DB_clipped, sigma=smoothing_sigma)

    return DB_smoothed

@visualizer_decorator
def visualize_acoustic_2node_melspectrogram_and_rgb(
    batch,
    output_dir="acoustic_melspec_plus_rgb_advanced",
    fps_out=10.0,
    save_pdf=True,
    acoustic_total_secs=5.4,  
    smoothing_sigma=(0,1.5),
    sr=16000
):
    """
    For each sample in the batch:
      - We assume shape [B, 2, 1, n_mels, T_acoustic].
      - For each of the 2 acoustic nodes, we:
          * z-score
          * power_to_db
          * percentile-clip
          * smooth with Gaussian
      - Then display with librosa.display.specshow(..., y_axis='mel').
      - We produce T_cam frames total in the final .mp4, stepping the mel
        spectrogram in sync with the camera if desired.
      - Also produce an optional PDF of the final mel plots (no camera).
    """
    # 1) Retrieve acoustic data
    modality_data = batch["modality_data"]
    acoustic_data = modality_data.get("acoustic", None)
    if acoustic_data is None:
        print("No 'acoustic' data found in this batch.")
        return
    if not isinstance(acoustic_data, torch.Tensor):
        acoustic_data = acoustic_data.get('mel_spectrogram', None)
        if acoustic_data is None or not isinstance(acoustic_data, torch.Tensor):
            print("No valid acoustic['mel_spectrogram'] found or not a tensor.")
            return

    B, acoustic_nodes, ch_dim, n_mels, T_acoustic = acoustic_data.shape
    if acoustic_nodes < 2:
        print("Warning: This function expects at least 2 acoustic nodes.")

    # 2) Retrieve camera data
    depthcam_data = modality_data.get("depthCamera", {})
    if "rgb_frames" not in depthcam_data or depthcam_data["rgb_frames"] is None:
        print("No 'rgb_frames' in 'depthCamera'.")
        return

    rgb_frames = depthcam_data["rgb_frames"]  # => shape [B, 1, T_cam, H, W, 3]
    if not isinstance(rgb_frames, torch.Tensor):
        print("Expected 'rgb_frames' to be a torch.Tensor.")
        return

    B2, N_rgb, T_cam, camH, camW, camC = rgb_frames.shape
    if B != B2:
        print("Mismatch in batch dimension: acoustic vs camera.")
        return

    # We'll map horizontal frames => [0..acoustic_total_secs]
    time_per_frame = acoustic_total_secs / T_acoustic

    user_ids   = batch["user_id"]
    activities = batch["activity"]

    for b_idx in range(B):
        user_id  = user_ids[b_idx]
        activity = activities[b_idx]
        act_str  = activity[0] if isinstance(activity, list) else activity

        # shape => [2, 1, n_mels, T_acoustic]
        node1_data = acoustic_data[b_idx, 0, 0].cpu().numpy()  # => [n_mels, T_acoustic]
        node2_data = acoustic_data[b_idx, 1, 0].cpu().numpy()  # => [n_mels, T_acoustic]

        # Process each node
        proc_n1 = _process_mel_spectrogram(node1_data, smoothing_sigma=smoothing_sigma)
        proc_n2 = _process_mel_spectrogram(node2_data, smoothing_sigma=smoothing_sigma)

        sample_rgb = rgb_frames[b_idx, 0]  # => [T_cam, H, W, 3]

        # === Build the MP4 ===
        frames_list = []
        out_mp4_name = f"acoustic_2node_{act_str}_user_{user_id}_ADVANCED.mp4"
        out_mp4_path = os.path.join(output_dir, out_mp4_name)

        for i_cam in range(T_cam):
            if T_cam > 1 and T_acoustic > 1:
                alpha = i_cam/(T_cam - 1)
                acoustic_float_idx = alpha*(T_acoustic - 1)
                acoustic_idx = floor(acoustic_float_idx)
            else:
                acoustic_idx = 0

            fig, axes = plt.subplots(1, 3, figsize=(14,4))
            fig.suptitle(
                f"User {user_id}, Activity={act_str}, Frame {i_cam+1}/{T_cam}",
                fontsize=14
            )

            # ------ Node 1 ------
            ax_n1 = axes[0]
            im1 = librosa.display.specshow(
                proc_n1,
                sr=sr,
                cmap='jet',
                x_axis='time',
                y_axis='mel',  # <--- correct for mel-spectrogram data
                fmax=sr/2,     # or your max freq
                ax=ax_n1
            )
            ax_n1.set_title("Node 1 (ADV)", fontsize=9)

            # optional vertical line => x= acoustic_idx * time_per_frame
            x_cur_1 = acoustic_idx * time_per_frame
            ax_n1.axvline(x=x_cur_1, color='white', linestyle='--', linewidth=1.5)
            ax_n1.set_xlabel("Time (sec)", fontsize=8)
            ax_n1.set_ylabel("Mel Frequency (Hz)", fontsize=8)

            cbar1 = fig.colorbar(im1, ax=ax_n1, format='%+2.0f dB')
            cbar1.ax.tick_params(labelsize=8)

            # ------ Node 2 ------
            ax_n2 = axes[1]
            im2 = librosa.display.specshow(
                proc_n2,
                sr=sr,
                cmap='jet',
                x_axis='time',
                y_axis='mel',
                fmax=sr/2,
                ax=ax_n2
            )
            ax_n2.set_title("Node 2 (ADV)", fontsize=9)

            x_cur_2 = acoustic_idx * time_per_frame
            ax_n2.axvline(x=x_cur_2, color='white', linestyle='--', linewidth=1.5)
            ax_n2.set_xlabel("Time (sec)", fontsize=8)
            ax_n2.set_ylabel("Mel Frequency (Hz)", fontsize=8)

            cbar2 = fig.colorbar(im2, ax=ax_n2, format='%+2.0f dB')
            cbar2.ax.tick_params(labelsize=8)

            # ------ Camera on the right ------
            ax_cam = axes[2]
            if i_cam < sample_rgb.shape[0]:
                rgb_frame = sample_rgb[i_cam].cpu().numpy().astype(np.uint8)
                ax_cam.imshow(rgb_frame)
                ax_cam.set_title("Camera", fontsize=9)
                ax_cam.axis('off')
            else:
                ax_cam.axis('off')

            # Convert fig => RGBA
            fig.canvas.draw()
            w_fig, h_fig = fig.canvas.get_width_height()
            img_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img_argb = img_argb.reshape((h_fig, w_fig, 4))
            # ARGB => RGBA
            img_rgba = img_argb[..., [1,2,3,0]]
            frames_list.append(img_rgba.copy())
            plt.close()

        if frames_list:
            height, width = frames_list[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(out_mp4_path, fourcc, fps_out, (width, height))
            for frame_img in frames_list:
                frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGBA2BGR)
                out_writer.write(frame_bgr)
            out_writer.release()
            print(f"[visualize_acoustic_2node_melspectrogram_and_rgb_advanced] => MP4 saved: {out_mp4_path}")
        else:
            print(f"No frames produced for user={user_id}, activity={act_str} (Acoustic).")

        # === Build the PDF (optional) ===
        if save_pdf and T_acoustic > 0:
            pdf_name = f"acoustic_2node_{act_str}_user_{user_id}_ADVANCED.pdf"
            pdf_path = os.path.join(output_dir, pdf_name)

            fig_pdf, axs_pdf = plt.subplots(1, 2, figsize=(12,5))
            fig_pdf.suptitle(
                f"Acoustic (Mel-Spec ADV) - User {user_id}, Activity={act_str}",
                fontsize=14
            )

            # --- Node 1 ---
            axp1 = axs_pdf[0]
            im1p = librosa.display.specshow(
                proc_n1,
                sr=sr,
                cmap='inferno',
                x_axis='time',
                y_axis='mel',
                fmax=sr/2,
                ax=axp1
            )
            axp1.set_title("Node 1", fontsize=10)
            axp1.set_xlabel("Time (second)", fontsize=9)
            axp1.set_ylabel("Mel Frequency (Hz)", fontsize=9)
            cb1p = fig_pdf.colorbar(im1p, ax=axp1, format='%+2.0f dB')
            cb1p.ax.tick_params(labelsize=8)

            # --- Node 2 ---
            axp2 = axs_pdf[1]
            im2p = librosa.display.specshow(
                proc_n2,
                sr=sr,
                cmap='inferno',
                x_axis='time',
                y_axis='mel',
                fmax=sr/2,
                ax=axp2
            )
            axp2.set_title("Node 2", fontsize=10)
            axp2.set_xlabel("Time (second)", fontsize=9)
            axp2.set_ylabel("Mel Frequency (Hz)", fontsize=9)
            cb2p = fig_pdf.colorbar(im2p, ax=axp2, format='%+2.0f dB')
            cb2p.ax.tick_params(labelsize=8)

            plt.tight_layout()
            fig_pdf.savefig(pdf_path, transparent=True, bbox_inches='tight')
            plt.close(fig_pdf)
            print(f"[visualize_acoustic_2node_melspectrogram_and_rgb_advanced] => PDF saved: {pdf_path}")

@visualizer_decorator
def visualize_polar_and_camera_batch(
    batch,
    output_dir="polar_hr_plus_rgb",
    fps_out=10.0,
    save_pdf=True,
    axis_label_fontsize=40,
    tick_label_fontsize=40,
    x_domain=(0, 5.5),   # e.g. from 0..5 seconds or indexes
    y_domain=(80, 100)  # e.g. from 80..100 BPM
):
    """
    Simplified for one polar node only.

    - No node2 logic (since you only have one node).
    - Fix the x-axis to x_domain, y-axis to y_domain.
    - Use tight_layout(), no titles, custom font sizes, transparent PDF.
    """
    user_ids   = batch["user_id"]     
    activities = batch["activity"]
    modality_data = batch["modality_data"]

    # 1) Pull out polar data => shape [B, 1, T_polar]
    polar_data = modality_data.get("polar", None)
    if polar_data is None or not isinstance(polar_data, torch.Tensor):
        print("No 'polar' data found or not a torch.Tensor.")
        return

    B, N_polar, T_polar = polar_data.shape
    if N_polar < 1:
        print("No polar node found.")
        return
    # We assume 1 node => polar_data[b_idx, 0, :]

    # 2) Pull out camera => shape [B, 1, T_cam, H, W, 3]
    depthcam_data = modality_data.get("depthCamera", {})
    rgb_frames = depthcam_data.get("rgb_frames", None)
    if not isinstance(rgb_frames, torch.Tensor):
        print("No 'rgb_frames' or not a torch.Tensor.")
        return

    B2, N_rgb, T_cam, camH, camW, camC = rgb_frames.shape
    if B != B2:
        print("Mismatch in batch dimension for polar vs camera.")
        return

    for b_idx in range(B):
        user_id  = user_ids[b_idx]
        activity = activities[b_idx]
        act_str  = activity[0] if isinstance(activity, list) else activity

        # shape => [T_polar], only node=0
        sample_polar = polar_data[b_idx, 0].cpu().numpy()

        # shape => [T_cam, H, W, 3]
        sample_rgb = rgb_frames[b_idx, 0]

        # We'll assume x-axis is "index" 
        T_p = sample_polar.shape[0]
        x_axis = np.arange(T_p)

        frames_list = []
        out_mp4_name = f"polar_hr_{act_str}_user_{user_id}.mp4"
        out_mp4_path = os.path.join(output_dir, out_mp4_name)

        # # ---------- Build MP4 frames ----------
        # for i_cam in range(T_cam):
        #     fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12,5))
        #     # No suptitle => removed

        #     # Left subplot => static polar data
        #     ax_left.scatter(
        #         x_axis,
        #         sample_polar,
        #         marker='D',    # diamond
        #         s=60,          # bigger
        #         facecolors='white',
        #         edgecolors='black',
        #         linewidths=1.2
        #     )
        #     # No subplot title => removed

        #     # Axis label + custom font size
        #     ax_left.set_xlabel("Time (index)", fontsize=axis_label_fontsize)
        #     ax_left.set_ylabel("Heart Rate (bpm)", fontsize=axis_label_fontsize)
        #     ax_left.tick_params(axis='both', labelsize=tick_label_fontsize)

        #     # Force domain
        #     ax_left.set_xlim(x_domain)
        #     ax_left.set_ylim(y_domain)

        #     # Right subplot => camera
        #     if i_cam < sample_rgb.shape[0]:
        #         rgb_frame = sample_rgb[i_cam].cpu().numpy().astype(np.uint8)
        #         ax_right.imshow(rgb_frame)
        #         ax_right.axis('off')
        #     else:
        #         ax_right.axis('off')

        #     fig.tight_layout()

        #     # Convert fig => RGBA
        #     fig.canvas.draw()
        #     w_fig, h_fig = fig.canvas.get_width_height()
        #     img_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        #     img_argb = img_argb.reshape((h_fig, w_fig, 4))
        #     # ARGB => RGBA
        #     img_rgba = img_argb[..., [1,2,3,0]]
        #     frames_list.append(img_rgba.copy())
        #     plt.close(fig)

        # if not frames_list:
        #     print(f"No frames for user={user_id}, activity={act_str}")
        #     continue

        # # Write MP4
        # height, width = frames_list[0].shape[:2]
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # writer = cv2.VideoWriter(out_mp4_path, fourcc, fps_out, (width, height))
        # for frame_img in frames_list:
        #     frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGBA2BGR)
        #     writer.write(frame_bgr)
        # writer.release()
        # print(f"[visualize_polar_and_camera_batch] => MP4: {out_mp4_path}")

        # ---------- Optional PDF -----------
        if save_pdf and T_p > 0:
            pdf_name = f"polar_hr_{act_str}_user_{user_id}.pdf"
            pdf_path = os.path.join(output_dir, pdf_name)

            fig_pdf, ax_pdf = plt.subplots(figsize=(12,8))
            # No title => removed

            ax_pdf.scatter(
                x_axis,
                sample_polar,
                marker='D',
                s=900,
                facecolors='red',
                edgecolors='red',
                linewidths=1.2
            )
            ax_pdf.set_xlabel("Time (second)", fontsize=axis_label_fontsize)
            ax_pdf.set_ylabel("Heart Rate (bpm)", fontsize=axis_label_fontsize)
            ax_pdf.tick_params(axis='both', labelsize=tick_label_fontsize)

            # Force domain
            ax_pdf.set_xlim(x_domain)
            ax_pdf.set_ylim(y_domain)

            plt.tight_layout()

            # Transparent background
            fig_pdf.patch.set_facecolor("none")
            ax_pdf.set_facecolor("none")

            fig_pdf.savefig(pdf_path, transparent=True, bbox_inches="tight")
            plt.close(fig_pdf)
            print(f"[visualize_polar_and_camera_batch] => PDF: {pdf_path}")


if __name__ == "__main__":
    '''The following is the complete list of valid users, activities, node_ids, and modalities for the dataset.
        'exp_list': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 101, 102, 104, 108, 111, 112, 113, 114, 115, 117, 118, 120, 121, 201, 202, 204, 208, 211, 213, 215, 217, 220, 221, 230],  # Specify which users to filter
        'activity_list' = [
            'sit', 'walk', 'bow', 'sleep', 'dance', 'jog', 'falldown', 'jump', 'jumpingjack',
            'squat', 'lunge', 'turn', 'pushup', 'legraise', 'airdrum', 'boxing', 'shakehead',
            'answerphone', 'eat', 'drink', 'wipeface', 'pickup', 'jumprope', 'moppingfloor',
            'brushhair', 'bicepcurl', 'playphone', 'brushteeth', 'type', 'thumbup', 'thumbdown',
            'makeoksign', 'makevictorysign', 'drawcircleclockwise', 'drawcirclecounterclockwise',
            'stopsign', 'pullhandin', 'pushhandaway', 'handwave', 'sweep', 'clap', 'slide',
            'drawzigzag', 'dodge', 'bowling', 'liftupahand', 'tap', 'spreadandpinch', 'drawtriangle',
            'sneeze', 'cough', 'stagger', 'yawn', 'blownose', 'stretchoneself', 'touchface',
            'handshake', 'hug', 'pushsomeone', 'kicksomeone', 'punchsomeone', 'conversation',
            'gym', 'freestyle'
        ]
        'node_id': [1, 2, 3, 4, 5], 
        'modality': [ 'mmWave', 'IRA', 'uwb', 'ToF', 'polar', 'wifi', 'depthCamera', 'seekThermal','acoustic', 'imu', 'vayyar']
    '''
    # Sample configuration and usage
    dataset_path = "dataset"
    data_config = {
        'exp_list': [5],  # Specify which users to filter
        'activity_list': ['dance'],  
        'node_id': [1], 
        'segmentation_flag': True,
        'modality': ['polar', 'depthCamera'],
        # 'mocap_downsample_num': 6
    }
    
    # Get the DataLoader
    dataset = get_dataset(data_config, dataset_path, data_config.get('mocap_downsample_num', None))
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate
    )

    for batch in dataloader:
        # dump_seekthermal_frames_as_png(batch, output_dir="validation_seekthermal")
        visualize_seekthermal_and_rgb_mosaic_batch_discard_excess(
            batch,
            output_dir='seekthermal_rgb_mosaic_videos',
            fps_out=8.80
        )
        # visualize_3_depth_3_rgb_mosaic_batch_discard_excess(
        #     batch,
        #     output_dir='depth_rgb_mosaic_discard',
        #     fps_out=10
        # )
        visualize_4wifi_time_subcarrier_with_camera(
        batch,
        output_dir='wifi_rgb_mosaic_videos',
        fps_out=10.0,
        BW="40MHz"
        )
        visualize_ira_and_rgb_mosaic_batch_downsample_cam(
        batch,
        output_dir='ira_rgb_mosaic_videos',
        fps_out=6.91
        )
        visualize_mocap_and_rgb_mosaic_batch_downsample_mocap(
        batch,
        output_dir='mocap_rgb_mosaic_videos',
        fps_out=10)
        visualize_tof_and_rgb_mosaic_batch_downsample_tof(
        batch,
        output_dir='tof_rgb_mosaic_videos',
        fps_out=7.32)

        visualize_fmcw_and_rgb_mosaic_batch_raw_fixed_axes(
            batch,
            output_dir='fmcw_rgb_mosaic',
            fps_out=8.81)
        visualize_vayyar_txrx_only_and_camera(
            batch,
            output_dir="vayyar_rgb_mosaic",
            fps_out=10.0
        )
        visualize_acoustic_2node_melspectrogram_and_rgb(
            batch,
            output_dir="acoustic_melspec_plus_rgb",
            fps_out=10.0
        )
        visualize_polar_and_camera_batch(
            batch,
            output_dir="polar_hr_plus_rgb",
            fps_out=10.0
        )
        visualize_imu_four_rows_no_zscore(
            batch,
            output_dir="imu_time_features_plus_rgb",
            fps_out=10.0
        )
        visualize_uwb_and_rgb_in_same_row_with_box(
            batch,
            output_dir="uwb_rgb_same_row_with_box",
            fps_out=10.0)
        break