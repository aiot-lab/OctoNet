import pandas as pd
import struct
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import numpy as np
import cv2
from datetime import datetime, timedelta
import torchaudio
import matplotlib.pyplot as plt
# from utils import get_CSI, preprocess_CSI, get_dfs, get_csi_dfs
import re
import time
from PIL import Image
from tqdm import tqdm
import threading

from func_utils import SubpageInterpolating
# Add global debugger flag at the top of the file, after imports
DEBUGGER = False  # Set to True to enable debug prints

class DataSelector:
    def __init__(self, config, metadata_df, dataset_path=""):
        self.config = config
        self.metadata_df = metadata_df
        self.dataset_path = dataset_path
        # Extract parameters from config
        self.selected_user_ids = config.get('user_list', [])
        self.selected_activities = config.get('activity_list', [])
        self.node_ids = config.get('node_id', [])
        self.modalities = config.get('modality', [])
        self.segmentation_flag = config.get('segmentation_flag')

        # Filter metadata
        self.filtered_metadata = self.filter_metadata()

        # Prepare data paths
        self.pivoted_data = self.prepare_pivoted_data()

        ##############3 START CHANGE: Remove empty columns
        # Remove columns that are entirely empty (all NaN)
        initial_columns = self.pivoted_data.shape[1]
        self.pivoted_data = self.pivoted_data.dropna(axis=1, how='all')
        removed_columns = initial_columns - self.pivoted_data.shape[1]
        if DEBUGGER:
            if removed_columns > 0:
                print(f"Removed {removed_columns} entirely empty columns.")
            else:
                print("No entirely empty columns removed.")
        ##############3 END CHANGE

        ##############3 START CHANGE: Validate rows
        # Validate rows: no missing entries and all files/dirs must have >100 bytes.
        self.pivoted_data = self.validate_and_filter_rows()
        ##############3 END CHANGE

    def filter_metadata(self):
        """Filter the metadata based on user ids and activities."""
        df = self.metadata_df.copy()

        if self.selected_user_ids:
            df = df[df['user_id'].isin(self.selected_user_ids)]
        if self.selected_activities:
            df = df[df['activity'].isin(self.selected_activities)]
        
        return df

    def prepare_pivoted_data(self):
        """Prepare the data in a pivoted format with modalities as separate columns."""
        # We don't need to extract timestamp from the 'data_path' column
        # Since you want to keep the original column names, we will just group by user_id, activity
        grouped = self.filtered_metadata.groupby(['user_id', 'activity'], as_index=False)

        # Initialize an empty list to store the processed data rows
        processed_rows = []

        for _, group in grouped:
            # Create a dictionary for the current row with user_id, activity, and cut_timestamps
            row_data = {
                'user_id': group['user_id'].iloc[0],
                'activity': group['activity'].iloc[0],
                'cut_timestamps': group['cut_timestamps'].iloc[0]
            }

            # Iterate through each modality and add the corresponding data_path to the row
            for modality in self.modalities:
                if modality == 'imu':
                    # Just one column:
                    column_name = f'{modality}_data_path'  # this should be 'imu_data_path'
                    if column_name in group.columns and not group[column_name].isna().all():
                        row_data[column_name] = group[column_name].iloc[0]
                    else:
                        row_data[column_name] = None
                elif modality == 'mocap':
                    # Just one column:
                    column_name = f'{modality}_data_path'  # this should be 'mocap_data_path'
                    if column_name in group.columns and not group[column_name].isna().all():
                        row_data[column_name] = group[column_name].iloc[0]
                    else:
                        row_data[column_name] = None
                else:
                    for node_id in self.node_ids:
                        # Construct the column name for the current modality and node
                        column_name = f'node_{node_id}_{modality}_data_path'

                        # Check if this column exists in the group and if it's not empty
                        if column_name in group.columns and not group[column_name].isna().all():
                            row_data[column_name] = group[column_name].iloc[0]
                        else:
                            row_data[column_name] = None

            # Add the processed row to the list
            processed_rows.append(row_data)

        # Create a new DataFrame from the processed rows
        pivoted_df = pd.DataFrame(processed_rows)
        # Prepend `dataset_path` to each path column if `dataset_path` is not empty
        if self.dataset_path:
            data_path_columns = [c for c in pivoted_df.columns if '_data_path' in c]
            for col in data_path_columns:
                pivoted_df[col] = pivoted_df[col].apply(
                    lambda rel_path: os.path.join(self.dataset_path, rel_path) 
                                    if pd.notna(rel_path) else rel_path
                )
        return pivoted_df

    ##############3 START CHANGE: Add a helper function to check file/dir validity
    def is_valid_file_or_dir(self, path, min_size=100):
        """Check if a file or directory exists and data size >= min_size bytes."""
        if not isinstance(path, str) or not os.path.exists(path):
            return False

        if os.path.isfile(path):
            # Check file size
            size = os.path.getsize(path)
            return size >= min_size
        elif os.path.isdir(path):
            # Check total size of directory
            total_size = 0
            for root, dirs, files in os.walk(path):
                for file in files:
                    fpath = os.path.join(root, file)
                    if os.path.exists(fpath):
                        total_size += os.path.getsize(fpath)
                        if total_size >= min_size:
                            return True
            return total_size >= min_size
        else:
            return False
    ##############3 END CHANGE

    ##############3 START CHANGE: Add a method to validate rows
    def validate_and_filter_rows(self):
        """
        Validate each row. If any required modality cell is missing or invalid,
        discard the entire row. Print out info about discarded rows.
        """
        df = self.pivoted_data.copy()
        # Identify columns that correspond to data paths
        data_path_columns = [c for c in df.columns if '_data_path' in c]

        discarded_rows = []
        valid_rows = []
        for idx, row in df.iterrows():
            user_id = row['user_id']
            activity = row['activity']

            # Check if any required column is NaN
            if any(pd.isna(row[col]) for col in data_path_columns):
                discarded_rows.append((idx, user_id, activity, "Missing entry (NaN)"))
                continue

            # Check file/directory validity
            all_valid = True
            for col in data_path_columns:
                path = row[col]
                if not self.is_valid_file_or_dir(path, min_size=100):
                    all_valid = False
                    discarded_rows.append((idx, user_id, activity, f"Invalid path or size <100 bytes in column {col}"))
                    break

            if all_valid:
                valid_rows.append(row)

        # Print discarded rows summary
        if discarded_rows:
            print("Discarded Rows Summary:")
            for (idx, uid, act, reason) in discarded_rows:
                print(f"  Row index {idx}, user_id {uid}, activity {act} discarded due to: {reason}")
        else:
            print("No rows discarded.")

        return pd.DataFrame(valid_rows)
    ##############3 END CHANGE

    def generate_filtered_csv(self, output_file):
        """Generate a new CSV file with the filtered data."""
        self.pivoted_data.to_csv(output_file, index=False)
        print(f"Filtered CSV file generated: {output_file}")
    
    def get_filtered_data(self):
        return self.pivoted_data

class OctonetDataset(Dataset):
    def __init__(self, metadata_df, transform=None, segmentation_flag = True, mocap_base_dir = None, mocap_downsample_num = None):
        """
        Args:
            csv_file (str): Path to the filtered CSV file with the metadata.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Load the filtered CSV into a pandas DataFrame
        # self.metadata_df = pd.read_csv(csv_file)
        self.metadata_df = metadata_df
        self.transform = transform
        self.segmentation_flag = segmentation_flag
        # Preprocess the CSV data into a dictionary (user_id -> activity -> modality_data_path)
        self.data_dict = self._preprocess_data()
        # Precompute segments for each row
        self.segments_per_row = []
        self.total_segments = 0
        # Initialize cache for depth camera data
        self.depthcamera_cache = {}
        self.depthcamera_cache_lock = threading.Lock()  # For thread safety
        # Initialize cache for SeekThermal data
        self.seekthermal_cache = {}
        self.seekthermal_cache_lock = threading.Lock()
        # ***CHANGED: New cache for acoustic data
        self.acoustic_cache = {}
        self.acoustic_cache_lock = threading.Lock()
        
        
        self.mocap_base_dir = mocap_base_dir
        self.mocap_downsample_num = mocap_downsample_num
         # mocap cache
        self.mocap_cache = {}
        self.mocap_cache_lock = threading.Lock()

        for idx, row in self.metadata_df.iterrows():
            cut_timestamps_str = row['cut_timestamps']
            if isinstance(cut_timestamps_str, str):
                # Convert string representation of list to actual list
                cut_timestamps_list = eval(cut_timestamps_str)
            else:
                # Handle None or NaN by setting to empty list (2025)
                cut_timestamps_list = [] if pd.isna(cut_timestamps_str) else cut_timestamps_str
            
            # Convert strings to datetime objects
            segment_boundaries = [datetime.strptime(ts, "%Y-%m-%d %H.%M.%S.%f") for ts in cut_timestamps_list]
            
            # Number of segments is the number of intervals between boundaries
            num_segments = len(segment_boundaries) - 1 if len(segment_boundaries) > 1 else 1

            # Discard the first and the last segments
            # Effective segments = num_segments - 3, discard the first two segment and the last timestamp
            # If num_segments <= 2, no valid "middle" segments remain
            effective_num_segments = max(num_segments - 3, 0)
            # If segmentation_flag is True but no valid middle segments exist, 
            # force at least one segment to ensure we have data.
            if self.segmentation_flag and effective_num_segments == 0:
                effective_num_segments = 1  # CHANGED LINE: Ensure at least one segment
                
            self.segments_per_row.append({
                'boundaries': segment_boundaries,
                'num_segments': effective_num_segments
            })
            if self.segmentation_flag:
                self.total_segments += effective_num_segments
            else:
                self.total_segments += 1
                
    def _preprocess_data(self):
        """
        Preprocess the data into a dictionary for fast access during __getitem__.
        Each row corresponds to a user_id + activity combination with modality data paths.
        """
        data_dict = {}

        # Iterate over each row of the dataframe to organize it into the dictionary
        for idx, row in self.metadata_df.iterrows():
            user_id = row['user_id']
            activity = row['activity']

            # Initialize the user_id -> activity structure if it doesn't exist
            if user_id not in data_dict:
                data_dict[user_id] = {}

            if activity not in data_dict[user_id]:
                data_dict[user_id][activity] = {}

            # Prepare a dictionary to hold data paths for all modalities
            modality_data = {}
            for col in row.index:
                if '_data_path' in col:  # We're only interested in data path columns
                    parts = col.split('_')
                    if len(parts) >= 3:
                        modality_name = parts[2]  # Extract modality (e.g., 'IRA', 'uwb')
                        data_path = row[col]

                        # If modality already exists, append to the list; otherwise, create a new list
                        if modality_name not in modality_data:
                            modality_data[modality_name] = []
                        modality_data[modality_name].append(data_path)

            # Store the modality data for each user-activity combination
            data_dict[user_id][activity] = modality_data

        return data_dict

    def __len__(self):
        if self.segmentation_flag:
            return self.total_segments
        else:
            return len(self.metadata_df)

    def load_mocap_data(self, csv_path):
        """
        1) Convert CSV path into .npy path by appending "_skeleton.npy".
        2) Load that .npy, which has { "timestamps": [...], "positions": [...] }.
        3) Convert timestamps to the format your segmentation code expects, e.g. "%Y-%m-%d %H.%M.%S.%f".
        4) Return { 'timestamps', 'frames', 'raw_data' } so that 'frames' can be used just like IMU data.
        """
        # 1) Build path to the .npy
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        npy_path = os.path.join(self.mocap_base_dir, base_name + "_skeleton.npy")

        if not os.path.exists(npy_path):
            print(f"[Mocap] Skeleton file not found: {npy_path}")
            return None

        try:
            # 2) Load data from the .npy
            data = np.load(npy_path, allow_pickle=True).item()
            # data["timestamps"] -> e.g. ["2024-05-25T16:16:59.314000", ...] or numeric
            # data["positions"]  -> shape (N_frames, n_joints, 3)

            positions = data["positions"]         # (N_frames, n_joints, 3)
            raw_timestamps = data["timestamps"]   # shape (N_frames,) array or list
            # print('positions.shape',positions.shape)
            # 3) Convert timestamps to the segmentation format: "%Y-%m-%d %H.%M.%S.%f"
            converted_timestamps = []
            for t in raw_timestamps:
                # If your data is ISO 8601 "2024-05-25T16:16:59.314000"
                #   -> parse with datetime.fromisoformat
                #   -> then reformat with strftime("%Y-%m-%d %H.%M.%S.%f")

                if isinstance(t, str):
                    # parse as ISO-8601
                    dt = datetime.fromisoformat(t)   # or use datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f") if strictly that pattern
                    # reformat
                    t_str = dt.strftime("%Y-%m-%d %H.%M.%S.%f")
                    converted_timestamps.append(t_str)
                elif isinstance(t, (float, int)):
                    # If it's numeric seconds, interpret as offset from 0 or something
                    # Possibly you just store it as the segmentation code can parse floats
                    # but if you want them consistent, pick a "zero" reference or do something like:
                    dt = datetime(1970,1,1) + timedelta(seconds=float(t))
                    t_str = dt.strftime("%Y-%m-%d %H.%M.%S.%f")
                    converted_timestamps.append(t_str)
                else:
                    # Fallback or handle other cases
                    converted_timestamps.append(str(t))

            return {
                "modality": "mocap",
                "timestamps": converted_timestamps,   # now a list of matching-format strings
                "frames": positions,
                "raw_data": data
            }

        except Exception as e:
            print(f"[Mocap] Error loading {npy_path}: {e}")
            return None

            
    def load_data_from_pickle(self, file_path):
        data = []
        with open(file_path, 'rb') as f:
            while True:
                try:
                    item = pickle.load(f)
                    data.append(item)
                except EOFError:
                    # We've reached the end of the file
                    break
                except pickle.UnpicklingError:
                    # If we hit an invalid load key, break or pass
                    # Decide whether to skip only the current read or skip the entire file.
                    print(f"Warning: Unpickling error encountered in {file_path}. Skipping invalid data.")
                    break
        return data
    def load_ira_data(self, data_path):
        if os.path.exists(data_path):
            data = self.load_data_from_pickle(data_path)
            
            timestamps = []
            frames = []
            ambient_temperatures = []
            
            for entry in data:
                timestamps.append(entry['timestamp'])
                frames.append(entry['Detected_Temperature'])
                ambient_temperatures.append(entry['Ambient_Temperature'])
            
            frames = np.array(frames)
            
            # Return data as a dictionary
            return {
                'timestamps': timestamps,
                'ambient_temperatures': ambient_temperatures,
                'frames': frames,
                'raw_data': data
            }
        else:
            return None  # Handle missing data
        
    def load_imu_data(self, data_path):
        if os.path.exists(data_path):
            data = self.load_data_from_pickle(data_path)
            
            timestamps = []
            frames = []
            
            for entry in data:
                # Validate and extract timestamps
                if 'timestamps' in entry and isinstance(entry['timestamps'], list):
                    timestamps.extend(entry['timestamps'])
                else:
                    print(f"Invalid or missing 'timestamps' in entry: {entry}")
                
                # Validate and extract data
                if 'data' in entry:
                    entry_data = np.array(entry['data'])
                    # print(f"Entry data shape before appending: {entry_data.shape}")
                    frames.append(entry_data)
                else:
                    print(f"Missing 'data' in entry: {entry}")

            # Convert frames to NumPy array
            if len(frames) == 1:
                frames = frames[0]  # Use directly without extra dimension
            else:
                frames = np.array(frames)  # Convert list to NumPy array for multiple entries
            
            # print(f"Frames array shape (after processing): {frames.shape}")
            
            return {
                'timestamps': timestamps,
                'frames': frames,
                'raw_data': data
            }
        else:
            return None 
        
    def load_vayyar_data(self, data_path):
        if os.path.exists(data_path):
            data = self.load_data_from_pickle(data_path)
            
            timestamps = []
            frames = []
            
            for entry in data:
                if 'timestamps' in entry and isinstance(entry['timestamps'], (list, np.ndarray)):
                    ts_array = np.array(entry['timestamps']).flatten()  # Convert to 1D array
                    timestamps.extend(ts_array.tolist())
                else:
                    print(f"Invalid or missing 'timestamps' in entry: {entry}")

                
                # Validate and extract data
                if 'data' in entry:
                    entry_data = np.array(entry['data'])
                    # print(f"Entry data shape before appending: {entry_data.shape}")
                    frames.append(entry_data)
                else:
                    print(f"Missing 'data' in entry: {entry}") 
            

            # Convert frames to NumPy array
            if len(frames) == 1:
                frames = frames[0]  # Use directly without extra dimension
            else:
                frames = np.array(frames)  # Convert list to NumPy array for multiple entries
            
            # Now reorder from (400, 100, T) => (T, 400, 100)
            if frames.ndim == 3 and frames.shape[0] == 400 and frames.shape[1] == 100:
                frames = frames.transpose((2, 0, 1))  # => (T, 400, 100)
            else:
                print(f"[Vayyar] Unexpected frames shape: {frames.shape}")
                # print(f"Frames array shape (after processing): {frames.shape}")
            
            return {
                'timestamps': timestamps,
                'frames': frames,
                'raw_data': data
            }
        else:
            return None 
        
    def load_wifi_data(self, data_path):
        if os.path.exists(data_path):
            data = self.load_data_from_pickle(data_path)
            
            timestamps = []
            frames = []
            
            for entry in data:
                # Check if 'timestamp' is a list
                if isinstance(entry['timestamp'], list):
                    # Extend the timestamps list with the entries
                    timestamps.extend(entry['timestamp'])
                    # Assuming 'entry['data']' corresponds to multiple frames, extend frames as well
                    frames.extend(entry['data'])  # Adjust this if 'data' needs to be handled differently
                else:
                    # Append the single timestamp
                    timestamps.append(entry['timestamp'])
                    frames.append(entry['data'])
            
            frames = np.array(frames)
            arr = np.array(frames)
            # Return data as a dictionary
            return {
                'timestamps': timestamps,
                'frames': frames,
                'raw_data': data
            }
        else:
            return None  # Handle missing data


    def load_uwb_data(self, data_path):
        if os.path.exists(data_path):
            data = self.load_data_from_pickle(data_path)
            
            timestamps = []
            frames = []
            
            for entry in data:
                timestamps.append(entry['timestamp'])
                frames.append(entry['frame'])
            arr = np.array(frames)
            frames = np.array(frames)
            
            # Return data as a dictionary
            return {
                'timestamps': timestamps,
                'frames': frames,
                'raw_data': data
            }
        else:
            return None  # Handle missing data

    def load_mmwave_data(self, data_path):
        if os.path.exists(data_path):
            data = self.load_data_from_pickle(data_path)

            timestamps = []
            all_points = []  # CHANGE: We'll store a list of Nx4 arrays here

            # Extract timestamps and frames
            for entry in data:
                timestamp = entry.get('timestamp')
                frame = entry.get('data')
                
                if timestamp is not None and frame is not None:
                    # CHANGE: 'frame' is expected to be a dict with keys like 'x', 'y', 'z', 'velocity'.
                    # We must confirm the structure. Assuming frame['x'], frame['y'], etc. are NumPy arrays.
                    # Combine them into Nx4 array.

                    # If 'frame' is a dict like:
                    # {
                    #     'numObj': int,
                    #     'x': np.array([...]),
                    #     'y': np.array([...]),
                    #     'z': np.array([...]),
                    #     'velocity': np.array([...])
                    # }
                    # We'll stack them along last dimension to shape [numObj, 4].
                    
                    if all(k in frame for k in ['x', 'y', 'z', 'velocity']):
                        x = frame['x']
                        y = frame['y']
                        z = frame['z']
                        v = frame['velocity']

                        # Ensure they are all the same length
                        num_points = len(x)
                        # Stack them
                        points = np.stack([x, y, z, v], axis=-1)  # shape [num_points,4]

                        timestamps.append(timestamp)
                        all_points.append(points)
                    else:
                        # If frame keys are missing, skip or handle as empty
                        timestamps.append(timestamp)
                        all_points.append(np.zeros((0,4), dtype=np.float32))
                else:
                    # Missing data, append empty
                    timestamps.append(timestamp if timestamp is not None else datetime.now())
                    all_points.append(np.zeros((0,4), dtype=np.float32))
            
            # Return data as a dictionary
            return {
                'timestamps': timestamps,
                'points': all_points,  # CHANGE: return 'points' instead of 'frames'
                'raw_data': data
            }
        else:
            return None  # Handle missing data path


    def load_tof_data(self, data_path):
        if os.path.exists(data_path):
            data = self.load_data_from_pickle(data_path)
            
            timestamps = []
            tof_depths = []
            tof_bins_list = []
            
            # Extract timestamps and frames
            for entry in data:
                timestamp = entry.get('timestamp')
                tof_depth = entry.get('tof_depth')
                tof_bins = entry.get('tof_bins')
                
                if timestamp is not None and tof_depth is not None and tof_bins is not None:
                    timestamps.append(timestamp)
                    tof_depths.append(tof_depth)
                    tof_bins_list.append(tof_bins)
                else:
                    # Handle missing data within the entry if necessary
                    pass
            
            # Convert lists to numpy arrays
            tof_depths = np.array(tof_depths)
            tof_bins_list = np.array(tof_bins_list)
            
            # Return data as a dictionary
            return {
                'timestamps': timestamps,
                'tof_depths': tof_depths,
                'tof_bins': tof_bins_list,
                'raw_data': data
            }
        else:
            return None  # Handle missing data path

    def load_polar_data(self, data_path):
        if os.path.exists(data_path):
            data = self.load_data_from_pickle(data_path)
            
            timestamps = []
            frames = []
            
            # Extract timestamps and frames
            for entry in data:
                timestamp = entry.get('timestamp')
                frame = entry.get('data')
                
                if timestamp is not None and frame is not None:
                    timestamps.append(timestamp)
                    frames.append(frame)
                else:
                    # Handle missing data within the entry if necessary
                    pass
            
            frames = np.array(frames)
            
            # Return data as a dictionary
            return {
                'timestamps': timestamps,
                'frames': frames,
                'raw_data': data
            }
        else:
            return None  # Handle missing data path

    def load_seekthermal_data(self, data_path):
        # Check if data_path is already cached
        with self.seekthermal_cache_lock:
            if data_path in self.seekthermal_cache:
                if DEBUGGER:
                    print(f"[SeekThermal] Using cached data for: {data_path}")
                return self.seekthermal_cache[data_path]
        if os.path.exists(data_path):
            timestamps = []
            thermal_images = []
            
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith('.png') and 'thermal' in file:
                        image_path = os.path.join(root, file)
                        # Extract timestamp from the filename (assuming format: 'thermal_<timestamp>.png')
                        try:
                            timestamp_str = file.replace("thermal_", "").replace(".png", "")
                            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                        except ValueError:
                            print(f"Error parsing timestamp from file: {file}")
                            continue
            
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        if image is not None:
                            # Convert the image to temperature using the png_to_temperature function
                            temperature = self.png_to_temperature(image, min_temp=15, max_temp=50)
                            
                            # ========== Start Normalization ========== 
                            # 1) Map to [0,1]
                            temperature = (temperature - 15.0) / (50.0 - 15.0)
                            temperature = np.clip(temperature, 0.0, 1.0)

                            # 2) Convert to float32
                            temperature = temperature.astype(np.float32)
                            # ========== End Normalization ==========

                            timestamps.append(timestamp)
                            thermal_images.append(temperature)
                        else:
                            print(f"Failed to read image: {file}")
                            
            # If there are any thermal images, return data as a dictionary
            if len(thermal_images) > 0:
                # NEW: Sort frames by timestamp
                combined = list(zip(timestamps, thermal_images))           # NEW
                combined.sort(key=lambda x: x[0])                          # NEW
                timestamps, thermal_images = zip(*combined)                # NEW
                timestamps = list(timestamps)                              # NEW
                thermal_images = list(thermal_images)                      # NEW
                data = {
                'timestamps': timestamps,
                'thermal_images': thermal_images
                }
                
                # Cache the loaded data
                with self.seekthermal_cache_lock:
                    self.seekthermal_cache[data_path] = data
                
                return data
            else:
                return None
        else:
            return None  # Handle missing data path

    def png_to_temperature(self, img, min_temp, max_temp):
        img = img.astype(np.float32) 
        temperature = (img / 255.0) * (max_temp - min_temp) + min_temp
        return temperature

    def load_depthcamera_data(self, data_path, 
                            downsample_size=(240, 320),
                            target_fps=1):
        """
        Loads and optionally downsamples depth-camera data in both space and time.
        - Spatial downsample via `cv2.resize` to `downsample_size`.
        - Temporal downsample via `target_fps` if original FPS is higher.

        Args:
            data_path       : directory containing .png depth images and an .mp4 RGB video
            downsample_size : (height, width) for spatial resizing, or None to skip
            target_fps      : (float) if original video > target_fps, skip frames 
                            to approximate target_fps in both RGB and depth.

        Returns:
            data dict with { 'timestamps', 'depth_images', 'rgb_frames' } or None
        """

        # Check cache
        with self.depthcamera_cache_lock:
            if data_path in self.depthcamera_cache:
                if DEBUGGER:
                    print(f"[DepthCamera] Using cached data for: {data_path}")
                return self.depthcamera_cache[data_path]

        if not os.path.exists(data_path):
            print(f"[DepthCamera] Data path does not exist: {data_path}")
            return None

        if DEBUGGER:
            print(f"Loading DepthCamera data from: {data_path}")
        start_total = time.time()

        # 1) Gather PNG depth files + an MP4 file
        depth_image_files = []
        video_path = None
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.png') and 'depth' in file:
                    depth_image_files.append(os.path.join(root, file))
                elif file.endswith('.mp4'):
                    video_path = os.path.join(root, file)

        # if not video_path:
        #     print(f"[DepthCamera] No .mp4 file found in {data_path}")
        #     return None

        # Sort depth files in alphabetical order => typically time order
        depth_image_files.sort()

        # 2) Parse timestamps from file names
        timestamps = []
        for file in depth_image_files:
            filename = os.path.basename(file)
            ts_str = filename.replace('depth_', '').replace('.png', '')
            try:
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                timestamps.append(ts)
            except ValueError:
                if DEBUGGER:
                    print(f"[DepthCamera] Skipping invalid timestamp: {filename}")

        # 3) Read the video into a list of frames
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)  # e.g. 30.0
        rgb_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Spatial resize
            if downsample_size is not None:
                frame_rgb = cv2.resize(frame_rgb,
                                    (downsample_size[1], downsample_size[0]),
                                    interpolation=cv2.INTER_NEAREST)
            
            # === ===
            # 1) 
            frame_rgb = frame_rgb.astype(np.float32) / 255.0

            # 2) 
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            # (x - mean[c]) / std[c]
            frame_rgb -= mean  
            frame_rgb /= std
            # === HIGHLIGHT (1) END ===
            rgb_frames.append(frame_rgb)
        cap.release()

        # 4) Read depth PNG images
        depth_images = []
        for file in depth_image_files:
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            if img is None:
                if DEBUGGER:
                    print(f"[DepthCamera] Failed to read {file}")
                continue
            # Convert raw to depth
            depth = self.png_to_depth(img, min_dist=0, max_dist=10)
            ## New segmentation:
            
            # (2) Crop and map to [0,1]
            # depth = np.clip(depth, 0, 10) / 10.0  
             # (A) Human segmentation
            mask = self.naive_human_segmentation(depth,
                                                 min_human_dist=1.0,
                                                 max_human_dist=4.0,
                                                 morph_kernel_size=5)
            # Set background=0, preserve original depth in human regions
            depth_segmented = depth * mask

            # (B) Map to [0,1]
            depth_segmented = np.clip(depth_segmented, 0, 10)/10.0

            # (C) (Optional) If you want to subtract 0.485 and divide by 0.229, write:
            depth_segmented -= 0.485
            depth_segmented /= 0.229
            depth = depth_segmented
            #------------------------------
            # depth -= 0.485
            # depth /= 0.229
             # If you want to keep it single-channel, don't stack it into 3 channels
             # This depth.shape is still [height, width]

            # Spatial resize
            if downsample_size is not None:
                depth = cv2.resize(depth,
                                (downsample_size[1], downsample_size[0]),
                                interpolation=cv2.INTER_NEAREST)
            depth_images.append(depth)

        # Align all lists if there's a mismatch
        min_len = min(len(rgb_frames), len(depth_images), len(timestamps))
        rgb_frames  = rgb_frames[:min_len]
        depth_images = depth_images[:min_len]
        timestamps   = timestamps[:min_len]

        # 5) Temporal downsampling – if original_fps is known and > target_fps
        if target_fps and original_fps > target_fps:
            factor = int(round(original_fps / target_fps))
            if factor < 1: 
                factor = 1
            # Slice each list with step = factor
            rgb_frames  = rgb_frames[::factor]
            depth_images = depth_images[::factor]
            timestamps   = timestamps[::factor]

        # Prepare output
        data = {
            'timestamps': timestamps,
            'depth_images': depth_images,
            'rgb_frames': rgb_frames
        }

        # Cache
        with self.depthcamera_cache_lock:
            self.depthcamera_cache[data_path] = data

        return data

    def naive_human_segmentation(self, depth_img,
                                 min_human_dist=0.3,
                                 max_human_dist=3.0,
                                 morph_kernel_size=5):
        """
        1) Create a mask: only keep depths within [min_human_dist, max_human_dist] range
        2) Use morphological operations to remove isolated noise points

        Returns: segmented_mask, same shape as depth_img, dtype float32 (0 or 1)
        """
        # Generate initial mask (True/False)
        mask = (depth_img >= min_human_dist) & (depth_img <= max_human_dist)

        # Convert to uint8 for OpenCV morphological operations
        mask_uint8 = mask.astype(np.uint8)

        # Morphological opening: erode then dilate to remove small noise
        kernel = np.ones((morph_kernel_size, morph_kernel_size), dtype=np.uint8)
        mask_clean = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

        # Convert back to float32 (0,1)
        mask_clean = mask_clean.astype(np.float32)
        return mask_clean

    def png_to_depth(self, img, min_dist, max_dist):
        """
        Converts a depth image from raw pixel values to actual depth distances.
        Args:
            img (numpy.ndarray): The raw depth image.
            min_dist (float): The minimum distance represented in the depth image.
            max_dist (float): The maximum distance represented in the depth image.
        Returns:
            numpy.ndarray: The depth image converted to depth distances.
        """
        # Ensure the image is in float32 format
        img = img.astype(np.float32)
        
        # Scale to the desired distance range
        depth = (img / 255.0) * (max_dist - min_dist) + min_dist
        # return depth
        return depth

    
    def load_acoustic_data(self, data_path):
        """
        Loads and processes acoustic data from a .wav file.
        Converts the raw audio waveform into a dictionary containing 'timestamps' and 'waveform'.
    
        Args:
            data_path (str): Path to the .wav file.
    
        Returns:
            dict or None: Dictionary containing waveform and timestamps.
        """
        # Check if data_path is already cached
        with self.acoustic_cache_lock:
            if data_path in self.acoustic_cache:
                if DEBUGGER:
                    print(f"[Acoustic] Using cached data for: {data_path}")
                return self.acoustic_cache[data_path]

        if os.path.exists(data_path):
            try:
                # Locate the corresponding .log file
                base_dir = os.path.dirname(data_path)
                base_name = os.path.splitext(os.path.basename(data_path))[0]
                log_file = os.path.join(base_dir, f"{base_name}.log")
                

                # Initialize start_time
                start_time = None

                # Read the log file and attempt to extract start_time
                with open(log_file, 'r') as f:
                    for line in f:
                        # **Format 1 Detection and Parsing**
                        if "NTP Time:" in line and "Start Recording" in line:
                            try:
                                # Extract the NTP Time using regex
                                ntp_time_match = re.search(r"NTP Time:\s*\[(.*?)\]: Start Recording", line)
                                if ntp_time_match:
                                    ntp_time_str = ntp_time_match.group(1)
                                    start_time = datetime.strptime(ntp_time_str, "%Y-%m-%d %H:%M:%S.%f")
                                    break  # Exit after finding the start time
                            except Exception as e:
                                print("Error parsing NTP Time from log file {log_file}: {e}") 
                                continue

                        # **Format 2 Detection and Parsing**
                        elif "Start Recording" in line and "NTP Time:" not in line:
                            try:
                                # Assume the timestamp is within square brackets at the beginning
                                # Example: 2024-05-25 16:46:23.997 | INFO     | __main__:<module>:19 - Start Recording ...
                                timestamp_str = line.split('|')[0].strip()
                                start_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                                break  # Exit after finding the start time
                            except Exception as e:
                                print(f"Error parsing Start Recording time from log file {log_file}: {e}") 
                                continue

                if not start_time:
                    print(f"Start time not found in log file: {log_file}") 
                    return None

                # Load audio
                waveform, sample_rate = torchaudio.load(data_path)  # Load audio
                target_sample_rate = 16000
                if sample_rate != target_sample_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                    waveform = resampler(waveform)
                    sample_rate = target_sample_rate

                # Convert waveform to numpy for easier manipulation
                waveform_np = waveform.squeeze(0).numpy()  # Shape: [samples]

                # Calculate timestamps for each sample
                timestamps = [start_time + timedelta(seconds=i / sample_rate) for i in range(len(waveform_np))]

                # Convert waveform back to tensor
                waveform_tensor = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0)  # Shape: [1, samples]

                # ============ Normalization here ============ 
            
                # (a) Amplitude normalization (optional, if you want to limit values to [-1,1])
                # max_val = waveform_tensor.abs().max()
                # if max_val > 1e-8:
                #     waveform_tensor /= max_val
                
                # (b) Zero mean, unit variance
                mean = waveform_tensor.mean()
                std = waveform_tensor.std() + 1e-8  # Prevent division by 0
                waveform_tensor = (waveform_tensor - mean) / std
                
                # ============ End of normalization ============

                # ***CHANGED: Store loaded data in acoustic_cache
                with self.acoustic_cache_lock:
                    self.acoustic_cache[data_path] = {
                    'modality':'acoustic',
                    'waveform': waveform_tensor,
                    'timestamps': timestamps
                }
                # Return as a dictionary
                return {
                    'modality':'acoustic',
                    'waveform': waveform_tensor,
                    'timestamps': timestamps
                }
            except Exception as e:
                print(f"Error loading acoustic data from {data_path}: {e}")  # ####################### Changed to logger
                return None
        else:
            print(f"Acoustic data path does not exist: {data_path}")  # ####################### Changed to logger
            return None
        
    def get_segment_index(self, idx):
        # If segmentation_flag is off, idx corresponds directly to a row
        if not self.segmentation_flag:
            return idx, None
        # If segmentation_flag is on, determine which row and which segment
        seg_count = 0
        for row_i, seg_info in enumerate(self.segments_per_row):
            if seg_count + seg_info['num_segments'] > idx:
                segment_idx = idx - seg_count + 1
                return row_i, segment_idx
            seg_count += seg_info['num_segments']
        raise IndexError("Requested segment index out of range.")

    def downsample_mocap_frames(self, frames, num_sampled_frames=6):
        """
        Downsample the mocap frames array (T, 20, 3) to 'num_sampled_frames' equally spaced frames.
        """
        T = frames.shape[0]
        if T == 0:
            return frames  # nothing to sample

        # e.g. pick 6 frames from [0..T-1], inclusive
        # Round to nearest int to avoid float indexing
        sampled_idxs = np.round(np.linspace(0, T - 1, num_sampled_frames)).astype(int)
        # Catch edge case: if T < num_sampled_frames
        sampled_idxs = np.clip(sampled_idxs, 0, T - 1)
        
        frames_downsampled = frames[sampled_idxs]  # shape => (num_sampled_frames, 20, 3)
        return frames_downsampled 

    def segment_modality_data(self, modality_data, segment_boundaries, segment_idx, user_id):
        # If segment_idx is None, that means no segmentation should occur. Just return the entire data.
        if segment_idx is None:
            if modality_data['modality'] == 'acoustic' and modality_data is not None:
                # Directly return the raw waveform and timestamps without converting to Mel Spectrogram
                waveform = modality_data['waveform']
                timestamps = modality_data['timestamps']
                return {
                    'waveform': waveform,
                    'timestamps': timestamps
                }
            elif modality_data['modality'] == 'mocap' and self.mocap_downsample_num is not None:
                print('-----------------------')
                frames = modality_data['frames']
                modality_data['frames'] = self.downsample_mocap_frames(frames, self.mocap_downsample_num)
                return modality_data
            else:
                return modality_data
            
            
         
            
        actual_segment_idx = segment_idx + 1
        start_time = segment_boundaries[actual_segment_idx]
        end_time = segment_boundaries[actual_segment_idx+1]
        if DEBUGGER:
            print('user_id:', user_id)
            print('actual_segment_idx:', actual_segment_idx)
            print('start_time:', start_time)
            print('end_time:', end_time)
        
        
        timestamps = modality_data['timestamps']
        
        # Convert timestamps if they are strings
        converted_ts = []
        for t in timestamps:
            if isinstance(t, str):
                converted_ts.append(datetime.strptime(t, "%Y-%m-%d %H.%M.%S.%f"))
            else:
                converted_ts.append(t)
        timestamps = converted_ts

        indices = [i for i, t in enumerate(timestamps) if t >= start_time and t < end_time]

        if len(indices) == 0:
            empty_result = {'timestamps': []}
            # Return an empty torch tensor for frames to ensure consistency, For IRA only
            if 'ambient_temperatures' in modality_data:
                # This is IRA => return shape (0,24,32)
                empty_frames = torch.zeros((0, 24, 32), dtype=torch.float32)
                print("Got empty IRA frames with shape (0,24,32)")
                return {
                    'timestamps': [],
                    'frames': empty_frames
                }
            elif 'depth_images' in modality_data or 'rgb_frames' in modality_data:
                # Return empty lists for each to keep consistent
                print("Got empty depthCamera frames.")
                empty_result['depth_images'] = torch.zeros((0, 240, 320), dtype=torch.float32)
                empty_result['rgb_frames'] = torch.zeros((0, 240, 320, 3), dtype=torch.float32)

                return {
                    'timestamps': [],
                    'depth_images': empty_result['depth_images'],
                    'rgb_frames': empty_result['rgb_frames']
                    }
            elif 'thermal_images' in modality_data:
                # Return empty lists for each to keep consistent
                print("Got empty thermalCamera frames.")
                empty_result['thermal_images'] = torch.zeros((0, 240, 320), dtype=torch.float32)
                
                return {
                    'timestamps': [],
                    'thermal_images': empty_result['thermal_images']
                    }
            # =============== Acoustic Case ===============
            elif 'waveform' in modality_data:
                # For acoustic, a typical shape is [1, samples], so let's do [1, 0].
                print("Got empty acoustic waveform.")
                empty_result['waveform'] = torch.zeros((1, 1024), dtype=torch.float32) # Which will solve the probelm of empty segment of acoustic using torch.zeros((1, 1024) rather than torch.zeros((1, 0).
                return {
                    'modality':'acoustic',
                    'timestamps': [],
                    'waveform': empty_result['waveform']
                }

        seg_data = {}
        seg_data['timestamps'] = [timestamps[i] for i in indices]

        if 'points' in modality_data and modality_data['points'] is not None:
            # 'points' is a list of Nx4 arrays, one per timestamp
            seg_data['points'] = [modality_data['points'][i] for i in indices]
        if 'frames' in modality_data and modality_data['frames'] is not None:
            seg_data['frames'] = modality_data['frames'][indices]
        if 'ambient_temperatures' in modality_data and modality_data['ambient_temperatures'] is not None:
            ambient = np.array(modality_data['ambient_temperatures'])
            seg_data['ambient_temperatures'] = ambient[indices].tolist()

        if 'thermal_images' in modality_data and modality_data['thermal_images'] is not None:
            seg_data['thermal_images'] = [modality_data['thermal_images'][i] for i in indices]

        # if 'tof_depths' in modality_data and modality_data['tof_depths'] is not None:
        #     seg_data['tof_depths'] = modality_data['tof_depths'][indices]
        if 'tof_bins' in modality_data and modality_data['tof_bins'] is not None:
            seg_data['tof_bins'] = modality_data['tof_bins'][indices]

        if 'depth_images' in modality_data and modality_data['depth_images'] is not None:
            
            seg_data['depth_images'] = [modality_data['depth_images'][i] for i in indices]
        if 'rgb_frames' in modality_data and modality_data['rgb_frames'] is not None:
            seg_data['rgb_frames'] = [modality_data['rgb_frames'][i] for i in indices]
        # Handle acoustic data: segment waveform without padding
        if 'waveform' in modality_data and modality_data['waveform'] is not None:
            # return {
            #     'waveform': None,
            #     'timestamps': []
            # }
            waveform = modality_data['waveform']  # torch.Tensor [1, samples]
            timestamps = modality_data['timestamps']

            # Compute the time offset for segmentation
            first_timestamp = timestamps[0]
            if start_time < first_timestamp:
                print(f"[{modality}] Segment start_time {start_time} is before acoustic data start_time {first_timestamp}")  # ####################### Changed to logger
                return {
                    'waveform': None,
                    'timestamps': []
                }
            # Find the indices corresponding to the segment
            indices = [i for i, t in enumerate(timestamps) if t >= start_time and t < end_time]

            if len(indices) == 0:
                return {
                    'waveform': None,
                    'timestamps': []
                }

            # Slice the waveform
            segmented_waveform = waveform[:, indices[0]:indices[-1]+1]  # [1, segment_samples]
            segmented_timestamps = [timestamps[i] for i in indices]

            # Return the segmented waveform and timestamps
            seg_data = {
                'waveform': segmented_waveform,
                'timestamps': segmented_timestamps
            }
        # For mocap downsampling
        if 'modality' in modality_data and modality_data['modality'] == 'mocap' and 'frames' in seg_data and self.mocap_downsample_num is not None:
            seg_data['frames'] = self.downsample_mocap_frames(seg_data['frames'], self.mocap_downsample_num)
        return seg_data

        
        
    #     return sample
    def __getitem__(self, idx):
        if self.segmentation_flag:
            ################# START OF CHANGES #################
            # Find row and segment index among the effective (middle) segments
            row_idx, segment_idx = self.get_segment_index(idx)
            
            ################# END OF CHANGES #################
        else:
            row_idx = idx
            segment_idx = None

        row = self.metadata_df.iloc[row_idx]
        user_id = row['user_id']
        activity = row['activity']

        sample = {
            'user_id': user_id,
            'activity': activity,
            'modality_data': {}
        }

        cut_timestamps_str = row['cut_timestamps']
        if isinstance(cut_timestamps_str, str):
            cut_timestamps_list = eval(cut_timestamps_str)
        else:
            # Handle None or NaN by setting to empty list (2025)
            cut_timestamps_list = [] if pd.isna(cut_timestamps_str) else cut_timestamps_str

        # Convert strings to datetime objects
        segment_boundaries = []
        for ts in cut_timestamps_list:
            try:
                segment_boundaries.append(datetime.strptime(ts, "%Y-%m-%d %H.%M.%S.%f"))
            except:
                continue
        # For depthcam partial loading purposes
        start_time, end_time = None, None
        if self.segmentation_flag and len(segment_boundaries) > 2:
            actual_segment_idx = segment_idx + 1
            start_time = segment_boundaries[actual_segment_idx]
            end_time = segment_boundaries[actual_segment_idx+1]
        # Check if we have valid "middle" segments:
        # Valid middle segments exist only if len(segment_boundaries) > 2.
        # If segmentation_flag is True but no valid segments, we just return entire data (no segmentation).
        # if self.segmentation_flag and len(segment_boundaries) > 2:
        #     row_idx, segment_idx = self.get_segment_index(idx)
        #     # Reload row for selected segment
        #     row = self.metadata_df.iloc[row_idx]
        # else:
        #     # No valid segments available, return entire data
        #     # Instead of performing segmentation, we just set segment_idx to None.
        #     row_idx = idx
        #     segment_idx = None
        for col_name in self.metadata_df.columns:
            # Handle IMU modality separately
            if col_name == 'imu_data_path':
                data_path = row[col_name]
                if pd.isna(data_path):
                    continue

                data = self.load_imu_data(data_path)

                if self.segmentation_flag:
                    if len(segment_boundaries) > 2 and data is not None:
                        data = self.segment_modality_data(data, segment_boundaries, segment_idx, user_id)

                # Add the IMU data under the 'imu' modality key
                if 'imu' not in sample['modality_data']:
                    sample['modality_data']['imu'] = []
                sample['modality_data']['imu'].append(data)

            elif col_name == 'mocap_data_path':
                data_path = row[col_name]
                if pd.isna(data_path):
                    continue
                mocap_data = self.load_mocap_data(data_path)
                if self.segmentation_flag and len(segment_boundaries) > 2 and mocap_data is not None:
                    mocap_data = self.segment_modality_data(mocap_data, segment_boundaries, segment_idx, user_id)
                if 'mocap' not in sample['modality_data']:
                    sample['modality_data']['mocap'] = []
                sample['modality_data']['mocap'].append(mocap_data)
            
            # Handle other modalities with node_{node_id}_{modality}_data_path format
            elif '_data_path' in col_name:
                data_path = row[col_name]
                if pd.isna(data_path):
                    continue

                parts = col_name.split('_')
                # Expecting at least 3 parts: node, node_id, modality, data_path
                if len(parts) >= 3:
                    modality_name = parts[2]
                    if modality_name == 'IRA':
                        data = self.load_ira_data(data_path)
                    elif modality_name == 'uwb':
                        data = self.load_uwb_data(data_path)
                    elif modality_name == 'mmWave':
                        data = self.load_mmwave_data(data_path)
                    elif modality_name == 'ToF':
                        data = self.load_tof_data(data_path)
                    elif modality_name == 'polar':
                        data = self.load_polar_data(data_path)
                    elif modality_name == 'seekThermal':
                        data = self.load_seekthermal_data(data_path)
                    elif modality_name == 'depthCamera':
                        data = self.load_depthcamera_data(data_path)
                    elif modality_name == 'acoustic':
                        data = self.load_acoustic_data(data_path)
                    elif modality_name == 'wifi':
                        data = self.load_wifi_data(data_path)
                    elif modality_name == 'vayyar':
                        data = self.load_vayyar_data(data_path)
                    else:
                        data = None

                    if self.segmentation_flag:
                        if len(segment_boundaries) > 2 and data is not None:
                            data = self.segment_modality_data(data, segment_boundaries, segment_idx, user_id)
                        # If no valid middle segments, return data unsegmented

                    if modality_name not in sample['modality_data']:
                        sample['modality_data'][modality_name] = []
                    sample['modality_data'][modality_name].append(data)

        if self.transform:
            sample = self.transform(sample)
        return sample

def custom_collate(batch):
    user_ids = []
    activities = []
    modality_data = {}

    for sample in batch:
        user_ids.append(sample['user_id'])
        activities.append(sample['activity'])
        
        for modality, data_list in sample['modality_data'].items():
            if modality not in modality_data:
                modality_data[modality] = []
            modality_data[modality].append(data_list)  # data_list is a list of data dicts

    # For mmWave, we now have data in 'points' format: a variable-length Nx4 array.
    # We will NOT attempt to stack them into a single tensor. Instead, we'll keep them as is.
    # That means each element in the batch for mmWave is a list of dicts (with 'points' keys).
    ##############3 END CHANGE

    # Example handling of mmWave modality:
    # Instead of trying to stack, we just leave them as a list of arrays.
    # if 'mmWave' in modality_data:
    #     mmwave_batch = []
    #     for sample_data in modality_data['mmWave']:
    #         # sample_data is a list (one entry per node)
    #         # Each entry is {'points': np.array([...])} or None
    #         node_arrays = []
    #         for node_item in sample_data:
    #             if node_item is not None and 'points' in node_item:
    #                 # points: np.array of shape [N,4]
    #                 node_arrays.append(node_item['points'])
    #             else:
    #                 # If missing, append an empty array or handle accordingly
    #                 node_arrays.append(np.zeros((0,4), dtype=np.float32))
    #         mmwave_batch.append(node_arrays)
    #     modality_data['mmWave'] = mmwave_batch

    # --------------------------------------------------------
    # For mmWave, let's gather each sample's data into a tensor
    # [num_nodes, time, max_points, 4], then store it so that
    # we can unify across the batch below.
    # --------------------------------------------------------
    # from rich import inspect
    # inspect(sample['modality_data']['mmWave'])
    if 'mmWave' in modality_data:
        mmwave_all_samples = []  # holds one padded tensor per sample: shape [N, T, P, 4]

        for sample_data_list in modality_data['mmWave']:
            # sample_data_list is the list of node-level dicts for *this* sample
            # e.g. [ { 'timestamps': [...], 'points': [arr0, arr1,...] },
            #        { 'timestamps': [...], 'points': [arr0, arr1,...] }, ... ]

            # 1) Identify max_time across all nodes
            max_time = 0
            for node_dict in sample_data_list:
                if node_dict is not None:
                    frames_list = node_dict.get('points', [])
                    if len(frames_list) > max_time:
                        max_time = len(frames_list)

            # 2) Identify max_points across all frames in this sample
            max_points = 0
            for node_dict in sample_data_list:
                if node_dict is not None:
                    frames_list = node_dict.get('points', [])
                    for frame_array in frames_list:
                        if frame_array is not None:
                            max_points = max(max_points, frame_array.shape[0])

            if max_time == 0:
                # Means no valid frames at all
                mmwave_all_samples.append(None)
                continue

            # 3) Build a tensor of shape [N, T, P, 4] for this sample
            node_tensors = []
            for node_dict in sample_data_list:
                if node_dict is None:
                    # If node missing, shape is [max_time, max_points, 4] of zeros
                    node_tensor = torch.zeros((max_time, max_points, 4), dtype=torch.float32)
                    node_tensors.append(node_tensor)
                    continue

                frames_list = node_dict.get('points', [])
                padded_frames = []
                for frame_array in frames_list:
                    if frame_array is None:
                        padded_frames.append(torch.zeros((max_points, 4), dtype=torch.float32))
                    else:
                        frame_tensor = torch.from_numpy(frame_array).float()
                        frame_tensor = torch.nan_to_num(frame_tensor, nan=0.0, posinf=0.0, neginf=0.0)

                        # pad up to max_points
                        if frame_tensor.shape[0] < max_points:
                            diff = max_points - frame_tensor.shape[0]
                            pad_ = torch.zeros((diff, 4), dtype=torch.float32)
                            frame_tensor = torch.cat([frame_tensor, pad_], dim=0)
                        padded_frames.append(frame_tensor)

                # If this node had fewer than max_time frames, pad time dimension
                frames_len = len(padded_frames)
                if frames_len < max_time:
                    zero_frame = torch.zeros((max_points, 4), dtype=torch.float32)
                    for _ in range(max_time - frames_len):
                        padded_frames.append(zero_frame)

                # stack => [T, max_points, 4]
                node_tensor = torch.stack(padded_frames, dim=0)
                node_tensors.append(node_tensor)

            # stack nodes => [N, T, max_points, 4]
            sample_tensor = torch.stack(node_tensors, dim=0)
            mmwave_all_samples.append(sample_tensor)
        
        # Now rewrite `modality_data['mmWave']` to be a list of these sample-level tensors
        modality_data['mmWave'] = mmwave_all_samples
    # ---------------------------------------------------------
    # For mmWave, unify across batch by node/time if needed
    # The same "max_nodes, max_time" logic you do for other modalities:
    # ---------------------------------------------------------
    if 'mmWave' in modality_data:
        mmwave_all_sample_tensors = modality_data['mmWave']  # list of length B
        valid_items = [x for x in mmwave_all_sample_tensors if x is not None]
        if len(valid_items) == 0:
            # no valid mmWave data in this batch
            modality_data['mmWave'] = None
        else:
            # find max_nodes and max_time
            max_nodes = 0
            max_time  = 0
            max_points = 0
            for item in valid_items:
                if item.shape[0] > max_nodes:
                    max_nodes = item.shape[0]
                if item.shape[1] > max_time:
                    max_time = item.shape[1]
                if item.shape[2] > max_points:
                    max_points = item.shape[2]
            
            padded_batch = []
            for item in mmwave_all_sample_tensors:
                if item is None:
                    # create zero
                    shape = (max_nodes, max_time, max_points, 4)
                    padded_batch.append(torch.zeros(shape, dtype=torch.float32))
                else:
                    n, t, p, c = item.shape  # c=4
                    pad_n = max_nodes - n
                    pad_t = max_time - t
                    pad_p = max_points - p
                    
                    # pad node dim
                    if pad_n > 0:
                        node_pad = torch.zeros((pad_n, t, p, c), dtype=torch.float32)
                        item = torch.cat((item, node_pad), dim=0)
                    # pad time dim
                    if pad_t > 0:
                        time_pad = torch.zeros((item.shape[0], pad_t, p, c), dtype=torch.float32)
                        item = torch.cat((item, time_pad), dim=1)
                    # pad points dim
                    if pad_p > 0:
                        points_pad = torch.zeros((item.shape[0], item.shape[1], pad_p, c), dtype=torch.float32)
                        item = torch.cat((item, points_pad), dim=2)
                    
                    padded_batch.append(item)
            
            # Now shape: [B, max_nodes, max_time, max_points, 4]
            mmwave_tensor = torch.stack(padded_batch, dim=0)
            modality_data['mmWave'] = mmwave_tensor

    # Prepare batched data
    for modality in modality_data:
        if modality == 'mmWave':
            continue

        if modality == 'acoustic':
            # Initialize lists to hold per-node waveforms and timestamps
            batched_waveforms_per_node = []
            batched_timestamps_per_node = []
            
            num_nodes = len(batch[0]['modality_data'][modality])  # Assuming all samples have the same number of nodes
            
            # Initialize lists for each node
            for _ in range(num_nodes):
                batched_waveforms_per_node.append([])
                batched_timestamps_per_node.append([])
            
            # Iterate over each sample in the batch
            for sample_data in modality_data[modality]:
                for node_idx, data in enumerate(sample_data):
                    if data is not None and 'waveform' in data:
                        batched_waveforms_per_node[node_idx].append(data['waveform'])  # [1, samples]
                        batched_timestamps_per_node[node_idx].append(data['timestamps'])  # List of timestamps
                    else:
                        # >>>>> Modified Start: Append a zero waveform for missing data
                        batched_waveforms_per_node[node_idx].append(torch.zeros((1, 1), dtype=torch.float32))
                        batched_timestamps_per_node[node_idx].append([])  # Empty timestamps
                        # <<<<< Modified End

            # Process each node separately
            mel_spectrograms_per_node = []
            time_frames_per_node = []
            for node_waveforms, node_timestamps in zip(batched_waveforms_per_node, batched_timestamps_per_node):
                # Determine the maximum length among waveforms
                max_length = max([w.shape[1] for w in node_waveforms])
                
                # Pad waveforms to max_length
                padded_waveforms = []
                for waveform in node_waveforms:
                    current_length = waveform.shape[1]
                    if current_length < max_length:
                        pad_length = max_length - current_length
                        padding = torch.zeros((1, pad_length), dtype=torch.float32)
                        padded_waveform = torch.cat((waveform, padding), dim=1)
                    else:
                        padded_waveform = waveform
                    padded_waveforms.append(padded_waveform)
                
                # Stack waveforms: [batch_size, 1, max_length]
                stacked_waveforms = torch.stack(padded_waveforms)  # [batch_size, 1, max_length]
                
                # Convert to Mel Spectrograms
                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=16000,
                    n_fft=1024,
                    hop_length=512,
                    n_mels=128,
                    f_min=0.0,
                    f_max=None
                )
                
                mel_spectrograms = mel_transform(stacked_waveforms)  # [batch_size, n_mels, time_frames]
                
                # Apply logarithmic scaling
                log_mel_spectrograms = torch.log1p(mel_spectrograms)  # [batch_size, n_mels, time_frames]
                
                mel_spectrograms_per_node.append(log_mel_spectrograms)
                time_frames_per_node.append(log_mel_spectrograms.shape[3])
            
            # Determine the global maximum number of time frames across all nodes
            global_max_time_frames = max(time_frames_per_node)
            # Pad each node's mel spectrograms to global_max_time_frames
            padded_mel_spectrograms_per_node = []
            for mel_spectrogram in mel_spectrograms_per_node:
                current_time_frames = mel_spectrogram.shape[3]
                if current_time_frames < global_max_time_frames:
                    pad_time = global_max_time_frames - current_time_frames
                    padding = torch.zeros((mel_spectrogram.shape[0], mel_spectrogram.shape[1], mel_spectrogram.shape[2], pad_time), dtype=torch.float32)
                    mel_spectrogram = torch.cat((mel_spectrogram, padding), dim=3)
                elif current_time_frames > global_max_time_frames:
                    # Truncate to desired_time_frames if exceeding
                    mel_spectrogram = mel_spectrogram[:, :, :, :global_max_time_frames]
                if DEBUGGER:
                    print(f"mel_spectrum.shape: {mel_spectrogram.shape}")
                padded_mel_spectrograms_per_node.append(mel_spectrogram)
            
            # Stack mel spectrograms along the node dimension
            # Each element in mel_spectrograms_per_node has shape [batch_size, n_mels, time_frames]
            # After stacking: [batch_size, num_nodes, n_mels, time_frames]
            batched_mel_spectrograms = torch.stack(padded_mel_spectrograms_per_node, dim=1)
            
            modality_data[modality] = {
                'mel_spectrogram': batched_mel_spectrograms  # [batch_size, num_nodes, 128, max_time_frames]
                # 'timestamps': batched_timestamps_padded  # Optionally include timestamps
            }
        else:
            if modality == 'mmWave':
                continue
            batched_sequences = []
            all_sample_tensors = []  # Will hold processed tensor(s) for each sample in the batch
            for data_list in modality_data[modality]:
                sequences = []
                for data in data_list:
                    if data is not None:
                        if modality == 'depthCamera':
                            depth_images = data.get('depth_images')
                            rgb_frames = data.get('rgb_frames')
                            if depth_images is not None and rgb_frames is not None:
                                # Convert to tensors
                                depth_images_tensor = torch.tensor(np.array(depth_images), dtype=torch.float32)
                                rgb_frames_tensor = torch.tensor(np.array(rgb_frames), dtype=torch.float32)
                                sequences.append({
                                    'depth_images': depth_images_tensor,
                                    'rgb_frames': rgb_frames_tensor
                                })
                            else:
                                sequences.append(None)
                        elif modality == 'seekThermal':
                            thermal_images = data.get('thermal_images')
                            if thermal_images is not None:
                                thermal_images_tensor = torch.tensor(np.array(thermal_images), dtype=torch.float32)
                                sequences.append(thermal_images_tensor)                        
                            else:
                                sequences.append(None)
                        elif modality in ['IRA', 'uwb', 'polar','wifi', 'imu', 'vayyar', 'mocap']:
                            frames = data.get('frames')
                            arr = np.array(frames)
                            if DEBUGGER:
                                if np.iscomplexobj(arr):
                                    print("Yes, it is a complex array.")
                            if frames is not None:
                                # Detect if frames is complex
                                if np.iscomplexobj(frames):
                                    # Convert to a complex PyTorch tensor
                                    frames_torch = torch.tensor(frames, dtype=torch.complex64)
                                else:
                                    # Non-complex data is safe to cast to float
                                    frames_torch = torch.tensor(frames, dtype=torch.float32)
                                sequences.append(frames_torch)
                        elif modality == 'ToF':
                            # Handle TOF data differently if needed
                            # tof_depths = data.get('tof_depths')
                            tof_bins = data.get('tof_bins')
                            # print("Original shape of tof_bins:", tof_bins.shape)  # Debugging
                            # if tof_depths is not None:
                            #     sequences.append(torch.tensor(tof_depths, dtype=torch.float32))
                            if tof_bins is not None:
                                # Ensure it's a tensor first
                                tof_bins = torch.tensor(tof_bins, dtype=torch.float32)

                                # Ensure reshaping is feasible
                                if tof_bins.shape[1:] == (64, 18):
                                    # Reshape only the last two dimensions (64, 18) → (8, 8, 18)
                                    tof_bins = tof_bins.view(tof_bins.shape[0], 8, 8, 18)
                                else:
                                    print(f"Warning: Unexpected shape {tof_bins.shape}, cannot reshape last two dims to (8, 8, 18)")


                                sequences.append(tof_bins)
                    else:
                        # Handle None data
                        sequences.append(None)
                if sequences:
                    if modality == 'depthCamera':
                        # Handle padding for depth images and RGB frames separately
                        depth_sequences = []
                        rgb_sequences = []

                        # Find maximum sequence length
                        max_seq_len = max([seq['depth_images'].shape[0] for seq in sequences if seq is not None])

                        # Pad sequences
                        for seq in sequences:
                            if seq is not None:
                                depth_seq = seq['depth_images']
                                rgb_seq = seq['rgb_frames']
                                seq_len = depth_seq.shape[0]
                                pad_seq_len = max_seq_len - seq_len

                                # Pad temporal dimension
                                if pad_seq_len > 0:
                                    # For depth images
                                    depth_padding = torch.zeros((pad_seq_len, *depth_seq.shape[1:]), dtype=torch.float32)
                                    depth_seq = torch.cat((depth_seq, depth_padding), dim=0)
                                    # For RGB frames
                                    rgb_padding = torch.zeros((pad_seq_len, *rgb_seq.shape[1:]), dtype=torch.float32)
                                    rgb_seq = torch.cat((rgb_seq, rgb_padding), dim=0)

                                depth_sequences.append(depth_seq)
                                rgb_sequences.append(rgb_seq)
                            else:
                                # Create zero tensors if sequence is None
                                depth_shape = (max_seq_len, *sequences[0]['depth_images'].shape[1:])
                                rgb_shape = (max_seq_len, *sequences[0]['rgb_frames'].shape[1:])
                                depth_sequences.append(torch.zeros(depth_shape, dtype=torch.float32))
                                rgb_sequences.append(torch.zeros(rgb_shape, dtype=torch.float32))

                        # Stack sequences
                        batched_depth_images = torch.stack(depth_sequences)
                        batched_rgb_frames = torch.stack(rgb_sequences)

                        # Instead of assigning directly, we append to 'all_sample_tensors'
                        all_sample_tensors.append({
                            'depth_images': batched_depth_images,
                            'rgb_frames': batched_rgb_frames
                        })
                    elif modality == 'seekThermal':
                        # Find maximum sequence length
                        if not any(seq is not None for seq in sequences):
                            # ***CHANGED LINES START***
                            # No valid data for seekThermal in this segment
                            modality_data[modality] = None
                            continue
                        # Existing padding for 'seekthermal' modality
                        # Find maximum sequence length
                        max_seq_len = max([seq.shape[0] for seq in sequences if seq is not None])
                        # Pad sequences
                        padded_sequences = []
                        for seq in sequences:
                            if seq is not None:
                                seq_len = seq.shape[0]
                                pad_seq_len = max_seq_len - seq_len
                                if pad_seq_len > 0:
                                    padding = torch.zeros((pad_seq_len, *seq.shape[1:]), dtype=torch.float32)
                                    seq = torch.cat((seq, padding), dim=0)
                                padded_sequences.append(seq)
                            else:
                                seq_shape = (max_seq_len, *sequences[0].shape[1:])
                                padded_sequences.append(torch.zeros(seq_shape, dtype=torch.float32))
                        # Stack sequences
                        batched_sequences = torch.stack(padded_sequences)
                        all_sample_tensors.append(batched_sequences)
                    else:
                        # For IRA, uwb, mmWave, polar, wifi, imu, vayyar
                        valid_sequences = [seq for seq in sequences if seq is not None]
                        if len(valid_sequences) == 0:
                            all_sample_tensors.append(None)
                        else:
                            # Pad sequences by time dimension if needed
                            # Find max_time for these sequences
                            ### CHANGED CODE START ###
                            # Assume shape: [nodes_within_sample_for_this_modality, time, ...]
                            # First, we might have different node counts per sample. So we must handle that.
                            # We'll handle varying node/time dimensions after all samples are collected.
                            # For now, just stack them as is.
                            # Actually, we need to return them as a single tensor per sample.
                            # Let's just pad by time dimension here. Node dimension is length of valid_sequences.
                            
                            max_time = 0
                            for seq in valid_sequences:
                                if seq.shape[0] > max_time:
                                    max_time = seq.shape[0]

                            # Pad each seq to max_time
                            padded_node_seq = []
                            for seq in valid_sequences:
                                time_len = seq.shape[0]
                                if time_len < max_time:
                                    time_pad = torch.zeros((max_time - time_len, *seq.shape[1:]), dtype=torch.float32)
                                    seq = torch.cat((seq, time_pad), dim=0)
                                padded_node_seq.append(seq)

                            # Now stack nodes for this one sample: shape [nodes, max_time, ...]
                            sample_tensor = torch.stack(padded_node_seq, dim=0)
                            all_sample_tensors.append(sample_tensor)
                else:
                    # batched_sequences.append(None)
                    all_sample_tensors.append(None)

            # Now we have all_sample_tensors for this modality: [batch_size] elements, each either None or [nodes, time, ...]
            # We must handle varying node/time across samples.
            valid_items = [x for x in all_sample_tensors if x is not None]

            if len(valid_items) == 0:
                modality_data[modality] = None
            else:
                if modality == 'depthCamera':
                    # Already dictionary of tensors
                    # Find max nodes and max time
                    # Extract depths and rgbs separately
                    depth_list = []
                    rgb_list = []
                    for item in all_sample_tensors:
                        if item is None:
                            depth_list.append(None)
                            rgb_list.append(None)
                        else:
                            depth_list.append(item['depth_images'])
                            rgb_list.append(item['rgb_frames'])

                    # Find max_nodes and max_time
                    max_nodes = 0
                    max_time = 0
                    for d_item in depth_list:
                        if d_item is not None:
                            if d_item.shape[0] > max_nodes:
                                max_nodes = d_item.shape[0]
                            if d_item.shape[1] > max_time:
                                max_time = d_item.shape[1]

                    # Pad all samples to max_nodes and max_time
                    padded_depth_batch = []
                    padded_rgb_batch = []
                    for d_item, r_item in zip(depth_list, rgb_list):
                        if d_item is None:
                            # create zeros
                            d_zeros = torch.zeros((max_nodes, max_time, *depth_list[0].shape[2:]), dtype=torch.float32)
                            r_zeros = torch.zeros((max_nodes, max_time, *rgb_list[0].shape[2:]), dtype=torch.float32)
                            padded_depth_batch.append(d_zeros)
                            padded_rgb_batch.append(r_zeros)
                        else:
                            # pad nodes
                            d_nodes = d_item.shape[0]
                            r_nodes = d_item.shape[0]
                            if d_nodes < max_nodes:
                                # Pad node dimension
                                node_pad_shape = (max_nodes - d_nodes, d_item.shape[1], *d_item.shape[2:])
                                d_item = torch.cat((d_item, torch.zeros(node_pad_shape, dtype=torch.float32)), dim=0)
                                r_item = torch.cat((r_item, torch.zeros(node_pad_shape[:-1] + (r_item.shape[-1],), dtype=torch.float32)), dim=0)
                            # pad time
                            if d_item.shape[1] < max_time:
                                time_pad_shape = (d_item.shape[0], max_time - d_item.shape[1], *d_item.shape[2:])
                                d_item = torch.cat((d_item, torch.zeros(time_pad_shape, dtype=torch.float32)), dim=1)
                                rgb_time_pad_shape = (r_item.shape[0], max_time - r_item.shape[1], *r_item.shape[2:])
                                r_item = torch.cat((r_item, torch.zeros(rgb_time_pad_shape, dtype=torch.float32)), dim=1)

                            padded_depth_batch.append(d_item)
                            padded_rgb_batch.append(r_item)

                    depth_batch = torch.stack(padded_depth_batch, dim=0) # [batch, max_nodes, max_time, H, W]
                    rgb_batch = torch.stack(padded_rgb_batch, dim=0)     # [batch, max_nodes, max_time, H, W, 3]
                    modality_data[modality] = {
                        'depth_images': depth_batch,
                        'rgb_frames': rgb_batch
                    }

                elif modality == 'seekThermal':
                    # valid_items: list of [nodes, time, H, W]
                    max_nodes = 0
                    max_time = 0
                    for item in valid_items:
                        if item.shape[0] > max_nodes:
                            max_nodes = item.shape[0]
                        if item.shape[1] > max_time:
                            max_time = item.shape[1]

                    # Pad all samples
                    padded_batch = []
                    for item in all_sample_tensors:
                        if item is None:
                            # create zero tensor
                            shape = (max_nodes, max_time, *valid_items[0].shape[2:])
                            padded_batch.append(torch.zeros(shape, dtype=torch.float32))
                        else:
                            pad_nodes = max_nodes - item.shape[0]
                            pad_time = max_time - item.shape[1]
                            padded_item = item
                            if pad_nodes > 0:
                                node_pad = torch.zeros((pad_nodes, item.shape[1], *item.shape[2:]), dtype=torch.float32)
                                padded_item = torch.cat((padded_item, node_pad), dim=0)
                            if pad_time > 0:
                                time_pad = torch.zeros((padded_item.shape[0], pad_time, *padded_item.shape[2:]), dtype=torch.float32)
                                padded_item = torch.cat((padded_item, time_pad), dim=1)
                            padded_batch.append(padded_item)

                    # Now stack along batch
                    modality_data[modality] = torch.stack(padded_batch, dim=0)

                else:
                    # For IRA, uwb, mmWave, polar, wifi, acoustic:
                    # valid_items: [nodes, time, ...]
                    max_nodes = 0
                    max_time = 0
                    for item in valid_items:
                        if item.shape[0] > max_nodes:
                            max_nodes = item.shape[0]
                        if item.shape[1] > max_time:
                            max_time = item.shape[1]

                    padded_batch = []
                    for item in all_sample_tensors:
                        if item is None:
                            # create zero
                            # Use shape from the first valid sample as a reference
                            ref = valid_items[0]
                            shape = (max_nodes, max_time, *ref.shape[2:])
                            padded_batch.append(torch.zeros(shape, dtype=torch.float32))
                        else:
                            pad_nodes = max_nodes - item.shape[0]
                            pad_time = max_time - item.shape[1]
                            padded_item = item
                            if pad_nodes > 0:
                                node_pad = torch.zeros((pad_nodes, item.shape[1], *item.shape[2:]), dtype=torch.float32)
                                padded_item = torch.cat((padded_item, node_pad), dim=0)
                            if pad_time > 0:
                                time_pad = torch.zeros((padded_item.shape[0], pad_time, *padded_item.shape[2:]), dtype=torch.float32)
                                padded_item = torch.cat((padded_item, time_pad), dim=1)
                            padded_batch.append(padded_item)
        
                    modality_data[modality] = torch.stack(padded_batch, dim=0)
    batch_collated = {
        'user_id': user_ids,
        'activity': activities,
        'modality_data': modality_data
    }

    return batch_collated

def get_dataset(config, dataset_path="", mocap_downsample_num = None) -> OctonetDataset:
    """
    Args:
        config: config file
        dataset_path: path to the dataset
        mocap_downsample_num: number of downsample for mocap data
    """
    file_path = os.path.join(dataset_path, "cut_manual.csv")
    metadata_df = pd.read_csv(file_path)  
    data_selector = DataSelector(config, metadata_df, dataset_path)
    # Generate the filtered CSV from the DataSelector
    # filtered_csv_file = "filtered_metadata.csv"
    # data_selector.generate_filtered_csv(filtered_csv_file)
    # Build the mocap_base_dir from dataset_path instead of hardcoding
    mocap_base_dir = os.path.join(dataset_path, "mocap_pose")
    data_fd = data_selector.get_filtered_data()
    # Instantiate the dataset with the filtered CSV
    dataset = OctonetDataset(data_fd, segmentation_flag=config['segmentation_flag'], mocap_base_dir=mocap_base_dir, mocap_downsample_num = config.get('mocap_downsample_num', mocap_downsample_num))
    return dataset

def get_dataloader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=custom_collate, config=None):
    if config is not None:
        # combine config with input arguments
        batch_size = config.get('batch_size', batch_size)
        shuffle = config.get('shuffle', shuffle)
        num_workers = config.get('num_workers', num_workers)
        drop_last = config.get('drop_last', False)
    else:
        # default values
        pass
    
    # get dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=drop_last
    )
    return dataloader