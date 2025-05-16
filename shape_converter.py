'''

Modality: polar
  ---Data shape: torch.Size([2, 1, 8])
  ---Data type:     torch.float32

Activities: ['dance', 'dance']
'''
import torch
import torch.nn.functional as F 
import numpy as np

# All_activities = [
#             'sit', 'walk', 'bow', 'sleep', 'dance', 'jog', 'falldown', 'jump', 'jumpingjack',
#             'squat', 'lunge', 'turn', 'pushup', 'legraise', 'airdrum', 'boxing', 'shakehead',
#             'answerphone', 'eat', 'drink', 'wipeface', 'pickup', 'jumprope', 'moppingfloor',
#             'brushhair', 'bicepcurl', 'playphone', 'brushteeth', 'type', 'thumbup', 'thumbdown',
#             'makeoksign', 'makevictorysign', 'drawcircleclockwise', 'drawcirclecounterclockwise',
#             'stopsign', 'pullhandin', 'pushhandaway', 'handwave', 'sweep', 'clap', 'slide',
#             'drawzigzag', 'dodge', 'bowling', 'liftupahand', 'tap', 'spreadandpinch', 'drawtriangle',
#             'sneeze', 'cough', 'stagger', 'yawn', 'blownose', 'stretchoneself', 'touchface',
#             'handshake', 'hug', 'pushsomeone', 'kicksomeone', 'punchsomeone', 'conversation',
#             'gym', 'freestyle'
#         ]
# [ 'mmWave', 'IRA', 'uwb', 'ToF', 'polar', 'wifi', 'depthCamera', 'seekThermal','acoustic', 'imu', 'vayyar']

# def padding_tool(batch, target_size):
#     """
#     Cuts or pads the data to match the target size.
    
#     Args:
#         batch: Tensor data of any shape
#         target_size: Target size tuple. If any dimension is -1, that dimension is unchanged.
#                      Example: target_size = (4, 100, -1) will only modify the first two dimensions.
    
#     Returns:
#         Padded or truncated tensor of size specified by the target_size parameter
#     """
#     # Convert input to tensor if it's not already
#     if not isinstance(batch, torch.Tensor):
#         batch = torch.tensor(batch)
    
#     # Get current batch shape
#     current_shape = batch.shape
    
#     # Convert any -1 in target_size to the corresponding dimension in current_shape
#     actual_size = list(target_size)
#     for i in range(min(len(current_shape), len(actual_size))):
#         if actual_size[i] == -1:
#             actual_size[i] = current_shape[i]
    
#     # Ensure actual_size has same number of dimensions as current_shape
#     while len(actual_size) < len(current_shape):
#         actual_size.append(current_shape[len(actual_size)])
    
#     # Create the result tensor
#     result = torch.zeros(actual_size, dtype=batch.dtype, device=batch.device)
    
#     # Calculate slices for copying data
#     slices_src = []
#     slices_dst = []
#     for i in range(len(current_shape)):
#         if i >= len(actual_size) or actual_size[i] == -1 or actual_size[i] >= current_shape[i]:
#             slices_src.append(slice(None))
#             slices_dst.append(slice(0, current_shape[i]))
#         else:
#             slices_src.append(slice(0, actual_size[i]))
#             slices_dst.append(slice(None))
    
#     # Copy data from batch to result
#     result[tuple(slices_dst)] = batch[tuple(slices_src)]
    
#     return result



class DataConverter:
    def __init__(self, data_config, model_config):
        self.data_config = data_config
        self.modalities = data_config['modality']
        self.activity_list = data_config['activity_list']
        # self.exp_list = data_config['exp_list']
        self.activity_id_mapping = {activity: i for i, activity in enumerate(self.activity_list)}
        self.id_activity_mapping = {i: activity for i, activity in enumerate(self.activity_list)}
        self.model_config = model_config    
        self.model_input_shape = model_config['model_input_shape']
        self.target_shapes = model_config.get('target_shape', {})
        
    def shape_convert(self, batch):
        
        
        user_id = self._convert_userid(batch['user_id'])
        activity_id = self._convert_activity(batch['activity'])
        modality_data = batch['modality_data']
        data_dict = {}
        for modality in modality_data:
            if modality == 'IRA':
                data_dict[modality] = self._convert_IRA(modality_data[modality])
            elif modality == 'mocap':
                data_dict[modality] = modality_data[modality]
            elif modality == 'uwb':
                data_dict[modality] = self._convert_uwb(modality_data[modality])
            elif modality == 'ToF':
                data_dict[modality] = self._convert_ToF(modality_data[modality])
            elif modality == 'depthCamera':
                # Check your new config flags:
                use_depth = self.model_config.get('use_depth', True)
                use_rgb   = self.model_config.get('use_rgb',   True)
                
                # If user wants to train on depth frames:
                if use_depth:
                    # Put the depth frames under the key "depthCamera"
                    # so that the normal line "input = input[modality]" still works:
                    data_dict[modality] = self._convert_depth_images(modality_data[modality])
                
                # Optionally store the RGB frames if needed:
                if use_rgb:
                    data_dict[modality] = self._convert_rgb_frames(modality_data[modality])

            elif modality == 'seekThermal':
                data_dict[modality] = self._convert_seekThermal(modality_data[modality])
            elif modality == 'acoustic':
                data_dict[modality] = self._convert_acoustic(modality_data[modality])
            elif modality == 'imu':
                data_dict[modality] = self._convert_imu(modality_data[modality])
            elif modality == 'vayyar':
                data_dict[modality] = self._convert_vayyar(modality_data[modality])
            elif modality == 'wifi':
                data_dict[modality] = self._convert_wifi(modality_data[modality])
            elif modality == 'mmWave':
                data_dict[modality] = self._convert_mmWave(modality_data[modality])
                print(data_dict[modality].shape)
                
                if torch.isnan(data_dict[modality]).any():
                    print("Found NaN in mmWave data!")
                if torch.isinf(data_dict[modality]).any():
                    print("Found Inf in mmWave data!")
            elif modality == 'polar':
                data_dict[modality] = self._convert_polar(modality_data[modality])
            # elif modality == 'mocap':
            #     data_dict[modality] = self._convert_mocap(modality_data[modality])
            else:
                raise ValueError(f'Invalid modality: {modality}')
            # Then pad to target shape if specified
            if modality in self.target_shapes:
                data_dict[modality] = self._pad_to_target(data_dict[modality], self.target_shapes[modality])
        return user_id, activity_id, data_dict, modality
    
    def _pad_to_target(self, batch, target_size):
        """Internal padding method using the padding_tool logic"""
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        
        current_shape = batch.shape
        actual_size = list(target_size)
        
        # Convert any -1 in target_size to corresponding dimension
        for i in range(min(len(current_shape), len(actual_size))):
            if actual_size[i] == -1:
                actual_size[i] = current_shape[i]
        
        # Create and fill result tensor
        result = torch.zeros(actual_size, dtype=batch.dtype, device=batch.device)
        
        # Calculate slices for copying
        slices_src = []
        slices_dst = []
        for i in range(len(current_shape)):
            if i >= len(actual_size) or actual_size[i] == -1 or actual_size[i] >= current_shape[i]:
                slices_src.append(slice(None))
                slices_dst.append(slice(0, current_shape[i]))
            else:
                slices_src.append(slice(0, actual_size[i]))
                slices_dst.append(slice(None))
        
        result[tuple(slices_dst)] = batch[tuple(slices_src)]
        return result

    def _convert_IRA(self, data):
        # infared array temperature map: torch.Size([2, 3, 64, 24, 32]), (batch_size, num_nodes, time, H, W)
        if self.model_input_shape == 'BCHW':
            # merge the second and third dimensions
            data = data.view(data.size(0), -1, data.size(3), data.size(4))  # (batch_size, num_nodes*time, H, W)
            #Clamp the IRA data:
            # data is now [B, C, H, W], where C = num_nodes * time
            # B, C, H, W = data.shape
            # # 2) Mode-based clamp ±15 on each frame
            # for b in range(B):
            #     for c in range(C):
            #         frame = data[b, c]  # shape (H, W)

            #         # Round and flatten
            #         flattened = torch.round(frame).view(-1)
            #         # Remove zeros (optional) if zeros dominate:
            #         nonzero = flattened[flattened != 0]
            #         if nonzero.numel() > 0:
            #             mode_val = torch.mode(nonzero).values.item()
            #         else:
            #             mode_val = 0  # fallback if all zeros

            #         frame.clamp_(mode_val - 15, mode_val + 15)
                    
        elif self.model_input_shape == 'BLC':
            # permute the second and third dimensions
            data = data.permute(0, 2, 1, 3, 4)
            data = data.reshape(data.size(0), data.size(1), -1)   # (batch_size, time, num_nodes*H*W)
        elif self.model_input_shape == 'BTCHW':
            data = data.permute(0, 2, 1, 3, 4)  # (batch_size, time, num_nodes, H, W)
        else:
            raise ValueError('Invalid model input shape for IRA modality')
        return data
    
    def _convert_uwb(self, data):
        # ultra wide band: torch.Size([2, 1, 158, 1535]), (batch_size, num_nodes, time, CIR)
        if self.model_input_shape == 'BCHW':
            pass
        elif self.model_input_shape == 'BLC':
            # merge the second and third dimensions
            data = data.view(data.size(0), -1, data.size(3)) # (batch_size, num_nodes*time, CIR)
        elif self.model_input_shape == 'BTCHW':
            # Calculate the padding needed
            padding_size = 1600 - data.shape[3]
            data = F.pad(data, (0, padding_size), mode='constant', value=0)  # (batch_size, 1, 158, 1600)
        else:
            raise ValueError('Invalid model input shape for UWB modality')
        return data
    
    def _convert_ToF(self, data):
        # Data shape: torch.Size([2, 1, 68, 8, 8, 18])  # (batch_size, num_nodes, time, H, W, C)
        if self.model_input_shape == 'BCHW':
            data = data.permute(0, 1, 2, 5, 3, 4)  # (batch_size, num_nodes, time, C, H, W)
            # Merge (num_nodes, time, C) into a single channel dimension
            data = data.reshape(data.size(0), -1, data.size(4), data.size(5))  # (batch_size, num_nodes*time*C, H, W)
        elif self.model_input_shape == 'BLC':
            # Permute to bring (H, W, C) together
            data = data.permute(0, 1, 2, 5, 3, 4)  # (batch_size, num_nodes, time, C, H, W)

            # Merge (num_nodes, time) into one dimension while keeping (C, H, W) together
            data = data.view(data.size(0), data.size(1) * data.size(2), data.size(3), data.size(4), data.size(5))  
            # (batch_size, num_nodes*time, C, H, W)

            # Flatten spatial dimensions (H, W) into one
            data = data.reshape(data.size(0), data.size(1), -1)  # (batch_size, num_nodes*time, C*H*W)

        elif self.model_input_shape == 'BTCHW':
            # Permute: (batch_size, num_nodes, time, H, W, C) → (batch_size, time, num_nodes, H, W, C)
            data = data.permute(0, 2, 1, 3, 4, 5)  # (batch_size, time, num_nodes, H, W, C)

            # Merge last dimension (C) into the channel dimension (num_nodes, C)
            data = data.view(data.size(0), data.size(1), -1, data.size(3), data.size(4))  # (batch_size, time, num_nodes*C, H, W)
        # elif self.model_input_shape == 'BLC':
        #     # merge the second and third dimensions
        #     data = data.view(data.size(0), -1, data.size(3), data.size(4)) # (batch_size, num_nodes*time, H, W) 
        #     data = data.view(data.size(0), -1, data.size(2)*data.size(3))  # (batch_size, num_nodes*time, H*W)
        # elif self.model_input_shape == 'BTCHW':
        #     data = data.permute(0, 2, 1, 3, 4)  # (batch_size, time, num_nodes, H, W)
        else:
            raise ValueError('Invalid model input shape for ToF modality')
        return data
    
    def _convert_depth_images(self, data):
        # Data shape: torch.Size([2, 2, 278, 480, 640]) # (batch_size, num_nodes, time, H, W)
        if isinstance(data, dict):  # Handle dict case
            data = data['depth_images']
        if self.model_input_shape == 'BCHW':
            data = data.view(data.size(0), -1, data.size(3), data.size(4)) # (batch_size, num_nodes*time, H, W)
        elif self.model_input_shape == 'BLC':
            data = data.permute(0, 2, 1, 3, 4)
            data = data.reshape(data.size(0), data.size(1), -1) # (batch_size, time, num_nodes*H*W)
        elif self.model_input_shape == 'BTCHW':
            data = data.permute(0, 1, 2, 4, 3)
        else:
            raise ValueError('Invalid model input shape for depth_images modality')
        return data
    
    def _convert_rgb_frames(self, data):
        # Data shape: torch.Size([2, 2, 278, 480, 640, 3]) # (batch_size, num_nodes, time, H, W, C)
        if isinstance(data, dict):  # Handle dict case
            data = data['rgb_frames']
        data = data.permute(0, 1, 2, 5, 3, 4)  # (batch_size, num_nodes, time, C, H, W)
        if self.model_input_shape == 'BCHW':
            data = data.reshape(data.size(0), -1, data.size(4), data.size(5)) # (batch_size, num_nodes* time*C, H, W)
        elif self.model_input_shape == 'BLC':
            data = data.permute(0, 2, 1, 3,4,5) # (batch_size, time, num_nodes, C, H, W)
            data = data.reshape(data.size(0), data.size(1), -1) # (batch_size, time, num_nodes*H*W*C)
        elif self.model_input_shape == 'BTCHW':
            data = data.permute(0, 2, 1, 3,4,5) # (batch_size, time, num_nodes, C, H, W)
            data = data.view(data.size(0), data.size(1),-1, data.size(4), data.size(5)) # (batch_size, time, num_nodes * C, H, W)
        else:
            raise ValueError('Invalid model input shape for rgb_frames modality')
        return data
    
    def _convert_seekThermal(self, data):
        # Data shape: torch.Size([2, 2, 72, 240, 320]) # (batch_size, num_nodes, time, H, W)
        if self.model_input_shape == 'BCHW':
            data = data.view(data.size(0), -1, data.size(3), data.size(4)) # (batch_size, num_nodes*time, H, W)
        elif self.model_input_shape == 'BLC':
            data = data.permute(0, 1, 3, 4, 2)  # (batch_size, num_nodes, H, W, time)
            data = data.reshape(data.size(0), -1, data.size(4))  # (batch_size, num_nodes*time, H*W)
        elif self.model_input_shape == 'BTCHW':
            data = data.permute(0, 1, 2, 4, 3)  # (batch_size, num_nodes, time, H, W)
        else:
            raise ValueError('Invalid model input shape for seekThermal modality')
        return data
    
    def _convert_acoustic(self, data):
        # Data shape: torch.Size([2, 2, 1, 128, 254]) # (batch_size, num_nodes, 1, num_mel_spectrum , max_time)
        if isinstance(data, dict):  # Handle dict case
            data = data['mel_spectrogram']
        if self.model_input_shape == 'BCHW':
            data = data.view(data.size(0), -1, data.size(3), data.size(4)) # (batch_size, num_nodes*1, num_mel_spectrum, max_time)
        elif self.model_input_shape == 'BLC':
            data = data.permute(0, 4, 1, 2, 3)  # (batch_size, max_time, num_nodes, 1, num_mel_spectrum)
            data = data.view(data.size(0), data.size(1), -1) # (batch_size, max_time, num_nodes*num_mel_spectrum)
        elif self.model_input_shape == 'BTCHW':
            pass  # (batch_size, num_nodes, 1, num_mel_spectrum, max_time)
        else:
            raise ValueError('Invalid model input shape for acoustic modality')
        return data
    
    def _convert_imu(self, data):
        # Data shape: torch.Size([2, 1, 487, 13, 17])   # (batch_size, num_nodes, time, num_features (x_y_z,acc, gyro,...), num_markers)
        if self.model_input_shape == 'BCHW':
            data = data.view(data.size(0), data.size(1), data.size(2), -1) # (batch_size, num_nodes, time, num_features*num_markers)
        elif self.model_input_shape == 'BLC':
            data = data.permute(0, 2, 1, 3, 4)  # (batch_size, time, num_nodes, num_features, num_markers)
            data = data.view(data.size(0), data.size(1), -1) # (batch_size, time, num_nodes*num_features*num_markers)
        elif self.model_input_shape == 'BTCHW':
            data = data.permute(0, 2, 1, 3, 4)  # (batch_size, time, num_nodes, num_features, num_markers)
        else:
            raise ValueError('Invalid model input shape for imu modality')
        return data

    def _convert_mocap(self, data):
        # Data shape: torch.Size([8, 1, 851, 20, 3])   # (batch_size, num_nodes, time, num_markers ,num_features (x_y_z))
        if self.model_input_shape == 'BCHW':
            data = data.permute(0, 1, 4, 2, 3)
            data = data.view(data.size(0), -1, data.size(3), data.size(4)) # (batch_size, num_nodes*num_features (x_y_z) , time, num_markers)
        elif self.model_input_shape == 'BLC':
            data = data.permute(0, 2, 1, 3, 4)  # (batch_size, time, num_nodes, num_features, num_markers)
            data = data.view(data.size(0), data.size(1), -1) # (batch_size, time, num_nodes*num_features*num_markers)
        elif self.model_input_shape == 'BTCHW':
            data = data.permute(0, 2, 1, 3, 4)  # (batch_size, time, num_nodes, num_features, num_markers)
        else:
            raise ValueError('Invalid model input shape for imu modality')
        return data
    
    def _convert_vayyar(self, data):
        # Data shape: torch.Size([2, 1, 25, 400, 100]), complex64 # (batch_size, num_nodes, time, tx*rx, ADC)
        phase = torch.angle(data)
        magnitude = torch.abs(data)
        data = torch.cat((magnitude, phase), dim=1)  # (batch_size, 2*num_nodes, time, tx*rx, ADC)
        if self.model_input_shape == 'BCHW':
            data = data.permute(0, 1, 3, 2, 4)  # (batch_size, 2*num_nodes, tx*rx, time, ADC)
            data = data.reshape(data.size(0), data.size(1), data.size(2), -1)  # (batch_size, 2*num_nodes, tx*rx, time*ADC)
        elif self.model_input_shape == 'BLC':
            data = data.permute(0, 1, 3, 2, 4)  # (batch_size, 2*num_nodes, tx*rx, time, ADC)
            data = data.reshape(data.size(0), data.size(1) * data.size(2), -1)  # (batch_size, 2*num_nodes*tx*rx, time*ADC)
            data = data.permute(0, 2, 1)  # (batch_size, time*ADC, 2*num_nodes*tx*rx)
        elif self.model_input_shape == 'BTCHW':
            data = data.permute(0, 2, 1, 3, 4) # (batch_size, time, num_nodes, tx*rx, ADC)
        else:
            raise ValueError('Invalid model input shape for vayyar modality')
        return data
    
    def _convert_wifi(self, data):
        # Data shape: torch.Size([2, 3, 708, 2, 114]), complex64 # (batch_size, num_nodes, time, tx*rx, subcarriers)
        # phase = torch.angle(data)
        data = torch.abs(data)  # (batch_size, num_nodes, time, tx*rx, subcarriers)
        if self.model_input_shape == 'BCHW':
            data = data.permute(0, 1, 3, 2, 4)  # (batch_size, num_nodes, tx*rx, time, subcarriers)
            data = data.reshape(data.size(0), -1, data.size(3), data.size(4))  # Safe approach  # (batch_size, num_nodes*tx*rx, time, subcarriers)
        elif self.model_input_shape == 'BLC':
            data = data.permute(0, 2, 1, 3, 4)  # (batch_size, time, num_nodes, tx*rx, subcarriers)
            data = data.reshape(data.size(0), data.size(1), -1) # (batch_size, time, num_nodes*tx*rx*subcarriers)
        elif self.model_input_shape == 'BTCHW':
            data = data.permute(0, 2, 1, 3, 4) # (batch_size, time, num_nodes, tx*rx, subcarriers)
        else:
            raise ValueError('Invalid model input shape for wifi modality')
        return data
    
    def _convert_mmWave(self, data):
        # Universal pre-processing: remove extreme values and clamp coordinates.
        try:
            # 1. Remove extreme abnormal values
            reasonable_mask = torch.abs(data) < 1e10
            data = data * reasonable_mask  # Set out-of-range values to 0

            # 2. Define coordinate ranges (for x, y, z, and velocity)
            coord_ranges = {
                'x': (-5, 5),      # x-axis: -5 to 5
                'y': (0, 3),       # y-axis: 0 to 3
                'z': (-5, 5),      # z-axis: -5 to 5
                'velocity': (-2, 2)  # velocity: -2 to 2
            }

            # 3. Clamp each coordinate dimension safely
            for i, (coord, (min_val, max_val)) in enumerate(coord_ranges.items()):
                if i < data.shape[-1]:  # Ensure the coordinate dimension exists
                    data[..., i] = torch.clamp(data[..., i], min=min_val, max=max_val)
        except Exception as e:
            print(f"Error processing data: {e}")
            print(f"Data shape: {data.shape}")
            raise e

        # Process based on the desired model input shape
        if self.model_input_shape == 'BCHW':
            # BCHW manipulation: permute and reshape
            if len(data.shape) == 5:
                data = data.permute(0, 1, 4, 2, 3)
                data = data.reshape(data.size(0), -1, data.size(3), data.size(4))
            else:
                print(f"Warning: Unexpected input shape: {data.shape}")
        elif self.model_input_shape == 'BLC':
            # Extend the same pre-processing to BLC.
            # Then perform BLC-specific permutation and reshaping.
            data = data.permute(0, 2, 1, 3, 4)  # Expected shape: (batch, time, num_nodes, num_points, 4)
            data = data.reshape(data.size(0), data.size(1), -1)  # Flatten last dimensions to: (batch, time, num_nodes*num_points*4)
        elif self.model_input_shape == 'BTCHW':
            data = data.permute(0, 2, 1, 3, 4)  # Adjust as needed for BTCHW format.
        else:
            raise ValueError('Invalid model input shape for mmWave modality')

        return data

    def _convert_userid(self, data):
        return np.array(data)
    
    def _convert_polar(self, data):
        # Data shape: torch.Size([2, 1, 8]) # (batch_size, num_nodes (1), time)
        # squeeze the num_nodes dimension
        data = data.squeeze(1) # (batch_size, time)
        return data
    
    def _convert_activity(self, data):
        return np.array([self.activity_id_mapping[activity] for activity in data])
    
    def inverse_convert_activity(self, activity_ids):
        return [self.id_activity_mapping[i] for i in activity_ids]
        
        
        