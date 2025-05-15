import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# get the id of the available i-th gpu
def get_gpu_ids(i):
    gpu_ids = []
    gpu_info = os.popen("nvidia-smi -L").readlines()
    for line in gpu_info:
        # print(line)
        ids = line.split("UUID: ")[-1].strip(" ()\n")
        if ids.startswith("GPU"):
            continue
        gpu_ids.append(ids)
    if i >= len(gpu_ids):
        print("The number of the gpu is not enough! using the 0 by default")
        return gpu_ids[0]
    return gpu_ids[i]

def SubpageInterpolating(subpage: np.ndarray) -> np.ndarray:
    """
    Interpolate zero (or <=0) values by averaging neighbors (up, down, left, right).
    Simple in-place approach for IRA arrays.
    """
    mat = subpage.copy()
    rows, cols = mat.shape
    for i in range(rows):
        for j in range(cols):
            if mat[i, j] > 0.0:
                continue
            neighbors = []
            if i-1 >= 0: 
                neighbors.append(mat[i-1, j])
            if i+1 < rows: 
                neighbors.append(mat[i+1, j])
            if j-1 >= 0: 
                neighbors.append(mat[i, j-1])
            if j+1 < cols: 
                neighbors.append(mat[i, j+1])
            if len(neighbors) > 0:
                mat[i, j] = sum(neighbors) / len(neighbors)
            else:
                mat[i, j] = 0.0
    return mat

# excute the model and get the predictions and the labels
def inference(model,data_loader,data_shape_converter,rotary_physcis_prior_embedding,config,device, criterion =None, metric = None):
    model.eval()  # Set the model to evaluation mode
    recordings = {
        'predicts': [],
        # 'attributes': [],
        'labels': [],
        'metrics': [],
    }
    
    loss_all = 0
    dataset = data_loader.dataset.dataset if hasattr(data_loader.dataset, 'dataset') else data_loader.dataset
    with torch.no_grad():  # Disable gradient computation during evaluation
        for i, sample in enumerate(tqdm(data_loader, disable= config['tqdm_disable'])):
            data = sample["modality_data"]
            label = sample["activity"]

            # Convert activity labels (strings) to numerical class indices
            if isinstance(label, list):  # If it's a list of labels, map all of them
                label = [data_shape_converter.activity_id_mapping[act] for act in label]
            else:  # If it's a single label, map it directly
                label = data_shape_converter.activity_id_mapping[label]
            # recordings['attributes'].append(attr)
            if data_shape_converter is not None:
                if rotary_physcis_prior_embedding is not None:
                    data = rotary_physcis_prior_embedding(data)
                user_id, activity_id, input, modality = data_shape_converter.shape_convert(sample)
                input = input[modality]
            else:
                input = data
            if config['format'] == 'complex':
                input = input.cfloat().to(device)
            elif (config['format'] == 'dfs' or config['format'] == 'dense_dfs' or config['format'] == 'dense_dfs_amp') and config['model_input_shape'] == 'BCHW-C':
                input = input.cfloat().to(device)   
            else:
                input = input.float().to(device)   
            output = model(input)
            predicts = output
                
            if criterion is not None: # if the loss function is not None, then calculate the loss
                if config['criterion'] == 'mse' or config['criterion'] == 'bone_loss':
                    label = label.float().to(device)
                else:
                    label = torch.tensor(label).long().to(device)
                loss = criterion(predicts, label)
                loss_all += loss.item()
            if metric is not None:
                metric_value = metric(predicts, label)
                recordings['metrics'].append(metric_value)
            recordings['labels'].append(label.cpu().numpy())
            recordings['predicts'].append(predicts.cpu().numpy())
    return recordings, loss_all


# model training
def train(model,train_loader,validation_loader,data_shape_converter,criterion,regularizer,regularizer_lambda,rotary_physcis_prior_embedding,optimizer,scheduler, metric,config,log_file_name,writer,device):
    best_loss = float('inf')
    best_model_weights = None
    best_model_epoch = 0
    
    # Retrieve dataset reference (assuming all splits belong to the same dataset instance)
    dataset = train_loader.dataset.dataset if hasattr(train_loader.dataset, 'dataset') else train_loader.dataset
    
    try:  # parameter for the gradient clipping
        max_norm = config['max_norm']
    except:
        max_norm = None

    for epoch in range(config['num_epochs']):
        model.train()  # Set the model to training mode
        loss_epoch = 0
        metric_epoch = 0
        for i, sample in enumerate(tqdm(train_loader ,disable= config['tqdm_disable'])):
            # print(sample.keys(), "sample.keys")
            data = sample["modality_data"]
            label = sample["activity"]
            # Convert activity labels (strings) to numerical class indices
            if isinstance(label, list):  # If it's a list of labels, map all of them
                label = [data_shape_converter.activity_id_mapping[act] for act in label]
            else:  # If it's a single label, map it directly
                label = data_shape_converter.activity_id_mapping[label]

            # user_id = sample["user_id"]    # Only if you need it
            if data_shape_converter is not None:
                if rotary_physcis_prior_embedding is not None:
                    data = rotary_physcis_prior_embedding(data)
                user_id, activity_id, input, modality = data_shape_converter.shape_convert(sample)
                input = input[modality]
                
            else:
                input = data
            loss = 0
            # with torch.autograd.detect_anomaly():
            if config['format'] == 'complex':
                input = input.cfloat().to(device)
            elif (config['format'] == 'dfs' or config['format'] == 'dense_dfs' or config['format'] == 'dense_dfs_amp') and config['model_input_shape'] == 'BCHW-C':
                input = input.cfloat().to(device)
            else:
                input = input.float().to(device) 
            
            if config['criterion'] == 'mse' or config['criterion'] == 'bone_loss':
                label = label.float().to(device)
            else:
                label = torch.tensor(label).long().to(device)  # Convert list to tensor and set long type
            # if the label is more than two dimensions: (Batch_size, dim_num, ...) -> (Batch_size, ...)
            if len(label.shape) > 2:
                label = label.squeeze()
                label = label.reshape(label.shape[0], -1)
            
            output = model(input)
            
            if 'rf_crate' in config['model_config']['model_name']:  # the output of the rf_crate model is a tuple that contains the predicts and the feature, i.e., output of the last transformer block.
                predicts, feature = output
            else:
                predicts = output
                
            loss = criterion(predicts, label)
            if regularizer is not None:
                loss += regularizer_lambda*regularizer(feature)
                
            loss_epoch += loss.item()
            optimizer.zero_grad()
            loss.backward()
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm, norm_type=2)
            optimizer.step()
            writer.add_scalar('train loss (per batch)', loss.item(), epoch * len(train_loader) + i)
            if metric is not None:
                metric_value = metric(predicts, label)
                metric_epoch += metric_value
                writer.add_scalar('train metric (per batch)', metric_value, epoch * len(train_loader) + i)
            # ✅ Clear cache safely
            # if hasattr(dataset, 'depthcamera_cache'):
                # dataset.depthcamera_cache.clear()
            # if hasattr(dataset, 'seekthermal_cache'):
            #     dataset.seekthermal_cache.clear()
            
        if scheduler is not None:
            scheduler.step()
        print(f'Epoch {epoch}, Average Loss (train set) {loss_epoch/ len(train_loader):.10f}')
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('train loss (per sample)', loss_epoch/ len(train_loader), epoch)
        if metric is not None:
            writer.add_scalar('train metric (per sample)', metric_epoch/ len(train_loader), epoch)
        
        recordings, loss_all = inference(model,validation_loader,data_shape_converter,rotary_physcis_prior_embedding,config,device, criterion, metric)
        

        metric_all = recordings['metrics']
        if len(metric_all) > 0:
            metric_all = np.mean(metric_all)
            writer.add_scalar('validation metric (per sample)', metric_all, epoch)
        writer.add_scalar('validation loss (per sample)', loss_all/ len(validation_loader), epoch)
        print(f'Epoch {epoch}, Average Loss (valid set) {loss_all/ len(validation_loader):.10f}')

        if loss_all<best_loss and (not np.isnan(loss_all)):
            best_model_weights = model.state_dict()
            best_loss = loss_all
            best_model_epoch = epoch
        else:
            if epoch - best_model_epoch >= config['early_stop']:    
                print('early stop with best model at epoch: ',best_model_epoch)
                break
        # if the loss is nan, then stop the training
        # if np.isnan(loss_all):
        #     print('early nan stop with best model at epoch: ',best_model_epoch)
        #     break
        
    return best_model_weights, best_model_epoch



# metrics calculation
def accuracy(predicts, labels):
    # predicts: (batch_size, num_classes)
    # labels: (batch_size, )
    if isinstance(predicts, torch.Tensor):
        predicts = predicts.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    predicts = np.argmax(predicts, axis=1)
    return np.sum(predicts == labels) / len(labels)

def precision(predicts, labels):
    # predicts: (batch_size, num_classes) ; num_classes = 2
    # labels: (batch_size, )
    if isinstance(predicts, torch.Tensor):
        predicts = predicts.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    predicts = np.argmax(predicts, axis=1)
    tp = np.sum((predicts == labels) & (predicts == 1))
    fp = np.sum((predicts != labels) & (predicts == 1))
    return tp / (tp + fp)

def recall(predicts, labels):
    # predicts: (batch_size, num_classes) ; num_classes = 2
    # labels: (batch_size, )
    if isinstance(predicts, torch.Tensor):
        predicts = predicts.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    predicts = np.argmax(predicts, axis=1)
    tp = np.sum((predicts == labels) & (predicts == 1))
    fn = np.sum((predicts != labels) & (predicts == 0))
    return tp / (tp + fn)

def f1_score(predicts, labels):
    # predicts: (batch_size, num_classes) ; num_classes = 2
    # labels: (batch_size, )
    if isinstance(predicts, torch.Tensor):
        predicts = predicts.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    predicts = np.argmax(predicts, axis=1)
    tp = np.sum((predicts == labels) & (predicts == 1))
    fp = np.sum((predicts != labels) & (predicts == 1))
    fn = np.sum((predicts != labels) & (predicts == 0))
    return tp / (tp + 0.5 * (fp + fn))


# mean per joint position error
def MPJPE_2D(predicts, labels, per_sample = False):
    # if the shape of the predicts and labels are : (batch_size, num_joints * 2)
    # reshape the predicts and labels to (batch_size, num_joints, dim)
    if isinstance(predicts, torch.Tensor):
        predicts = predicts.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    predicts = predicts.reshape(predicts.shape[0], -1, 2)
    labels = labels.reshape(labels.shape[0], -1, 2)
    per_sample_mpjpe = np.mean(np.linalg.norm(predicts - labels, axis=2), axis=1)
    if per_sample:
        return per_sample_mpjpe
    return np.mean(per_sample_mpjpe, axis=0)

def MPJPE_3D(predicts, labels, per_sample = False):
    # if the shape of the predicts and labels are : (batch_size, num_joints * 3) 
    # reshape the predicts and labels to (batch_size, num_joints, dim)
    if isinstance(predicts, torch.Tensor):
        predicts = predicts.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    predicts = predicts.reshape(predicts.shape[0], -1, 3)
    labels = labels.reshape(labels.shape[0], -1, 3)
    per_sample_mpjpe = np.mean(np.linalg.norm(predicts - labels, axis=2), axis=1)
    if per_sample:
        return per_sample_mpjpe
    return np.mean(per_sample_mpjpe)

# loss function and regularization

class BoneLength_Position_Loss(nn.Module):
    def __init__(self):
        super(BoneLength_Position_Loss, self).__init__()
        self.BONE_CONNECTIONS_3D = torch.tensor([
            (0,1),(1,2),(2,3),  # head and torso
            (4,5), (5,6), (6,7), # right arm
            (8,9), (9,10), (10,11), # left arm
            (12,13), (13,14), (14,15), # right leg
            (16,17), (17,18), (18,19), # left leg
            (0,16), (0,12), (12,16), # torso and legs
        ])  # the bone connections for the 3D human pose in octonetmini dataset
        
        self.BONE_CONNECTIONS_2D = torch.tensor([
            (6,7), # head
            (6,11), (11,12), (12,13), # right arm
            (6,8), (8,9), (9,10), # left arm
            (6,3), (3,4), (4,5), # right leg
            (6,0), (0,1), (1,2), # left leg
        ])

    def forward(self, pred, gt):
        # pred: (batch_size, num_joints * dim)
        # gt: (batch_size, num_joints * dim)
        
        # Extract joints based on connections for both pred and gt
        if pred.shape[1] == 28:   # 2D human pose: 14 joints
            BONE_CONNECTIONS = self.BONE_CONNECTIONS_2D
            pred = pred.reshape(pred.shape[0], -1, 2)
            gt = gt.reshape(gt.shape[0], -1, 2)
        else:
            BONE_CONNECTIONS = self.BONE_CONNECTIONS_3D
            pred = pred.reshape(pred.shape[0], -1, 3)
            gt = gt.reshape(gt.shape[0], -1, 3)
        pred_joints1 = pred[:, BONE_CONNECTIONS[:, 0]]
        pred_joints2 = pred[:, BONE_CONNECTIONS[:, 1]]
        gt_joints1 = gt[:, BONE_CONNECTIONS[:, 0]]
        gt_joints2 = gt[:, BONE_CONNECTIONS[:, 1]]
        # Calculate bone lengths for both pred and gt
        pred_bone_lengths = torch.norm(pred_joints1 - pred_joints2, dim=2)
        gt_bone_lengths = torch.norm(gt_joints1 - gt_joints2, dim=2)
        # Calculate the mean square error of the bone lengths
        length_loss = torch.mean((pred_bone_lengths - gt_bone_lengths) ** 2)
        
        # calculate the position loss
        position_loss = torch.mean((pred - gt) ** 2)
        return length_loss + position_loss
    
    
class Subspace_Regularization(nn.Module):
    def __init__(self, num_subspace, dim):
        # num_subspace: the number of the subspaces which euqals to the number of the heads in the transformer of RF_CRATE
        # dim: the dimension of the feature, which is the same as that in the RF_CRATE
        super(Subspace_Regularization, self).__init__()
        self.num_subspace = num_subspace
        self.dim = dim
        
    def forward(self, feature):
        # feature: torch.complex, (batch_size, seq_len (CLS, patch1, patch2, ..), dim)
        if feature.shape[2] % self.num_subspace != 0:
            print("The dimension of the feature is not divisible by the number of the subspace")
            return None
        feature = feature.reshape(feature.shape[0], feature.shape[1], self.num_subspace, self.dim // self.num_subspace) # (batch_size, seq_len, num_subspace, dim // num_subspace)
        # except the CLS tocken
        feature = feature[:, 1:, :, :]  # (batch_size, seq_len - 1, num_subspace, dim // num_subspace)
        # comput the magnitude of the feature
        feature_magnitude = torch.norm(feature, dim=-1) # (batch_size, seq_len - 1, num_subspace)
        # comput the average of the magnitude of the feature in each subspace
        feature_magnitude = torch.mean(feature_magnitude, dim=1) # (batch_size, num_subspace)
        # get the mean across the batch
        feature_magnitude = torch.mean(feature_magnitude, dim=0) # (num_subspace, )
        # normalize the feature magnitude
        feature_magnitude = feature_magnitude / torch.max(feature_magnitude)
        # enforce the average of the magnitude of the feature in each subspace to be 1
        regularization_loss = torch.mean((feature_magnitude - 1) ** 2)
        return regularization_loss


class CSI_subcarrerier_frequence():
    def __init__(self, device_name):
        # device_name: the name of the device, e.g., '802.11n_csitool', '802.11ax_octonetmini_csitool'
        self.device_name = device_name
        # band information: https://en.wikipedia.org/wiki/List_of_WLAN_channels
        if device_name == '802.11n_csitool': # channel 165 at 5.825 GHz in Widar3.0/Wigait datasets with 20 MHz bandwidth
            self.band_center_frequency = 5825    # MHz
            self.subcarrier_spacing = 312.5      # KHz
            self.num_subcarriers = 30
            self.carrier_grouping = [-28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28]
        elif device_name == '802.11ax_octonetmini_csitool': # channel 36 at 5.18 GHz in Octonetmini dataset with 40 MHz bandwidth
            self.band_center_frequency = 5190    # MHz  # for 40 MHz bandwidth it is 5190 MHz, wihle for 20 MHz bandwidth it is 5180 MHz
            self.subcarrier_spacing = 312.5      # KHz
            self.num_subcarriers = 114
            self.carrier_grouping = [-58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
            

    def _calculate_subcarrier_center_frequency(self,band_center_frequency, subcarrier_spacing, num_subcarriers, carrier_grouping):
        # band_center_frequency: the center frequency of the band
        # subcarrier_spacing: the spacing between the subcarriers
        # num_subcarriers: the number of the subcarriers
        # carrier_grouping: the index of group that the subcarrier belongs to. Check the Table 7-25f in the 802.11n-2009 standard or the 802.11ax standard
        # return the center frequency of each subcarrier
        if num_subcarriers != len(carrier_grouping):
            print("The number of the subcarriers is not equal to the length of the carrier_grouping")
        center_frequencies = []
        for i in range(num_subcarriers):
            frequency_shift = carrier_grouping[i] * subcarrier_spacing / 1000   # MHz
            center_frequency = band_center_frequency + frequency_shift  # MHz
            center_frequencies.append(center_frequency)
        return center_frequencies 
    
    def get_subcarrier_center_frequency(self):
        return self._calculate_subcarrier_center_frequency(self.band_center_frequency, self.subcarrier_spacing, self.num_subcarriers, self.carrier_grouping)


class RotaryPhysicPriorEmbedding(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        if config['dataset_name'] == 'widar3' or config['dataset_name'] == 'widar_gait' or config['dataset_name'] == 'OctoNetMini':  # all wifi csi datasets
            if config['format'] == 'dense_dfs' or config['format'] == 'dense_dfs_amp':
                # (256 or 300, 121, 30, 3, 2) widar3 and widargait dense dfs: (time_bin, freq_bin, subcarrier, rx*tx, complex (real, imag)), time_step: 10/1000 = 0.01s
                # Octonetmini dense dfs: (60, 61, 114, 4, 2) (time_bin, freq_bin, subcarrier, rx*tx, complex (real, imag)), time_step: 20/80 = 0.25s
                rx_dim = 4  # take the maximum number of the rx*tx antennas
                freq_dim = 121
                time_dim = 1500  # 300*0.01 = 3s, 60*0.25 = 15s:: time_bins = 15s/0.01s = 1500
                subcarrier_dim = 2119  # considering the real center frequency of the subcarriers across both widar3 and octonetmini datasets
                total_dim = (rx_dim, subcarrier_dim, freq_dim, time_dim)  # based on the physical prior of the csi data
                
                if config['dataset_name'] == 'widar3' or config['dataset_name'] == 'widar_gait':  # they use the same device setting: 802.11n CSItool
                    select_rx_index = [0, 1, 2]
                    select_subcarrier_index = [2062, 2064, 2066, 2068, 2070, 2072, 2074, 2076, 2078, 2080, 2082, 2084, 2086, 2088, 2089, 2091, 2093, 2095, 2097, 2099, 2101, 2103, 2105, 2107, 2109, 2111, 2113, 2115, 2117, 2118]
                    select_freq_index = list(range(121))
                    select_time_index = list(range(256)) if config['dataset_name'] == 'widar3' else list(range(300))
                    selected_dim = [select_rx_index, select_subcarrier_index, select_freq_index, select_time_index]
                    rotary_range = (0, np.pi)  # the range of the rotary values
                else:  # OctoNetMini dataset
                    select_rx_index = [0, 1, 2, 3]
                    select_subcarrier_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116]
                    select_freq_index = [i+30 for i in range(61)]
                    # get the time index based on the time_step: from 0 to 1500 every 25, then we have 60 time bins
                    select_time_index = [i*25 for i in range(60)]
                    selected_dim = [select_rx_index, select_subcarrier_index, select_freq_index, select_time_index]
                    rotary_range = (0, np.pi)  # the range of the rotary values
                rotary_embedding = self._generate_rotary_embedding(total_dim, selected_dim, rotary_range)
                # change the dims to match the input shape
                rotary_embedding = rotary_embedding.permute(3, 2, 1, 0)  # (time_dim, freq_dim, subcarrier_dim, rx_dim,)    
            else:
                raise ValueError(f'Invalid format: {config["format"]}')
        else:
            raise ValueError(f'Invalid dataset name: {config["dataset_name"]}')
        # Register the rotary_embedding as a buffer
        rotary_embedding = rotary_embedding.to(device)
        self.register_buffer('rotary_embedding', rotary_embedding)
    
    def _generate_rotary_embedding(self, total_dim, selected_dim, rotary_range):
        # Calculate the total number of rotary values
        total_rotary_num = np.prod(total_dim)
        # Generate evenly spaced rotary values within the specified range
        rotarys = torch.linspace(rotary_range[0], rotary_range[1], total_rotary_num)
        # Reshape the rotary values to match the total_dim
        rotarys = rotarys.reshape(total_dim)
        # Use advanced indexing to select the desired rotary values
        selected_indices = torch.meshgrid(
            *[torch.tensor(dim_indices) for dim_indices in selected_dim],
            indexing='ij'
        )
        selected_rotarys = rotarys[selected_indices]
        # Convert the selected rotary values to complex embeddings
        rotary_embedding = torch.exp(1j * selected_rotarys)
        return rotary_embedding

    def forward(self, batch):
        batch = batch.to(self.rotary_embedding.device)
        if self.config['dataset_name'] == 'widar3' or self.config['dataset_name'] == 'widar_gait' or self.config['dataset_name'] == 'OctoNetMini':  # all wifi csi datasets
            if self.config['format'] == 'dense_dfs': #batch  dense dfs format: (batchsize, time_bin, freq_bin, subcarrier, rx*tx, complex (real, imag))
                # self.rotary_embedding: (time_dim, freq_dim, subcarrier_dim, rx_dim,)
                real_part = batch[:,:,:,:,:,0]
                img_part = batch[:,:,:,:,:,1]
                batch = real_part + 1j*img_part
                # dot product with the rotary embedding
                batch = batch * self.rotary_embedding
                # separate the real and imaginary parts
                real_part = batch.real
                img_part = batch.imag
                batch = torch.stack((real_part, img_part), dim=-1)
                return batch
            elif self.config['format'] == 'dense_dfs_amp': #batch  dense dfs format: (batchsize, time_bin, freq_bin, subcarrier, rx*tx, 1 (magnitude))
                batch = batch[:,:,:,:,:,0]
                batch_time_len = batch.shape[1]
                initial_rotary_embedding_time_len = self.rotary_embedding.shape[0]
                if batch_time_len != initial_rotary_embedding_time_len:
                    self.rotary_embedding = self.rotary_embedding[:batch_time_len]
                real_part = batch*torch.cos(self.rotary_embedding)
                img_part = batch*torch.sin(self.rotary_embedding)
                # print(self.rotary_embedding.shape, batch.shape, real_part.shape, img_part.shape)
                # print(torch.stack((real_part, img_part), dim=-1).shape)
                return torch.stack((real_part, img_part), dim=-1)
            else:
                raise ValueError(f'Invalid format: {config["format"]}')
        else:
            raise ValueError(f'Invalid dataset name: {config["dataset_name"]}')

