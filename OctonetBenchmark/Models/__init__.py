from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .densenet import densenet121, densenet169, densenet201, densenet161
from .swin_transformer import swin_t, swin_s, swin_b, swin_v2_s, swin_v2_b, swin_v2_t
from .vision_transformer import vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14
from .crate import crate_small, crate_base, crate_large, crate_tiny

from .units import units_standard
from .stfnet import stfnet_standard
from .rfnet import rf_net
from .slnet import slnet_standard
from .widar3 import widar3_standard
from .laxcat import laxcat_standard

from .rf_crate import rf_crate_tiny, rf_crate_small, rf_crate_base, rf_crate_large, RF_CRATE


registered_models= {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,

    'densenet121': densenet121,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'densenet161': densenet161,

    'swin_t': swin_t,
    'swin_s': swin_s,
    'swin_b': swin_b,
    'swin_v2_s': swin_v2_s, 
    'swin_v2_b': swin_v2_b,
    'swin_v2_t': swin_v2_t,

    'crate_small': crate_small,
    'crate_base': crate_base,
    'crate_large': crate_large,
    'crate_tiny': crate_tiny,

    'units_standard': units_standard,
    'stfnet_standard': stfnet_standard,
    'rf_net': rf_net,
    'slnet_standard': slnet_standard,
    'widar3_standard': widar3_standard,
    }

def get_registered_models(model_name, config):
    if 'resnet' in model_name or 'densenet' in model_name or 'swin' in model_name or 'vit' in model_name:
        return registered_models[model_name](num_classes = config['num_classes'], in_channels = config['in_channels'])
    elif 'crate' in model_name and 'rf' not in model_name:  # the crate model
        return registered_models[model_name](num_classes = config['num_classes'], in_channels = config['in_channels'], 
                                             image_size = config['image_size'],patch_size = config['patch_size'])
    elif 'units' in model_name:
        return registered_models[model_name](num_classes = config['num_classes'], in_channels = config['in_channels'], 
                                             time_length = config['time_length'])
    elif 'stfnet' in model_name:
        sensor_num = config['sensor_num']  # the number of sensors, e.g., for wifi csi, sensor_num = num_rx_antennas * num_tx_antennas
        feature_dim = config['feature_dim']   # !!!!!! feature_dim % sensor_num == 0
        if feature_dim % sensor_num != 0:
            print("The feature_dim % sensor_num != 0")
            return None
        sensor_in_channels = config['in_channels'] // sensor_num   # the number of input channels for each sensor
        if sensor_in_channels * sensor_num != config['in_channels']:
            print("The in_channels should be divisible by sensor_num")
            return None
        seq_length = config['time_length']    # the sequence length of the input data
        num_classes = config['num_classes']     # the number of classes
        batch_size = config['batch_size']
        # GEN_FFT_N = [8, 16, 32, 64]
        # GEN_FFT_N2 = [6, 12, 24, 48]
        GEN_FFT_N = [16, 32, 64, 128]
        GEN_FFT_N2 = [12, 24, 48, 96]

        act_domain = config['act_domain']  # the domain of the activation function, 'freq' or 'time'
        if act_domain == 'freq':  # # for wifi: the CFR data
            freq_conv_flag = True
            filter_flag = False 
        elif act_domain == 'time':   # for time series
            freq_conv_flag = False
            filter_flag = True
        else:
            print("The activation domain for stfnet is wrong!, it should be 'freq' or 'time'")
            return None
        return registered_models[model_name](
            seq_length=seq_length, num_classes=num_classes, sensor_num=sensor_num, sensor_in_channels=sensor_in_channels,
            feature_dim=feature_dim, GEN_FFT_N=GEN_FFT_N, GEN_FFT_N2=GEN_FFT_N2, freq_conv_flag=freq_conv_flag, filter_flag=filter_flag,
            act_domain=act_domain, batch_size=batch_size)
    elif 'rf_net' in model_name:
        return registered_models[model_name](num_classes = config['num_classes'], in_channels = config['in_channels'], 
                                             seq_length = config['seq_length'] if 'seq_length' in config else config['time_length'])
    elif 'slnet' in model_name:
        num_classes = config['num_classes']     # the number of classes
        sensor_num = config['sensor_num']  # the number of sensors, e.g., for wifi csi, sensor_num = num_rx_antennas * num_tx_antennas
        sensor_in_channels = config['in_channels'] // sensor_num   # the number of input channels for each sensor, e.g., the No. subcarriers for wifi csi
        if sensor_in_channels * sensor_num != config['in_channels']:
            print("The in_channels should be divisible by sensor_num")
            return None
        freq_bins = config['freq_bins']    # stft freq bins
        time_steps = config['time_steps']     # stft time steps
        return registered_models[model_name](num_classes=num_classes, sensor_num=sensor_num, 
                                             sensor_in_channels=sensor_in_channels,freq_bins=freq_bins, time_steps=time_steps,)
    elif 'widar3' in model_name:    
        hight = config['hight']
        width = config['width']
        num_classes = config['num_classes']
        time_step = config['time_step']
        in_channels = config['in_channels']
        batch_size = config['batch_size']
        return registered_models[model_name](time_step, in_channels, hight,width, num_classes, batch_size)
    elif 'laxcat' in model_name:
        return registered_models[model_name](config['time_length'], config['in_channels'], config['num_classes'])
    elif 'rf_crate' in model_name:
        return registered_models[model_name](num_classes = config['num_classes'], in_channels = config['in_channels'], 
                                             image_size = config['image_size'],patch_size = config['patch_size'],
                                             feedforward=config['feedforward'], relu_type=config['relu_type'],
                                             patch_embedding_method=config['patch_embedding_method'],
                                             mlp_head_type=config['mlp_head_type'], output_type=config['output_type'])
    else:
        print("The model name is not registered!")
        return None