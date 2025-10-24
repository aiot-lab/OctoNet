import cv2
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset,random_split
import yaml
import os
import argparse
import torch
from tqdm import tqdm
import math
import pickle
from torch.utils.tensorboard import SummaryWriter
import os
import timm
from timm.scheduler import CosineLRScheduler
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy
from lion_pytorch import Lion
# Datasets
# from Datasets import *
from shape_converter import DataConverter
# models
from Models import *
from func_utils import *
from octonet.Octonet import get_dataset, custom_collate
# from dataset_call import get_dataset, custom_collate

'''testing for reproducibility'''
# reproducibility helpers -------------------------------------------------
import random
def seed_everything(seed: int):
    """Fix torch / numpy / python randomness in *this* process."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # cuDNN & deterministic flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training and Testing')
    parser.add_argument("--config_file", type=str, help="Configuration YAML file")
    parser.add_argument("--cuda_index", type=int, default=0, help="The index of the cuda device")
    parser.add_argument("--mode", type=int, default=0, help="0: train + test, 1: test only, 2: finetune + test, 3: check the pipeline")
    parser.add_argument("--pretrained_model", type=str, default=None, help="The file path of the pretrained model weights")
    args = parser.parse_args()
    
    # loading the configuration file ########################################
    config_file_name = args.config_file + '.yaml'
    all_yaml_files = []
    all_yaml_file_paths = []
    for root, dirs, files in os.walk("Configurations"):
        for file in files:
            if file.endswith(".yaml"):
                all_yaml_files.append(file)
                all_yaml_file_paths.append(os.path.join(root, file))
    if config_file_name not in all_yaml_files:
        print("Configuration file name is: ",config_file_name)
        print("The configuration file is not found! Please check the file name!")
        exit
    else:
        # if there are multiple configuration files with the same name, print all the locations
        if all_yaml_files.count(config_file_name) > 1:
            print("The configuration file is: ",config_file_name)
            print("There are multiple configuration files with the same name!")
            for i in range(all_yaml_files.count(config_file_name)):
                print(all_yaml_file_paths[all_yaml_files.index(config_file_name, i)])
            exit
        else:
            config_file_path = all_yaml_file_paths[all_yaml_files.index(config_file_name)]
            print("The configuration file is found at: ",config_file_path)
            with open(config_file_path, 'r') as fd:
                config = yaml.load(fd, Loader=yaml.FullLoader)
    # traverse the configuration file if any value is a string: 'None', convert it to None
    for key, value in config.items():
        if value == 'None':
            config[key] = None
    # end the loading of the configuration file ########################################        
            
    config['cuda_index'] = args.cuda_index
    
    if args.mode == 1:
        print("Testing only!")
        if args.pretrained_model == None:
            print("Please provide the pretrained model weights!")
            exit
    elif args.mode == 2:
        print("Finetuning and Testing!")
        if args.pretrained_model == None:
            print("Please provide the pretrained model weights!")
            exit
    elif args.mode == 3:
        print("Check the pipeline!")
        config['num_epochs'] = 1
    else:
        print("Training and Testing!")
        

    
    tensorboard_folder = config['tensorboard_folder']
    if not os.path.exists(tensorboard_folder):
        os.makedirs(tensorboard_folder)
    trained_model_folder = config['trained_model_folder']
    if not os.path.exists(trained_model_folder):
        os.makedirs(trained_model_folder)
    log_folder = config['log_folder']   # save all the outputs records when testing
    log_enable = config['log_enable']   # if log_enable is False, the log folder will not be created
    if not os.path.exists(log_folder) and log_enable:
        os.makedirs(log_folder)

    model_save_enable = config['model_save_enable'] # if model_save_enable is False, the model weights will not be saved
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['cuda_index'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("The device is: ",device)
    
    # fix the seed for reproducibility
    rng_generator = torch.manual_seed(config['init_rand_seed'])
    torch.cuda.manual_seed(config['init_rand_seed'])
    np.random.seed(config['init_rand_seed'])
    # testing for reproducibility
    seed_everything(config['init_rand_seed'])
    
    localtime = time.localtime(time.time())
    if args.mode == 1:
        log_file_name = f"{args.config_file}_{config['model_config']['model_name']}_{time.strftime('%m%d%H%M%S', localtime)}_test"
    elif args.mode == 2:
        log_file_name = f"{args.config_file}_{config['model_config']['model_name']}_{time.strftime('%m%d%H%M%S', localtime)}_finetune"
    else:
        log_file_name = f"{args.config_file}_{config['model_config']['model_name']}_{time.strftime('%m%d%H%M%S', localtime)}"
    print("The log filename is: ",log_file_name)

    writer = SummaryWriter(config['tensorboard_folder'] + log_file_name)
    
    # Create data shape converter instance
    data_shape_converter = DataConverter(config['data_config'], config['model_config'])
    # If you do not need them:
    regularizer = None
    regularizer_lambda = None
    rotary_physcis_prior_embedding = None
    # configur the dataset and dataloader  ########################################
    train_ratio = config['data_split'][0]
    val_ratio = config['data_split'][1]
    test_ratio = config['data_split'][2]

    dataset = get_dataset(config['data_config'], config['dataset_path'], config.get('mocap_downsample_num', None))
    train_num = int(len(dataset)*train_ratio)
    val_num = int(len(dataset)*val_ratio)
    test_num = int(len(dataset)) - train_num - val_num
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_num, val_num, test_num], generator=rng_generator)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=custom_collate,
        generator=rng_generator,
        worker_init_fn=seed_worker,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=config['batch_size'],
        shuffle=False,
        worker_init_fn=seed_worker,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=config['batch_size'],
        shuffle=False,
        worker_init_fn=seed_worker,
        collate_fn=custom_collate
    )
    

    # configure the model, optimizer, scheduler, criterion, and metric  ########################################
    model_config = config['model_config']
    model_name = config['model_config']['model_name']
    model = get_registered_models(model_name, model_config)
    model.to(device)
    
    if args.mode == 1 or args.mode == 2:
        model.load_state_dict(torch.load(args.pretrained_model))
        print("The pretrained model weights are loaded!")


    if config['optimizer'] == "AdamW":
        try:
            momentum = config['momentum']
        except:
            momentum = 0.9
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(momentum, momentum + momentum/10), weight_decay=config['weight_decay'])                      
    elif config['optimizer'] == "Adam":
        try:
            momentum = config['momentum']
        except:
            momentum = 0.9
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(momentum, momentum + momentum/10), weight_decay=config['weight_decay'])
    else:
        try:
            momentum = config['momentum']
        except:
            momentum = 0.9
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=momentum, weight_decay=config['weight_decay'])
    try:
        warmup_steps = config['warmup_steps'] 
    except:
        warmup_steps = 10
    lr_func = lambda step: min((step + 1) / (warmup_steps + 1e-8), 0.5 * (math.cos(step / config['num_epochs'] * math.pi) + 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)

    if config['criterion'] == 'mse':
        criterion = nn.MSELoss()
    elif config['criterion'] == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif config['criterion'] == 'label_smoothing':
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    else:
        raise NotImplementedError
    criterion.to(device)

    # 'accuracy', 'f1_score', 'precision', 'recall', 'mpjpe_2d', 'mpjpe_3d'
    if config['metric'] == 'accuracy':
        metric = accuracy
    elif config['metric'] == 'f1_score':
        metric = f1_score
    elif config['metric'] == 'precision':
        metric = precision
    elif config['metric'] == 'recall':
        metric = recall
    elif config['metric'] == 'mpjpe_2d':
        metric = MPJPE_2D
    elif config['metric'] == 'mpjpe_3d':
        metric = MPJPE_3D
    else:
        raise NotImplementedError
    
    # training/finetuning  ########################################
    if args.mode == 0:
        print("Start training!")
        best_model_weights, best_model_epoch = train(model,train_loader,val_loader,data_shape_converter,criterion,regularizer,regularizer_lambda,rotary_physcis_prior_embedding,optimizer,scheduler,metric,config,log_file_name,writer,device)
        print("Training is done!")
        print("The best model is at epoch: ",best_model_epoch)
        model.load_state_dict(best_model_weights)
        if model_save_enable:
            saved_model_path = config['trained_model_folder']+ log_file_name + '.pth'
            torch.save(best_model_weights, saved_model_path)
    elif args.mode == 1:
        print("Start testing!")
    elif args.mode == 2 or args.mode == 3:
        print("Start finetuning or checking!")
        best_model_weights, best_model_epoch = train(model,train_loader,val_loader,data_shape_converter,criterion,regularizer,regularizer_lambda,rotary_physcis_prior_embedding,optimizer,scheduler,metric,config,log_file_name,writer,device)
        print("Finetuning or checking is done!")
        print("The best model is at epoch: ",best_model_epoch)
        model.load_state_dict(best_model_weights)
        if model_save_enable:
            saved_model_path = config['trained_model_folder']+ log_file_name + '.pth'
            torch.save(best_model_weights, saved_model_path)
    else:
        print("The mode is wrong!")
    
    # in domain testing ########################################
    if args.mode == 1:
        best_model_epoch = 0
    model.eval()
    recordings, loss_all = inference(model,test_loader,data_shape_converter,rotary_physcis_prior_embedding,config,device, criterion, metric)
    print(f'Average Loss (test set) {loss_all/ len(test_loader):.10f}')
    metric_all = recordings['metrics']
    if len(metric_all) > 0:
        metric_all = np.mean(metric_all)
        writer.add_scalar('test metric (per sample)', metric_all, best_model_epoch)
    writer.add_scalar('test loss (per sample)', loss_all/ len(test_loader), best_model_epoch)
    
    if log_enable:
        test_result_save_path = config['log_folder']+ log_file_name + '.pkl'
        pickle.dump(recordings, open(test_result_save_path, 'wb'))
        print("The test results are saved at: ",test_result_save_path)

    if hasattr(dataset, "rgbcamera_cache"):
        dataset.rgbcamera_cache.clear()
    if hasattr(dataset, "depthcamera_cache"):
        dataset.depthcamera_cache.clear()
    
    # ----------------------------------------------------------------------
    # Cross-domain testing (if present)
    # ----------------------------------------------------------------------
    try:
        cross_scene_data_config = config['cross_scene_data_config']
    except KeyError:
        cross_scene_data_config = None

    if cross_scene_data_config is not None:
        cross_scene_dataset = get_dataset(cross_scene_data_config, config['dataset_path'], config.get('mocap_downsample_num', None))
        cross_scene_loader = DataLoader(
            dataset=cross_scene_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            collate_fn=custom_collate,
            num_workers=config.get('num_workers', 0)
        )
        model.eval()
        recordings, loss_all = inference(model,
                                         cross_scene_loader,
                                         data_shape_converter,
                                         rotary_physcis_prior_embedding,
                                         config,
                                         device,
                                         criterion,
                                         metric)
        print(f'Average Loss (cross-domain test set) {loss_all / len(cross_scene_loader):.10f}')
        metric_all = recordings['metrics']
        if len(metric_all) > 0:
            metric_value = np.mean(metric_all)
            writer.add_scalar('cross-domain test metric (per sample)', metric_value, best_model_epoch)
        writer.add_scalar('cross-domain test loss (per sample)',
                          loss_all / len(cross_scene_loader),
                          best_model_epoch)

        if log_enable:
            test_result_save_path = os.path.join(log_folder, log_file_name + '_cross_domain.pkl')
            pickle.dump(recordings, open(test_result_save_path, 'wb'))
            print("The cross-domain test results are saved at: ", test_result_save_path)

    # ----------------------------------------------------------------------
    #  Cross-user testing (if present)
    # ----------------------------------------------------------------------
    try:
        cross_user_data_config = config['cross_user_data_config']
    except KeyError:
        cross_user_data_config = None

    if cross_user_data_config is not None:
        cross_user_dataset = get_dataset(cross_user_data_config, config['dataset_path'], config.get('mocap_downsample_num', None))
        cross_user_loader = DataLoader(
            dataset=cross_user_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            collate_fn=custom_collate,
            num_workers=config.get('num_workers', 0)
        ) 
        model.eval()
        recordings, loss_all = inference(model,
                                         cross_user_loader,
                                         data_shape_converter,
                                         rotary_physcis_prior_embedding,
                                         config,
                                         device,
                                         criterion,
                                         metric)
        print(f'Average Loss (cross-user test set) {loss_all / len(cross_user_loader):.10f}')
        metric_all = recordings['metrics']
        if len(metric_all) > 0:
            metric_value = np.mean(metric_all)
            writer.add_scalar('cross-user test metric (per sample)', metric_value, best_model_epoch)
        writer.add_scalar('cross-user test loss (per sample)',
                          loss_all / len(cross_user_loader),
                          best_model_epoch)

        if log_enable:
            test_result_save_path = os.path.join(log_folder, log_file_name + '_cross_user.pkl')
            pickle.dump(recordings, open(test_result_save_path, 'wb'))
            print("The cross-user test results are saved at: ", test_result_save_path)
    
    writer.close()
    print("All done!")
