# OctoNet Toolbox Quick Start Guide
<div style="text-align:center;">
  <a href="https://aiot-lab.github.io/OctoNet/" target="_blank">
    <img src="https://img.shields.io/badge/Project%20Page-Visit-blue" alt="Project Page" style="margin-right:10px;">
  </a>
  <a href="https://github.com/aiot-lab/OctoNet" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-Visit-lightgrey" alt="GitHub" style="margin-right:10px;">
  </a>
  <a href="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License" style="margin-right:10px;">
    <img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License" style="margin-right:10px;">
  </a>
</div>

This toolbox contains visualization code for selecting and visualizing the OctoNet dataset.

> **Note:** It is recommended to run the code in Python Jupyter Notebook `demo.ipynb`.

## Dataset Download
Download the dataset from [huggingface](https://huggingface.co/datasets/hku-aiot/OctoNet) using the provided script.

```bash
bash download_octonet.sh
```

**The following is the directory structure of the dataset:**
```bash
./dataset
├── mocap_csv_final          # Data: Final motion capture data in CSV format.
├── mocap_pose               # Data: Final motion capture data in npy format.
├── node_1                   # Data: Data related to multi-modal sensor node 1.
├── node_2                   # Data: Data related to multi-modal sensor node 2.
├── node_3                   # Data: Data related to multi-modal sensor node 3.
├── node_4                   # Data: Data related to multi-modal sensor node 4.
├── node_5                   # Data: Data related to multi-modal sensor node 5.
├── imu                      # Data: Inertial measurement unit data.
├── vayyar_pickle            # Data: vayyar mmWave radar data.
└── cut_manual.csv           # Manually curated data cuts.
```

<details>
<summary>📋Metadata of OctoNet Dataset</summary>
Note: <br>
<ul>
<li>Gender is denoted by male (M) and female (F).</li>
<li>PA&F indicates that a subject performed both Programmed Aerobics and Freestyle. The asterisk (*) marks subjects who performed only Programmed Aerobics (no Freestyle).</li>
<li>For Scene&hArr;Exp ID mapping, Scene 1: 1-99, Scene 2: 101-199, Scene 3: 201-299.</li>
</ul>
<br>

| User (Gender) | Exp ID                   | Scene 1: Activity IDs | Scene 1: PA&F | Scene 2: Activity IDs | Scene 2: PA&F | Scene 3: Activity IDs   | Scene 3: PA&F |
|---------------|--------------------------|:---------------------:|:-------------:|:---------------------:|:-------------:|:-----------------------:|:-------------:|
| 1 (M)         | 1, 11, 101, 201          | all 62                | ✓             | 1–23                  |               | 1–23, 57–62             | ✓*            |
| 2 (M)         | 2, 12, 102, 112, 202     | all 62                | ✓             | 9–29                  | ✓             | 9–29                    |               |
| 3 (M)         | 3, 13, 113, 213          | all 62                | ✓             |                       | ✓             |                         | ✓             |
| 4 (F)         | 4, 14, 104, 114, 204     | all 62                | ✓             | 30–56                 | ✓             | 30–56                   |               |
| 5 (M)         | 5, 15, 115, 215          | all 62                | ✓             |                       | ✓             |                         | ✓             |
| 6 (F)         | 6, 16                    | all 62                | ✓             |                       |               |                         |               |
| 7 (M)         | 7, 17, 117, 217          | all 62                | ✓             |                       | ✓             |                         | ✓             |
| 8 (M)         | 8, 18, 108, 118          | all 62                | ✓             | 24–62                 | ✓             | 24–62                   |               |
| 9 (M)         | 9                        | all 62                |               |                       |               |                         |               |
| 10 (M)        | 10, 20, 120, 220         | all 62                | ✓             |                       | ✓             |                         | ✓             |
| 11 (F)        | 21                       |                       | ✓             |                       |               |                         |               |
| 12 (M)        | 22                       |                       | ✓             |                       |               |                         |               |
| 13 (F)        | 23                       |                       | ✓             |                       |               |                         |               |
| 14 (M)        | 24                       |                       | ✓             |                       |               |                         |               |
| 15 (F)        | 25                       |                       | ✓             |                       |               |                         |               |
| 16 (F)        | 26                       |                       | ✓             |                       |               |                         |               |
| 17 (F)        | 27                       |                       | ✓             |                       |               |                         |               |
| 18 (F)        | 28                       |                       | ✓             |                       |               |                         |               |
| 19 (F)        | 29                       |                       | ✓             |                       |               |                         |               |
| 20 (F)        | 30, 230                  |                       | ✓             |                       |               |                         | ✓             |
| 21 (M)        | 31                       |                       | ✓             |                       |               |                         |               |
| 22 (M)        | 32                       |                       | ✓             |                       |               |                         |               |
| 23 (F)        | 33                       |                       | ✓             |                       |               |                         |               |
| 24 (M)        | 34                       |                       | ✓             |                       |               |                         |               |
| 25 (M)        | 35                       |                       | ✓             |                       |               |                         |               |
| 26 (M)        | 36                       |                       | ✓             |                       |               |                         |               |
| 27 (M)        | 37                       |                       | ✓             |                       |               |                         |               |
| 28 (F)        | 38                       |                       | ✓             |                       |               |                         |               |
| 29 (F)        | 39                       |                       | ✓             |                       |               |                         |               |
| 30 (M)        | 40                       |                       | ✓             |                       |               |                         |               |
| 31 (M)        | 41                       |                       | ✓             |                       |               |                         |               |
| 32 (F)        | 42                       |                       | ✓             |                       |               |                         |               |
| 33 (F)        | 43                       |                       | ✓             |                       |               |                         |               |
| 34 (F)        | 44                       |                       | ✓             |                       |               |                         |               |
| 35 (M)        | 45                       |                       | ✓             |                       |               |                         |               |
| 36 (M)        | 46                       |                       | ✓             |                       |               |                         |               |
| 37 (M)        | 47                       |                       | ✓             |                       |               |                         |               |
| 38 (F)        | 48                       |                       | ✓             |                       |               |                         |               |
| 39 (F)        | 49                       |                       | ✓             |                       |               |                         |               |
| 40 (M)        | 111, 211                 |                       |               | 1–8                   | ✓             | 1–8                     | ✓             |
| 41 (F)        | 121, 221                 |                       |               |                       | ✓             |                         | ✓             |
</details>


## Environment Setup
Create Anaconda environment `octonet` using the environment.yaml file.

```bash
conda env create -f environment.yaml
pip install -r requirements.txt
# conda activate octonet
```

Then, start the jupyter notebook `demo.ipynb` using octonet environment.

## Sample Data Selection
In `dataset_loader.py`, we define function `get_dataset` to easily load the dataset using a config file (example see the following section).

```python
def get_dataset(config, dataset_path="", mocap_downsample_num = None) -> OctonetDataset:
    """
    Args:
        config: config file
        dataset_path: path to the dataset
        mocap_downsample_num: number of downsample for mocap data, could be shadowed by config['mocap_downsample_num']
    Returns:
        OctonetDataset: a dataset object
    """
    ...
```

### Config for dataset selection
A full version of the config is shown below:
```python
config = {
    'exp_list': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 101, 102, 104, 108, 111, 112, 113, 114, 115, 117, 118, 120, 121, 201, 202, 204, 208, 211, 213, 215, 217, 220, 221, 230]
    'activity_list': ['sit', 'walk', 'bow', 'sleep', 'dance', 'jog', 'falldown', 'jump', 'jumpingjack', 'thunmbup'
        'squat', 'lunge', 'turn', 'pushup', 'legraise', 'airdrum', 'boxing', 'shakehead',
        'answerphone', 'eat', 'drink', 'wipeface', 'pickup', 'jumprope', 'moppingfloor',
        'brushhair', 'bicepcurl', 'playphone', 'brushteeth', 'type', 'thumbup',
        'makeoksign', 'makevictorysign', 'drawcircleclockwise', 'drawcirclecounterclockwise',
        'stopsign', 'pullhandin','pushhandaway', 'handwave', 'sweep', 'clap', 'slide',
        'drawzigzag', 'dodge', 'bowling', 'liftupahand', 'tap', 'spreadandpinch', 'drawtriangle',
        'sneeze', 'cough', 'stagger', 'yawn', 'blownose', 'stretchoneself', 'touchface',
        'handshake', 'hug', 'pushsomeone', 'kicksomeone', 'punchsomeone', 'conversation', 'gym', 'freestyle'],  # Specify which activities to filter
    'node_id': [1, 2, 3, 4, 5], 
    'segmentation_flag': True, # whether to include segmentation in the dataset
    'modality': [ 'mmWave', 'IRA', 'uwb', 'ToF', 'polar', 'wifi', 'depthCamera', 'seekThermal','acoustic', 'imu', 'vayyar', 'mocap'] # depthCamera is RGB-D camera
}
```

To select a subset of the dataset, you can modify the config file. For example:
```python
config = {
    'exp_list': [1],  # select exp 1
    'activity_list': ['dance'],  # select activity 'dance'
    'node_id': [1, 2, 3, 4, 5],  # select all nodes
    'segmentation_flag': True, # data is segmented
    'modality': [ 'mmWave', 'IRA', 'uwb', 'ToF', 'polar', 'wifi', 'depthCamera', 'seekThermal','acoustic', 'imu', 'vayyar', 'mocap'], # select all modalities
    # 'modality': ['polar', 'depthCamera'], # select polar and depthCamera modalities
    # 'mocap_downsample_num': 6 # downsample the mocap data to 6 frames per second
}
```

> **Note:** `get_dataset` will try it best to include all the data that satisfies the config.

### Visualization
The visualization code is provided in `demo.ipynb`. It will generate figures and videos in the `vis_output` folder in the root directory.

```python
# Sample configuration and usage
dataset_path = "dataset"
data_config = {
    'exp_list': [1],  # Specify which experiments to filter
    'activity_list': ['dance'],  
    'node_id': [1, 2, 3, 4, 5], 
    'segmentation_flag': True,
    'modality': [ 'mmWave', 'IRA', 'uwb', 'ToF', 'polar', 'wifi', 'depthCamera', 'seekThermal','acoustic', 'imu', 'vayyar', 'mocap'],
    # 'modality': ['polar', 'depthCamera'],
    # 'mocap_downsample_num': 6
}

# Get the DataLoader
dataset = get_dataset(data_config, dataset_path)
dataloader = get_dataloader(dataset, batch_size=1, shuffle=False, config=data_config)

for batch in dataloader:
    dump_seekthermal_frames_as_png(
        batch, 
        output_dir="validation_seekthermal"
    )
    visualize_seekthermal_and_rgb_mosaic_batch_discard_excess(
        batch,
        output_dir='seekthermal_rgb_mosaic_videos',
        fps_out=8.80
    )
    visualize_3_depth_3_rgb_mosaic_batch_discard_excess(
        batch,
        output_dir='depth_rgb_mosaic_discard',
        fps_out=10
    )
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
        fps_out=10
    )
    visualize_tof_and_rgb_mosaic_batch_downsample_tof(
        batch,
        output_dir='tof_rgb_mosaic_videos',
        fps_out=7.32
    )
    visualize_fmcw_and_rgb_mosaic_batch_raw_fixed_axes(
        batch,
        output_dir='fmcw_rgb_mosaic',
        fps_out=8.81
    )
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
        fps_out=10.0,
        y_domain=None
    )
    visualize_imu_four_rows_no_zscore(
        batch,
        output_dir="imu_time_features_plus_rgb",
        fps_out=10.0
    )
    visualize_uwb_and_rgb_in_same_row_with_box(
        batch,
        output_dir="uwb_rgb_same_row_with_box",
        fps_out=10.0
    )
    break
```

## License
This project is licensed under the GPL-3.0 License. See the [LICENSE](./LICENSE) file for details.
