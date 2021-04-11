# MinkLoc++: Lidar and Monocular Image Fusion for Place Recognition

Paper: [MinkLoc++: Lidar and Monocular Image Fusion for Place Recognition](http://) IJCNN 2021

[Jacek Komorowski](mailto:jacek.komorowski@pw.edu.pl), Monika Wysoczańska, Tomasz Trzciński

Warsaw University of Technology

![Overview](media/overview.jpg)

### Introduction
We present a discriminative multimodal descriptor based on a pair of sensor readings: a point cloud from a LiDAR and an image from an RGB camera.
Our descriptor, named MinkLoc++, can be used for place recognition, re-localization and loop closure purposes in robotics or autonomous vehicles applications.
We use late fusion approach, where each modality is processed separately and fused in the final part of the processing pipeline.
The proposed method achieves state-of-the-art performance on standard place recognition benchmarks.
We also identify dominating modality problem when training a multimodal descriptor. 
The problem manifests itself when the network focuses on a modality with a larger overfit to the training data. This drives the loss down during the training but leads to suboptimal performance on the evaluation set.
In this work we describe how to detect and mitigate such risk when using a deep metric learning approach to train a multimodal neural network.
### Citation
If you find this work useful, please consider citing:

    @InProceedings{Komorowski_2021_IJCNN,
        author    = {XXX},
        title     = {XXXXX},
        booktitle = {XXXX},
        month     = {XXXX},
        year      = {2021},
        pages     = {XXXXX}
    }

### Environment and Dependencies
Code was tested using Python 3.8 with PyTorch 1.7.1 and MinkowskiEngine 0.5.2 on Ubuntu 18.04 with CUDA 10.2.

The following Python packages are required:
* PyTorch (version 1.7.1 or above)
* MinkowskiEngine (version 0.5.2)
* pytorch_metric_learning (version 0.9.98 or above)
* tensorboard
* bitarray


Modify the `PYTHONPATH` environment variable to include absolute path to the project root folder: 
```export PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/.../MinkLoc3D
```

### Datasets

**MinkLoc++** is a multimodal descriptor based on two inputs:
- a 3D point cloud constructed by aggregating multiple 2D LiDAR scans from Oxford RobotCar dataset 
- a corresponding RGB image from the stereo-center camera 

We use 3D point clouds built by authors of *PointNetVLAD: Deep Point Cloud Based Retrieval for Large-Scale Place Recognition* paper ([link](https://arxiv.org/pdf/1804.03492)).
Each point cloud is built by aggregating 2D LiDAR scans during the 20 meter vehicle traversal.
For details see PointNetVLAD paper or their github repository ([link](https://github.com/mikacuy/pointnetvlad)).
You can download training and evaluation point clouds from 
[here](https://drive.google.com/open?id=1rflmyfZ1v9cGGH0RL4qXRrKhg-8A-U9q) 
([alternative link](https://drive.google.com/file/d/1-1HA9Etw2PpZ8zHd3cjrfiZa8xzbp41J/view?usp=sharing)). 
Extract the folder in the same directory as the project code. Thus, in that directory you must have two folders: 1) benchmark_datasets and 2) MinkLoc3DRGB.

RGB images are taken directly from Oxford RobotCar dataset. 
First, you need to download stereo camera images from Oxford RobotCar dataset and then
run ```generate_rg_for_lidar.py``` script that finds 20 closest RGB images for each 3D point cloud and saves them in the 
target directory. 
During the training an input to the network consists of a 3D point cloud and one RGB image randomly chosen 
from these 20 corresponding images.
During the evaluation, a network input consists of a 3D point cloud and one RGB image with the closest timestamp.

Before the network training or evaluation, run the below code to generate training pickles (with positive and negative point 
clouds for each anchor point cloud) and evaluation pickles. The procedure for generating training and 
evaluation pickles is the same as proposed in PointNetVLAD paper. 
 
```generate pickles
cd generating_queries/ 

# Generate training tuples for the Baseline Dataset
python generate_training_tuples_baseline.py

# Generate training tuples for the Refined Dataset
python generate_training_tuples_refine.py

# Generate evaluation tuples
python generate_test_sets.py
```

### Training
To train **MinkLoc++** network, download and decompress the dataset and generate training pickles as described above.
Edit the configuration file (`config_baseline.txt` or `config_refined.txt`). 
Set `dataset_folder` parameter to the dataset root folder.
Modify `batch_size_limit` parameter depending on available GPU memory. 
Default limit (=256) requires at least 11GB of GPU RAM.

To train the network, run:

```train baseline
cd training

# To train minkloc3d model on the Baseline Dataset
python train.py --config ../config/config_baseline.txt --model_config ../models/minkloc3d.txt

# To train minkloc3d model on the Refined Dataset
python train.py --config ../config/config_refined.txt --model_config ../models/minkloc3d.txt
```

### Pre-trained Models

Pretrained models are available in `weights` directory
- `minkloc_multimodal.pth` 3D+RGB model trained on the Baseline Dataset with corresponding RGB images 
- `minkloc3d_baseline.pth` 3D model trained on the Baseline Dataset 
- `minkloc3d_refined.pth` 3D model trained on the Refined Dataset 

### Evaluation

To evaluate pretrained models run the following commands:

```eval baseline
cd eval

# To evaluate the model trained on the Baseline Dataset
python evaluate.py --config ../config/config_baseline.txt --model_config ../models/minkloc3d.txt --weights ../weights/minkloc3d_baseline.pth

# To evaluate the model trained on the Refined Dataset
python evaluate.py --config ../config/config_refined.txt --model_config ../models/minkloc3d.txt --weights ../weights/minkloc3d_refined.pth
```

## Results

**MinkLoc++** performance (measured by Average Recall@1\%) compared to state-of-the-art:

### Trained on Baseline Dataset

| Method         | Oxford  | U.S. | R.A. | B.D |
| ------------------ |---------------- | -------------- |---|---|
| PointNetVLAD [1] |     80.3     |   72.6 | 60.3 | 65.3 |
| PCAN [2] |     83.8     |   79.1 | 71.2 | 66.8 |
| DAGC [3] |     87.5     |   83.5 | 75.7 | 71.2 |
| LPD-Net [4] |     94.9   |   96.0 | 90.5 | **89.1** |
| EPC-Net [5] |     94.7   |   **96.5** | 88.6 | 84.9 |
| SOE-Net [6] |     96.4   |   93.2 | **91.5** | 88.5 |
| MinkLoc3D  |     97.9     |   95.0 | 91.2 | 88.5 |
| MinkLoc++ (RGB-only)  |     **98.3**     |    |  |  |


### Trained on Refined Dataset

| Method         | Oxford  | U.S. | R.A. | B.D |
| ------------------ |---------------- | -------------- |---|---|
| PointNetVLAD [1] |     80.1     |   94.5 | 93.1 | 86.5 |
| PCAN [2] |     86.4     |   94.1 | 92.3 | 87.0 |
| DAGC [3] |     87.8     |   94.3 | 93.4 | 88.5 |
| LPD-Net [4] |     94.9     |   98.9 | 96.4 | 94.4 |
| SOE-Net [6] |     96.4   |   **97.7** | 95.9 | 92.6 |
| **MinkLoc++ (RGB-only)**  |     **98.5**     |   **99.7** | **99.3** | **96.7** |


### Trained on 3D point cloud + RGB

| Method         | Oxford  | 
| ------------------ |-------|
| CORAL  |     96.1     |   
| PIC-Net  |     98.2   |  
| **MinkLoc++ (our)**  | 99.1 |      

1. M. A. Uy and G. H. Lee, "PointNetVLAD: Deep Point Cloud Based Retrieval for Large-Scale Place Recognition," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
2. W. Zhang and C. Xiao, "PCAN: 3D Attention Map Learning Using Contextual Information for Point Cloud Based Retrieval," 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
3. Q. Sun et al., "DAGC: Employing Dual Attention and Graph Convolution for Point Cloud based Place Recognition", Proceedings of the 2020 International Conference on Multimedia Retrieval
4. Z. Liu et al., "LPD-Net: 3D Point Cloud Learning for Large-Scale Place Recognition and Environment Analysis," 2019 IEEE/CVF International Conference on Computer Vision (ICCV)
5. L. Hui et al., "Efficient 3D Point Cloud Feature Learning for Large-Scale Place Recognition." arXiv preprint arXiv:2101.02374 (2021)
6. Y. Xia et al., "SOE-Net: A Self-Attention and Orientation Encoding Network for Point Cloud based Place Recognition." arXiv preprint arXiv:2011.12430 (2020)
* J. Komorowski, "MinkLoc3D: Point Cloud Based Large-Scale Place Recognition", Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), (2021)

### License
Our code is released under the MIT License (see LICENSE file for details).
