# LIDAR-BASED TREE-CLEARANCE ANALYSIS

This is a demonstration of the clear height analysis above roads. 
It shows images and point clouds one after the other. It is intended to explain the idea of the algorithm.
The initial computing takes some time.

### Installation:
- download the pandaset data set from Kaggle (https://www.kaggle.com/datasets/usharengaraju/pandaset-dataset?resource=download)
- create new conda environment
- install the requirements and the pandaset devkit (from https://github.com/scaleapi/pandaset-devkit)

### Structure of Demonstration:

1. point cloud visualizations (with Open3D): 
   - one LiDAR sweep
   - scene of concatenated sweeps
   - semantic segmentation
   - only classes vegetation and road
   - sampled street points
   - contour of street
   - road polygon (green) and tree points to be trimmed (red)
2. the projection of the results (where to trim the trees) on images that were recorded simultaneously

### Files:

- **main_labels.py** : Main file of the demo using the labels of the dataset
- **main_inference.py** : Main file of the demo using the predictions of the RandLANet

- **data_handling.py** : functions for general data and point cloud processing (creating and concatenating point clouds, creating line sets, etc.)
- **contour_functions.py** : functions for the algorithm for finding the contours of the road plane
- **color_palette.py** : get the colors for each semantic class

- **open3d_visualization.py** : takes a list of point clouds, functions for visualizing them one after the other
- **projection.py** : functions for projecting and showing the resulting points on the photos

- **experiments_parameters.py** : functions for running the experiments presented in the paper

### Example:
run the system for sequence 001 on Linux:
```python main.py -p <path-to-pandaset> -s 001 -w False```
