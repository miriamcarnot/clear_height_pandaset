# Semantic Segmentation of the PandaSet with RandLANet in Open3D-ML

1. Follow the installation steps for Open3D-ML (only works on Linux), we used the Tensorflow version
2. Follow the steps in the Open3d-ML repository to add Pandaset as a new dataset within the framework, the data set and config file can be found in the folder "to_include_in_Open3D-ML" of this repo:
https://github.com/isl-org/Open3D-ML/blob/main/docs/howtos.md#adding-a-new-dataset
2. for training, use the predefined script: 
```python scripts/run_pipeline.py tf -c _ml3d/configs/randlanet_pandaset.yml --dataset.dataset_path <path to pandaset> --pipeline SemanticSegmentation --dataset.use_cache True```
3. for testing our model, run the run_inference.py program

Thanks go to the issue of github user SulekBartek for sharing files to add the pandaset as dataset :)