# from ml3d.vis import Visualizer, LabelLUT
import open3d.ml as _ml3d
import open3d.ml.tf as ml3d

import os
import numpy as np
from statistics import mean
import pickle
import argparse


def reduce_labels(original_labels):
    """
    reduce the full amount of labels to only road (0), vegetation (1) and others (2)
    Parameters
    ----------
    original_labels

    Returns
    -------
    transformed labels list
    """
    labels = np.array(original_labels)
    labels[labels == 5] = 1
    labels[labels == 7] = 0
    labels[~np.isin(labels, [0, 1])] = 2
    return labels


def run_and_visualize(model, cfg, dataset, n_vis):
    """
    runs the inference for n_vis point clouds and visualizes the results using Open3D
    Parameters
    ----------
    model: the trained model
    cfg: config file
    dataset: pandaset dataset
    n_vis: number of point clouds to visualize
    """
    print("Creating the pipeline")
    pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, **cfg.pipeline)

    print("Loading the checkpoint")
    ckpt_path = "./trained_tf_model/ckpt_randlanet_pandaset"
    pipeline.load_ckpt(ckpt_path=ckpt_path)

    print("Initializing the Visualizer")
    vis = ml3d.vis.Visualizer()
    reduced_class_names = {0: 'Road', 1: 'Vegetation', 2: 'Other'}
    lut = ml3d.vis.LabelLUT()
    for val in sorted(reduced_class_names.keys()):
        lut.add_label(reduced_class_names[val], val)
    vis.set_lut("pred", lut)

    test_split = dataset.get_split("test")

    all_vis_data = []
    for idx in range(n_vis):
        data = test_split.get_data(idx)
        result = pipeline.run_inference(data)

        labels = reduce_labels(data['label'])

        vis_d = {
            "name": 'test_' + str(idx),
            "points": data['point'],  # n x 3
            "labels": labels,  # n
            "pred": result['predict_labels'],  # n
        }
        all_vis_data.append(vis_d)

    vis.visualize(all_vis_data)


def calculate_accuracy(preds, labels):
    """
    calculates the percentage of correctly predicted labels
    Parameters
    ----------
    preds: model predictions
    labels: true labels

    Returns
    -------
    accuracy in percent
    """
    flat_labels = [item for sublist in labels for item in sublist]

    # Calculate the percentage of identical numbers
    matches = sum(p == l for p, l in zip(preds, flat_labels))
    total_elements = len(preds)
    percentage_identical = (matches / total_elements) * 100

    return percentage_identical


def run_all_tests(model, cfg, dataset):
    """
    lets the model make predictions for every test point cloud, calculates the average accuracy,
    saves results in pickle file

    Parameters
    ----------
    model: trained model
    cfg: config file
    dataset: pandaset dataset
    """
    print("Creating the pipeline")
    pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, **cfg.pipeline)

    print("Loading the checkpoint")
    ckpt_path = "./trained_tf_model/ckpt_randlanet_pandaset"
    pipeline.load_ckpt(ckpt_path=ckpt_path)

    test_split = dataset.get_split("test")
    accs = []
    all_results = []

    for idx in range(len(test_split)):
        print("Point Cloud number ", idx)
        data = test_split.get_data(idx)

        result = pipeline.run_inference(data)

        labels = reduce_labels(data['label'])
        preds = result['predict_labels']

        acc = calculate_accuracy(preds, labels)
        accs.append(acc)
        all_results.append({'test_' + str(idx): [preds]})

    print("All accuricies: ", [int(item) for item in accs])
    print("Avg accuracy: ", mean(accs))

    with open('test_results.pkl', 'wb') as file:
        pickle.dump(all_results, file)


def main():
    parser = argparse.ArgumentParser(description='Run tests')
    parser.add_argument('-p', '--path', help='path to the pandas dataset', default=os.path.join(os.getcwd(), '..', 'pandaset'))
    parser.add_argument('-t', '--tests', help='True for running all test point clouds and calculating accuracies', default="False")
    parser.add_argument('-v', '--vis', help='True for visualization, else False', default="True")
    parser.add_argument('-n', '--n_vis', help='Number of test point clouds to visualize', default=10)
    args = parser.parse_args()

    cfg_file = "to_include_in Open3D-ML/randlanet_pandaset.yml"
    cfg = _ml3d.utils.Config.load_from_file(cfg_file)
    model = ml3d.models.RandLANet(**cfg.model)

    print("Reading the dataset")
    cfg.dataset['dataset_path'] = args.path
    dataset = ml3d.datasets.Pandaset(cfg.dataset.pop('dataset_path', None), **cfg.dataset)

    if args.vis == "True":
        run_and_visualize(model, cfg, dataset, int(args.n_vis))
    if args.tests == "True":
        run_all_tests(model, cfg, dataset)


if __name__ == "__main__":
    main()
