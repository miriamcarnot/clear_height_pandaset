from projection import *
from contour_functions import *
from open3d_visualization import *
from data_handling import *

import os
import argparse
import pandas as pd
import numpy as np

import open3d.ml as _ml3d
import open3d.ml.tf as ml3d
import open3d as o3d


def filter_pcd(merged_pcds):
    # separate points per class
    veg_data = merged_pcds[merged_pcds['class'] == 1]
    road_data = merged_pcds[merged_pcds['class'] == 0]
    focus_data = pd.concat([veg_data, road_data])

    return focus_data, veg_data, road_data


def concatenate(step, seq):
    # get the points from both sensors
    seq.lidar.set_sensor(0)
    selected_data_sensor0 = seq.lidar[::step]
    seq.lidar.set_sensor(1)
    selected_data_sensor1 = seq.lidar[::step]

    # merge the points from both sensors in one dataframe
    both_sensors_pcd_list = []
    for i in range(len(selected_data_sensor0)):
        both_in_one_pcd = pd.concat([selected_data_sensor0[i], selected_data_sensor1[i]], axis=0)
        both_sensors_pcd_list.append(both_in_one_pcd)

    return both_sensors_pcd_list


def run_inference_pcd_list(pcd_list, pandaset_path):
    print("Loading the config file and the dataset")
    cfg_file = "./semantic_segmentation/to_include_in Open3D-ML/randlanet_pandaset.yml"
    cfg = _ml3d.utils.Config.load_from_file(cfg_file)
    model = ml3d.models.RandLANet(**cfg.model)
    cfg.dataset['dataset_path'] = pandaset_path
    dataset = ml3d.datasets.Pandaset(cfg.dataset.pop('dataset_path', None), **cfg.dataset)

    print("Creating the pipeline")
    pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, **cfg.pipeline)

    print("Loading the checkpoint")
    ckpt_path = "./semantic_segmentation/trained_tf_model/ckpt_randlanet_pandaset"
    pipeline.load_ckpt(ckpt_path=ckpt_path)

    all_pcds_with_classes = []
    for df_pcd in pcd_list:
        pcd_dict = {
            'point': df_pcd[['x', 'y', 'z']].values.astype(np.float32),
            'intensity': df_pcd['i'].values.astype(np.float32),
            'label': df_pcd[['d']].values.astype(np.int32)
        }
        result = pipeline.run_inference(pcd_dict)
        df_pcd['class'] = result['predict_labels']

        all_pcds_with_classes.append(df_pcd)

    merged_pcds_df = pd.concat(all_pcds_with_classes, ignore_index=True)

    return merged_pcds_df


def data_to_pcds_semseg(one_frame_data, merged_pcds_df, focus_data):
    color_palette = color_palette_reduced()

    one_frame_pcd = o3d.geometry.PointCloud()
    one_frame_pcd.points = o3d.utility.Vector3dVector(one_frame_data[:, :3])

    concatenated_pcd_before_semseg = o3d.geometry.PointCloud()
    concatenated_pcd_before_semseg.points = o3d.utility.Vector3dVector(merged_pcds_df.to_numpy()[:, :3])

    concatenated_pcd = o3d.geometry.PointCloud()
    concatenated_pcd.points = o3d.utility.Vector3dVector(merged_pcds_df.to_numpy()[:, :3])
    colors = [color_palette[sem_class] for sem_class in merged_pcds_df['class']]
    colors = np.asarray(colors)
    concatenated_pcd.colors = o3d.utility.Vector3dVector(colors)

    only_focus_pcd = o3d.geometry.PointCloud()
    only_focus_pcd.points = o3d.utility.Vector3dVector(focus_data.to_numpy()[:, :3])
    colors = [color_palette[sem_class] for sem_class in focus_data['class']]
    colors = np.asarray(colors)
    only_focus_pcd.colors = o3d.utility.Vector3dVector(colors)

    return [[one_frame_pcd], [concatenated_pcd_before_semseg], [concatenated_pcd], [only_focus_pcd]]


def color_palette_reduced():
    number_of_classes = 3
    colors = [[0.0, 0.0, 0.0]] * number_of_classes
    colors[0] = [0.13, 0.13, 0.13]  # road in grey
    colors[1] = [0.1, 0.35, 0.1]  # vegetation in green
    colors[2] = [0.9, 0.63, 0.43]  # others in beige
    return colors


def process_sequence(pandaset_path, seq):
    sequence = load_one_sequence(pandaset_path, seq)
    both_sensors_pcd_list = concatenate(step=10, seq=sequence)
    merged_pcds_df = run_inference_pcd_list(both_sensors_pcd_list, pandaset_path)
    focus_data, veg_data, road_data = filter_pcd(merged_pcds_df)
    # sample street points, find contour points, and crop the vegetation point cloud
    sampled_pcd = sample_points(data=road_data, nb_spl_points=1000)
    cont_pcd, road_without_contours_pcd = find_contour_points(sampled_pcd, 6)
    vege_inliers_pcd, vege_outliers_pcd, ls_road, ls_cl_h = process_road_pcd(cont_voxels=cont_pcd.points,
                                                                             veg_data=veg_data)
    return (sequence, merged_pcds_df, focus_data, veg_data, road_data, sampled_pcd, cont_pcd, road_without_contours_pcd,
            vege_inliers_pcd, vege_outliers_pcd, ls_road, ls_cl_h)


def visualize_results_for_one_seq(pandaset_path, seq):

    (sequence, merged_pcds_df, focus_data, veg_data, road_data, sampled_pcd, cont_pcd, road_without_contours_pcd,
     vege_inliers_pcd, vege_outliers_pcd, ls_road, ls_cl_h) = process_sequence(pandaset_path, seq)

    one_frame_data = get_points_of_one_frame(sequence=sequence, frame_number=0)
    pcds_list = data_to_pcds_semseg(one_frame_data, merged_pcds_df, focus_data)
    pcds_list.append([sampled_pcd])

    road_pcd = o3d.geometry.PointCloud()
    road_pcd.points = o3d.utility.Vector3dVector(road_data.to_numpy()[:, :3])
    road_pcd.colors = o3d.utility.Vector3dVector([[0.45, 0.45, 0.45] for _ in range(len(road_pcd.points))])

    multiple_geometries_list = [[road_without_contours_pcd, cont_pcd],
                                [road_pcd, vege_inliers_pcd, vege_outliers_pcd, ls_road, ls_cl_h]]

    all_geometries = pcds_list + multiple_geometries_list

    pid = os.fork()
    if pid:
        visualize_pcds_one_after_another(all_geometries)
    else:
        project_greenery_to_cut_onto_image(vege_inliers_pcd, sequence)


def main():
    parser = argparse.ArgumentParser(description='Run Clear Height Analysis')
    parser.add_argument('-p', '--path', help='path to the dataset')
    parser.add_argument('-s', '--sequence', help='dataset sequence to run, check readme for options', default="046")
    args = parser.parse_args()

    pandaset_path = args.path
    sequence_number = args.sequence
    # test_seqs = ['046', '027', '013', '029']

    visualize_results_for_one_seq(pandaset_path, sequence_number)


if __name__ == "__main__":
    main()
