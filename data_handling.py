import open3d as o3d
import pandaset
import pandas as pd
import numpy as np


def load_one_sequence(pandaset_path, sequence_number):
    """
    loads one sequence (including multiple frames = recording points) from the pandaset data set
    Parameters
    ----------
    pandaset_path: path to the pandaset folder
    sequence_number: number of the sequence to load

    Returns
    -------
    the loaded sequence
    """
    dataset = pandaset.DataSet(pandaset_path)
    sequence = dataset[sequence_number]
    sequence.load()
    return sequence


def get_points_of_one_frame(sequence, frame_number):
    """
    gets the points of both Lidar sensors for a frame of the sequence
    Parameters
    ----------
    sequence: the loaded pandaset sequence
    frame_number: the frame number of interest (one sequence includes multiple frames)

    Returns
    -------
    the x,y,z coordinates of the points belonging to this frame
    """
    # get the content for the visualization
    sequence.lidar.set_sensor(0)
    frame_points = sequence.lidar[frame_number].to_numpy()
    sequence.lidar.set_sensor(1)
    np.append(frame_points, sequence.lidar[frame_number].to_numpy())

    return frame_points[:, :3]


def concatenate_multiple_frames(step, seq):
    """

    Parameters
    ----------
    step
    seq

    Returns
    -------

    """
    # get the points from both sensors
    seq.lidar.set_sensor(0)
    selected_data_sensor0 = seq.lidar[::step]
    _ = list(map(lambda xy: xy[1].insert(3, 'f', xy[0]),
                 enumerate(selected_data_sensor0)))  # Add column 'f' to each data frame in order

    seq.lidar.set_sensor(1)
    selected_data_sensor1 = seq.lidar[::step]  # Take every 5th frame from sequence
    _ = list(map(lambda xy: xy[1].insert(3, 'f', xy[0]),
                 enumerate(selected_data_sensor1)))  # Add column 'f' to each data frame in order

    # get the annotations for each frame
    segmentations = seq.semseg[::step]
    selected_data = []

    # merge the data points table with the annotations
    for i in range(len(segmentations)):
        temp_sensor0 = selected_data_sensor0[i]
        temp_sensor1 = selected_data_sensor1[i]
        temp_merge = pd.concat([temp_sensor0, temp_sensor1])
        temp_seg = segmentations[i]

        temp_data_with_class = pd.concat([temp_merge, temp_seg], axis=1)
        selected_data.append(temp_data_with_class)

    # print("Number of concatenated point clouds: ")
    nb_pcds = len(selected_data)
    selected_data = pd.concat(selected_data)
    return selected_data, nb_pcds


def concatenate_and_focus(sequence, step=10):
    """
    concatenates multiple frames from one sequence, takes only every 10th frame (adjustable)
    Parameters
    ----------
    sequence: the sequence including the frames to be concatenated
    step: how often to pick a frame (concatenating all frames takes a long time and is difficult to hold in mempory)

    Returns
    -------
    the entire concatenated data, the data including only vegetation, road and sidewalk points, and the 3 classes separated
    """
    # all points from the selected frames
    data, _ = concatenate_multiple_frames(step=step, seq=sequence)

    # only points from the classes 5, 7, and 11
    focus_data = pd.concat([data[data['class'] == 5], data[data['class'] == 7]])

    # separate points per class
    veg_data, road_data = focus_data[focus_data['class'] == 5], focus_data[focus_data['class'] == 7]

    return data, focus_data, veg_data, road_data


def sample_points(data, nb_spl_points=1000):
    """
    creates a point cloud from given data points, removes outliers, samples the points (less points, equally distributed)
    Parameters
    ----------
    nb_spl_points: number of points after the sampling
    data: the data points to be sampled (road points)

    Returns
    -------
    the sampled point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data.to_numpy()[:, :3])

    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
    gt_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=2.0)
    sampled_pcd = gt_mesh.sample_points_poisson_disk(nb_spl_points)
    sampled_pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals

    sampled_pcd.estimate_normals()
    sampled_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(sampled_pcd.points))])
    return sampled_pcd

def create_o3d_point_cloud(data, colors_by_class):
    """
    creates the point cloud in open3D with different colors per class

    Parameters
    ----------
    data = points of the cloud
    colors_by_class = color palette with colors for each class

    Returns
    -------
    the mesh (arrows) and the point cloud
    """

    # Create the point cloud
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(data.to_numpy()[:, :3])

    # set the colors
    colors = [colors_by_class[sem_class] for sem_class in data['class']]
    colors = np.asarray(colors)
    o3d_pc.colors = o3d.utility.Vector3dVector(colors)

    return o3d_pc
