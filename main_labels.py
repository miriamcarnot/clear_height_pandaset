from color_palette import *
from projection import *
from contour_functions import *
from open3d_visualization import *
from data_handling import *

import os
import open3d as o3d
import argparse


def data_to_pcds(data_list, sequence):
    """
    create point clouds from the data contained in the given list
    Parameters
    ----------
    data_list: a list with data points (for one frame, multiple frames, and only the 3 classes of interest = focus)
    sequence: the sequence the data was taken from

    Returns
    -------
    a list of point clouds (same as the data list)
    """
    one_frame_data, multiple_frames_data, focus_data = data_list
    colors_no_focus = get_color_palette(sequence, focus=False)
    colors_with_focus = get_color_palette(sequence, focus=True)

    one_frame_pcd = o3d.geometry.PointCloud()
    one_frame_pcd.points = o3d.utility.Vector3dVector(one_frame_data[:, :3])
    # one_frame_pcd.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(one_frame_data.shape[0])])

    multiple_one_color_pcd = o3d.geometry.PointCloud()
    multiple_one_color_pcd.points = o3d.utility.Vector3dVector(multiple_frames_data.to_numpy()[:, :3])
    # multiple_one_color_pcd.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(one_frame_data.shape[0])])

    multiple_no_focus_pcd = o3d.geometry.PointCloud()
    multiple_no_focus_pcd.points = o3d.utility.Vector3dVector(multiple_frames_data.to_numpy()[:, :3])
    colors = [colors_no_focus[sem_class] for sem_class in multiple_frames_data['class']]
    colors = np.asarray(colors)
    multiple_no_focus_pcd.colors = o3d.utility.Vector3dVector(colors)

    only_focus_pcd = o3d.geometry.PointCloud()
    only_focus_pcd.points = o3d.utility.Vector3dVector(focus_data.to_numpy()[:, :3])
    colors = [colors_with_focus[sem_class] for sem_class in focus_data['class']]
    colors = np.asarray(colors)
    only_focus_pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd_list = [[one_frame_pcd], [multiple_one_color_pcd], [multiple_no_focus_pcd], [only_focus_pcd]]

    return pcd_list


def main():
    parser = argparse.ArgumentParser(description='Run Clear Height Analysis')
    parser.add_argument('-p', '--path', help='path to the dataset', default='/home/miriam/Dokumente/datasets/pandaset')
    parser.add_argument('-s', '--sequence', help='dataset sequence to run, check readme for options', default="001")

    args = parser.parse_args()
    windows = False

    # get a sequence of lidar files (one scene)
    print("load sequence")
    pandaset_path = args.path
    sequence_number = args.sequence
    # only those sequences have labels (30 in total):
    # '001', '002', '003', '005', '011', '013', '015', '016', '017', '019', '021', '023', '024', '027', '028', '029',
    # '030', '032', '033', '034', '035', '037', '038', '039', '040', '041', '042', '043', '044', '046'
    sequence = load_one_sequence(pandaset_path, sequence_number)

    # get one and multiple frames
    print("get points for one and for multiple frames")
    one_frame_data = get_points_of_one_frame(sequence=sequence, frame_number=0)
    multiple_frames_data, focus_data, veg_data, road_data = concatenate_and_focus(sequence=sequence, step=10)
    data_list = [one_frame_data, multiple_frames_data, focus_data]
    pcds_list = data_to_pcds(data_list, sequence)

    # sample street points, find contour points, and crop the vegetation point cloud
    sampled_pcd = sample_points(data=road_data, nb_spl_points=1000)
    cont_pcd, road_without_contours_pcd = find_contour_points(sampled_pcd)
    vege_inliers_pcd, vege_outliers_pcd, ls_road, ls_cl_h = process_road_pcd(cont_voxels=cont_pcd.points,
                                                                             veg_data=veg_data)

    num_red_points = len(vege_inliers_pcd.points)
    print("number of found red points: ", num_red_points)

    pcds_list.append([sampled_pcd])
    road_pcd = o3d.geometry.PointCloud()
    road_pcd.points = o3d.utility.Vector3dVector(road_data.to_numpy()[:, :3])
    road_pcd.colors = o3d.utility.Vector3dVector([[0.33, 0.33, 0.33] for _ in range(len(road_pcd.points))])

    multiple_geometries_list = [[road_without_contours_pcd, cont_pcd], [road_pcd, vege_inliers_pcd, vege_outliers_pcd,
                                                                        ls_road, ls_cl_h]]
    # focus_data_crop_pcd = crop_point_cloud(vege_inliers_pcd, geometries_list[4])
    print("finished gathering all geometries")

    all_geometries = pcds_list + multiple_geometries_list

    # show images of a car equipped with lidar
    car_images = Image.open('combined_images.png')
    car_images.show()

    if windows:
        visualize_pcds_one_after_another(all_geometries)
        # visualize the greenery to cut on the images
        print("visualizing the greenery to cut on the images")
        project_greenery_to_cut_onto_image(vege_inliers_pcd, sequence)
    else:
        pid = os.fork()
        if pid:
            visualize_pcds_one_after_another(all_geometries)
        else:
            # visualize the greenery to cut on the images
            print("visualizing the greenery to cut on the images")
            project_greenery_to_cut_onto_image(vege_inliers_pcd, sequence)


if __name__ == "__main__":
    main()
