from data_handling import *
from contour_functions import *

from statistics import mean
import time


def concatenation_exp(sequence, step):
    """
    runs the expirements for the concatenation step, prints the results
    Parameters
    ----------
    sequence: the sequence to test
    step: the concatenation step to test

    Returns
    -------

    """
    concat_exec_time = []
    full_exec_time = []
    nb_veg_inliners = []

    for i in range(10):
        # concatenation
        start_time = time.time()
        data, nb_pcds = concatenate_multiple_frames(step=step, seq=sequence)
        concat_end_time = time.time()
        concat_exec_time.append(concat_end_time - start_time)
        # focus data
        focus_data = pd.concat([data[data['class'] == 5], data[data['class'] == 7]])
        veg_data, road_data = focus_data[focus_data['class'] == 5], focus_data[focus_data['class'] == 7]
        # sampling
        sampled_pcd = sample_points(road_data)
        # contours
        cont_pcd, road_without_contours_pcd = find_contour_points(sampled_pcd)
        # inliers
        vege_inliers_pcd, vege_outliers_pcd, ls_road, ls_cl_h = process_road_pcd(cont_voxels=cont_pcd.points,
                                                                                 veg_data=veg_data)
        full_exec_time.append(time.time() - start_time)
        nb_veg_inliners.append(len(vege_inliers_pcd.points))

    print("Step: ", step)
    print("Number of pcds: ", nb_pcds)
    print("Nb points: ", len(data))
    print("Avg vegetation inliers: ", mean(nb_veg_inliners))
    print("Avg concatenation time: ", mean(concat_exec_time))
    print("Avg total time: ", mean(full_exec_time))
    print("=======================================")


def sampling_exp(sequence, spl_points):
    """
    runs the expirements for the number of sampling points, prints the results
    Parameters
    ----------
    spl_points: the number of sampling points to test
    sequence: the sequence to test

    Returns
    -------

    """
    sampling_time = []
    full_exec_time = []
    nb_veg_inliners = []

    for i in range(10):
        start_time = time.time()
        # concatenating
        multiple_frames_data, focus_data, veg_data, road_data = concatenate_and_focus(sequence=sequence, step=10)
        # sampling
        spl_start_time = time.time()
        sampled_pcd = sample_points(data=road_data, nb_spl_points=spl_points)
        sampling_time.append(time.time() - spl_start_time)
        # contours
        cont_pcd, road_without_contours_pcd = find_contour_points(sampled_pcd)
        # inliers
        vege_inliers_pcd, vege_outliers_pcd, ls_road, ls_cl_h = process_road_pcd(cont_voxels=cont_pcd.points,
                                                                                 veg_data=veg_data)
        full_exec_time.append(time.time() - start_time)
        nb_veg_inliners.append(len(vege_inliers_pcd.points))

    print("Nb sampling points: ", spl_points)
    print("Avg vegetation inliers: ", mean(nb_veg_inliners))
    print("Avg sampling time: ", mean(sampling_time))
    print("Avg total time: ", mean(full_exec_time))
    print("=======================================")


def radius_exp(sequence, radius):
    """
    runs the experiments for the neighborhood radius, prints the results
    Parameters
    ----------
    sequence: the sequence to test
    radius: the nighborhood radius to test

    Returns
    -------

    """
    contour_time = []
    full_exec_time = []

    for i in range(10):
        start_time = time.time()
        # concatenating
        multiple_frames_data, focus_data, veg_data, road_data = concatenate_and_focus(sequence=sequence, step=10)
        # sampling
        sampled_pcd = sample_points(data=road_data)
        # contours
        cont_start_time = time.time()
        cont_pcd, road_without_contours_pcd = find_contour_points(sampled_pcd, radius)
        contour_time.append(time.time() - cont_start_time)
        # inliers
        vege_inliers_pcd, vege_outliers_pcd, ls_road, ls_cl_h = process_road_pcd(cont_voxels=cont_pcd.points,
                                                                                 veg_data=veg_data)
        full_exec_time.append(time.time() - start_time)

    print("Radius: ", radius)
    print("Avg time for contour algorithm: ", mean(contour_time))
    print("Avg total time: ", mean(full_exec_time))
    print("=======================================")


def main():
    pandaset_path = "/home/miriam/Dokumente/datasets/pandaset"
    sequence_number = "003"
    sequence = load_one_sequence(pandaset_path, sequence_number)

    for step in [40, 20, 10, 5, 1]:
        concatenation_exp(sequence, step)

    for spl_points in [100, 250, 500, 1000, 2000]:
        sampling_exp(sequence, spl_points)

    for radius in [2, 4, 6, 8, 10]:
        radius_exp(sequence, radius)


if __name__ == "__main__":
    main()
