from main_labels import *
from statistics import mean
import time
import pandas as pd


def concatenation_exp(seqs, pandaset_path):
    """
    runs the expirements for the concatenation step, prints the results, saves result as tsv file
    Parameters
    ----------
    pandaset_path: path to the dataset
    seqs: the sequences to test

    Returns
    -------

    """
    # Initialize an empty DataFrame
    df = pd.DataFrame(
        columns=['sequence', 'step', 'number of pcds', 'number of points', 'vegetation inliers', 'concatenation time',
                 'total time'])

    for seq in seqs:
        print("Sequence: ", seq)
        sequence = load_one_sequence(pandaset_path, seq)

        for step in [40, 20, 10, 5, 1]:
            print("--> Step: ", step)
            # concatenation
            start_time = time.time()
            data, nb_pcds = concatenate_multiple_frames(step=step, seq=sequence)
            concat_time = time.time() - start_time
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
            full_exec_time = time.time() - start_time

            # Append a new row to the DataFrame
            df = pd.concat([df, pd.DataFrame([{
                'sequence': seq,
                'step': step,
                'number of pcds': nb_pcds,
                'number of points': len(data),
                'vegetation inliers': len(vege_inliers_pcd.points),
                'concatenation time': concat_time,
                'total time': full_exec_time
            }])], ignore_index=True)

    # saving as tsv file
    df.to_csv('experiments/concatenation_experiment.tsv', sep="\t")

    # Calculate averages for each column based on 'Step'
    avg_df = df.groupby('step').agg({
        'number of pcds': 'mean',
        'number of points': 'mean',
        'vegetation inliers': 'mean',
        'concatenation time': 'mean',
        'total time': 'mean'
    }).reset_index()

    # Print the final DataFrame with averages
    print(avg_df)
    avg_df.to_csv('experiments/avg_concatenation_experiment.tsv', sep="\t")

    # print("Step: ", step)
    # print("Number of pcds: ", nb_pcds)
    # print("Nb points: ", len(data))
    # print("Avg vegetation inliers: ", mean(nb_veg_inliners))
    # print("Avg concatenation time: ", mean(concat_exec_time))
    # print("Avg total time: ", mean(full_exec_time))
    # print("=======================================")


def sampling_exp(seqs, pandaset_path):
    """
    runs the expirements for the number of road sampling points, saves result as tsv file
    Parameters
    ----------
    pandaset_path: path to the dataset
    seqs: the sequences to test

    Returns
    -------

    """
    # Initialize an empty DataFrame
    df = pd.DataFrame(
        columns=['sequence', 'sampling points', 'vegetation inliers', 'sampling time', 'total time'])

    for seq in seqs:
        print("Sequence: ", seq)
        sequence = load_one_sequence(pandaset_path, seq)

        for spl_points in [100, 250, 500, 1000, 2000]:
            print("--> Sampling points: ", spl_points)
            start_time = time.time()
            # concatenating
            multiple_frames_data, focus_data, veg_data, road_data = concatenate_and_focus(sequence=sequence, step=10)
            # sampling
            spl_start_time = time.time()
            sampled_pcd = sample_points(data=road_data, nb_spl_points=spl_points)
            sampling_time = time.time() - spl_start_time
            # contours
            cont_pcd, road_without_contours_pcd = find_contour_points(sampled_pcd)
            # inliers
            vege_inliers_pcd, vege_outliers_pcd, ls_road, ls_cl_h = process_road_pcd(cont_voxels=cont_pcd.points,
                                                                                     veg_data=veg_data)
            full_exec_time = time.time() - start_time

            # Append a new row to the DataFrame
            df = pd.concat([df, pd.DataFrame([{
                'sequence': seq,
                'sampling points': spl_points,
                'vegetation inliers': len(vege_inliers_pcd.points),
                'sampling time': sampling_time,
                'total time': full_exec_time
            }])], ignore_index=True)

    # saving as tsv file
    df.to_csv('experiments/sampling_experiment.tsv', sep="\t")
    # Calculate averages for each column based on 'Step'
    avg_df = df.groupby('sampling points').agg({
        'vegetation inliers': 'mean',
        'sampling time': 'mean',
        'total time': 'mean'
    }).reset_index()

    avg_df.to_csv('experiments/avg_sampling_experiment.tsv', sep="\t")

    # print("Nb sampling points: ", spl_points)
    # print("Avg vegetation inliers: ", mean(nb_veg_inliners))
    # print("Avg sampling time: ", mean(sampling_time))
    # print("Avg total time: ", mean(full_exec_time))
    # print("=======================================")


def radius_exp(seqs, pandaset_path):
    """
    runs the experiments for the neighborhood radius, saves result as tsv file
    Parameters
    ----------
    seqs: the sequences to test
    pandaset_path: path to the dataset

    Returns
    -------

    """
    # Initialize an empty DataFrame
    df = pd.DataFrame(
        columns=['sequence', 'radius', 'contour time', 'total time'])

    for seq in seqs:
        print("Sequence: ", seq)
        sequence = load_one_sequence(pandaset_path, seq)

        for radius in [2, 4, 6, 8, 10]:
            print("--> Radius: ", radius)
            start_time = time.time()
            # concatenating
            multiple_frames_data, focus_data, veg_data, road_data = concatenate_and_focus(sequence=sequence, step=10)
            # sampling
            sampled_pcd = sample_points(data=road_data)
            # contours
            cont_start_time = time.time()
            cont_pcd, road_without_contours_pcd = find_contour_points(sampled_pcd, radius)
            contour_time = time.time() - cont_start_time
            # inliers
            vege_inliers_pcd, vege_outliers_pcd, ls_road, ls_cl_h = process_road_pcd(cont_voxels=cont_pcd.points,
                                                                                     veg_data=veg_data)
            full_exec_time = time.time() - start_time

            # Append a new row to the DataFrame
            df = pd.concat([df, pd.DataFrame([{
                'sequence': seq,
                'radius': radius,
                'contour time': contour_time,
                'total time': full_exec_time
            }])], ignore_index=True)

    # saving as tsv file
    df.to_csv('experiments/radius_experiment.tsv', sep="\t")
    # Calculate averages for each column based on 'Step'
    avg_df = df.groupby('radius').agg({
        'contour time': 'mean',
        'total time': 'mean'
    }).reset_index()

    avg_df.to_csv('experiments/avg_radius_experiment.tsv', sep="\t")

    # print("Radius: ", radius)
    # print("Avg time for contour algorithm: ", mean(contour_time))
    # print("Avg total time: ", mean(full_exec_time))
    # print("=======================================")


def total_duration(seqs, pandaset_path):
    """
    measures the total duration for processing all 30 sequences
    Parameters
    ----------
    pandaset_path: path to the dataset
    seqs: all sequences to process

    Returns
    -------

    """
    start_time = time.time()
    for seq in seqs:
        print("Sequence: ", seq)
        sequence = load_one_sequence(pandaset_path, seq)
        # concatenating
        multiple_frames_data, focus_data, veg_data, road_data = concatenate_and_focus(sequence=sequence, step=10)
        # sampling
        sampled_pcd = sample_points(data=road_data)
        # contours
        cont_pcd, road_without_contours_pcd = find_contour_points(sampled_pcd, 6)
        # inliers
        vege_inliers_pcd, vege_outliers_pcd, ls_road, ls_cl_h = process_road_pcd(cont_voxels=cont_pcd.points,
                                                                                 veg_data=veg_data)
    print("Duration for all Sequences: ", time.time() - start_time)


def main():
    pandaset_path = "/home/miriam/Dokumente/datasets/pandaset"
    all_seqs = ['001', '002', '003', '005', '011', '013', '015', '016', '017', '019', '021', '023', '024', '027', '028',
                '029', '030', '032', '033', '034', '035', '037', '038', '039', '040', '041', '042', '043', '044', '046']

    print("\nConcatenation Experiment")
    concatenation_exp(all_seqs, pandaset_path)
    print("\nSampling Experiment")
    sampling_exp(all_seqs, pandaset_path)
    print("\nRadius Experiment")
    radius_exp(all_seqs, pandaset_path)
    print("Duration Experiment")
    total_duration(all_seqs, pandaset_path)


if __name__ == "__main__":
    main()
