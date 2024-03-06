from main_inference import *
import os
import time


def compare_preds_and_labels(pandaset_path, test_seqs):
    found_percentage = []
    # Initialize an empty DataFrame
    df = pd.DataFrame(
        columns=['sequence', 'vegetation inliers with inference', 'total time with inference',
                 'vegetation inliers without inference', 'total time without inference'])

    for seq in test_seqs:
        print("Sequence: ", seq)

        # experiment with running the model inference
        start_time_with_inf = time.time()
        (sequence, merged_pcds_df, focus_data, veg_data, road_data, sampled_pcd, cont_pcd, road_without_contours_pcd,
         vege_inliers_pcd, vege_outliers_pcd, ls_road, ls_cl_h) = process_sequence(pandaset_path, seq)
        nb_inliers_with_inf = len(vege_inliers_pcd.points)
        full_exec_time_with_inf = time.time() - start_time_with_inf

        # experiment using the dataset annotations
        start_time_without_inf = time.time()
        sequence = load_one_sequence(pandaset_path, seq)
        multiple_frames_data, focus_data, veg_data, road_data = concatenate_and_focus(sequence=sequence, step=10)
        sampled_pcd = sample_points(data=road_data)
        cont_pcd, road_without_contours_pcd = find_contour_points(sampled_pcd, 6)
        vege_inliers_pcd, vege_outliers_pcd, ls_road, ls_cl_h = process_road_pcd(cont_voxels=cont_pcd.points,
                                                                                 veg_data=veg_data)
        nb_inliers_without_inf = len(vege_inliers_pcd.points)
        full_exec_time_without_inf = time.time() - start_time_without_inf

        # Append a new row to the DataFrame
        df = pd.concat([df, pd.DataFrame([{
            'sequence': seq,
            'vegetation inliers with inference': nb_inliers_with_inf,
            'total time with inference': full_exec_time_with_inf,
            'vegetation inliers without inference': nb_inliers_without_inf,
            'total time without inference': full_exec_time_without_inf,
        }])], ignore_index=True)

    # saving as tsv file
    df.to_csv('experiments/comparison_preds_annos.tsv', sep="\t")


def main():
    pandaset_path = os.path.join(os.getcwd(), 'pandaset')
    print(pandaset_path)
    test_seqs = ['046', '027', '013', '029']
    print("Experiment to compare the results when using the annotations or the predictions")
    compare_preds_and_labels(pandaset_path, test_seqs)


if __name__ == "__main__":
    main()
