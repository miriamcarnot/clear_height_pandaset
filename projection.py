from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandaset


def make_img_from_plot(plt):
    """
    converts the plot into an image (easier for concatenation and visualization after)
    Parameters
    ----------
    plt: the plot that will get converted

    Returns
    -------
    the converted image
    """
    # Get the figure and its axes
    fig = plt.gcf()
    fig.tight_layout()
    axes = plt.gca()

    # Draw the content
    fig.canvas.draw()

    # Get the RGB values
    rgb = fig.canvas.tostring_rgb()

    # Get the width and height of the figure
    width, height = fig.canvas.get_width_height()

    # Convert the RGB values to a PIL Image
    img = Image.frombytes('RGB', (width, height), rgb)

    return img


def plot_projection(sequence, seq_idx, pcd, camera_view, n_plot):
    """
    the heart of the projection, makes a plot for each image
    Parameters
    ----------
    sequence:
    seq_idx: frame index in this sequence
    pcd: the vegetation points that lay inside the clear height
    camera_view: the view of the camera at this point
    n_plot: tha name of the plot (to not plot on top of other plots)

    Returns
    -------
    the final plot
    """
    choosen_camera = sequence.camera[camera_view]
    projected_points2d, camera_points_3d, inner_indices = pandaset.geometry.projection(
        lidar_points=np.asarray(pcd.points)[:, :3],
        camera_data=choosen_camera[seq_idx],
        camera_pose=choosen_camera.poses[seq_idx],
        camera_intrinsics=choosen_camera.intrinsics,
        filter_outliers=True)
    # print("projection 2d-points inside image count:", projected_points2d.shape)
    # Plot the points
    ori_image = sequence.camera[camera_view][seq_idx]
    # print("small img size:", ori_image.size) is (1920,1080)
    plt.figure(n_plot)
    plt.imshow(ori_image)
    plt.scatter(projected_points2d[:, 0], projected_points2d[:, 1], s=1, color='red', alpha=0.8)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

    # plt.set_title(f"{camera_view}", fontsize=10)
    plt.xlabel(camera_view)

    return plt


def get_concat_h_cut(img_list):
    """
    concatenates a list of images horizontally
    Parameters
    ----------
    img_list: list of images

    Returns
    -------
    concatenated images
    """
    total_width = len(img_list) * img_list[0].width

    dst = Image.new('RGB', (total_width, img_list[0].height))

    tmp_paste_width = 0
    for img in img_list:
        dst.paste(img, (tmp_paste_width, 0))
        tmp_paste_width += img.width
    return dst


def get_concat_v_cut_with_margin(img_list):
    """
    concatenates a list of images vertically
    Parameters
    ----------
    img_list: list of images

    Returns
    -------
    concatenated images
    """
    total_height = len(img_list) * img_list[0].height

    dst = Image.new('RGB', (img_list[0].width, total_height))

    tmp_paste_height = 0
    for img in img_list:
        dst.paste(img, (0, tmp_paste_height))
        tmp_paste_height += img.height
    return dst


def project_greenery_to_cut_onto_image(vege_inliers_pcd, sequence):
    """
    projects the greenery that needs to be trimmed onto the images (3 positions x 6 cameras on the car)
    shows the final images
    Parameters
    ----------
    vege_inliers_pcd: point cloud with the vegetation points that lay inside the clear height
    sequence: the sequence of lidar sweeps of interest

    Returns
    -------

    """
    camera_views = ['back_camera', 'left_camera', 'front_left_camera', 'front_camera', 'front_right_camera',
                    'right_camera']

    all_seq_idx = [30, 50, 70]
    all_concatenated_imgs = []
    n_plot = 0

    for seq_idx in all_seq_idx:
        seq_idx_images = []
        for i, camera_view in enumerate(camera_views):
            projected_plt = plot_projection(sequence, seq_idx, vege_inliers_pcd, camera_view, n_plot)
            n_plot += 1
            img = make_img_from_plot(projected_plt)
            seq_idx_images.append(img)
        concatenated_img = get_concat_h_cut(seq_idx_images)
        all_concatenated_imgs.append(concatenated_img)

    final_img = get_concat_v_cut_with_margin(all_concatenated_imgs)
    # print("final img size: ", final_img.width, final_img.height) is (3840, 1440)
    final_img.show()
