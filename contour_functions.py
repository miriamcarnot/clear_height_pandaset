import numpy as np
import math
import open3d as o3d
import ray
from scipy.spatial import KDTree
import copy


def points_in_radius(query_point, all_points, radius, tree):
    """
    finds all the points in the radius of the query point
    Parameters
    ----------
    query_point
    all_points
    radius
    tree

    Returns
    -------
    the nearest points in a list
    """
    # Find the nearest neighbors to the query point
    indices = tree.query_ball_point(query_point, r=radius)
    # distances, indices = tree.query(query_point, k=k+1)

    # Get the coordinates of the nearest neighbors, skip the point itself (always the first)
    nearest_neighbors = all_points[indices]

    return nearest_neighbors


def unit_vector(vector):
    """
    normalizes a given vector to a length of one
    :param vector: any vector
    :return: its unit vector
    """
    if len(vector) == 0:
        return 0
    return vector / np.linalg.norm(vector)


def angle_between_vectors(vector1, vector2):
    v1, v2 = unit_vector(vector1[:2]), unit_vector(vector2[:2])
    if np.array_equal(v1, v2):
        return 0
    determinant = np.linalg.det([v1, v2])
    dot_product = np.dot(v1, v2)
    angle = np.math.atan2(determinant, dot_product)
    angle = np.degrees(angle)
    if angle < 0:
        angle = 360 + angle
    return angle


# %%
@ray.remote
def is_cont_candidate(query_point, all_points, radius, tree, angle_thresh=135):
    # get the k nearest neighbors
    nearest_neighbors = points_in_radius(query_point, all_points, radius, tree)

    # Find the index of the query_point in the nearest neighbors
    # Create a boolean mask that indicates which elements to keep
    mask = ~np.all(nearest_neighbors == query_point, axis=1)
    # Use the boolean mask to create a new array without the element
    nearest_neighbors = nearest_neighbors[mask]
    # print('Nearest neighbors: ', nearest_neighbors)

    length_nn = len(nearest_neighbors)
    if length_nn == 1:
        return query_point
    elif length_nn == 0:
        return False

    # get the vectors from query point to the neighbors
    vectors = [[nn[0] - query_point[0], nn[1] - query_point[1], nn[2] - query_point[2]] for nn in nearest_neighbors]
    # print('Vectors: ', vectors)

    # check all possible pairs one after the other
    vector = vectors[0]
    # copy of vectors, take out vectors that were already looked at
    tmp_vectors = vectors.copy()
    smallest_angles = []
    while len(tmp_vectors) > 1:
        # print("vector to check: ", vector)
        # print("available vectors: ", tmp_vectors)
        # remove this element from temporary list
        tmp_vectors.remove(vector)
        # get the smallest angle between this vector and any other vector
        angles = [angle_between_vectors(vector, tmp_vec) for tmp_vec in tmp_vectors]
        # print("Angles to others: ", angles)
        # smallest = min(angles)
        # smallest_angles.append(smallest)

        positive_angles = [angle for angle in angles if angle > 0]
        smallest_clockwise_angle = min(positive_angles)
        smallest_angles.append(smallest_clockwise_angle)

        # print("Chosen angle: ", smallest_clockwise_angle)

        # change working vector to the one with the smallest angle
        vector = tmp_vectors[angles.index(smallest_clockwise_angle)]
        print()
    # compare last to first
    vector = tmp_vectors[0]
    first_vector = vectors[0]
    smallest_angles.append(angle_between_vectors(vector, first_vector))
    # print('smallest angles: ', smallest_angles)
    # print('sum of angles: ', sum(smallest_angles))

    # Iterate over the angles
    for angle in smallest_angles:
        # Check if the current angle is greater than 90 degrees
        if angle > angle_thresh:
            return query_point

    return False


def sort_points_dist(poly_points):
    """
    sorts the contour points to have adjacent points following each other in the list
    Parameters
    ----------
    poly_points

    Returns
    -------
    the sorted list of points
    """
    # Convert the list of points to a numpy array
    poly_points = np.array(poly_points)

    # start at point 0 and check for the shortest connection
    p1 = poly_points[0]
    tmp_points = poly_points.copy()[1:]
    sorted_points = []

    while len(tmp_points) > 0:
        sorted_points.append(p1)

        # get the point that is closest to p1
        first_p = tmp_points[0]
        min_dist = math.dist(p1, first_p)
        min_pt = first_p
        for p2 in tmp_points[1:]:
            tmp_dist = math.dist(p1, p2)
            if tmp_dist < min_dist:
                min_dist = tmp_dist
                min_pt = p2

        # check if the distance to the starting point is shorter
        # if yes stop the while loop
        # > 0 to not stop after first iteration
        # < min_dist/3 to avoid points that were left out and would come at the end
        dist_to_start = math.dist(p1, poly_points[0])
        if 0 < dist_to_start < min_dist / 3 and len(poly_points) // 2 > len(tmp_points):
            break
        # if not continue with the next point
        else:
            p1 = min_pt
            # remove p1 from tmp_points
            mask = ~np.all(tmp_points == p1, axis=1)
            tmp_points = tmp_points[mask]
        # p1 = min_pt
        # # remove p1 from tmp_points
        # mask = ~np.all(tmp_points == p1, axis=1)
        # tmp_points = tmp_points[mask]

    return sorted_points


def find_contour_points(sampled_pcd, radius=6, angle_thresh=90):
    """
    creates a tree of all points and checks which of them are contour points
    uses ray for multiprocessing
    creates a new point cloud including all the contour points
    Parameters
    ----------
    radius: size of the neighborhood radius
    angle_thresh: threshold for the angle between the neighbors
    sampled_pcd: the sampled point cloud of the road

    Returns
    -------
    a point cloud including all contour points
    """
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    cont_candidates = []
    points = np.asarray(sampled_pcd.points)
    tree = KDTree(points)

    for voxel in points:
        cont_candidates.append(is_cont_candidate.remote(query_point=voxel, all_points=points, radius=radius, tree=tree,
                                                        angle_thresh=angle_thresh))

    # Wait for the tasks to complete and retrieve the results
    results = ray.get(cont_candidates)
    cont_voxels = [x for x in results if isinstance(x, np.ndarray)]
    contour_points = np.vstack(cont_voxels)

    cont_pcd = o3d.geometry.PointCloud()
    cont_pcd.points = o3d.utility.Vector3dVector(contour_points)
    cont_pcd.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(cont_pcd.points))])

    # make a point cloud of points that are not contour points (better for separating the colors)
    dists = np.asarray(sampled_pcd.compute_point_cloud_distance(cont_pcd))
    indices = np.where(dists > 0.00001)[0]
    road_without_contours_pcd = sampled_pcd.select_by_index(indices)

    return cont_pcd, road_without_contours_pcd


def remove_points_by_indices(pcd, indices):
    """
    removes points from a point cloud given their indices
    aim: not show points double (not predictable which one is gonna show, for the demonstration they have to be in a certain color)
    Parameters
    ----------
    pcd: the point cloud in which to remove points
    indices: indices of points to remove

    Returns
    -------
    a point cloud with the remaining points
    """
    pcd_points = np.asarray(pcd.points)
    remaining_points = np.delete(pcd_points, indices, axis=0)

    remaining_pcd = o3d.geometry.PointCloud()
    remaining_pcd.points = o3d.utility.Vector3dVector(remaining_points)

    return remaining_pcd


def line_set_from_poly(poly):
    # Define the points of the polygon
    points = np.array(poly)

    # Define the edges of the polygon
    lines = [[i, (i + 1) % len(points)] for i in range(len(points))]

    # Create a LineSet object
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    return line_set


def process_road_pcd(cont_voxels, veg_data):
    """
    cuts out the vegetation points that lay inside the clear height
    Parameters
    ----------
    cont_voxels: the voxels of the road contour
    veg_data: the data of the vegetation points

    Returns
    -------
    a point cloud with the vegetation points inside the clear height,
    a point cloud with the vegetation points outside the clear height,
    the line sets of the two polygons
    """
    # sort the polygon points, make a polygone using the points, and another 4m above
    sorted_contour_poly = sort_points_dist(cont_voxels)
    polygon_clearance_height = copy.deepcopy(sorted_contour_poly)
    for i in range(len(polygon_clearance_height)):
        polygon_clearance_height[i][-1] += 4

    # create line sets for both polygons
    ls_road = line_set_from_poly(sorted_contour_poly)
    ls_cl_h = line_set_from_poly(polygon_clearance_height)

    # make a point cloud only with vegetation
    vege_pcd = o3d.geometry.PointCloud()
    vege_pcd.points = o3d.utility.Vector3dVector(veg_data.to_numpy()[:, :3])

    # Calculate the average z-value
    z_values = np.array([point[2] for point in sorted_contour_poly])
    average_z = np.mean(z_values)

    # Create SelectionPolygonVolume object
    vol = o3d.visualization.SelectionPolygonVolume()
    vol.orthogonal_axis = "Z"
    vol.axis_min = average_z
    vol.axis_max = average_z + 4
    vol.bounding_polygon = o3d.utility.Vector3dVector(sorted_contour_poly)

    # Crop point cloud
    vege_inliers_pcd = vol.crop_point_cloud(vege_pcd)

    # Get indices of points inside the volume
    dists = np.asarray(vege_pcd.compute_point_cloud_distance(vege_inliers_pcd))
    indices = np.where(dists > 0.00001)[0]
    vege_outliers_pcd = vege_pcd.select_by_index(indices)

    # set colors
    vege_inliers_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(vege_inliers_pcd.points))])
    vege_outliers_pcd.colors = o3d.utility.Vector3dVector(
        [[0.3, 0.47, 0.23] for _ in range(len(vege_outliers_pcd.points))])
    ls_road.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(ls_road.lines))])
    ls_cl_h.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(ls_cl_h.lines))])

    return vege_inliers_pcd, vege_outliers_pcd, ls_road, ls_cl_h


def get_road_polygon(points_df):
    """

    Parameters
    ----------
    points_df

    Returns
    -------

    """
    # check start and end x position
    min_x_row = points_df.loc[points_df['x'].idxmin()]
    max_x_row = points_df.loc[points_df['x'].idxmax()]

    min_x, max_x = min_x_row['x'], max_x_row['x']
    beginning_z, ending_z = min_x_row['z'], max_x_row['z']

    x_range = max_x - min_x
    n_steps = 500  # how many steps where we check width of street
    step_size = x_range / n_steps  # how big the distance is between these steps
    outer_points_left, outer_points_right = [], []  # this is what we want

    for step in range(n_steps):
        # create a slice for each x level
        x_level = step * step_size + min_x
        min_x_slice, max_x_slice = math.floor(x_level), math.ceil(x_level)
        mask = (points_df['x'] > min_x_slice) & (points_df['x'] < max_x_slice)
        x_slice = points_df[mask]
        # only look at slices that contain at least 100 points
        if len(x_slice) < 20:
            continue
        # get the points where the y coordinate is min and max
        point_side_left, point_side_right = x_slice.loc[x_slice['y'].idxmax()].tolist(), x_slice.loc[
            x_slice['y'].idxmin()].tolist()
        outer_points_left.append(point_side_left)
        outer_points_right.append(point_side_right)

    polygon = outer_points_right + outer_points_left[::-1]

    return polygon
