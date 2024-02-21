import open3d as o3d


def visualize_pcds_one_after_another(geometries_lists):
    """
    takes a list of pointclouds and shows them in open3d one after another
    Parameters
    ----------
    geometries_lists: list of pointclouds to show

    Returns
    -------

    """
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    idx = 0
    pcd = geometries_lists[idx]
    for g in pcd:
        # print(type(g))
        vis.add_geometry(g)
    # print("")

    def right_click(vis):
        nonlocal idx
        idx = idx + 1
        vis.clear_geometries()
        if len(geometries_lists) <= idx:
            vis.destroy_window()
            return
        pcd = geometries_lists[idx]
        for g in pcd:
            # print(type(g))
            vis.add_geometry(g)
        # print("")

    def left_click(vis):
        nonlocal idx
        idx = idx - 1
        vis.clear_geometries()
        pcd = geometries_lists[idx]
        for g in pcd:
            # print(type(g))
            vis.add_geometry(g)
        # print("")

    def exit_key(vis):
        vis.destroy_window()

    vis.register_key_callback(262, right_click)
    vis.register_key_callback(263, left_click)
    vis.register_key_callback(32, exit_key)
    vis.poll_events()
    vis.run()
