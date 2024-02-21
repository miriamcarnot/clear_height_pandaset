import random

def get_color_palette(seq, focus=True):
    """
    defines the colors for the different classes
    Parameters
    ----------
    sequence = sequence of frames
    focus = all classes in black except for road, side walk and vegetation, otherwise random (with exceptions)

    Returns
    -------
    the color palette
    """
    # {'1': 'Smoke', '2': 'Exhaust', '3': 'Spray or rain', '4': 'Reflection', '5': 'Vegetation', '6': 'Ground',
    # '7': 'Road', '8': 'Lane Line Marking', '9': 'Stop Line Marking', '10': 'Other Road Marking', '11': 'Sidewalk',
    # '12': 'Driveway', '13': 'Car', '14': 'Pickup Truck', '15': 'Medium-sized Truck', '16': 'Semi-truck',
    # '17': 'Towed Object', '18': 'Motorcycle', '19': 'Other Vehicle - Construction Vehicle',
    # '20': 'Other Vehicle - Uncommon', '21': 'Other Vehicle - Pedicab', '22': 'Emergency Vehicle', '23': 'Bus',
    # '24': 'Personal Mobility Device', '25': 'Motorized Scooter', '26': 'Bicycle', '27': 'Train', '28': 'Trolley',
    # '29': 'Tram / Subway', '30': 'Pedestrian', '31': 'Pedestrian with Object', '32': 'Animals - Bird',
    # '33': 'Animals - Other', '34': 'Pylons', '35': 'Road Barriers', '36': 'Signs', '37': 'Cones', '38':
    # 'Construction Signs', '39': 'Temporary Construction Barriers', '40': 'Rolling Containers', '41': 'Building',
    # '42': 'Other Static Object'}

    number_of_classes = len(seq.semseg.classes)
    if focus:
        # all black except 3 classes of interest
        colors = [[0.0, 0.0, 0.0]] * (number_of_classes + 1)
        colors[5] = [0.3, 0.47, 0.23]  # vegetation in green
        colors[7] = [0.45, 0.45, 0.45]  # road (clear height: 4m)
        colors[11] = [0.61, 0.27, 0.12]  # sidewalk (clear height: 2.5m)
    else:
        # all random
        colors = [[random.random(), random.random(), random.random()] for _ in range(number_of_classes + 1)]
        colors[5] = [0.3, 0.47, 0.23]  # vegetation in green
        colors[6] = [0.49, 0.27, 0.18]
        colors[7] = [0.33, 0.33, 0.33]  # road (clear height: 4m) in grey
        colors[11] = [0.61, 0.27, 0.12]  # sidewalk (clear height: 2.5m) in brown
        colors[13] = [0, 0, 0.80]  # cars in dark blue
        colors[30] = [0.86, 0.66, 0.67]  # pedestrians in pink
        colors[36] = [0.7, 0.09, 0.1]  # signs in red
        colors[41] = [0.71, 0.64, 0.46]  # buildings in beige

    return colors