import os
from os.path import join
import numpy as np
import pandas as pd
from pathlib import Path
import logging

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import make_dir, DATASET

log = logging.getLogger(__name__)


class Pandaset(BaseDataset):
    """ This class is used to create a dataset based on the Pandaset autonomous
    driving dataset.

    https://pandaset.org/

    The dataset includes 42 semantic classes and covers more than 100 scenes,
    each of which is 8 seconds long.

    """

    def __init__(self,
                 dataset_path,
                 name="Pandaset",
                 cache_dir="./logs/cache",
                 use_cache=False,
                 ignored_label_inds=[],
                 test_result_folder='./logs/test_log',
                 test_split=['046', '027', '013', '029'],
                 training_split=['035', '042', '028', '043', '019', '038', '011', '016', '037', '005', '044', '002',
                                 '003', '001',
                                 '033', '023', '041', '040', '024', '034', '039', '030', '017', '032'],
                 validation_split=['021', '015'],
                 all_split=['035', '042', '028', '043', '019', '038', '011', '016', '037', '005', '044', '002', '003',
                            '001', '033', '023', '041', '040', '024', '034', '039', '030', '017', '032', '046', '027',
                            '013', '029', '021', '015'],
                 **kwargs):

        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset.
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            ignored_label_inds: A list of labels that should be ignored in the dataset.
        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         ignored_label_inds=ignored_label_inds,
                         test_result_folder=test_result_folder,
                         test_split=test_split,
                         training_split=training_split,
                         validation_split=validation_split,
                         all_split=all_split,
                         **kwargs)
        cfg = self.cfg
        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {'1': 'Road',  # before class 7
                          '2': 'Vegetation',  # before class 5
                          '3': 'Other'}
        return label_to_names

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return PandasetSplit(self, split=split)

    def get_split_list(self, split):
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The split name should be one of
            'training', 'test', 'validation', or 'all'.
        """
        cfg = self.cfg
        dataset_path = cfg.dataset_path
        file_list = []

        if split in ['train', 'training']:
            seq_list = cfg.training_split
        elif split in ['test', 'testing']:
            seq_list = cfg.test_split
        elif split in ['val', 'validation']:
            seq_list = cfg.validation_split
        elif split in ['all']:
            seq_list = cfg.all_split
        else:
            raise ValueError("Invalid split {}".format(split))

        for seq_id in seq_list:
            pc_path = join(dataset_path, seq_id, 'lidar')
            for f in np.sort(os.listdir(pc_path)):
                if f.split('.')[-1] == 'gz':
                    file_list.append(join(pc_path, f))

        return file_list

    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.

        Args:
            dataset: The current dataset to which the datum belongs to.
            attr: The attribute that needs to be checked.

        Returns:
            If the dataum attribute is tested, then return the path where the
            attribute is stored; else, returns false.
        """
        pass

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the
            attribute passed.
            attrs: The attributes that correspond to the outputs passed in
            results.
        """
        cfg = self.cfg
        pred = results['predict_labels']
        name = attr['name']

        test_path = join(cfg.test_result_folder, 'sequences')
        make_dir(test_path)
        save_path = join(test_path, name, 'predictions')
        make_dir(save_path)
        pred = results['predict_labels']

        for ign in cfg.ignored_label_inds:
            pred[pred >= ign] += 1

        store_path = join(save_path, name + '.label')

        pred = pred.astype(np.uint32)
        pred.tofile(store_path)


class PandasetSplit(BaseDatasetSplit):
    """This class is used to create a split for Pandaset dataset.

    Args:
        dataset: The dataset to split.
        split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.
        **kwargs: The configuration of the model as keyword arguments.

    Returns:
        A dataset split object providing the requested subset of the data.
    """

    def __init__(self, dataset, split='train'):
        super().__init__(dataset, split=split)
        log.info("Found {} pointclouds for {}".format(len(self.path_list),
                                                      split))

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        label_path = pc_path.replace('lidar', 'annotations/semseg')

        points = pd.read_pickle(pc_path)
        labels = pd.read_pickle(label_path)

        intensity = points['i'].to_numpy().astype(np.float32)
        points = points.drop(columns=['i', 't', 'd']).to_numpy().astype(np.float32)
        labels = labels.to_numpy().astype(np.int32)

        # Define conditions
        condition_5 = (labels == 5)
        condition_7 = (labels == 7)

        # Update values based on conditions
        labels[condition_5] = 2
        labels[condition_7] = 1

        # Set the remaining values to 3
        labels[~(condition_5 | condition_7)] = 3

        data = {
            'point': points,
            'intensity': intensity,
            'label': labels
        }

        return data

    def get_attr(self, idx):
        pc_path = self.path_list[idx]
        value = (pc_path).split('/')[-3]  # was 9 before, has to be -3
        name = Path(pc_path).name.split('.')[0]
        name = value + '_' + name

        attr = {'name': name, 'path': pc_path, 'split': self.split}
        return attr


DATASET._register_module(Pandaset)
