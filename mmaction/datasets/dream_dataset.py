from .registry import DATASETS
from .base import BaseDataset

import os.path as osp
import numpy as np
import copy
from collections import defaultdict

from torch.utils.data import Dataset

import mmcv


@DATASETS.register_module()
class DreamVideoDataset(BaseDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self, ann_file, pipeline, split=None, start_index=0, **kwargs):
        self.split = split
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)
        
    def parse_by_class(self):
        video_infos_by_class = defaultdict(list)
        for item in self.video_infos:
            label = item['label']
            video_infos_by_class[label].append(item)
        return video_infos_by_class
    
    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        if self.sample_by_class:
            # Then, the idx is the class index
            samples = self.video_infos_by_class[idx]
            results = copy.deepcopy(np.random.choice(samples))
        else:
            results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        if self.sample_by_class:
            # Then, the idx is the class index
            samples = self.video_infos_by_class[idx]
            results = copy.deepcopy(np.random.choice(samples))
        else:
            results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)
    
    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.pkl')
        return self.load_pkl_annotations()

    def load_pkl_annotations(self):
        data = mmcv.load(self.ann_file)

        if self.split:
            split, data = data['split'], data['annotations']
            identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
            split = set(split[self.split])
            data = [x for x in data if x[identifier] in split]

        for item in data:
            # Sometimes we may need to load anno from the file
            if 'filename' in item:
                item['filename'] = osp.join(self.data_prefix, item['filename'])
                item['rgb'] = np.load(f"{item['filename']}.npy")
            if 'frame_dir' in item:
                item['frame_dir'] = osp.join(self.data_prefix, item['frame_dir'])
                item['rgb'] = np.load(f"{item['frame_dir']}.npy")
        return data


@DATASETS.register_module()
class DREAMDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        super().__init__()
        
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        
    def __len__(self):
        return self.dataset1.__len__()
    
    def __getitem__(self, idx):
        data1 = self.dataset1.__getitem__(idx)
        data2 = self.dataset2.__getitem__(idx)
        
        return data1, data2