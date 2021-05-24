# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
from PIL import Image
import sys
import torchvision

from datasets import base
from platforms.platform import get_platform

import csv
import torch 

class CIFAR10(torchvision.datasets.CIFAR10):
    """A subclass to suppress an annoying print statement in the torchvision CIFAR-10 library.

    Not strictly necessary - you can just use `torchvision.datasets.CIFAR10 if the print
    message doesn't bother you.
    """

    def download(self):
        if get_platform().is_primary_process:
            with get_platform().open(os.devnull, 'w') as fp:
                sys.stdout = fp
                super(CIFAR10, self).download()
                sys.stdout = sys.__stdout__
        get_platform().barrier()


class Dataset(base.ImageDataset):
    """The CIFAR-10 dataset."""

    @staticmethod
    def num_train_examples(): return 50000

    @staticmethod
    def num_test_examples(): return 10000

    @staticmethod
    def num_classes(): return 10

    @staticmethod
    def get_train_set(use_augmentation):
        augment = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, 4)]
        train_set = CIFAR10(train=True, root=os.path.join(get_platform().dataset_root, 'cifar10'), download=True)
        difficulties = []
        diff_file = '/afs/crc.nd.edu/user/j/jlalor1/data/artificial-crowd-rps/cifar_diffs_train.csv'
        with open(diff_file, 'r') as infile:
            diffreader = csv.reader(infile, delimiter=',')

            for row in diffreader:
                difficulties.append((eval(row[0]), eval(row[1])))
            difficulties_sorted = sorted(difficulties, key=lambda x:x[0])
            difficulties = [d[1] for d in difficulties_sorted]

            difficulties = torch.tensor(difficulties, dtype=torch.float)
            indices = list(range(50000))
        return Dataset(train_set.data, np.array(train_set.targets), augment if use_augmentation else [], idx=indices, diffs=difficulties)

    @staticmethod
    def get_test_set():
        test_set = CIFAR10(train=False, root=os.path.join(get_platform().dataset_root, 'cifar10'), download=True)
        return Dataset(test_set.data, np.array(test_set.targets), idx=list(range(10000)), diffs=[0] * 10000)

    def __init__(self,  examples, labels, image_transforms=None, idx=None, diffs=None):
        super(Dataset, self).__init__(examples, labels, image_transforms or [],
                                      [torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])], idx=idx, diffs=diffs)

    def example_to_image(self, example):
        return Image.fromarray(example)


DataLoader = base.DataLoader
