# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from PIL import Image
import torchvision

from datasets import base
from platforms.platform import get_platform
import csv
import torch 

class Dataset(base.ImageDataset):
    """The MNIST dataset."""

    @staticmethod
    def num_train_examples(): return 60000

    @staticmethod
    def num_test_examples(): return 10000

    @staticmethod
    def num_classes(): return 10

    @staticmethod
    def get_train_set(use_augmentation):
        # No augmentation for MNIST.
        train_set = torchvision.datasets.MNIST(
            train=True, root=os.path.join(get_platform().dataset_root, 'mnist'), download=True)
        diff_file = '/afs/crc.nd.edu/user/j/jlalor1/data/artificial-crowd-rps/mnist_diffs_train.csv'
        difficulties = []
        with open(diff_file, 'r') as infile:
            diffreader = csv.reader(infile, delimiter=',')

            for row in diffreader:
                difficulties.append((eval(row[0]), eval(row[1])))
            difficulties_sorted = sorted(difficulties, key=lambda x:x[0])
            difficulties = [d[1] for d in difficulties_sorted]

        difficulties = torch.tensor(difficulties, dtype=torch.float)
        indices = list(range(60000))
        return Dataset(train_set.data, train_set.targets, idx=indices, diffs=difficulties)

    @staticmethod
    def get_test_set():
        test_set = torchvision.datasets.MNIST(
            train=False, root=os.path.join(get_platform().dataset_root, 'mnist'), download=True)
        return Dataset(test_set.data, test_set.targets, list(range(10000)), [0] * 10000)

    def __init__(self,  examples, labels, idx=None, diffs=None):
        tensor_transforms = [torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])]
        super(Dataset, self).__init__(examples, labels, [], tensor_transforms, idx=idx, diffs=diffs)

    def example_to_image(self, example):
        return Image.fromarray(example.numpy(), mode='L')


DataLoader = base.DataLoader
