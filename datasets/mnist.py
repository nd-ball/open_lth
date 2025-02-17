# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from PIL import Image
import torchvision

from datasets import base
from platforms.platform import get_platform
from datasets.my_data_loaders import my_MNIST 

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
        return Dataset(train_set.data, train_set.targets, [0]*60000, [0]*60000)

    @staticmethod
    def get_test_set():
        test_set = torchvision.datasets.MNIST(
            train=False, root=os.path.join(get_platform().dataset_root, 'mnist'), download=True)
        return Dataset(test_set.data, test_set.targets, [0]*10000, [0]*10000)

    def __init__(self,  examples, labels, idx, diffs):
        tensor_transforms = [torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])]

    def example_to_image(self, example):
        return Image.fromarray(example.numpy(), mode='L')

    def __getitem__(self, index):
        return self._examples[index], self._labels[index], self._idx[index], self._diffs[index]



DataLoader = base.DataLoader
