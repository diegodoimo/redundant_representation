from torchvision.datasets.vision import VisionDataset
import torch
import os
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union

#*******************************************************************************
class MNIST_pc(VisionDataset):

    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,):
        super(MNIST_pc, self).__init__(root, transform=transform,
                                    target_transform=target_transform)

        self.train = train  # training set or test set
        self.root = root

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.root, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class adv(VisionDataset):

    def __init__(self,
            root: str,
            data_file: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,):
        super(adv, self).__init__(root, transform=transform,
                                    target_transform=target_transform)

        self.root = root
        self.data, self.targets = torch.load(os.path.join(self.root, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
