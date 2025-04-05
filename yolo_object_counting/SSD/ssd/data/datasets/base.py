# ssd/data/datasets/base.py
class BaseDataset:
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError