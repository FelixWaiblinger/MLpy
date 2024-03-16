"""
"""

from mlpy.types import Dataset


class ImageFolderDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __len__(self) -> int:
        return super().__len__()


class CSVDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __len__(self) -> int:
        return super().__len__()