from .registry import DATASETS

from torch.utils.data import Dataset


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