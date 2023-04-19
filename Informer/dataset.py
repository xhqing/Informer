import torch
from torch.utils.data import Dataset, DataLoader

class Dataset_Custom(Dataset):
    def __init__(self, data):
        self.data_x = data[0]
        self.data_y = data[1]
        self.data_x_mark = data[2]
        self.data_y_mark = data[3]

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, idx):
        seq_x = self.data_x[idx]
        seq_y = self.data_y[idx]
        seq_x_mark = self.data_x_mark[idx]
        seq_y_mark = self.data_y_mark[idx]
        return seq_x, seq_y, seq_x_mark, seq_y_mark


