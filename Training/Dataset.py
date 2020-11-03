import torch
from torch.utils import data
import numpy as np



class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'

  def __init__(self, x_data, p_data, r_data, labels, train=False):
        self.x_data = torch.from_numpy(x_data)
        self.p_data = torch.from_numpy(p_data)
        self.r_data = torch.from_numpy(r_data)
        self.labels = torch.from_numpy(labels)
        length = len(self.x_data)
        self.list_IDs = range(0, length)
        self.train = train


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)


  def __getitem__(self, index):
        'Generates one sample of data'
        ID = index

        X = self.x_data[ID]
        P = self.p_data[ID]
        R = self.r_data[ID]
        y = self.labels[ID]

        if self.train == True:
            if np.random.choice([True, False]):
                X = torch.flip(X, [2])
                y[1] = -y[1]  # Y
                y[3] = -y[3]  # Relative YAW

        return X, P, R, y
