import torch, os, pickle, gzip
import numpy as np
from torch.utils.data import Dataset

class ExpertSet(Dataset):
    def __init__(self, csv_file, max_episode=0, test_iteration=0):
        self.file_name = csv_file
        if os.path.exists(self.file_name):
            print("loading file...")
            self.load(self.file_name)
       
    def save(self, file_name):
        with gzip.open(file_name, 'wb') as f:
            pickle.dump(self.data, f)

    def load(self, file_name):
        with gzip.open(file_name, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state = torch.tensor(self.data[idx][0])
        action0 = torch.tensor(self.data[idx][1])
        action1 = torch.tensor(self.data[idx][2])
        return state, action0, action1

    def sample(self, batch_size):
        ind = np.random.randint(0, 35000, size=batch_size)
        x, y, u, r, d = [], [], [], [], []
        for i in ind:
            X, Y, U, R, D = self.data[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
        return np.array(x), np.array(y), np.array(u), np.array(r), np.array(d).reshape(-1, 1)
