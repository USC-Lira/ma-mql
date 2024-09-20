import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch, os, pickle, gzip

class ExpertSet(Dataset):
    def __init__(self, csv_file, max_episode=0, test_iteration=0):
        self.file_name = csv_file
        if os.path.exists(self.file_name):
            print("loading file...")
            self.load(self.file_name)

    def temperature_softmax(self, logits, temperature=1.0):
        return F.softmax(logits / temperature, dim=-1)

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
        ind = np.random.randint(0, 20000, size=batch_size)
        x = np.array([self.data[i][0] for i in ind])
        y = np.array([self.data[i][1] for i in ind])
        u = np.array([self.data[i][2] for i in ind], copy=False)
        r = np.array([self.data[i][3] for i in ind], copy=False)
        d = np.array([self.data[i][4] for i in ind], copy=False).reshape(-1, 1)
        return x, y, u, r, d
