# 半监督采样
# 训练集 62295:250(EventualStop)
# 验证集 5700: 300*19
# 测试集 24700: 1300*19
import numpy as np
import torch
from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label=label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data = torch.tensor(self.data[index], dtype=torch.float)
        label = torch.tensor(self.label[index], dtype=torch.int)
        return data,label

gt=np.load('GroundTruth.npy')
total_gt=gt.shape[0]
train_indices = np.random.choice(total_gt, size=62295, replace=False)
remain_indices = np.setdiff1d(np.arange(total_gt), train_indices)
valid_indices = np.random.choice(remain_indices, size=5700, replace=False)
remain_indices = np.setdiff1d(remain_indices, valid_indices)
test_indices = np.random.choice(remain_indices, size=24700, replace=False)

anomaly=[np.load(f'Atype{i}.npy') for i in range(1,20)]
train_set_ES=[]
valid_set=[]
test_set=[]
for i  in range(1,20):
    total_typei=anomaly[i-1].shape[0]
    v_i=np.random.choice(total_typei, size=300, replace=False)
    valid_set=np.concatenate((valid_set, anomaly[i-1][v_i]))
    r_i = np.setdiff1d(np.arange(total_typei), v_i)
    t_i=np.random.choice(r_i, size=1300, replace=False)
    test_set=np.concatenate((test_set, anomaly[i-1][t_i]))
    if(i==9):
        r_i=np.setdiff1d(r_i, t_i)
        train_indices_ES=np.random.choice(r_i, size=250, replace=False)
        train_set_ES=np.concatenate((train_set_ES, anomaly[i-1][train_indices_ES]))



torch.save(MyDataset(np.concatenate((gt[train_indices], train_set_ES)),np.concatenate((np.ones(62295), -np.ones(250)))), 'train_dataset.pt')
torch.save(MyDataset(np.concatenate((gt[valid_indices], valid_set)),np.concatenate((np.ones(5700), -np.ones(300*19)))), 'valid_dataset.pt')
torch.save(MyDataset(np.concatenate((gt[test_indices], test_set)),np.concatenate((np.ones(24700), -np.ones(1300*19)))), 'test_dataset.pt')