# 对GroundTruth的数据处理

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

def build_seq(path):
    df=pd.read_json(path,orient='records', lines=True)
    sender_messages = {}
    for index, row in df.iterrows():
        sender = row['sender']
        pos_spd = row['pos'][:2] + row['spd'][:2]
    
        if sender not in sender_messages:
            sender_messages[sender] = []
    
        sender_messages[sender].append(pos_spd)
    tmp=[]
    for sender,seqs in sender_messages.items():
        for i in range(int((len(seqs)-10)/10)):
            tmp.append(seqs[10*i:10*i+20])
    return tmp

root_dir=r'D:\SRP Projects\origin datas\seq_data'

data=[]
for filepath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.startswith('traceGroundTruthJSON'):
            data.extend(build_seq(os.path.join(filepath,filename)))


data=scale(np.array(data).reshape(-1,4))
data=data.reshape(-1,20,4)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.tensor(self.data[index], dtype=torch.float)
        return data

train_data, test_data = train_test_split(data, test_size=0.15, random_state=42)
train_dataset = MyDataset(train_data)
test_dataset = MyDataset(test_data)

torch.save(train_dataset, 'train_dataset.pt')  #85%用于训练重构器
torch.save(test_dataset, 'test_dataset.pt') #15%用于测试