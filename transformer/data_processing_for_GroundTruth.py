# 对GroundTruth的数据处理

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

root=''   #下载数据集的根目录路径

sender_messages = {}
def process4Json(path):
    df=pd.read_json(path,orient='records', lines=True)
    
    for index, row in df.iterrows():
        sender = row['sender']
        
        if sender not in sender_messages:
            sender_messages[sender] = []
        
        if len(sender_messages[sender])<200:
            sendtime_pseudoID_pos_spd_acl_hed=np.concatenate(([row['sendTime']], [row['senderPseudo']], row['pos'][:2], row['spd'][:2], row['acl'][:2], row['hed'][:2]))
            sender_messages[sender].append(sendtime_pseudoID_pos_spd_acl_hed)

for dirname in os.listdir(root): 
    dirpath=os.path.join(root,dirname)
    for filename in os.listdir(dirpath):
        if filename.startswith('traceGroundTruthJSON'):
            process4Json(os.path.join(dirpath,filename))
            break


def process4Msgs(msgs):
    msgs=np.array(msgs)
    msgs[1:,0]=[(msgs[i,0]-msgs[i-1,0]) for i in range(1,len(msgs))]
    msgs[:,1]=len(set(msgs[:,1]))
    msgs[:,2:]=scale(msgs[:,2:])
    if msgs.shape[0]<200:
        msgs_makeup=np.zeros((200,10))
        msgs_makeup[:msgs.shape[0],:]=msgs
        return msgs_makeup
    return msgs

data=[]
for msgs in sender_messages.values():
    if len(msgs)>10:
        data.append(process4Msgs(msgs))
# save
np.save('GroundTruth.npy',np.array(data))