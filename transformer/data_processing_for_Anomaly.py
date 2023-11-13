import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

# 找到所有恶意车辆的ID
def get_anomaly_vehicles(root):
    anomaly_vehicles = {}
    for dirname in os.listdir(root): 
        for filename in os.listdir(os.path.join(root,dirname)):
            if filename.startswith('traceJSON'):
                r=re.findall(r'\d+', filename)
                id,type=int(r[0]),int(r[2])
                if type!=0:
                    anomaly_vehicles[id]=type
    return anomaly_vehicles

root=''

anomaly_vehicles=get_anomaly_vehicles(root)
merged_df = pd.DataFrame()
for dirname in os.listdir(root): 
    dirpath=os.path.join(root,dirname)
    for filename in os.listdir(dirpath):
        if filename.startswith('traceJSON'):
            df=pd.read_json(os.path.join(dirpath,filename),orient='records', lines=True)
            if 'sender' in df.columns:
                filtered_df = df[(df['type'] == 3) & (df['sender'].isin(anomaly_vehicles))]
                merged_df = pd.concat([merged_df, filtered_df], ignore_index=True)
df = merged_df.drop_duplicates(subset='messageID') # 筛选掉重复的messageID行
sender_messages = {}
for index, row in df.iterrows():
        sender = row['sender']
        
        if sender not in sender_messages:
            sender_messages[sender] = []
        
        if len(sender_messages[sender])<200:
            sendtime_pseudoID_pos_spd_acl_hed=np.concatenate(([row['sendTime']], [row['senderPseudo']], row['pos'][:2], row['spd'][:2], row['acl'][:2], row['hed'][:2]))
            sender_messages[sender].append(sendtime_pseudoID_pos_spd_acl_hed)

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

data={}
for sender,msgs in sender_messages.items():
    if len(msgs)>10:
        tp=anomaly_vehicles[sender]
        if tp not in data:
            data[tp]=[]
        data[tp].append(process4Msgs(msgs))
for tp,dt in data.items():
    np.save(f'Atype{tp}.npy',np.array(dt))
