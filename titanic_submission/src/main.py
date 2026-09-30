import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as op
import pickle

df = pd.read_csv("train.csv")
def f1(data):
    data = data.copy() # 数据规约
    data["T"] = data["Name"].str.extract(r" ([A-Za-z]+)\.")
    title_map = {
        "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"
    }
    data["T"] = data["T"].map(title_map).fillna("No")
    data["NC"] = data["Cabin"].fillna("U").astype(str).str[0]
    data["family"] = data["SibSp"]+ data["Parch"] + 1
    data["alone"] = (data["family"]==1).astype(int)
    return data
df = f1(df)
y = df["Survived"].values

def sjj (y, Ceshi_size=0.3, seed=42):# 数据集划分
  
    y = np.asarray(y)
    r = np.random.default_rng(seed)
    idx = np.arange(len(y)) #生成0-n-1的行号数组
    xunlian_idx,ceshi_idx = [],[]
    n = int(round(len(y)*Ceshi_size))
    r.shuffle(idx)
    xunlian_idx.extend(idx[n:])
    ceshi_idx.extend(idx[:n])
    xunlian_idx = np.array(xunlian_idx)
    ceshi_idx = np.array(ceshi_idx)
    return xunlian_idx,ceshi_idx 
xunlian_idx, ceshi_idx= sjj(y, Ceshi_size=0.3, seed=42)
xunlian_df = df.iloc[xunlian_idx].copy()
ceshi_df = df.iloc[ceshi_idx].copy()

def QSZ (xunlian,ceshi):
    xunlian,ceshi = xunlian.copy(),ceshi.copy()
    for a in [xunlian,ceshi]:
        a["Age"] = a["Age"].fillna(xunlian["Age"].median())
        a["Embarked"] = a["Embarked"].fillna(xunlian["Embarked"].mode()[0])
        a["Fare"] = a["Fare"].fillna(xunlian["Fare"].median())
    return xunlian,ceshi
xunlian_df, ceshi_df = QSZ(xunlian_df, ceshi_df)
xunlian_x_df = xunlian_df.drop(columns=["PassengerId", "Name", "Ticket","Cabin", "Survived"])
ceshi_x_df = ceshi_df.drop(columns=["PassengerId", "Name", "Ticket","Cabin", "Survived"])

leibie_cols = ["Pclass", "Sex", "Embarked", "T","NC", "alone"]

num_cols = ["Fare","family","Age"]
def lbbm(df,cols):
    maps = {}
    for col in cols:# 赋名
        b = sorted(df[col].astype(str).unique())# 取值
        maps[col] = {bm:i + 1   for i, bm in enumerate(b)}# 对应
    return maps
l_maps = lbbm(xunlian_df,leibie_cols)
def zhnum(df, cols, maps):
    arr = np.zeros((len(df), len(cols)),dtype=np.int64)
    for c, col in enumerate(cols):
        arr[:, c] = (df[col].astype(str).map(maps[col]).values)
    return arr
x_ceshicol = zhnum(ceshi_x_df,leibie_cols,l_maps)
x_xunliancol = zhnum(xunlian_x_df,leibie_cols,l_maps)
num_mean = xunlian_x_df[num_cols].mean()
num_std = xunlian_x_df[num_cols].std().replace(0, 1)
x_xunliannum = ((xunlian_x_df[num_cols] - num_mean) / num_std).values
x_ceshinum = ((ceshi_x_df[num_cols] - num_mean) / num_std).values
x_xunlian_f =np.hstack([x_xunliancol, x_xunliannum]).astype(np.float32)
x_ceshi_f = np.hstack([x_ceshicol, x_ceshinum]).astype(np.float32)
y_xunlian = xunlian_df["Survived"].values
y_ceshi= ceshi_df["Survived"].values
class mydata(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]
    def __len__(self):
        return len(self.y)
xunlian_ds= mydata(x_xunlian_f, y_xunlian)
ceshi_ds = mydata(x_ceshi_f, y_ceshi)
xunlian_ld = DataLoader(dataset=xunlian_ds,batch_size=32,shuffle=True,num_workers=0,drop_last=False)
ceshi_ld = DataLoader(dataset=ceshi_ds,batch_size=32,shuffle=False,num_workers=0,drop_last=False)
class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
 
        self.fc1 = nn.Linear(9,128)
        self.fc2 = nn.Linear(128,2)
        self.relu = nn.ReLU()
    def forward(self,x):
        x =  self.fc1(x)  
        x =  self.relu(x)
        x =  self.fc2(x)
        return x
mod = model()
sunshi = nn.CrossEntropyLoss()
youhuaqi = torch.optim.Adam(mod.parameters(), lr=0.05)
epochs = 100
a,b = [],[]
for epochs in range(epochs):
    mod.train() #训练模式
    train_loss = 0 #累计训练损失
    train_correct = 0 #训练正确数量
    train_total = 0 #训练总数量
    for x_xunlian_f, y_xunlian in xunlian_ld:
        youhuaqi.zero_grad() #梯度清空
        outputs = mod(x_xunlian_f) #前向传播
        loss = sunshi(outputs,y_xunlian) #计算损失
        loss.backward() #反向传播
        youhuaqi.step() #更新参数
        train_loss += loss.item() #累计损失
        _,predicted = torch.max(outputs,1) #寻找预测类别
        train_correct += (predicted == y_xunlian ).sum().item() #正确数量
        train_total += y_xunlian.size(0)#统计总数
        Accuracy1 = train_correct/train_total
    a.append(Accuracy1)
    mod.eval() #测试模式
    train_correct = 0 
    train_total = 0
    with torch.no_grad(): #关闭梯度
        for x_ceshi_f, y_ceshi in ceshi_ld:
            outputs = mod(x_ceshi_f)
            _,predicted = torch.max(outputs,1)
            train_correct += (predicted == y_ceshi ).sum().item()
            train_total += y_ceshi.size(0)
            Accuracy2 = train_correct/train_total
        b.append(Accuracy2) 
torch.save(mod.state_dict(), "titanic.pth")
preprocess = {
    "l_maps": l_maps,                     
    "num_mean": num_mean,                 
    "num_std": num_std,                   
    "age_median": xunlian_df["Age"].median(),
    "fare_median": xunlian_df["Fare"].median(),
    "embarked_mode": xunlian_df["Embarked"].mode()[0],
    "leibie_cols": leibie_cols,           
    "num_cols": num_cols,               
}
with open("preprocess.pkl", "wb") as f:
    pickle.dump(preprocess, f)

epoch1 = range(1, 101)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure()
plt.plot(epoch1,a,label="训练准确率", color="blue")
plt.plot(epoch1,b,label="测试准确率", color="red")
plt.title("训练轮次与准确率变化图")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.show()       