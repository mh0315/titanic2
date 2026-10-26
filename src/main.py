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
    data = data.copy() 
    data["T"] = data["Name"].str.extract(r" ([A-Za-z]+)\.") #r是原始字符串前缀，表示匹配一个真正的.，如果不加\则会把.识别成任意一个字符，+表示前面的字符类出现一次或多次
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

def sjj (y, test_size=0.3, seed=42):# 数据集划分
  
    y = np.asarray(y)
    train_idx,test_idx = [],[]
    np.random.seed(seed) 
    idx = np.arange(0,len(y))
    np.random.shuffle(idx)
    n = int(round(len(y)*test_size))
    train_idx = idx[n:]
    test_idx = idx[:n]
    return train_idx,test_idx

train_idx, test_idx= sjj(y, test_size=0.3, seed=42)
train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

def QSZ (train,test):
    train,test = train.copy(),test.copy()
    for a in [train,test]:
        a["Age"] = a["Age"].fillna(train["Age"].median())
        a["Embarked"] = a["Embarked"].fillna(train["Embarked"].mode()[0])
        a["Fare"] = a["Fare"].fillna(train["Fare"].median())
    return train,test
train_df, test_df = QSZ(train_df, test_df)

train_x_df = train_df.drop(columns=["PassengerId", "Name", "Ticket","Cabin", "Survived"])
test_x_df = test_df.drop(columns=["PassengerId", "Name", "Ticket","Cabin", "Survived"])

leibie_cols = ["Pclass", "Sex", "Embarked", "T","NC", "alone"]

num_cols = ["Fare","family","Age"]

def lbbm(df,cols):
    maps = {}
    for col in cols:
        b = sorted(df[col].astype(str).unique())
        maps[col] = {bm:i + 1  for i, bm in enumerate(b)}
    return maps

l_maps = lbbm(train_df,leibie_cols)

def zhnum(df, cols, maps):
    arr = []
    for i in cols:
        col_num= df[i].astype(str).map(maps[i]).values
        arr.append(col_num)
    return np.column_stack(arr)

x_testcol = zhnum(test_x_df,leibie_cols,l_maps)
x_traincol = zhnum(train_x_df,leibie_cols,l_maps)

num_mean = train_x_df[num_cols].mean()
num_std = train_x_df[num_cols].std().replace(0, 1)

x_trainnum = ((train_x_df[num_cols] - num_mean) / num_std).values
x_testnum = ((test_x_df[num_cols] - num_mean) / num_std).values

x_train_f =np.hstack([x_traincol, x_trainnum])
x = np.hstack([x_testcol, x_testnum])
y_train= train_df["Survived"].values

y= test_df["Survived"].values
class mydata(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]
    def __len__(self):
        return len(self.y)
train_ds= mydata(x_train_f, y_train)
test_ds = mydata(x, y)

train_ld = DataLoader(dataset=train_ds,batch_size=32,shuffle=True,num_workers=0,drop_last=False)
test_ld = DataLoader(dataset=test_ds,batch_size=32,shuffle=False,num_workers=0,drop_last=False)
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
Loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mod.parameters(), lr=0.05)
epochs = 100
a,b = [],[]
for epochs in range(epochs):
    mod.train() #训练模式
    train_loss = 0 #累计训练损失
    train_correct = 0 #训练正确数量
    train_total = 0 #训练总数量
    for x, y in train_ld:
        optimizer.zero_grad() #梯度清空
        outputs = mod(x.float()) #前向传播
        loss = Loss(outputs,y) #计算损失
        loss.backward() #反向传播
        optimizer.step() #更新参数
        train_loss += loss.item() #累计损失
        _,predicted = torch.max(outputs,dim=1) #寻找预测类别
        train_correct += (predicted == y ).sum().item() #正确数量
        train_total += y.size(0)#统计总数
    Accuracy1 = train_correct/train_total
    a.append(Accuracy1)
    mod.eval() #测试模式
    test_correct = 0 
    test_total = 0
    with torch.no_grad(): #关闭梯度
        for x, y in test_ld:
            outputs = mod(x.float())
            _,predicted = torch.max(outputs,1)
            test_correct += (predicted == y ).sum().item()
            test_total += y.size(0)
        Accuracy2 = test_correct/test_total
        b.append(Accuracy2) 
print(a,b)
torch.save(mod.state_dict(), "titanic.pth")

preprocess = {
    "l_maps": l_maps,                     
    "num_mean": num_mean,                 
    "num_std": num_std,                   
    "age_median": train_df["Age"].median(),
    "fare_median": train_df["Fare"].median(),
    "embarked_mode": train_df["Embarked"].mode()[0],
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