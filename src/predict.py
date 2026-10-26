import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

mod = model()
mod.load_state_dict(torch.load("titanic.pth", weights_only=True))
mod.eval()

with open("preprocess.pkl", "rb") as f:
    preprocess = pickle.load(f)


l_maps = preprocess["l_maps"] 
num_mean = preprocess["num_mean"]
num_std = preprocess["num_std"]
age_median = preprocess["age_median"]
fare_median = preprocess["fare_median"]
embarked_mode = preprocess["embarked_mode"]
leibie_cols = preprocess["leibie_cols"]
num_cols = preprocess["num_cols"]

prep = {
    "l_maps" : l_maps,          
    "num_std" : num_std,
    "age_median" : age_median,
    "fare_median"  : fare_median,
    "embarked_mode" : embarked_mode,
    "leibie_cols"  : leibie_cols,
    "num_cols" : num_cols,
    "num_mean" : num_mean
}

def f1(data):
    data = data.copy()
    data["T"] = data["Name"].str.extract(r" ([A-Za-z]+)\.")
    title_map = {
        "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"
    }
    data["T"] = data["T"].map(title_map).fillna("No")
    data["NC"] = data["Cabin"].fillna("U").astype(str).str[0]
    data["family"] = data["SibSp"] + data["Parch"] + 1
    data["alone"] = (data["family"] == 1).astype(int)   
    return data

def zhnum(df, cols, maps):
    arr = []
    for i in cols:
        col_num= df[i].astype(str).map(maps[i]).values
        arr.append(col_num)
    return np.column_stack(arr)


def preprocess_new_sample(new_df, prep):
    new_df = f1(new_df)
    new_df["Age"] = new_df["Age"].fillna(prep["age_median"])
    new_df["Fare"] = new_df["Fare"].fillna(prep["fare_median"])
    new_df["Embarked"] = new_df["Embarked"].fillna(prep["embarked_mode"])
    X_leibie = zhnum(new_df, prep["leibie_cols"], prep["l_maps"])
    X_num = ((new_df[prep["num_cols"]] - prep["num_mean"]) / prep["num_std"]).values
    X = np.hstack([X_leibie, X_num]).astype(np.float32)
    return X

def yuce(mod, prep, passenger_dict):
    new_df = pd.DataFrame([passenger_dict])
    X = preprocess_new_sample(new_df, prep)

    x_tensor = torch.tensor(X).float()

    with torch.no_grad():
        outputs = mod(x_tensor)
        _, predicted = torch.max(outputs, 1)
        pred = predicted.item()

    return pred


def main():
    
    print("=" * 50)
    print("泰坦尼克号幸存预测系统")
    print("=" * 50)
    print("请依次输入乘客信息：\n")

    pclass = int(input("舱位等级 Pclass (1/2/3): "))
    name = input("姓名 Name（例如 'Braund, Mr. Owen Harris'）: ")
    sex = input("性别 Sex (male/female): ")
    age = float(input("年龄 Age: "))
    sibsp = int(input("兄弟姐妹/配偶数 SibSp: "))
    parch = int(input("父母/子女数 Parch: "))
    ticket = input("票号 Ticket（例如 'A/5 21171'）: ")
    fare = float(input("票价 Fare: "))
    cabin = input("船舱 Cabin（例如 'C85'）: ")
    embarked = input("登船港口 Embarked (C/Q/S): ")

    passenger = {
        "Pclass": pclass,
        "Name": name,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": ticket,
        "Fare": fare,
        "Cabin": cabin ,
        "Embarked": embarked,
    }
    pred = yuce(mod, prep, passenger)

    print("\n" + "=" * 50)
    print(f"预测结果：{pred}  ({'存活' if pred == 1 else '未存活'})")
    print("=" * 50)

main()
