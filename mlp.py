import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import csv
import random
import os 
import cv2
from PIL import Image

epoch = 100
hid1 = 50
hid2 = 20
group_num = 5
dis_val = 0.4

data = []
label_list = []
with open("D:/aaacode/faceornot/train_data.csv", 'r')as f, \
     open("D:/aaacode/faceornot/train_label.csv", 'r')as l:
    reader_f = csv.reader(f)
    reader_l = csv.reader(l)
    for row in reader_f:
        data.append(np.array(row, dtype=np.float64))
    for label in reader_l:
        if int(label[0]) == 1:
            label_list.append([1])
        else:   
            label_list.append([0])

group_size = int(400/group_num)
def random_group(arr, group_size):
    random.shuffle(arr)
    return [arr[i:i+group_size] for i in range(0, len(arr), group_size)]

arr_1 = list(range(400))
arr_0 = list(range(400,800))
result_1 = random_group(arr_1, group_size)
result_0 = random_group(arr_0, group_size)
for j in range(group_num):
    result_1[j].extend(result_0[j]) 
result = result_1

class Mlp(nn.Module):

    def __init__(self, input, hidden1, hidden2, output, dropout=0.2):
        super().__init__()
        self.hid1 = nn.Linear(input, hidden1, bias=True)
        self.hid2 = nn.Linear(hidden1, hidden2, bias=True)
        self.out = nn.Linear(hidden2, output, bias=True)
        self.act = nn.GELU()
        self.fnc = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
    
    def init_normal(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.hid1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.hid2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.out(x)
        x = self.fnc(x)
        return x

acc_list = [[], [],  [], [], []]
for n in range(group_num):
    #定义模型加初始化
    model = Mlp(361, hid1, hid2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    # lr_rate = StepLR(optimizer, step_size=4, gamma=0.8, verbose=True)
    lr_rate = CosineAnnealingLR(optimizer, T_max=epoch, eta_min=5e-4) 
    loss_fn = nn.BCELoss()
    
    result_d = []
    result_d.extend(result)
    val_index = result_d.pop(n)
    train_inex = np.array(result_d)
    train_inex = train_inex.flatten()
    print("第", n, "组")

    for epoch_id in range(epoch):
        model.train()
        for index in list(train_inex):
            predict = model(torch.Tensor(data[index]))
            label_i = torch.Tensor(label_list[index])
            loss = loss_fn(predict, label_i)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        lr_rate.step()
        if (epoch_id+1)%20 == 0:
            print('epoch={}, loss={}'.format(epoch_id+1, loss.detach().numpy()))
        
            model.eval()
            print("模型评估")
            i = 0
            acc = 0
            for i_v in val_index:
                pre = model(torch.Tensor(data[i_v]))
                if pre >= dis_val: pred = [1]
                else: pred = [0]
                i += 1
                if pred == label_list[i_v]:
                    acc += 1
            print("acc={}".format(acc/i))
            acc_list[n].append(acc/i)
print(acc_list)
for k in range(group_num):
    acc_array = np.zeros_like(np.array(acc_list[0]))
    acc_array += np.array(acc_list[k]) 
    acc_array = acc_array/group_num
print(acc_array)
