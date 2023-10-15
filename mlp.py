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

epoch = 40
hid1 = 100
hid2 = 50
group_num = 5
dis_val = 0.6

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
            label_list.append(1)
        else:   
            label_list.append(0)

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

    def __init__(self, input, hidden1, hidden2, output, dropout=0.1):
        super().__init__()
        self.hid1 = nn.Linear(input, hidden1, bias=True)
        self.hid2 = nn.Linear(hidden1, hidden2, bias=True)
        self.out = nn.Linear(hidden2, output, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.hid1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.hid2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.out(x)
        return x

model = Mlp(361, hid1, hid2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
lr_rate = StepLR(optimizer, step_size=4, gamma=0.8, verbose=True)
# cosine_lr = CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-3, verbose=True) 
loss_fn = torch.nn.BCELoss()
fn = nn.Sigmoid()

for n in range(group_num):
    val_index = result.pop(n)
    train_inex = np.array(result)
    train_inex = train_inex.flatten()
    print("第", n, "组")
    acc_list = [[]]
    for epoch_id in range(epoch):
        model.train()
        for index in list(train_inex):
            predict = model(torch.Tensor(data[index]))
            predict = fn(predict)
            loss = loss_fn(predict, torch.Tensor(label_list[index]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_rate.step()
        if epoch_id%5 == 0:
            print('epoch={}, loss={}'.format(epoch_id, loss.flatten().np()))
        
        model.eval()
        print("模型评估")
        i = 0
        acc = 0
        for index in val_index:
            pre = model(data[index])
            if pre >= dis_val: pre = 1
            else: pre = 0
            i += 1
            if pre == label[index]:
                acc +=1
            print("acc={}".format(acc/i))
            acc_list[n].append(acc/i)
