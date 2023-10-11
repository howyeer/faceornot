import numpy as np
import random 
import os
from PIL import Image
import csv
import cv2
import shutil


if not os.path.exists("D:/aaacode/faceornot/img"): 
    os.mkdir("D:/aaacode/faceornot/img")
else: 
    shutil.rmtree("D:/aaacode/faceornot/img/")
    os.mkdir("D:/aaacode/faceornot/img")

train_data = []
train_label = []
with open("D:/aaacode/faceornot/train_data.csv", 'r', encoding='UTF-8', errors='ignore')as f, \
     open("D:/aaacode/faceornot/train_label.csv", 'r', encoding='UTF-8', errors='ignore')as l:
    reader_f = csv.reader(f)
    reader_l = csv.reader(l)
    for row in reader_f:
        train_data.append(row)
        print(len(row))
    for label in reader_l:
        train_label.append(label)
       

# for i, row in enumerate(train_data):
#     array = np.array(row, dtype=np.float64)
#     array = array*255
#     img = array.reshape(16, 16)
#     img_path = os.path.join("D:/aaacode/faceornot/img",  "{}.png".format(i))
#     cv2.imwrite(img_path, img)
