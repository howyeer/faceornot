import numpy as np
import random 
import os
from PIL import Image
import csv

train_data = []
with open("D:/aaacode/faceornot/train_data.csv", "r") as f:
    reader = csv.reader(f)
    x = list(reader)
    Array = np.array(x, dtype=np.float64)

test_data = []
with open("D:/aaacode/faceornot/test_data.csv", "r") as f:
    reader = csv.reader(f)
    x = list(reader)
    Array = np.array(x, dtype=np.float64)