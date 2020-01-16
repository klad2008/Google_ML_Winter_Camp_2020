import torch
import os
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
# from utils impoirt transformer as tf
# from utils import transformer_usl
# import pandas as pd
# import config as cf
import cv2
from matplotlib import pyplot as plt

data_dir = '/home/charlesxujl/data/test/masks/'
new_dir = '/home/charlesxujl/data/test/labels/'

for file in os.listdir(data_dir):
    # print(file)
    file_dir = os.path.join(data_dir, file)
    img = cv2.imread(file_dir, cv2.IMREAD_UNCHANGED)
    alpha = img[:,:,3]
    # print(alpha)
    # break
    alpha[alpha > 0] = 1
    print(alpha)
    cv2.imwrite(os.path.join(new_dir,file), alpha)
