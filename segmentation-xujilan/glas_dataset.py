import torch
import os
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import transformer as tf
# from utils import transformer_usl
import pandas as pd
import config as cf
import cv2
from matplotlib import pyplot as plt

data_dir = '/home/charlesxujl/data/train/'


class dataset(Dataset):
    def __init__(self, root_dir, mask_dir, mode='train'):
        super(dataset, self).__init__()
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.mode = mode
        self.imgs, self.masks = self.get_image()
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])
        self.crop_size = None
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        #pass
        img_dir = self.imgs[index]
        img_name = img_dir.split('/')[-1].split('.')[0]
        if self.mode == 'inference':
            img = cv2.imread(img_dir)
            w, h = img.shape[:2]
            new_h  = h // 32 * 32
            new_w = w * new_h // h
            img = cv2.resize(img, (new_h, new_w), cv2.INTER_LINEAR)

            img = self.transforms(img).float()
            return img, img_name
        
        mask_dir = self.masks[index]
        # img = Image.open(img_dir)#.convert('RGB')
        img = cv2.imread(img_dir)
        mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
        # cv2.imwrite('debug_raw.png',img)
        # print(img.shape, mask.shape)
        # mask = Image.open(mask_dir).convert('L')
        #print(img_name)

        
        img, mask = self.augument_image(img, mask)
        #cv2.imwrite('debug_img.png', img)
        #cv2.imwrite('debug_mask.png', mask*255)

        img = self.transforms(img).float()
        mask = torch.LongTensor(np.array(mask))
        mask[mask > 0] = 1
        return img, mask

    def get_image(self):
        #pass
        imgs = [os.path.join(self.root_dir, x) for x in os.listdir(self.root_dir) if x[0] != '.']
        masks = [os.path.join(self.mask_dir, x) for x in os.listdir(self.mask_dir) if x[0] != '.']
        imgs.sort()
        masks.sort()
        # print(imgs[:5])
        # print(masks[:5])
        return imgs, masks

    def augument_image(self, img, mask):
        # w, h = img.size
        w, h = img.shape[:2]

        new_h  = h // 32 * 32
        new_w = w * new_h // h
        
        img = cv2.resize(img, (new_h, new_w), cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_h, new_w), cv2.INTER_NEAREST)
        new_w, new_h = img.shape[:2]
        
        # img = img.resize((new_w, new_h), Image.BILINEAR)
        # mask = mask.resize((new_w, new_h), Image.NEAREST)
        # new_w, new_h = img.size
        # print(new_w, new_h)
        # random_crop = tf.RandomCrop(self.crop_size)
        random_scale = tf.RandScale((0.8, 1.25))
        random_crop = tf.Crop((new_w, new_h), padding=[0, 0, 0])
        random_flip = tf.RandomHorizontalFlip(0.5)
        # random_resizedcrop = tf.RandomResizedCrop(self.crop_size, ratio=(1, 1))

        transformers = tf.Compose([
            random_scale,
            random_crop,
            random_flip,
            # random_rotate,
            # random_elastic
        ])
        if self.mode=='train':
            img, mask = transformers(img, mask)
        # print(img.shape, mask.shape, contour.shape)
        return img, mask

if __name__ == '__main__':
    root_dir = '/home/charlesxujl/data/test/imgs/'
    mask_dir = '/home/charlesxujl/data/test/labels/'
    dst = dataset(root_dir, mask_dir, mode='test')
    imgs, masks = dst.__getitem__(2)
    print(imgs.shape)
    print(masks.shape)


