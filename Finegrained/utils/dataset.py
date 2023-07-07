import os
import json
from os.path import join

import numpy as np
import scipy
from scipy import io
import scipy.misc
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

class Serifs():
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform

        with open(os.path.join(self.root, 'images.txt')) as img_txt_file:
            img_name_list = []
            for line in img_txt_file:
                img_name_list.append(line.strip().split(' ')[-1])
        
        with open(os.path.join(self.root, 'image_class_labels.txt')) as label_txt_file:
            label_list = []
            for line in label_txt_file:
                label_list.append(int(line.strip().split(' ')[-1]) - 1)

        with open(os.path.join(self.root, 'train_test_split.txt')) as train_val_file:
            train_test_list = []
            for line in train_val_file:
                train_test_list.append(int(line.strip().split(' ')[-1]))

        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]

        if self.is_train:
            self.train_img = [plt.imread(os.path.join(self.root, train_file)) for train_file in
                              train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
            self.train_imgname = [x for x in train_file_list[:data_len]]

        if not self.is_train:
            self.test_img = [plt.imread(os.path.join(self.root, test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
            self.test_imgname = [x for x in test_file_list[:data_len]]

    def __getitem__(self, index):
        if self.is_train:
            img, target, imgname = self.train_img[index], self.train_label[index], self.train_imgname[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)
        else:
            img, target, imgname = self.test_img[index], self.test_label[index], self.test_imgname[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)