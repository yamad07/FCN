import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import glob
import pandas as pd

class ListDataset(data.Dataset):
    def __init__(self, image_root, batch_size, label_file, label_root, train, transform):
        '''
        Args:
            image_root(String) : 実際に使用する画像
            label_root(String) : 白と黒で構成される画像
            train(Boolean) : トレーニングかどうか
            transform(Transform.Compose) : データ前処理のCompose
        '''

        self.image_root = image_root
        self.batch_size = batch_size
        self.label_root = label_root
        self.train = train
        self.transform = transform
        self.label_file = pd.read_csv(label_file) 
        self.num_samples = self.label_file.shape[0]

    def __getitem__(self, idx):
        image_label_set = self.label_file.iloc[idx, :]
        # image
        
        img = Image.open(self.image_root + str(image_label_set.file_name) + '.jpg')
        img = self.transform(img)
         
        # label
        label = Image.open(self.label_root + str(image_label_set.label_name) + '.jpg')
        label = label.convert('L')
        label = label.resize((768, 512))
        label = np.asarray(label)
        label.flags.writeable = True
        label[label == 255] = 1
        label[label != 255] = 0
        label = torch.LongTensor(label)
        return img, label

    def __len__(self):
        return self.num_samples
