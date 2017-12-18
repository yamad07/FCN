import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

class FCN(nn.Module):

    def __init__(self):
        super(FCN, self).__init__()

        self.conv1_1  = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.max1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.max2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.max3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.max4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.max5 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True)

        self.deconv1_1 = nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.deconv1_2 = nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.deconv1_3 = nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.unpool1 = nn.MaxUnpool2d((2, 2), stride=2)

        self.deconv2_1 = nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.deconv2_2 = nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.deconv2_3 = nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        
        self.deconv3_1 = nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.deconv3_2 = nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.deconv3_3 = nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        
        self.deconv4_1 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.deconv4_2 = nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        
        self.deconv5_1 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.deconv5_2 = nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.unpool5 = nn.MaxUnpool2d(2, stride=2)

        self.deconv6_1 = nn.ConvTranspose2d(64, 10, kernel_size=(3, 3), stride=1, padding=1)
        
    def forward(self, h):
        h, indeces1 = self.max1(self.conv1_2(self.conv1_1(h)))
        h, indeces2 = self.max2(self.conv2_2(self.conv2_1(h)))
        h, indeces3 = self.max3(self.conv3_3(self.conv3_2(self.conv3_1(h))))
        h, indeces4 = self.max4(self.conv4_3(self.conv4_2(self.conv4_1(h))))
        h, indeces5 = self.max5(self.conv5_3(self.conv5_2(self.conv5_1(h))))
        out5_size = h.size()
        print(h.size())
        print(indeces5.size())
        print(h.size())
        h = self.deconv1_3(self.deconv1_2(self.deconv1_1(h)))
        print(h.size())
        h = self.unpool1(h, output_size=out5_size)
        h = self.unpool2(self.deconv2_3(self.deconv2_2(self.deconv2_1(h))), indeces4)
        h = self.unpool3(self.deconv3_3(self.deconv3_2(self.deconv3_1(h))), indeces3)
        h = self.unpool4(self.deconv4_2(self.deconv4_1(h)), indeces2)
        h = self.unpool5(self.deconv5_2(self.deconv5_1(h)), indeces1)
        h = nn.relu(self.deconv6_1(h))
        return h
