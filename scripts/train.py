import sys
sys.path.append('../')
from modules.model import FCN
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch
import argparse
from modules.data_generator import ListDataset 

parser = argparse.ArgumentParser(description='FCN coding by yamad')

parser.add_argument('--use_cuda', default=False,help='gpu or cpu')
args = parser.parse_args()
batch_size=2
transform = transforms.Compose([transforms.Resize((512, 768)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
datasets = ListDataset('../data/images/', batch_size, '../data/labels/sample.jpg',  train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=True)


model = FCN()
if args.use_cuda:
    model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader, 0):
        if args.use_cuda:
            images = images.cuda()
            labels = labels.cuda()

        images = Variable(images)
        labels = Variable(labels)
        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, labels)
        print(loss.data[0])
        loss.backward()
start_epoch = 0
for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
