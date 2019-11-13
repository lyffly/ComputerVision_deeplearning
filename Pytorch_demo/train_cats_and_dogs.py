# coding by liuyf
# 
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision.datasets import ImageFolder
from torchvision import models
from torchvision import transforms
import time

def imshow(inp,cmap=None):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    inp = std*inp + mean
    inp = np.clip(inp,0,1)
    plt.imshow(inp,cmap)


is_cuda = False
if torch.cuda.is_available():
    is_cuda = True

simple_transforms = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                    ])

train = ImageFolder("train/",simple_transforms)
print(train.class_to_idx)
print(train.classes)

#imshow(train[1000][0])
#plt.show()

train_data_loader = torch.utils.data.DataLoader(train,batch_size=32,num_workers=2,shuffle=True)



"""
cats_names = glob.glob("train/cat.*.jpg")
print("there {} cats".format(len(cats_names)))

dogs_names = glob.glob("train/dog.*.jpg")
print("there {} dogs".format(len(dogs_names)))

class DogsAndCatsDataset(Dataset):
    def __init__(self,root_dir,size=(224,224)):
        self.files = glob.glob(root_dir)
        self.size = size
    def __len__(self):
        return len(self.files)
    def __getitem__(self,idx):
        img = np.asarray(Image.open(self.files[idx]).resize(self.size))
        label = self.files[idx].split(".")[-3].split("/")[-1]
        return img,label

dogset = DogsAndCatsDataset("train/dog*.jpg")
catset = DogsAndCatsDataset("train/cat*.jpg")

dogs_loader = DataLoader(dogset,batch_size=32,num_workers=2)
cats_loader = DataLoader(catset,batch_size=32,num_workers=2)
"""

vgg = models.vgg16(pretrained = True)

for param in vgg.features.parameters():
    param.requires_grad = False

vgg.classifier[6].out_features = 2
model = vgg
print(model)

if is_cuda:
    model.cuda()

optimizer = optim.SGD(model.classifier.parameters(),lr=0.001,momentum=0.5)


def fit(epoch,model,data_loader,phase='train',volatile=False):
    if phase =="train":
        model.train()
    
    running_loss = 0.0
    running_correct = 0

    for batch_idx,(data,target) in enumerate(data_loader):
        if is_cuda:
            data = data.cuda()
            target = target.cuda()
        data = Variable(data,volatile)
        target = Variable(target)
        if phase == "train":
            optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output,target)
        
        if batch_idx % 10 == 1:
            print(F.cross_entropy(output,target,reduction='sum').item())

        running_loss += F.cross_entropy(output,target,reduction='sum').item()
        preds = output.data.max(dim = 1,keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()

        if phase =="train":
            loss.backward()
            optimizer.step()
    
    loss = running_loss/len(data_loader.dataset)
    acc = 100.0 *(running_correct/len(data_loader.dataset))

    print(f'{phase} loss is {loss} and {phase} accuracy is {acc}')
    
    return loss,acc

        


train_losses =[]
train_acc = []
for epoch in range(1,20):
    print("epoch",epoch)
    epoch_loss,epoch_acc = fit(epoch,model,train_data_loader,phase="train")
    train_losses.append(epoch_loss)
    train_acc.append(epoch_acc)  
    


plt.plot(range(1,len(train_losses)+1),train_losses,'bo',label = 'training loss')
plt.show()



