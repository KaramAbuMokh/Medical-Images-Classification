import os

DATASET_NAME = 'classification_cc_vids'
MAX_IMAGES = 128
BATCH_SIZE = 32

EPOCH_LENGTH = MAX_IMAGES//BATCH_SIZE * 15

WIDTH = 64
HEIGHT = 64
NUM_CHANNELS = 1
NUM_CLASSES = 6


import pandas as pd

lst=os.listdir('data/resized')

human_readable =  pd.DataFrame(lst)

str(human_readable.loc[0])



from matplotlib import pylab as plt
import matplotlib.image as mpimg

img = mpimg.imread('data/resized/AbdomenCT/000000.jpeg')
plt.imshow(img)

print("Label:", human_readable.loc[0])



from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import random


transform_list = [transforms.Resize((WIDTH,HEIGHT)),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5,), (0.5,))]

transform = transforms.Compose(transform_list)

train_set = datasets.ImageFolder("data/resized" , transform=transform, target_transform=None, loader=Image.open)
valid_set = datasets.ImageFolder("valid-data", transform=transform, target_transform=None, loader=Image.open)

train_loader_folder = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)
valid_loader_folder = DataLoader(valid_set, batch_size=64, shuffle=True, drop_last=True)



from time import time

sample_images = iter(train_loader_folder)

start = time()

for i in range(10):
    x = next(sample_images)
    print("Loaded:", type(x[0]), x[0].shape, type(x[1]), x[1].shape)

end = time()

print("Time Elapsed:", end-start, "seconds")




from time import time

sample_images = iter(train_loader_folder)

start = time()

for i in range(10):
    x = next(sample_images)
    print("Loaded:", type(x[0]), x[0].shape, type(x[1]), x[1].shape)

end = time()

print("Time Elapsed:", end-start, "seconds")







import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np

from dali_workshop_utilities import Reshape

def get_flat_size(in_size, fts):
    f = fts(Variable(torch.ones(1,*in_size)))
    return int(np.prod(f.size()[1:]))

def add_conv_layer(layers, size_in=64, size_out=64, maxpool=True):
    layers.append(nn.Conv2d(size_in,size_out,3))
    layers.append(nn.ReLU())
    layers.append(nn.BatchNorm2d(size_out))
    if maxpool:
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

def get_model():
    layers = []

    add_conv_layer(layers, size_in=NUM_CHANNELS)
    add_conv_layer(layers)
    add_conv_layer(layers)

    size = get_flat_size((NUM_CHANNELS,WIDTH,HEIGHT), nn.Sequential(*layers))
    layers.append(Reshape(-1, size))

    layers.append(nn.Linear(size,32))
    layers.append(nn.Dropout(.6))
    layers.append(nn.Linear(32,NUM_CLASSES))
    layers.append(nn.Softmax(dim=1))

    model = nn.Sequential(*layers)
    model.to("cuda")
    return model

model = get_model()




from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from dali_workshop_utilities import create_custom_supervised_trainer, add_progress_bar, add_progress_bar_eval

from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
import torch.optim as optim

def get_metrics(loss):
    return {'accuracy' : Accuracy(), 'nll' : Loss(loss)}

def prepare_batch(batch, device, non_blocking=True):
    return batch[0].cuda(), batch[1].cuda()

def get_trainer(model, progress_bar=False):
    opt = optim.Adam(model.parameters(), lr=0.0001)
    loss = nn.NLLLoss()

    trainer = create_custom_supervised_trainer(model, opt, loss, metrics=get_metrics(loss), prepare_batch=prepare_batch, device="cuda")
    evaluator = create_supervised_evaluator(model, metrics=get_metrics(loss), device="cuda")

    if progress_bar:
        add_progress_bar(trainer, evaluator, valid_loader_folder, epoch_length=EPOCH_LENGTH)

    return trainer, opt, loss

trainer, _, _ = get_trainer(model, True)





trainer.run(train_loader_folder, max_epochs=5, epoch_length=EPOCH_LENGTH)





loss = nn.NLLLoss()
evaluator = create_supervised_evaluator(model.to("cpu"), metrics=get_metrics(loss), device="cpu")
add_progress_bar_eval(evaluator, valid_loader_folder)
evaluator.run(valid_loader_folder)