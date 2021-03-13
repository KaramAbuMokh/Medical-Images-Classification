import os

import pandas as pd
from PIL import Image

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
import torch.nn as nn
import torch.nn.functional as F
from ignite.metrics import Accuracy, Loss
import torch.optim as optim


import torch
if torch.cuda.is_available():  # Make sure GPU is available
    dev = torch.device("cuda:0")
    kwar = {'num_workers': 8, 'pin_memory': True}
    cpu = torch.device("cpu")
else:
    print("Warning: CUDA not found, CPU only.")
    dev = torch.device("cpu")
    kwar = {}
    cpu = torch.device("cpu")


USE_AMP = True
USE_PARALLEL = True
OPT_LEVEL="O2"
#GPUS = ["cuda:0", "cuda:1"]
GPUS = ["cuda:0"]


DATASET_NAME = 'data/resized/'
MAX_IMAGES = 256
BATCH_SIZE = 64

EPOCH_LENGTH = MAX_IMAGES//BATCH_SIZE * 15

WIDTH,HEIGHT=Image.open('data/resized/AbdomenCT/000000.jpeg').size
NUM_CHANNELS = 1
NUM_CLASSES = 6


def get_labels():
    human_label_dir = 'idx_human.csv'
    human_readable = pd.read_csv(human_label_dir, header=None, usecols=[1])
    str(human_readable.loc[0])
    print(human_readable)
    return human_readable

human_readable=get_labels()


def get_pipeline(folder="train", custom_reader=None):
    pipe = Pipeline(batch_size=64, num_threads=1, device_id=0)

    if custom_reader:
        raw_files, labels = custom_reader
    else:
        raw_files, labels = fn.file_reader(file_root="%s/yt_bb_classification_%s" % (DATASET_NAME, folder),
                                           random_shuffle=True)

    decode = fn.image_decoder(raw_files, device=dev, output_type=types.RGB)
    resize = fn.resize(decode, device=dev, image_type=types.RGB,
                       interp_type=types.INTERP_LINEAR, resize_x=WIDTH, resize_y=HEIGHT)

    hsv = fn.hsv(resize, hue=fn.uniform(range=(-10, 10)), saturation=fn.uniform(range=(-.5, .5)),
                 value=fn.uniform(range=(0.9, 1.2)), device=dev, dtype=types.UINT8)
    bc = fn.brightness_contrast(hsv, device=dev, brightness=fn.uniform(range=(.9, 1.1)))

    cmn = fn.crop_mirror_normalize(bc, device=dev, output_dtype=types.FLOAT,
                                   output_layout=types.NHWC,
                                   image_type=types.RGB,
                                   mean=[255 // 2, 255 // 2, 255 // 2],
                                   std=[255 // 2, 255 // 2, 255 // 2])

    rot = fn.rotate(cmn, angle=fn.uniform(range=(-40, 40)), device=dev, keep_size=True)

    tpose = fn.transpose(rot, perm=(2, 0, 1), device=dev)  # Reshaping to a format PyTorch likes

    pipe.set_outputs(tpose, labels)
    pipe.build()

    dali_iter = DALIClassificationIterator([pipe], -1)

    return dali_iter



class MedNet(nn.Module):
    def __init__(self, xDim, yDim, numC):  # Pass image dimensions and number of labels when initializing a model
        super(MedNet, self).__init__()  # Extends the basic nn.Module to the MedNet class
        # The parameters here define the architecture of the convolutional portion of the CNN. Each image pixel
        # has numConvs convolutions applied to it, and convSize is the number of surrounding pixels included
        # in each convolution. Lastly, the numNodesToFC formula calculates the final, remaining nodes at the last
        # level of convolutions so that this can be "flattened" and fed into the fully connected layers subsequently.
        # Each convolution makes the image a little smaller (convolutions do not, by default, "hang over" the edges
        # of the image), and this makes the effective image dimension decreases.

        numConvs1 = 5
        convSize1 = 7
        numConvs2 = 10
        convSize2 = 7
        numNodesToFC = numConvs2 * (xDim - (convSize1 - 1) - (convSize2 - 1)) * (
                    yDim - (convSize1 - 1) - (convSize2 - 1))

        # nn.Conv2d(channels in, channels out, convolution height/width)
        # 1 channel -- grayscale -- feeds into the first convolution. The same number output from one layer must be
        # fed into the next. These variables actually store the weights between layers for the model.

        self.cnv1 = nn.Conv2d(1, numConvs1, convSize1)
        self.cnv2 = nn.Conv2d(numConvs1, numConvs2, convSize2)

        # These parameters define the number of output nodes of each fully connected layer.
        # Each layer must output the same number of nodes as the next layer begins with.
        # The final layer must have output nodes equal to the number of labels used.

        fcSize1 = 400
        fcSize2 = 80

        # nn.Linear(nodes in, nodes out)
        # Stores the weights between the fully connected layers

        self.ful1 = nn.Linear(numNodesToFC, fcSize1)
        self.ful2 = nn.Linear(fcSize1, fcSize2)
        self.ful3 = nn.Linear(fcSize2, numC)

    def forward(self, x):
        # This defines the steps used in the computation of output from input.
        # It makes uses of the weights defined in the __init__ method.
        # Each assignment of x here is the result of feeding the input up through one layer.
        # Here we use the activation function elu, which is a smoother version of the popular relu function.

        x = F.elu(self.cnv1(x))  # Feed through first convolutional layer, then apply activation
        x = F.elu(self.cnv2(x))  # Feed through second convolutional layer, apply activation
        x = x.view(-1, self.num_flat_features(x))  # Flatten convolutional layer into fully connected layer
        x = F.elu(self.ful1(x))  # Feed through first fully connected layer, apply activation
        x = F.elu(self.ful2(x))  # Feed through second FC layer, apply output
        x = self.ful3(x)  # Final FC layer to output. No activation, because it's used to calculate loss
        return x

    def num_flat_features(self, x):  # Count the individual nodes in a layer
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



model = MedNet(WIDTH,HEIGHT,NUM_CLASSES)





if USE_PARALLEL:
    model = torch.nn.DataParallel(model.cuda(device = dev), device_ids=[0])



from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from dali_workshop_utilities import create_custom_supervised_trainer, add_progress_bar

# -----------------------------------------------------------------
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform_list = [transforms.Resize((WIDTH,HEIGHT)),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5,), (0.5,))]

transform = transforms.Compose(transform_list)

valid_set = datasets.ImageFolder("valid-data", transform=transform, target_transform=None, loader=Image.open)
valid_loader_folder = DataLoader(valid_set, batch_size=64, shuffle=True, drop_last=True)


# ---------------------------------------------------

def prepare_batch(batch, device, non_blocking=True):
    x = batch[0]["data"].to(device)
    y = batch[0]["label"]
    y = y.squeeze().long().to(device)
    return x, y

def get_metrics(loss):
    return {'accuracy': Accuracy(), 'nll': Loss(loss)}



def get_trainer(model, progress_bar=False):
    opt = optim.Adam(model.parameters(), lr=0.0001)
    loss = nn.NLLLoss()

    trainer = create_custom_supervised_trainer(model, opt, loss, metrics=get_metrics(loss), prepare_batch=prepare_batch,
                                               device=dev)
    evaluator = create_supervised_evaluator(model, metrics=get_metrics(loss), device=dev)

    if progress_bar:
        add_progress_bar(trainer, evaluator, valid_loader_folder, epoch_length=EPOCH_LENGTH)

    return trainer, opt, loss









from apex import amp
_, opt, model_loss = get_trainer(model)

if USE_AMP:
    if USE_PARALLEL:
        opt_level = "O1"
    else:
        opt_level = OPT_LEVEL

    model, opt = amp.initialize(model.cuda(device = dev), opt, opt_level=opt_level)
    model.to("cuda:0")


dali_iter = get_pipeline(folder="train")
dali_iter_valid = get_pipeline(folder="validation")




trainer = create_custom_supervised_trainer(model, opt, model_loss,
                                           metrics=get_metrics(model_loss),
                                           prepare_batch=prepare_batch,
                                           device=dev, scale_loss=USE_AMP)

evaluator = create_supervised_evaluator(model,
                                        prepare_batch=prepare_batch,
                                        metrics=get_metrics(model_loss),
                                        device=dev)

add_progress_bar(trainer,
                 evaluator,
                 dali_iter_valid,
                 epoch_length=EPOCH_LENGTH)




trainer.run(dali_iter,
            max_epochs=5,
            epoch_length=EPOCH_LENGTH)


torch.save(model.state_dict(), 'model')













'''

"cuda"

def prepare_batch(batch, device, non_blocking=True):
    return batch[0].cuda(), batch[1].cuda()








trainer = create_custom_supervised_trainer(model, opt, model_loss, metrics=get_metrics(model_loss),
                                           prepare_batch=prepare_batch, device="cuda:0", scale_loss=USE_AMP)

evaluator = create_supervised_evaluator(model, prepare_batch=prepare_batch, metrics=get_metrics(model_loss),
                                        device="cuda")

add_progress_bar(trainer, evaluator, dali_iter_valid, epoch_length=EPOCH_LENGTH)





trainer, _, _ = get_trainer(model, True)

trainer.run(train_loader_folder, max_epochs=30, epoch_length=EPOCH_LENGTH)
loss = nn.NLLLoss()
evaluator = create_supervised_evaluator(model.to("cpu"), metrics=get_metrics(loss), device="cpu")
add_progress_bar_eval(evaluator, valid_loader_folder)
evaluator.run(valid_loader_folder)'''