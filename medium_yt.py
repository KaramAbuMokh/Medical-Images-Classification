import os

USE_AMP = True
USE_PARALLEL = True
OPT_LEVEL="O2"
#GPUS = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
# GPUS = ["cuda:0", "cuda:1"]
GPUS = ["cuda:0"]


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




from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import matplotlib.pylab as plt
import nvidia.dali.fn as fn

import nvidia.dali.types as types

pipe = Pipeline(batch_size = 64, num_threads = 1, device_id = 0)

raw_files, labels = fn.file_reader(file_root = "data/resized", random_shuffle = True)

decode = fn.image_decoder(raw_files, device = "mixed", output_type = types.GRAY)
resize = fn.resize(decode, device = "gpu", image_type = types.GRAY,
                                interp_type = types.INTERP_LINEAR, resize_x=WIDTH, resize_y=HEIGHT)
cmn = fn.crop_mirror_normalize(resize, device="gpu",output_dtype=types.FLOAT,
                                                          output_layout=types.NCHW,
                                                        image_type=types.GRAY,
                                                        mean=[ 255//2],
                                                        std=[255//2])

pipe.set_outputs(cmn, labels)
pipe.build()



from nvidia.dali.plugin.pytorch import DALIClassificationIterator
dali_iter = DALIClassificationIterator([pipe], -1)

output = next(dali_iter)[0]
output["data"].shape, output["label"].shape



import numpy as np
import matplotlib.pylab as plt

def DALI_label_to_dataset(label):
    return int(sorted([str(s) for s in range(24)])[label])

def batch_to_image(batch):
    return np.array(batch[0]["data"].cpu()).transpose((0,2,3,1))[0], DALI_label_to_dataset(np.array(batch[0]["label"][0])[0])

batch = next(dali_iter)
img, label = batch_to_image(batch)




from time import time

def time_pipe(dali_iter):
    start = time()

    for i in range(10):
        sample = next(dali_iter)[0]
        print("Loaded:", type(sample["data"]), sample["data"].shape, type(sample["label"]), sample["label"].shape)

    end = time()

    print("Time Elapsed:", end - start, "seconds")

time_pipe(dali_iter)


def get_pipeline(folder="train", custom_reader=None):
    pipe = Pipeline(batch_size=64, num_threads=1, device_id=1)

    if custom_reader:
        raw_files, labels = custom_reader
    else:
        raw_files, labels = fn.file_reader(file_root="%s" % folder,
                                           random_shuffle=True)

    decode = fn.image_decoder(raw_files, device="mixed", output_type=types.GRAY)
    resize = fn.resize(decode, device="gpu", image_type=types.RGB,
                       interp_type=types.INTERP_LINEAR, resize_x=WIDTH, resize_y=HEIGHT)

    hsv = fn.hsv(resize, hue=fn.uniform(range=(-10, 10)), saturation=fn.uniform(range=(-.5, .5)),
                 value=fn.uniform(range=(0.9, 1.2)), device="gpu", dtype=types.UINT8)
    bc = fn.brightness_contrast(hsv, device="gpu", brightness=fn.uniform(range=(.9, 1.1)))

    cmn = fn.crop_mirror_normalize(bc, device="gpu", output_dtype=types.FLOAT,
                                   output_layout=types.NHWC,
                                   image_type=types.GRAY,
                                   mean=[255 // 2],
                                   std=[255 // 2])

    rot = fn.rotate(cmn, angle=fn.uniform(range=(-40, 40)), device="gpu", keep_size=True)

    tpose = fn.transpose(rot, perm=(2, 0, 1), device="gpu")  # Reshaping to a format PyTorch likes

    pipe.set_outputs(tpose, labels)
    pipe.build()

    dali_iter = DALIClassificationIterator([pipe], -1)

    return dali_iter


dali_iter = get_pipeline()

# Different rotation every run
for i in range(5):
    batch = next(dali_iter)

    img, label = batch_to_image(batch)

    plt.imshow(img)
    plt.title("Class Label: %s %s" % (label, human_readable.loc[label]))
    plt.show()




time_pipe(dali_iter)


from basic_yt import get_model, get_trainer, get_metrics


model = get_model()

model

from apex import amp

_, opt, model_loss = get_trainer(model)

if USE_AMP:
    if USE_PARALLEL:
        opt_level = "O1"
    else:
        opt_level = OPT_LEVEL

    model, opt = amp.initialize(model, opt, opt_level=opt_level)
    model.to("cuda:0")









import torch

if USE_PARALLEL:
    model = torch.nn.DataParallel(model, device_ids=GPUS)



dali_iter = get_pipeline(folder="data/resized")
dali_iter_valid = get_pipeline(folder="valid-data")

from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from dali_workshop_utilities import create_custom_supervised_trainer, add_progress_bar


def prepare_batch(batch, device, non_blocking=True):
    x = batch[0]["data"].to(device)
    y = batch[0]["label"]
    y = y.squeeze().long().to(device)
    return x, y


trainer = create_custom_supervised_trainer(model, opt, model_loss, metrics=get_metrics(model_loss),
                                           prepare_batch=prepare_batch, device="cuda:0", scale_loss=USE_AMP)

evaluator = create_supervised_evaluator(model, prepare_batch=prepare_batch, metrics=get_metrics(model_loss),
                                        device="cuda")

add_progress_bar(trainer, evaluator, dali_iter_valid, epoch_length=EPOCH_LENGTH)



trainer.run(dali_iter, max_epochs=5, epoch_length=EPOCH_LENGTH)