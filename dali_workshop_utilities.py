# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
from IPython import get_ipython
from ignite.engine import Engine, _prepare_batch
from ignite.engine import Events
from ignite.contrib.handlers import ProgressBar
from torch.cuda import amp

import tqdm

from ignite.metrics import Accuracy, Loss, RunningAverage

import torch.nn as nn

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    
def create_custom_supervised_trainer(model, optimizer, loss_fn, metrics={}, device=None , prepare_batch=None, scale_loss=False):
    """
    We need to make some changes to the default trainer so we can use running metrics and consume Tensors from DALI
    """


    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        if not prepare_batch:
            x, y = _prepare_batch(batch, device=device)
        else:
            x, y = prepare_batch(batch, device=device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        
        if scale_loss:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
            
        optimizer.step()
        return loss.item(), y_pred, y

    def _metrics_transform(output):
        return output[1], output[2]

    engine = Engine(_update)

    for name, metric in metrics.items():
        metric._output_transform = _metrics_transform
        metric.attach(engine, name)

    return engine

def add_progress_bar(trainer, evaluator, validation_loader, epoch_length):
    """
    "I can't believe it's not Keras"
    Running average accuracy and loss metrics + TQDM progressbar
    """
    training_history = {'accuracy':[],'loss':[]}
    validation_history = {'accuracy':[],'loss':[]}
    last_epoch = []

    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')
    RunningAverage(Accuracy(output_transform=lambda x: (x[1], x[2]))).attach(trainer, 'accuracy')
    
    prog_bar = ProgressBar()
    prog_bar.attach(trainer, ['loss', 'accuracy'])
    prog_bar.pbar_cls=tqdm.tqdm

    prog_bar_vd = ProgressBar()
    prog_bar_vd.attach(evaluator)
    prog_bar_vd.pbar_cls=tqdm.tqdm
    
    from ignite.handlers import Timer

    timer = Timer(average=True)
    timer.attach(trainer,start=Events.EPOCH_STARTED,
            resume=Events.EPOCH_STARTED,
            pause=Events.EPOCH_COMPLETED,
            step=Events.EPOCH_COMPLETED)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        metrics = trainer.state.metrics
        accuracy = metrics['accuracy']*100
        loss = metrics['nll']
        last_epoch.append(0)
        training_history['accuracy'].append(accuracy)
        training_history['loss'].append(loss)
        train_msg = "Train Epoch {}:  acc: {:.2f}% loss: {:.2f}, train time: {:.2f}s".format(trainer.state.epoch, accuracy, loss, timer.value())
            
        evaluator.run(validation_loader, epoch_length=epoch_length)
        metrics = evaluator.state.metrics
        accuracy = metrics['accuracy']*100
        loss = metrics['nll']
        validation_history['accuracy'].append(accuracy)
        validation_history['loss'].append(loss)
        val_msg = "Valid Epoch {}:  acc: {:.2f}% loss: {:.2f}".format(trainer.state.epoch, accuracy, loss)
        
        prog_bar_vd.log_message(train_msg+" --- "+val_msg)

        
def add_progress_bar_eval(evaluator, validation_loader):
    """
    "I can't believe it's not Keras"
    Running average accuracy and loss metrics + TQDM progressbar
    """
    validation_history = {'accuracy':[],'loss':[]}
    last_epoch = []

    RunningAverage(output_transform=lambda x: x[0]).attach(evaluator, 'loss')    
    RunningAverage(Accuracy(output_transform=lambda x: (x[0], x[1]))).attach(evaluator, 'accuracy')
    
    prog_bar = ProgressBar()
    prog_bar.attach(evaluator, ['accuracy'])
#     prog_bar.pbar_cls=tqdm.tqdm
    
    from ignite.handlers import Timer

    timer = Timer(average=True)
    timer.attach(evaluator,start=Events.EPOCH_STARTED,
            resume=Events.EPOCH_STARTED,
            pause=Events.EPOCH_COMPLETED,
            step=Events.EPOCH_COMPLETED)

    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_validation_results(evaluator):
        metrics = evaluator.state.metrics
        accuracy = metrics['accuracy']*100
        loss = metrics['nll']
        validation_history['accuracy'].append(accuracy)
        validation_history['loss'].append(loss)
        val_msg = "Valid Epoch {}:  acc: {:.2f}% loss: {:.2f}, eval time: {:.2f}s".format(evaluator.state.epoch, accuracy, loss, timer.value())
        prog_bar.log_message(val_msg)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)