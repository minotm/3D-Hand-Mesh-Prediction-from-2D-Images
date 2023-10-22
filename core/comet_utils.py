from comet_ml import Experiment, ExistingExperiment
from pprint import pprint
import socket
import core.sys_utils as sys_utils
from tqdm import tqdm
import torch
import numpy as np
import torch
import os.path as op
import time
from pytorch_lightning.loggers.comet import CometLogger


def log_dict(experiment, metric_dict, step, postfix=None):
    if experiment is None:
        return
    for key, value in metric_dict.items():
        if postfix is not None:
            key = key + postfix
        if isinstance(value, torch.Tensor) and len(value.view(-1)) == 1:
            value = value.item()

        if isinstance(value, (int, float, np.float32)):
            experiment.log_metric(key, value, step=step)


def fetch_key_from_experiment(experiment):
    if experiment is not None:
        key = str(experiment.get_key())
        key = key[:9]
        experiment.set_name(key)
    else:
        import random

        hash = random.getrandbits(128)
        key = "%032x" % (hash)
        key = key[:9]
    return key


def push_images(experiment, all_im_list, global_step=None, no_tqdm=False, verbose=True):
    if verbose:
        print("Pushing PIL images")
        tic = time.time()
    iterator = all_im_list if no_tqdm else tqdm(all_im_list)
    for im in iterator:
        im_np = np.array(im["im"])
        if "fig_name" in im.keys():
            experiment.log_image(im_np, im["fig_name"], step=global_step)
        else:
            experiment.log_image(im_np, "unnamed", step=global_step)
    if verbose:
        toc = time.time()
        print("Done pushing PIL images (%.1fs)" % (toc - tic))
