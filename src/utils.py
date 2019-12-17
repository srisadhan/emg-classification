import sys
from contextlib import contextmanager

import torch
import deepdish as dd
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd 

class SkipWith(Exception):
    pass


@contextmanager
def skip_run(flag, f):
    """To skip a block of code.

    Parameters
    ----------
    flag : str
        skip or run.

    Returns
    -------
    None

    """
    @contextmanager
    def check_active():
        deactivated = ['skip']
        p = ColorPrint()  # printing options
        if flag in deactivated:
            p.print_skip('{:>12}  {:>2}  {:>12}'.format(
                'Skipping the block', '|', f))
            raise SkipWith()
        else:
            p.print_run('{:>12}  {:>3}  {:>12}'.format('Running the block',
                                                       '|', f))
            yield

    try:
        yield check_active
    except SkipWith:
        pass


class ColorPrint:
    @staticmethod
    def print_skip(message, end='\n'):
        sys.stderr.write('\x1b[88m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_run(message, end='\n'):
        sys.stdout.write('\x1b[1;32m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_warn(message, end='\n'):
        sys.stderr.write('\x1b[1;33m' + message.strip() + '\x1b[0m' + end)


# For saving data
def save_data(path, dataset, save):
    """save the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataset : dataset
        pytorch dataset.
    save : Bool

    """
    if save:
        dd.io.save(path, dataset)

    return None


def save_trained_pytorch_model(trained_model,
                               trained_model_info,
                               save_path,
                               save_model=True):
    """Save pytorch model and info.

    Parameters
    ----------
    trained_model : pytorch model
    trained_model_info : dict
    save_path : str

    """

    if save_model:
        time_stamp = datetime.now().strftime("%Y_%b_%d_%H_%M_%S")
        torch.save(trained_model, save_path + '/model_' + time_stamp + '.pth')
        torch.save(trained_model_info,
                   save_path + '/model_info_' + time_stamp + '.pth')
        # Save time also
        with open(save_path + '/time.txt', "a") as f:
            f.write(time_stamp + '\n')

    return None


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap="YlGnBu"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    sns.set()

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    ax = sns.heatmap(cm, fmt = 'd' , cmap=cmap, cbar = False,  annot=True)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')

    plt.xticks(0.5 + np.arange(len(classes)), classes)
    plt.yticks(0.5 + np.arange(len(classes)), classes)

    plt.tight_layout()
    return ax
