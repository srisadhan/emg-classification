import torch
import torch.nn as nn
from torchnet.logger import VisdomPlotLogger
from torch.nn.init import xavier_normal_


def weights_init(model):
    """Xavier normal weight initialization for the given model.

    Parameters
    ----------
    model : pytorch model for random weight initialization
    Returns
    -------
    pytorch model with xavier normal initialized weights

    """
    if isinstance(model, nn.Conv2d):
        xavier_normal_(model.weight.data)


def calculate_accuracy(model, data_iterator, key):
    """Calculate the classification accuracy.

    Parameters
    ----------
    model : pytorch object
        A pytorch model.
    data_iterator : pytorch object
        A pytorch dataset.
    key : str
        A key to select which dataset to evaluate

    Returns
    -------
    float
        accuracy of classification for the given key.

    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        total = 0
        length = 0
        for x, y in data_iterator[key]:
            model.eval()
            out_put = model(x.to(device))
            out_put = out_put.cpu().detach()
            total += (out_put.argmax(dim=1) == y.argmax(dim=1)).float().sum()
            length += len(y)
        accuracy = total / length

    return accuracy.numpy()


def classification_accuracy(model, data_iterator):
    """Calculate the classification accuracy of all data_iterators.

    Parameters
    ----------
    model : pytorch object
        A pytorch model.
    data_iterator : dict
        A dictionary with different datasets.

    Returns
    -------
    list
        A dictionary of accuracy for all datasets.

    """
    accuracy = []
    keys = data_iterator.keys()
    for key in keys:
        accuracy.append(calculate_accuracy(model, data_iterator, key))

    return accuracy


def visual_log(title):
    """Return a pytorch tnt visual loggger.

    Parameters
    ----------
    title : str
        A title to describe the logging.

    Returns
    -------
    type
        pytorch visual logger.

    """
    visual_logger = VisdomPlotLogger(
        'line',
        opts=dict(legend=['Training', 'Validation', 'Testing'],
                  xlabel='Epochs',
                  ylabel='Accuracy',
                  title=title))

    return visual_logger


def create_model_info(config, loss_func, accuracy):
    """Create a dictionary of relevant model info.

    Parameters
    ----------
    param : dict
        Any parameter relevant for logging.
    accuracy_log : dict
        A dictionary containing accuracies.

    Returns
    -------
    type
        Description of returned object.

    """
    model_info = {
        'training_accuracy': accuracy[:, 0],
        'validation_accuracy': accuracy[:, 1],
        'testing_accuracy': accuracy[:, 2],
        'model_parameters': config,
        'loss function': loss_func
    }

    return model_info
