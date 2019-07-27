import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import (get_model_path, figure_asthetics, annotate_significance)


def plot_average_model_accuracy(experiment, config, variation=False):
    """Plots the average accuracy of the pytorch model prediction.

    Parameters
    ----------
    config: yaml file
        Configuration file with all parameters
    variation : bool
        Plot variation (std) along with mean.

    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    fig, ax = plt.subplots()

    keys = ['training_accuracy', 'validation_accuracy', 'testing_accuracy']
    colors = ['#BC0019', '#2C69A9', '#40A43A']
    for i, key in enumerate(keys):
        accuracy = np.empty((0, config['NUM_EPOCHS']))
        for j in range(5):
            model_path, model_info_path = get_model_path(experiment, j)
            model_info = torch.load(model_info_path, map_location=device)
            accuracy = np.vstack((model_info[key], accuracy))
        # Calculate the average
        average = np.mean(accuracy, axis=0)
        print(average[-1])
        # Plot variation
        if variation:
            min_val = average - np.min(accuracy, axis=0)
            max_val = np.max(accuracy, axis=0) - average
            ax.fill_between(range(config['NUM_EPOCHS']),
                            average - min_val,
                            average + max_val,
                            alpha=0.25,
                            color=colors[i],
                            edgecolor=colors[i])
        ax.plot(range(config['NUM_EPOCHS']),
                average,
                color=colors[i],
                label='average' + ' ' + key.replace('_', ' '))

    ax.set_ylim(top=1.0)
    # Specifications
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    figure_asthetics(ax)
    plt.show()

    return None


def plot_model_accuracy(experiment, config, model_number):
    """Plot training, validation, and testing acurracy.

    Parameters
    ----------
    model_path : str
        A path to saved pytorch model.

    """

    model_path, model_info_path = get_model_path(experiment, model_number)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_info = torch.load(model_info_path, map_location=device)
    training_accuracy = model_info['training_accuracy']
    validation_accuracy = model_info['validation_accuracy']
    testing_accuracy = model_info['testing_accuracy']
    epochs = np.arange(training_accuracy.shape[0])

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(epochs, training_accuracy, color=[0.69, 0.18, 0.45, 1.00])
    ax.plot(epochs, validation_accuracy, color=[0.69, 0.69, 0.69, 1.00])
    ax.plot(epochs, testing_accuracy, color=[0.12, 0.27, 0.59, 1.00])
    ax.set_ylim(top=1.0)
    # Specifications
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    figure_asthetics(ax)
    plt.show()

    return None


def plot_bar(config, dataframe, independent, dependent):
    """Bar plot of the dataframe.

    Parameters
    ----------
    config : yaml
        The configuration file.
    dataframe : dataframe
        A pandas dataframe containing the dependent and independent data
        with group data (usually this is the subject).
    dependent : str
        A string stating which variable to use as dependent variable
    independent : str
        A string stating which variable to use as independent variable

    Returns
    -------
    None

    """

    # sns.set(font_scale=1.4)

    ax = sns.barplot(x=dependent,
                     y=independent,
                     hue='damping',
                     data=dataframe,
                     capsize=.1)

    # Add significance
    if independent == 'velocity':
        y = 0.14
        annotate_significance(0.8, 1.20, y, 0.005)
        ax.set_ylim([0, y + y * 0.1])

    else:
        # Within Fine motion
        y = 5
        annotate_significance(-0.2, 0.2, y, 0.005)
        ax.set_ylim([0, y + y * 0.1])

        # Within Gross motion
        y = 16
        annotate_significance(0.8, 1.20, y, 0.005)
        ax.set_ylim([0, y + y * 0.1])

    # Other figure information
    ax.set_ylabel(independent.replace('_', ' '))
    ax.set_xlabel('task type information')

    return None
