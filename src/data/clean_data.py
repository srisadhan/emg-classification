import collections
from pathlib import Path

import deepdish as dd
import mne
import numpy as np
import pandas as pd
import yaml


def one_hot_encode(label_length, category):
    """Generate one hot encoded value of required length and category.

    Parameters
    ----------
    label_length : int
        required lenght of the array.
    category : int
        Caterory e.g: category=2, [0, 1, 0] in 3 class system

    Returns
    -------
    array
        One hot encoded array.

    """
    y = np.zeros((label_length, len(category)))
    y[:, category.index(1)] = 1

    return y


def convert_to_array(subject, trial, config):
    """Converts the edf files in eeg and robot dataset into arrays.

    Parameter
    ----------
    subject : str
        String of subject ID e.g. 0001.
    trial : str
        Trail e.g. HighFine, LowGross.
    config : yaml
        The configuration file.

    Returns
    -------
    array
        An array of feature (x) and lables (y)

    """

    # Read path
    emg_path = str(Path(__file__).parents[2] / config['epoch_emg_data'])

    # Load the data
    data = dd.io.load(emg_path, group='/' + 'subject_' + subject)
    epochs = data['emg'][trial]

    # Get array data
    x_array = epochs.get_data()

    if trial == 'HighFine':
        category = [1, 0, 0]
    if trial == 'LowGross':
        category = [0, 1, 0]
    if (trial == 'HighGross') or (trial == 'LowFine'):
        category = [0, 0, 1]

    # In order to accomodate testing
    try:
        y_array = one_hot_encode(x_array.shape[0], category)
    except:
        y_array = np.zeros((x_array.shape[0], 3))

    return x_array, y_array


def clean_epoch_data(subjects, trials, config):
    """Create feature dataset for all subjects.

    Parameter
    ----------
    subject : str
        String of subject ID e.g. 0001.
    trials : list
        A list of differet trials

    Returns
    -------
    tensors
        All the data from subjects with labels.

    """
    # Initialize the numpy array to store all subject's data
    features_dataset = collections.defaultdict(dict)

    # Parameters
    epoch_length = config['epoch_length']
    sfreq = config['sfreq']

    for subject in subjects:
        # Initialise for each subject
        x_temp = np.empty((0, config['n_electrodes'], epoch_length * sfreq))
        y_temp = np.empty((0, config['n_class']))
        for trial in trials:
            # Concatenate the data corresponding to all trials types
            x_array, y_array = convert_to_array(subject, trial, config)
            x_temp = np.concatenate((x_temp, x_array), axis=0)
            y_temp = np.concatenate((y_temp, y_array), axis=0)

        # Append to the big dataset
        features_dataset['subject_' + subject]['features'] = np.float16(x_temp)
        features_dataset['subject_' + subject]['labels'] = np.float16(y_temp)

    return features_dataset
