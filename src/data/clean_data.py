import collections
from pathlib import Path

import deepdish as dd
import numpy as np
import sys

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


def convert_to_array(subject, trial, path, sensor, config):
    """Converts the edf files in eeg and robot dataset into arrays.

    Parameter
    ----------
    subject : str
        String of subject ID e.g. 0001.
    trial : str
        Trail e.g. HighFine, LowGross.
    config : yaml
        The configuration file.
    n_class: int
        The number of classes
    path : string
        The path of file
    sensor : string
        The selection of data type between 'EMG' and 'PB'

    Returns
    -------
    array
        An array of feature (x) and lables (y)

    """

    # Read path
    # emg_path = str(Path(__file__).parents[2] / config['epoch_emg_data'])

    # Load the data
    data = dd.io.load(path, group='/' + 'subject_' + subject)
    epochs = data[sensor][trial]

    # Get array data
    x_array = epochs.get_data()

    if config['n_class'] == 3:

        # 3-class encoding
        if trial == 'HighFine':
            category = [1, 0, 0]
        elif trial == 'LowGross':
            category = [0, 1, 0]
        elif (trial == 'HighGross') or (trial == 'LowFine'):
            category = [0, 0, 1]

    elif config['n_class'] == 4:
        # 4-class encoding
        if trial == 'HighFine':
            category = [1, 0, 0, 0]
        elif trial == 'LowGross':
            category = [0, 1, 0, 0]
        elif trial == 'HighGross':
            category = [0, 0, 1, 0]
        elif trial == 'LowFine':
            category = [0, 0, 0, 1]

    # In order to accomodate testing
    try:
        y_array = one_hot_encode(x_array.shape[0], category)
    except ImportError:
        y_array = np.zeros((x_array.shape[0], config['n_class']))

    return x_array, y_array


def clean_epoch_data(subjects, trials, sensor, config):
    """Create feature dataset for all subjects.

    Parameter
    ----------
    subject : str
        String of subject ID e.g. 0001.
    trials : list
        A list of differet trials
    sensor : str
        Selection of data from sensor: 'EMG' or 'PB'
    Returns
    -------
    tensors
        All the data from subjects with labels.

    """
    # Initialize the numpy array to store all subject's data
    features_dataset = collections.defaultdict(dict)

    for subject in subjects:
        # Initialise for each subject
        x_temp = []
        y_temp = []
        for trial in trials:
            if (sensor == 'EMG'):
                path = str(Path(__file__).parents[2] / config['epoch_emg_data'])
            elif (sensor == 'PB'):
                path = str(Path(__file__).parents[2] / config['epoch_PB_data'])

            # Concatenate the data corresponding to all trials types
            x_array, y_array = convert_to_array(subject, trial, path, sensor, config)
            x_temp.append(x_array)
            y_temp.append(y_array)

        # Convert to array
        x_temp = np.concatenate(x_temp, axis=0)
        y_temp = np.concatenate(y_temp, axis=0)

        # Append to the big dataset
        features_dataset['subject_' + subject]['features'] = np.float32(x_temp)
        features_dataset['subject_' + subject]['labels'] = np.float32(y_temp)

    return features_dataset


def convert_combined_data_to_array(subject, trial, path1, path2, n_class, config):
    """Converts the edf files in eeg and robot dataset into arrays.

    Parameter
    ----------
    subject : str
        String of subject ID e.g. 0001.
    trial : str
        Trail e.g. HighFine, LowGross.
    config : yaml
        The configuration file.
    path1 : string
        The path of file
    path2 : string
        The path of the file
    n_class : int
        label the data into these many classes

    Returns
    -------
    array
        An array of feature (x) and lables (y)

    """

    # Read path
    # emg_path = str(Path(__file__).parents[2] / config['epoch_emg_data'])

    # Load the data
    data1 = dd.io.load(path1, group='/' + 'subject_' + subject)
    data2 = dd.io.load(path2, group='/' + 'subject_' + subject)
    
    epochs1 = data1['EMG'][trial]
    epochs2 = data2['PB'][trial]

    # Get array data
    EMG_array = epochs1.get_data()
    PB_array  = epochs2.get_data()

    # match the PB and EMG data size    
    array_len = np.min([EMG_array.shape[0], PB_array.shape[0]])
    EMG_array = EMG_array[:array_len, :, :]
    PB_array  = PB_array[:array_len, 2:, :] 
    
    PB_array  = PB_array.mean(axis=2) # consider average location in the epoch.

    if n_class == 3:

        # 3-class encoding
        if trial == 'HighFine':
            category = [1, 0, 0]
        elif trial == 'LowGross':
            category = [0, 1, 0]
        elif (trial == 'HighGross') or (trial == 'LowFine'):
            category = [0, 0, 1]

    elif n_class == 4:
        # 4-class encoding
        if trial == 'HighFine':
            category = [1, 0, 0, 0]
        elif trial == 'LowGross':
            category = [0, 1, 0, 0]
        elif trial == 'HighGross':
            category = [0, 0, 1, 0]
        elif trial == 'LowFine':
            category = [0, 0, 0, 1]

    # In order to accomodate testing
    try:
        y_array = one_hot_encode(EMG_array.shape[0], category)
    except ImportError:
        y_array = np.zeros((EMG_array.shape[0], config['n_class']))

    return EMG_array, PB_array, y_array


def clean_combined_data(subjects, trials, n_class, config):
    """Create combined EMG and Force feature set for all subjects.

    Parameter
    ----------
    subject : str
        String of subject ID e.g. 0001.
    trials : list
        A list of differet trials
    n_class : int
        label the data into these many classes

    Returns
    -------
    tensors
        All the data from subjects with labels.

    """
    # Initialize the numpy array to store all subject's data
    features_dataset = collections.defaultdict(dict)

    for subject in subjects:
        # Initialise for each subject
        EMG_temp = []
        PB_temp  = []
        y_temp   = []

        for trial in trials:
            path1 = str(Path(__file__).parents[2] / config['epoch_emg_data'])
            path2 = str(Path(__file__).parents[2] / config['epoch_PB_data'])

            # Concatenate the data corresponding to all trials types
            EMG_array, PB_array, y_array = convert_combined_data_to_array(subject, trial, path1, path2, n_class, config)
            
            EMG_temp.append(EMG_array)
            PB_temp.append(PB_array)
            y_temp.append(y_array)

        # Convert to array
        EMG_temp = np.concatenate(EMG_temp, axis=0)
        PB_temp  = np.concatenate(PB_temp, axis=0)
        y_temp   = np.concatenate(y_temp, axis=0)

        # Append to the big dataset
        features_dataset['subject_' + subject]['EMG'] = np.float32(EMG_temp)
        features_dataset['subject_' + subject]['PB'] = np.float32(PB_temp)
        features_dataset['subject_' + subject]['labels'] = np.float32(y_temp)

    return features_dataset

