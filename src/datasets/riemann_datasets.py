from pathlib import Path

import deepdish as dd
import numpy as np
import random 

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


def train_test_data(features, labels, leave_tags, config):
    """Short summary.

    Parameters
    ----------
    features : array
        An array of features.
    labels : array
        True labels.
    leave_tags : array
        An array specifying whether a subject was left out of training.
    config : yaml
        The configuration file.

    Returns
    -------
    dict
        A dict containing the train and test data.

    """
    data = {}
    labels = np.argmax(labels, axis=1)  # Convert to class int

    # Train test split
    id = np.arange(features.shape[0])
    if (leave_tags == 0).any():
        train_id = np.nonzero(leave_tags)[0]
        test_id = np.nonzero(1 - leave_tags)[0]
    else:
        train_id, test_id, _, _ = train_test_split(id,
                                                   id * 0,
                                                   test_size=2 *
                                                   config['TEST_SIZE'])

    # Training
    data['train_x'] = features[train_id, :, :]
    data['train_y'] = labels[train_id]

    # Testing
    data['test_x'] = features[test_id, :, :]
    data['test_y'] = labels[test_id]

    return data


def subject_pooled_EMG_data(subjects, path, config):
    """Get subject independent data (pooled data).

    Parameters
    ----------
    subjects : list
        List of strings containing subject identifiers
    config : yaml
        The configuration file
    path : str
        path to the file
    Returns
    -------
    features, labels, tags
        2 arrays features and labels.
        A tag determines whether the data point is used in training.

    """

    # path = str(Path(__file__).parents[2] / config['clean_emg_data'])
    data = dd.io.load(path)

    # Subject information
    subjects = config['subjects']

    # Empty array (list)
    x = []
    y = []
    tags = np.empty((0, 1))

    for subject in subjects:
        x_temp = data['subject_' + subject]['features']
        y_temp = data['subject_' + subject]['labels']
        x.append(x_temp)
        y.append(y_temp)
        tags = np.concatenate((tags, y_temp[:, 0:1] * 0 + 1), axis=0)

    # Convert to array
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(y, y)

    # Store them in dictionary
    features = x[rus.sample_indices_, :, :]
    labels = y[rus.sample_indices_, :]
    tags = tags[rus.sample_indices_, :]

    return features, labels, tags


def subject_pooled_EMG_PB_data(subjects, path, config):
    """Get subject independent data (pooled data).

    Parameters
    ----------
    subjects : list
        List of strings containing subject identifiers
    config : yaml
        The configuration file
    path : str
        path to the file
    Returns
    -------
    features_emg, pos, labels
        2 arrays features and labels.

    """

    # path = str(Path(__file__).parents[2] / config['clean_emg_data'])
    data = dd.io.load(path)

    # Subject information
    subjects = config['subjects']

    # Empty array (list)
    emg = []
    pb = []
    y = []
    # tags = np.empty((0, 1))

    for subject in subjects:
        emg_temp = data['subject_' + subject]['EMG']
        pb_temp  = data['subject_' + subject]['PB']
        y_temp  = data['subject_' + subject]['labels']
        
        emg.append(emg_temp)
        pb.append(pb_temp)
        y.append(y_temp)
        # tags = np.concatenate((tags, y_temp[:, 0:1] * 0 + 1), axis=0)

    # Convert to array
    emg = np.concatenate(emg, axis=0)
    pb = np.concatenate(pb, axis=0)
    y  = np.concatenate(y, axis=0)
    
    return emg, pb, y

    # uncomment the following line if you want to balance the data before splitting
    # Balance the dataset
    # rus = RandomUnderSampler()
    # rus.fit_resample(y, y)

    # Store them in dictionary
    # features_emg = emg[rus.sample_indices_, :, :]
    # pos = pb[rus.sample_indices_, :]
    # labels = y[rus.sample_indices_, :]
    # tags = tags[rus.sample_indices_, :]

    # return features_emg, pos, labels


def subject_dependent_data(config):
    """Get subject dependent data.

    Parameters
    ----------
    config : yaml
        The configuration file

    Returns
    -------
    features, labels
        2 arrays features and labels

    """

    path = str(Path(__file__).parents[2] / config['clean_emg_data'])
    data = dd.io.load(path)

    # Parameters
    subjects = config['subjects']
    epoch_length = config['epoch_length']
    sfreq = config['sfreq']

    # Subject information
    subjects = config['subjects']
    x = np.empty((0, config['n_electrodes'], epoch_length * sfreq))
    y = np.empty((0, config['n_class']))
    tags = np.empty((0, 1))

    for subject in subjects:
        x_temp = data['subject_' + subject]['features']
        y_temp = data['subject_' + subject]['labels']
        x = np.concatenate((x, x_temp), axis=0)
        y = np.concatenate((y, y_temp), axis=0)
        if subject in config['test_subjects']:
            tags = np.concatenate((tags, y_temp[:, 0:1] * 0), axis=0)
        else:
            tags = np.concatenate((tags, y_temp[:, 0:1] * 0 + 1), axis=0)

    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(y, y)

    # Store them in dictionary
    features = x[rus.sample_indices_, :, :]
    labels = y[rus.sample_indices_, :]
    tags = tags[rus.sample_indices_, :]

    return features, labels, tags


def subject_specific_data(subject, config):
    """Get subject specific data.

    Parameters
    ----------
    config : yaml
        The configuration file

    Returns
    -------
    features, labels
        2 arrays features and labels

    """

    path = str(Path(__file__).parents[2] / config['clean_emg_data'])
    data = dd.io.load(path)

    # Get the data
    x = data['subject_' + subject]['features']
    y = data['subject_' + subject]['labels']
    tags = y[:, 0:1] * 0 + 1

    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(y, y)

    # Store them in dictionary
    features = x[rus.sample_indices_, :, :]
    labels = y[rus.sample_indices_, :]
    tags = tags[rus.sample_indices_, :]

    return features, labels, tags


def split_pooled_EMG_PB_data_train_test(subjects, path, config):
    """ Split the EMG data into training and testing data and get the
        corresponding position data

    Parameters
    ----------
    subjects : list
        List of strings containing subject identifiers
    config : yaml
        The configuration file
    path : str
        path to the file

    Returns
    -------
    Data : datastructure
        Data[train] : EMG, pos, labels
        Data[test]  : EMG, pos, labels

    """
    Data = {}
    training_data = {}
    testing_data = {}

    # pool the data
    features_emg, pos, labels = subject_pooled_EMG_PB_data(subjects, path, config)
    # print("Total pooled data: %f" %(labels.shape[0]))

    # split the shuffled data for training and testing
    indices = np.arange(0, labels.shape[0], dtype=int)
    random.shuffle(indices)
    cutoff = round(0.75 * labels.shape[0])

    # training set
    train_emg       = features_emg[indices[:cutoff],:,:]
    train_pos       = pos[indices[:cutoff],:]
    train_labels    = labels[indices[:cutoff],:]

    # print("Total data after splitting the pooled data: %f" %(train_labels.shape[0]))

    # Balancing the training data
    rus = RandomUnderSampler(return_indices=True)
    rus.fit_resample(train_labels, train_labels)

    train_emg    = train_emg[rus.sample_indices_,:,:]
    train_pos    = train_pos[rus.sample_indices_,:]
    train_labels = train_labels[rus.sample_indices_,:]
    train_labels = np.dot(train_labels, np.array(np.arange(1, config['n_class']+1)))

    training_data['X']   = train_emg
    training_data['pos'] = train_pos
    training_data['y']   = train_labels

    # check if the classes are balanced
    print("Training data - Class 1: %f, Class 2: %f, Class 3: %f, Class 4: %f" %(train_labels[train_labels == 1].shape[0], train_labels[train_labels == 2].shape[0], train_labels[train_labels == 3].shape[0], train_labels[train_labels == 4].shape[0]))

    # test set
    test_emg        = features_emg[indices[cutoff:],:,:]
    test_pos        = pos[indices[cutoff:],:]
    test_labels     = labels[indices[cutoff:],:]
    test_labels     = np.dot(test_labels, np.array(np.arange(1, config['n_class']+1)))

    testing_data['X']   = test_emg
    testing_data['pos'] = test_pos
    testing_data['y']   = test_labels

    Data['train'] = training_data
    Data['test']  = testing_data

    return Data
    