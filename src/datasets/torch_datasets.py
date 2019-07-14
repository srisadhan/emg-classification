from pathlib import Path

import torch
import numpy as np
import deepdish as dd

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SubjectSpecificDataset(Dataset):
    """A subject specific dataset class.

    Parameters
    ----------
    x : array
        Input array of the .

    Attributes
    ----------
    length : int
        Length of the dataset.

    """

    def __init__(self, x_data):
        super(SubjectSpecificDataset, self).__init__()
        self.length = features.shape[0]
        self.features = features

    def __getitem__(self, index):
        # Convert to torch tensors
        x = torch.from_numpy(self.featurs[index, :, :]).type(torch.float32)
        return x

    def __len__(self):
        return self.length


class PooledDataset(Dataset):
    """All subject dataset class.

    Parameters
    ----------
    ids_list : list
        ids list of training or validation or traning data.

    Attributes
    ----------
    ids_list

    """

    def __init__(self, features, labels, ids_list):
        super(PooledDataset, self).__init__()
        self.ids_list = ids_list
        self.features = features[self.ids_list, :, :]
        self.labels = labels[self.ids_list, :]

    def __getitem__(self, index):
        # Read only specific data and convert to torch tensors
        x = torch.from_numpy(self.features[index, :, :]).type(torch.float32)
        y = torch.from_numpy(self.labels[index, :]).type(torch.float32)
        return x, y

    def __len__(self):
        return self.features.shape[0]


def subject_pooled_data(config):
    """Get subject independent data (pooled data).

    Parameters
    ----------
    config : yaml
        The configuration file

    Returns
    -------
    features, labels, tags
        2 arrays features and labels.
        A tag determines whether the data point is used in training.

    """

    path = str(Path(__file__).parents[2] / config['clean_emg_data'])
    data = dd.io.load(path)

    # Parameters
    epoch_length = config['epoch_length']
    subjects = config['subjects']
    sfreq = config['sfreq']

    # Empty arrays for data
    x = np.empty((0, config['n_electrodes'], epoch_length * sfreq))
    y = np.empty((0, config['n_class']))
    tags = np.empty((0, 1))

    for subject in subjects:
        x_temp = data['subject_' + subject]['features']
        y_temp = data['subject_' + subject]['labels']
        x = np.concatenate((x, x_temp), axis=0)
        y = np.concatenate((y, y_temp), axis=0)
        tags = np.concatenate((tags, y_temp[:, 0:1] * 0 + 1), axis=0)

    # Balance the dataset
    rus = RandomUnderSampler()
    id = np.expand_dims(np.arange(x.shape[0]), axis=1)
    rus.fit_resample(y, y)

    # Form the features, labels
    features = x[rus.sample_indices_, :, :]
    labels = y[rus.sample_indices_, :]
    tags = tags[rus.sample_indices_, :]

    return features, labels, tags


def data_split_ids(labels, test_size=0.15):
    """Generators training, validation, and training indices to be used by Dataloader.

    Parameters
    ----------
    labels : array
        An array of labels.
    test_size : float
        Test size e.g. 0.15 is 15% of whole data.

    Returns
    -------
    dict
        A dictionary of ids corresponding to train, validate, and test.

    """
    id = np.arange(labels.shape[0])
    # Create an empty dictionary
    ids_list = {}

    # Split train, validation, and testing id
    train_id, test_id, _, _ = train_test_split(id,
                                               id * 0,
                                               test_size=2 * test_size)
    test_id, validate_id, _, _ = train_test_split(test_id,
                                                  test_id * 0,
                                                  test_size=0.5)

    ids_list['training'] = train_id
    ids_list['validation'] = validate_id
    ids_list['testing'] = test_id

    return ids_list


def pooled_data_iterator(config):
    """Create data iterators for torch models.

    Parameters
    ----------
    config : yaml
        The configuration file.

    Returns
    -------
    dict
        A dictionary contaning traning, validation, and testing iterator.

    """

    # Parameters
    BATCH_SIZE = config['BATCH_SIZE']
    TEST_SIZE = config['TEST_SIZE']

    # Get the features and labels
    features, labels, tags = subject_pooled_data(config)

    # Get training, validation, and testing ids_list
    ids_list = data_split_ids(labels, test_size=TEST_SIZE)

    # Initialise an empty dictionary
    data_iterator = {}

    # Create train, validation, test datasets and save them in a dictionary
    train_data = PooledDataset(features, labels, ids_list['training'])
    data_iterator['training'] = DataLoader(train_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=10)

    valid_data = PooledDataset(features, labels, ids_list['validation'])
    data_iterator['validation'] = DataLoader(valid_data,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True,
                                             num_workers=10)

    test_data = PooledDataset(features, labels, ids_list['testing'])
    data_iterator['testing'] = DataLoader(test_data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=10)

    return data_iterator
