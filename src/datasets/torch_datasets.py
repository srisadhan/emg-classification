from pathlib import Path

import torch
import numpy as np
import deepdish as dd
import numpy as np

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

import sys

class SubjectSpecificDataset(Dataset):
    """A subject specific dataset class.

    Parameters
    ----------
    x : array
        Input array of the features.

    Attributes
    ----------
    length : int
        Length of the dataset.

    """
    def __init__(self, x_data):
        super(SubjectSpecificDataset, self).__init__()
        self.length = x_data.shape[0]
        self.features = x_data

    def __getitem__(self, index):
        # Convert to torch tensors
        x = torch.from_numpy(self.features[index, :, :]).type(torch.float32)
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
        if len(features.shape) > 2:
            self.features = features[self.ids_list, :, :]
        else:
            self.features = features[self.ids_list, :]

        self.labels = labels[self.ids_list, :]

    def __getitem__(self, index):
        # Read only specific data and convert to torch tensors
        if len(self.features.shape) > 2:
            x = torch.from_numpy(self.features[index, :, :]).type(torch.float32)
        else:
            x = torch.from_numpy(self.features[index, :]).type(torch.float32)

        y = torch.from_numpy(self.labels[index, :]).type(torch.float32)
        return x, y

    def __len__(self):
        return self.features.shape[0]


def subject_pooled_EMG_data(config):
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

    if config['n_class'] == 3:
        save_path = str(Path(__file__).parents[1] / config['clean_emg_data_3class'])
    elif config['n_class'] == 4:
        save_path = str(Path(__file__).parents[1] / config['clean_emg_data_4class'])
        
    data = dd.io.load(path)

    # Parameters
    subjects = config['subjects']

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


def data_split_ids(labels, test_size=0.15):
    """Generators training, validation, and training
    indices to be used by Dataloader.

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
    features, labels, tags = subject_pooled_EMG_data(config)

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


def extract_average_RMS(emg_data):
    """extract the RMS value of each channel and then average it across all the channels for each emg epoch
    
    Parameters
    ----------
    emg_data : numpy array
        epoched emg data with size epochs x channels x samples

    Returns
    -------
    rms_array : numpy array
        average rms values calculated for each epoch across all the channels
    """
    rms_array = []
    for data in emg_data:
        # calculate the RMS values of each channels using the epochs
        channel_rms = np.sqrt(np.mean(np.square(data, dtype=float), axis=1))
        # calculate the average RMS value of all the channels
        emg_rms     = np.mean(channel_rms, axis=0)
        # append the values to a list
        rms_array.append(emg_rms)
    
    return np.array(rms_array)


def classify_tangentSpace_features(clf, emg_data, flag):
    """extract the tangent space features from the epochs and 
    obtain the maximum log-likelihood
    
    Parameters
    ----------
    emg_data : numpy array
        epoched emg data with size epochs x channels x samples
    clf : trained sklearn classifier
        A sklearn classifier model such as SVM or RF previously trained on the user provided data
    flag : string
        Predict maximum log-likelihood if flag=log_proba otherwise just the predicted label
    Returns
    -------
    rms_array : numpy array
        average rms values calculated for each epoch across all the channels
    """
    cov = Covariances().fit_transform(emg_data)
    ts  = TangentSpace().fit_transform(cov)

    if flag == 'log_proba':
        return np.amax(clf.predict_log_proba(ts), axis=1)
    else:
        return clf.predict(ts)


def pool_classifier_output_NN(data, subjects, trials, clf, config):
    """ Pool the history of classifier output along with the RMS values for training the self-correction algorithm

    Parameters
    ----------
    subjects : list
        List of strings containing subject identifiers
    clf : trained sklearn classifier
        A sklearn classifier model such as SVM or RF previously trained on the user provided data
    config : yaml
        The configuration file
    filepath : str
        path to the file

    Returns
    -------
    features_emg , labels
        2 arrays features and labels.

    """
    # data = dd.io.load(filepath)

    # Empty array (list)
    features = np.empty((0, config['SELF_CORRECTION_NN']['INPUTS'])) # maximum log-likehood and average values stacked up together
    y = []

    for subject in subjects:
        for trial in trials:
            emg_temp = data['subject_' + subject][trial]['EMG']
            y_temp  = data['subject_' + subject][trial]['labels']
            
            # extract the average RMS values of the epochs
            rms_values = extract_average_RMS(emg_temp)

            # FIXME:Check if log probabilities should be used or class labels
            # predict log-probabilities   
            log_prob  = classify_tangentSpace_features(clf, emg_temp, 'log_proba')

            win_len   = config['SELF_CORRECTION_NN']['WIN_LEN']
            temp      = np.zeros((log_prob.shape[0] - win_len, config['SELF_CORRECTION_NN']['INPUTS']))

            # arrange the t:t-10 (11*2 values) of log_prob and rms_values 
            for i in range(win_len , log_prob.shape[0]):
                temp[i-win_len, :win_len+1]    = log_prob[i-win_len:i+1]
                temp[i-win_len, win_len+1:]    = rms_values[i-win_len:i+1]
                y.append(y_temp[i,:])

            features = np.concatenate((features, temp), axis=0)
    y = np.array(y).reshape(-1,3)

    return features, y


def pooled_data_SelfCorrect_NN(data, clf, config):
    """Prepare dataset for training the self-correcting Neural Network
    
    Parameters
    ----------
    clf : trained sklearn classifier
        A sklearn classifier model such as SVM or RF previously trained on the user provided data
    config : dictionary
        a dictionary of parameters loaded from the config.yaml file
    """

    # Parameters
    BATCH_SIZE = config['BATCH_SIZE']
    TEST_SIZE  = config['TEST_SIZE']

    subjects_train = list(set(config['subjects']) ^ set(config['test_subjects']))
    subjects_test  = config['test_subjects']
    trials         = list(set(config['trials']) ^ set(config['comb_trials']))

    print('Subjects for training:', subjects_train)
    print('Subjects for testing:', subjects_test)

    # ------ Training data preparation
    features_train, labels_train = pool_classifier_output_NN(data, subjects_train, trials, clf, config)

    # Balance the training dataset if not done previously
    rus = RandomUnderSampler()
    rus.fit_resample(labels_train, labels_train)
    features_train = features_train[rus.sample_indices_, :]
    labels_train = labels_train[rus.sample_indices_, :]

    # Get training, validation, and testing ids_list
    ids_list = data_split_ids(labels_train, test_size=TEST_SIZE)

    # combine the training and testing data as there is a separate dataset for testing
    ids_list['training'] = np.concatenate((ids_list['training'], ids_list['testing']), axis=0)
    
    # Initialise an empty dictionary
    data_iterator = {}

    # Split the train, validation datasets from the training data and save them in a dictionary
    train_data = PooledDataset(features_train, labels_train, ids_list['training'])
    data_iterator['training'] = DataLoader(train_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=10)

    valid_data = PooledDataset(features_train, labels_train, ids_list['validation'])
    data_iterator['validation'] = DataLoader(valid_data,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True,
                                             num_workers=10)

    # ------ Testing dataset preparation 
    features_test, labels_test = pool_classifier_output_NN(data, subjects_test, trials, clf, config)
    test_data = PooledDataset(features_test, labels_test, np.arange(0, labels_test.shape[0]))

    data_iterator['testing'] = DataLoader(test_data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=10)

    return data_iterator