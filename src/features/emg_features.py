import numpy as np
import deepdish as dd
from pathlib import Path
import collections

import pysiology
from sampen import sampen2, normalize_data

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from imblearn.under_sampling import RandomUnderSampler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt
import sys
import copy

def sampEntropy(rawEMGSignal):
    """
    Parameters
    ----------
    rawEMGSignal : ndarray
        an epoch of raw emg-signal

    Returns
    -------
    feature_value : float
        sample entropy calculated from the rawEMGSignal for the parameters m = 2 and r = 0.2 * std

    """
    copy_rawEMGSignal    = copy.copy(rawEMGSignal) # shallow copy
    copy_feature         = sampen2(normalize_data(list(copy_rawEMGSignal)))[2][1]
    if copy_feature is None:
        feature_value = 0
    else:
        feature_value = copy_feature

    return feature_value

def extract_emg_features(config, sort_channels):
    """ Load the EMG data and extract the features
    Parameters
    ----------
    config : yaml
        configuration file
    sort_channels : bool
        sort the channels according to the highest activity observed (high variance)
    Return
    ------
    Data : dictionary
        dictionary of feature and label data from all the subjects
    """
    if sort_channels:
        path = str(Path(__file__).parents[2] / config['emg_channel_order'])
        channel_order = dd.io.load(path)

    Data = collections.defaultdict(dict)
    # load the data from the path
    if config['n_class'] == 3:
        save_path = Path(__file__).parents[2] / config['clean_emg_data_3class']
    elif config['n_class'] == 4:
        save_path = Path(__file__).parents[2] / config['clean_emg_data_4class']
        
    data = dd.io.load(str(save_path))

    for subject in config['subjects']:

        emg_vec    = data['subject_'+subject]['features']
        labels     = data['subject_'+subject]['labels']

        # coverting one hot encoding back to class labels
        # if config['n_class'] == 3:
        labels     = np.sum( np.multiply(np.array(np.arange(1, config['n_class']+1),ndmin=2) , labels) , axis=1)
        # elif config['n_class'] == 4:
        #     labels     = np.sum( np.multiply(np.array([1,2,3,4],ndmin=2) , labels) , axis=1)

        # A 3d array with dimensions representing trial_samples x emg_channels x epochs
        data_shape = emg_vec.shape
        # initialize the feature array - samples x features
        features1   = np.zeros((data_shape[0],config['n_electrodes'] * config['n_features']))
        features2   = np.zeros((data_shape[0],config['n_electrodes'] * (config['n_features'] - 1)))

        for i in range(data_shape[0]):
            for j in range(data_shape[1]):
                rawEMGSignal         = emg_vec[i,j,:]

                # feature set 1
                features1[i,4*j]     = pysiology.electromyography.getWL(rawEMGSignal)
                features1[i,4*j+1]   = pysiology.electromyography.getZC(rawEMGSignal, config['threshold'])
                features1[i,4*j+2]   = pysiology.electromyography.getSSC(rawEMGSignal, config['threshold'])
                features1[i,4*j+3]   = pysiology.electromyography.getMAV(rawEMGSignal)

                # feature set 2
                features2[i,3*j]     = pysiology.electromyography.getRMS(rawEMGSignal)
                features2[i,3*j+1]   = pysiology.electromyography.getWL(rawEMGSignal)
                features2[i,3*j+2]   = sampEntropy(rawEMGSignal)

                # features[i,6*j+5]   = sampen2(normalize_data(copy_rawEMGSignal))[2][1] # m = 2, r = 0.2 * std

        # Min-Max scaling
        min_max_scaler = preprocessing.MinMaxScaler()
        features1      = min_max_scaler.fit_transform(features1)
        features2      = min_max_scaler.fit_transform(features2)

        if sort_channels:
            # load the channel sort order
            emg_order   = channel_order['subject_'+subject]['channel_order']

            # initialize the feature order array
            feat1_order = np.zeros(config['n_electrodes'] * config['n_features'])
            feat2_order = np.zeros(config['n_electrodes'] * (config['n_features'] - 1))
            # iterate through all the 8 channels to properly fit in features
            for chan in range(0,8):
                feat1_order[4*chan]   = 4*emg_order[chan]
                feat1_order[4*chan+1] = 4*emg_order[chan]+1
                feat1_order[4*chan+2] = 4*emg_order[chan]+2
                feat1_order[4*chan+3] = 4*emg_order[chan]+3

                feat2_order[3*chan]   = 3*emg_order[chan]
                feat2_order[3*chan+1] = 3*emg_order[chan]+1
                feat2_order[3*chan+2] = 3*emg_order[chan]+2

            feat1_order = feat1_order.astype(int)
            feat2_order = feat2_order.astype(int)

            features1 = features1[:,feat1_order[:]]
            features2 = features2[:,feat2_order[:]]

        Data['subject_'+subject]['features1'] = features1
        Data['subject_'+subject]['features2'] = features2

        Data['subject_'+subject]['labels']   = labels

    return Data

def pool_subject_emg_features(config):
    """ Pool the data from all the subjects together
    Parameters
    ----------
    config : yaml
        configuration file

    Return
    ------
    X1_data : array
        an array of features from Feature set 1.
    X2_data : array
        an array of features from Feature set 2.
    Y_data : array
        An array of true labels.
    """

    path = str(Path(__file__).parents[2] / config['subject_emg_features'])
    data = dd.io.load(path)

    for subject in config['subjects']:
        # concatenate array to X_data if it exist, otherwise initialize X_data with the values
        if 'X1_data' in locals():
            X1_data = np.concatenate((X1_data, data['subject_'+subject]['features1']),axis=0)
            X2_data = np.concatenate((X2_data, data['subject_'+subject]['features2']),axis=0)
        else:
            X1_data = data['subject_'+subject]['features1']
            X2_data = data['subject_'+subject]['features2']

        if 'Y_data' in locals():
            Y_data = np.concatenate((Y_data, data['subject_'+subject]['labels']),axis=0)
        else:
            Y_data = data['subject_'+subject]['labels']

    return X1_data, X2_data, Y_data

def balance_pooled_emg_features(config):
    """
    Parameters
    ----------
    config : yaml
        configuration file

    Return
    ------
    X_res : array
        resmapled features
    Y_res : array
        resampled labels
    """

    path = str(Path(__file__).parents[2] / config['pooled_emg_features'])
    data = dd.io.load(path)

    X1   = data['X1']
    X2   = data['X2']
    Y    = data['Y']

    rus = RandomUnderSampler()
    X1_res, Y_res = rus.fit_resample(X1, Y)

    # Store them in dictionary
    # X1_res = X1[rus.sample_indices_, :]
    # Y_res  = Y[rus.sample_indices_]
    X2_res = X2[rus.sample_indices_, :]

    return X1_res, X2_res, Y_res

def svm_cross_validated_pooled_emg_features(X, Y, config):
    """ Load the EMG data and extract the features
    Parameters
    ----------
    X      :  array
        array of features
    Y      : array
        array of labels
    config : yaml
        configuration file

    Returns
    -------
    """

    clf  = SVC(C=1.0, kernel='rbf', gamma='auto', decision_function_shape='ovr')
    scores = cross_val_score(clf, X, Y, cv=KFold(5, shuffle=True))

    print('5-fold cross validation Average accuracy: %0.4f (+/- %0.4f)' % ( np.mean(scores), np.std(scores) ))

def lda_cross_validated_pooled_emg_features(X, Y, config):
    """ Load the EMG data and extract the features
    Parameters
    ----------
    X      :  array
        array of features
    Y      : array
        array of labels
    config : yaml
        configuration file

    Returns
    -------
    """

    clf = LinearDiscriminantAnalysis(solver='svd')
    scores = cross_val_score(clf, X, Y, cv=KFold(5, shuffle=True))

    print('5-fold cross validation Average accuracy: %0.4f (+/- %0.4f)' % ( np.mean(scores), np.std(scores) ))
