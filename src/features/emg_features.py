import numpy as np
import deepdish as dd
from pathlib import Path
import collections
from numba import jit

import pysiology
from sampen import sampen2, normalize_data

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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

# @jit(nopython=True)
def extract_emg_features(data, config, scale=False):
    """ Load the EMG data and extract the features
    Parameters
    ----------
    data : dictionary
        epoched emg data
    config : yaml
        configuration file
    scale : bool
        use min-max scaling if scale=True
        
    Return
    ------
    Data : dictionary
        dictionary of feature and label data from all the subjects
    """

    Data = collections.defaultdict(dict)    

    for subject in config['subjects']:
        emg_vec    = data['subject_'+subject]['EMG']
        labels     = data['subject_'+subject]['labels']

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

        if scale:
            # print('Min-Max scaling the emg-features')
            # Min-Max scaling
            min_max_scaler = preprocessing.MinMaxScaler()
            features1      = min_max_scaler.fit_transform(features1)
            features2      = min_max_scaler.fit_transform(features2)

        Data['subject_'+subject]['features1'] = features1
        Data['subject_'+subject]['features2'] = features2

        Data['subject_'+subject]['labels']   = labels

    return Data

def pool_subject_emg_features(data, subjects, config):
    """ Pool the data from all the subjects together
    Parameters
    ----------
    config : yaml
        configuration file
    subjects: list
        list of strings consisting of subject identifiers
    Return
    ------
    X1_data : array
        an array of features from Feature set 1.
    X2_data : array
        an array of features from Feature set 2.
    Y_data : array
        An array of true labels.
    """

    for subject in subjects:
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
    scores = cross_val_score(clf, X, Y, cv=KFold(10, shuffle=True))

    print('10-fold cross validation Average accuracy: %0.4f (+/- %0.4f)' % ( np.mean(scores), np.std(scores) * 2))

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
    scores = cross_val_score(clf, X, Y, cv=KFold(10, shuffle=True))

    print('10-fold cross validation Average accuracy: %0.4f (+/- %0.4f)' % ( np.mean(scores), np.std(scores) * 2 ))
    

def RF_cross_validated_pooled_emg_features(X, Y, config):
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

    clf = RandomForestClassifier(n_estimators=100, oob_score=True)
    scores = cross_val_score(clf, X, Y, cv=KFold(10, shuffle=True))

    print('10-fold cross validation Average accuracy: %0.4f (+/- %0.4f)' % ( np.mean(scores), np.std(scores) * 2))