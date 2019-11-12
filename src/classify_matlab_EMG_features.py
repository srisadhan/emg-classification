import yaml
import hdf5storage
import collections
import copy
import random
import seaborn
import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import preprocessing
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from umap import UMAP
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix

import sys

Kfolds = 5

def convert_mat2h5(filepath, config):
    """ Converts the mat file generated using Matlab to a python dictionary and
        saves in HDF format
    Parameters:
    ----------
    filename : string
        location and name of the mat file
    config   : yaml
        configuration file

    saves the dictionary in .h5 format into the data folder
    """
    Data = collections.defaultdict(dict)
    mat = hdf5storage.loadmat(filepath)

    subjects = config['subjects']
    for subject in range(0,len(subjects)):
        temp = mat['Data'][0][subject] # mat file storage dimension 1 x 10
        data = collections.defaultdict(dict)

        for feature in config['features']:
            temp1             = temp[feature]
            data[feature]     = temp1

        data['label'] = temp['Y']

        Data['subject_'+subjects[subject]] = data

    dd.io.save(config['save_h5_file'], Data)

def pool_DB5_features_normalize(config, feature_set, subjects):
    """ normalize the emg features from individual subjects and Pool them

    Parameters:
    ----------
    config  : yaml
        configuration file
    feature_set  : list
        list of a string of features. For e.g. 'mDWT', 'TD', 'RMS', 'HIST'
    subjects : list
        list of a string of subjects. For e.g. '7707'

    Returns:
    --------
    X       : numpy array
        Pooled data
    Y       : numpy array
        Pooled labels
    """
    Data = dd.io.load(config['save_h5_file'])
    min_max_scaler = preprocessing.MinMaxScaler()
    Features = np.empty((0,0))
    Labels   = np.empty((0,0))

    for subject in subjects:
        temp1 = np.empty((0,0))
        label1= np.empty((0,0))

        for feature in feature_set:
            temp2 = min_max_scaler.fit_transform(Data['subject_'+subject][feature])
            if temp1.size == 0:
                temp1 = temp2
            else:
                temp1 = np.append(temp1, temp2, axis=1)

        label1 = Data['subject_'+subject]['label']

        if Features.size == 0:
            Features    = temp1
            Labels      = label1
        else:
            Labels      = np.append(Labels, label1, axis=0)
            Features    = np.append(Features, temp1, axis=0)

    return Features, Labels

def svm_train_test(X_train, X_test, Y_train, Y_test, split_data=False, split_size=0.25):
    """Train an SVM classifier on the (X_train, Y_train) and test it on (X_test, Y_test).

    Parameters
    ----------
    X_train : numpy array
        an nd-array of features for training the classifier.
    X_test  : numpy array
        an nd-array of features for testing the classifier.
    Y_train : numpy array
        an nd-array of class labels for training the classifier.
    Y_test  : numpy array
        an nd-array of class labels for testing the classifier.
    split_data : bool
        boolean to say if the data should be split into train and test. (default = False)
    split_size : float
        percentage test size. (default = 0.25)
    Returns
    -------
    type
        Description of returned object.

    """
    # only split the data if mentioned. (only in the case of pooled data)
    if split_data:
        X_train, X_test, Y_train, Y_test = train_test_split(X_train,Y_train, test_size=split_size)

    # create a SVM classifier with rbf kernel
    clf  = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')
    test_score = clf.fit(X_train, Y_train.ravel()).score(X_test, Y_test.ravel())
    print('SVM score for 75 train and 25 test: %0.4f '%(test_score))

def svm_cross_validate(X, Y, Kfolds):
    """ Task type classification using Support Vector Machine
        Obtain the mean classification accuracy using k-fold cross-validation

    Parameters
    ----------
    X : Resampled feature matrix
    Y : Resampled class labels

    """

    # create a SVM classifier with rbf kernel
    clf  = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')

    # cross validation
    scores = cross_val_score(clf, X, Y, cv=KFold(5, shuffle=True), n_jobs=-1)
    print("Accuracy using SVM: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print('\n')

def random_forest_train_test(X_train, X_test, Y_train, Y_test, split_data=False, split_size=0.25):
    """Train an Random Forrest classifier on the (X_train, Y_train) and test it on (X_test, Y_test).

    Parameters
    ----------
    X_train : numpy array
        an nd-array of features for training the classifier.
    X_test  : numpy array
        an nd-array of features for testing the classifier.
    Y_train : numpy array
        an nd-array of class labels for training the classifier.
    Y_test  : numpy array
        an nd-array of class labels for testing the classifier.
    split_data : bool
        boolean to say if the data should be split into train and test. (default = False)
    split_size : float
        percentage test size. (default = 0.25)
    Returns
    -------
    type
        Description of returned object.

    """
    # only split the data if mentioned. (only in the case of pooled data)
    if split_data:
        X_train, X_test, Y_train, Y_test = train_test_split(X_train,Y_train, test_size=split_size)

    # create a SVM classifier with rbf kernel
    clf = RandomForestClassifier(n_estimators=100, oob_score=True) #  max_depth=10,
    test_score = clf.fit(X_train, Y_train.ravel()).score(X_test, Y_test.ravel())
    print('Random Forest score for 75 train : %0.4f and 25 test : %0.4f '%(clf.oob_score_, test_score))
    conf_mat = confusion_matrix(Y_test.ravel(), clf.predict(X_test))
    print(conf_mat)
    seaborn.heatmap(conf_mat)
    # print([estimator.tree_.max_depth for estimator in clf.estimators_])

def random_forest_cross_validate(X, Y, Kfolds):
    """ Task type classification using Random forest classifier
        Obtain the mean classification accuracy using k-fold cross-validation

    Parameters
    ----------
    X : Resampled feature matrix
    Y : Resampled class labels

    """

    # create a SVM classifier with rbf kernel
    clf = RandomForestClassifier(n_estimators=100) #, max_depth=10)

    # cross validation
    scores = cross_val_score(clf, X, Y, cv=KFold(5, shuffle=True), n_jobs=-1)
    print(scores)
    print("Accuracy using Random Forest: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print('\n')

def resample_data(X, Y,resample=True):
    """ Resample the data to balance the classes

    Parameters
    ----------
    X       : numpy array
        features
    Y       : numpy array
        labels
    resample: bool (default: True)

    Returns
    -------
    X_resampled : Resampled feature matrix
    Y_resampled : Resampled class labels
    """
    if resample:
        # balance the data by under sampling
        ros = RandomUnderSampler()
        X_resampled, Y_resampled = ros.fit_resample(X, Y)
        print('The data is balanced')
    else:
        X_resampled = X
        Y_resampled = Y
        print('The data is not balanced')

    print('Class 1: ', Y_resampled[Y_resampled==1].shape,
          'Class 2: ', Y_resampled[Y_resampled==2].shape,
          'Class 3: ', Y_resampled[Y_resampled==3].shape,
          'Class 4: ', Y_resampled[Y_resampled==4].shape)

    return X_resampled, Y_resampled


# -------------Main function--------------#
if __name__ == '__main__':
    # function execution booleans
    split_data              = True   # True : train-test if true; False : cross-validate
    save_h5_file            = True   # True : load the matlab file and save it to a h5 format
    run_classifier          = True   # True : load the features and pass it to the classifiers
    test_transferability    = False  # True : Train the classifier on only 8 subjects and test it on the rest

    # overlap percentages used for evaluating features
    overlaps                = [75] #[25, 50, 75]

    # open the configuration file
    with open("config.yml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    for iter in overlaps:
        if save_h5_file:
            print('Saving the features into a dictionary with h5 format....')
            filepath = config['mat_file_loc'] + str(iter) + '_features.mat'
            convert_mat2h5(filepath, config)

        if run_classifier:
            feature_set  = ['RMS', 'TD', 'HIST', 'mDWT']

            if test_transferability:
                print('Training the classifier on 8 subjects and testing it on 4')
                sub_list = copy.copy(config['subjects'])
                random.shuffle(sub_list)

                [X_train,Y_train] = pool_DB5_features_normalize(config, feature_set, sub_list[0:8])
                [X_test,Y_test]   = pool_DB5_features_normalize(config, feature_set, sub_list[8:])

                print('-------------------------------')
                svm_train_test(X_train, X_test, Y_train, Y_test)

                print('-------------------------------')
                random_forest_train_test(X_train, X_test, Y_train, Y_test)
                sys.exit()
            else:
                print('Pooling the data from all subjects and cross-validating the classifier')
                [X,Y] = pool_DB5_features_normalize(config, feature_set, config['subjects'])
                [X,Y] = resample_data(X,Y)
                
                print('-------------------------------')
                print('Calculating the accuracy for %f overlap' %(iter))
                print(' # of data samples: %0.4f, # of features: %.4f ' %(X.shape[0],X.shape[1]))

                if split_data:
                    print('-------------------------------')
                    svm_train_test(X, [], Y, [], split_data=True)

                    print('-------------------------------')
                    random_forest_train_test(X, [], Y, [], split_data=True)

                else:
                    print('-------------------------------')
                    svm_cross_validate(X,Y.ravel(), Kfolds)

                    print('-------------------------------')
                    random_forest_cross_validate(X,Y.ravel(), Kfolds)

    plt.show()
            # print('-------------------------------')
            # print('t-SNE based data visualization')
            # ind = np.arange(0,Y.shape[0]).reshape(Y.shape[0],1)
            # temp1 = ind[Y == 1]
            # temp2 = ind[Y == 2]
            # temp3 = ind[Y == 3]

            # X_embedded = TSNE(n_components=2, perplexity=100, learning_rate=50.0).fit_transform(X)
            #
            # plt.figure()
            # plt.plot(X_embedded[temp1,0],X_embedded[temp1,1],'bo')
            # plt.plot(X_embedded[temp2,0],X_embedded[temp2,1],'ro')
            # plt.plot(X_embedded[temp3,0],X_embedded[temp3,1],'ko')

            # for neighbor in [10, 30, 60 ]:
            #     fit = UMAP(n_neighbors=neighbor, min_dist=0.0, n_components=3,metric='chebyshev')
            #     X_embedded = fit.fit_transform(X)
            #
            #     fig = plt.figure()
            #     ax = Axes3D(fig)
            #
            #     ax.plot(X_embedded[temp1,0],X_embedded[temp1,1],X_embedded[temp1,2],'bo')
            #     ax.plot(X_embedded[temp2,0],X_embedded[temp2,1],X_embedded[temp2,2],'ro')
            #     ax.plot(X_embedded[temp3,0],X_embedded[temp3,1],X_embedded[temp3,2],'ko')
            # plt.show()
