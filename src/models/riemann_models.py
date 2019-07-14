from pathlib import Path

import deepdish as dd
import mne
import numpy as np
from pyriemann.embedding import Embedding
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def svm_tangent_space_classifier(features, labels):
    """A tangent space classifier with svm for 3 classes.

    Parameters
    ----------
    features : array
        A array of features
    labels : array
        True labels

    Returns
    -------
    sklearn classifier
        Learnt classifier.

    """
    # Construct sklearn pipeline
    n_components = 3  # pick some components
    clf = Pipeline([('xdawn_transform',
                     XdawnCovariances(n_components, estimator='lwf')),
                    ('tangent_space', TangentSpace(metric='riemann')),
                    ('svm_classify', SVC(kernel='rbf', gamma='auto'))])
    # cross validation
    clf.fit(features, labels)

    return clf


def svm_tangent_space_prediction(clf, features, true_labels):
    """Predict from learnt tangent space classifier.

    Parameters
    ----------
    clf : sklearn classifier
        Learnt sklearn classifier.
    features : array
        A array of features
    true_labels : array
        True labels

    Returns
    -------
    array
        Predicted labels from the model.

    """

    # Predictions
    predictions = clf.predict(features)
    print('Classification accuracy = ', accuracy_score(true_labels,
                                                       predictions), '\n')

    return predictions


def svm_tangent_space_cross_validate(data):
    """A cross validated tangent space classifier with svm.

    Parameters
    ----------
    data : dict
        A dictionary containing training and testing data

    Returns
    -------
    cross validated scores
        A list of cross validated scores.

    """

    # Combine the dataset
    x = np.concatenate((data['train_x'], data['test_x']), axis=0)
    y = np.concatenate((data['train_y'], data['test_y']), axis=0)

    # Construct sklearn pipeline
    n_components = 3  # pick some components
    clf = Pipeline([('xdawn_transform',
                     XdawnCovariances(n_components, estimator='lwf')),
                    ('tangent_space', TangentSpace(metric='riemann')),
                    ('svm_classify', SVC(kernel='rbf', gamma='auto'))])
    # cross validation
    scores = cross_val_score(clf, x, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print('\n')

    return scores


def xdawn_embedding(data):
    """Perform embedding of EEG data in 2D Euclidean space with Laplacian Eigenmaps.

    Parameters
    ----------
    data : dict
        A dictionary containing training and testing data

    Returns
    -------
    array
        Embedded

    """

    nfilter = 3
    xdwn = XdawnCovariances(estimator='scm', nfilter=nfilter)
    covs = xdwn.fit(data['train_x'], data['train_y']).transform(data['test_x'])

    lapl = Embedding(metric='riemann', n_components=3)
    embd = lapl.fit_transform(covs)

    return embd
