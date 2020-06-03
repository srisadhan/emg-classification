import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import deepdish as dd
import sys
import random
import statistics
import copy
import math
import torch
# import hdf5storage
import h5py
import numpy as np
from pathlib import Path
import pandas as pd 

# PyRiemann package
from pyriemann.estimation import Covariances, Shrinkage, Coherences
from pyriemann.tangentspace import TangentSpace, FGDA
from pyriemann.utils.distance import distance, distance_riemann, distance_logeuclid
from pyriemann.utils.mean import mean_riemann
from pyriemann.classification import MDM, TSclassifier
from pyriemann.channelselection import ElectrodeSelection
from pyriemann.embedding import Embedding
from pyriemann.spatialfilters import SPoC, CSP
# from pyriemann.utils.viz import plot_confusion_matrix

# RPA Package (Riemann Procrustes Analysis)
from rpa import transfer_learning as TL 

from pathlib import Path
import collections
from data.clean_data import (clean_epoch_data, clean_combined_data, clean_intersession_test_data, 
                            clean_combined_data_for_fatigue_study, clean_correction_data, balance_correction_data, pool_correction_data,
                            convert_to_array)
from data.create_data import (create_emg_data, create_emg_epoch, create_PB_data,
                              create_PB_epoch, create_IMU_data, create_IMU_epoch,
                              create_robot_dataframe, sort_order_emg_channels)
from data.create_data_sri import read_Pos_Force_data, epoch_raw_emg, pool_emg_data

from datasets.riemann_datasets import (subject_pooled_EMG_data, 
                                       train_test_data, 
                                       subject_dependent_data,
                                       subject_pooled_EMG_PB_IMU_data,
                                       split_pooled_EMG_PB_data_train_test)

from datasets.torch_datasets import pooled_data_iterator, pooled_data_SelfCorrect_NN, pool_classifier_output_NN
from datasets.statistics_dataset import matlab_dataframe

from models.riemann_models import (tangent_space_classifier,
                                   svm_tangent_space_cross_validate,
                                   tangent_space_prediction)
from models.statistical_models import mixed_effect_model
from models.torch_models import train_torch_model, train_correction_network
from models.torch_networks import ShallowERPNet, ShallowCorrectionNet
from sklearn.model_selection import cross_val_score, KFold, train_test_split, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.covariance import ShrunkCovariance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.metrics import h_fbeta_score, multi_labeled

from features.emg_features import (extract_emg_features, pool_subject_emg_features,
                                svm_cross_validated_pooled_emg_features,
                                balance_pooled_emg_features, hudgins_features,
                                lda_cross_validated_pooled_emg_features,
                                RF_cross_validated_pooled_emg_features)

from features.force_features import (extract_passivity_index, pool_force_features)

from visualization.visualise import (plot_average_model_accuracy, plot_bar)
from utils import (skip_run, save_data, save_trained_pytorch_model, plot_confusion_matrix)

from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
import scipy  
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import treegrad as tgd 
import joblib
from tqdm import tqdm

# projection modules
from sklearn.manifold import TSNE
from umap import UMAP
from pydiffmap import diffusion_map

# The configuration file
config = yaml.load(open('src/config.yml'), Loader=yaml.SafeLoader)

##############################################################
### --------------- Preprocessing data --------------------###
##############################################################
# ------------------- Create EMG data ---------------------- #
with skip_run('skip', 'create_emg_data') as check, check():
    data = create_emg_data(config['subjects'], config['trials'], config)

    # Save the dataset
    save_path = Path(__file__).parents[1] / config['raw_emg_data']
    save_data(str(save_path), data, save=True)

with skip_run('skip', 'create_emg_epoch') as check, check():
    # file path
    read_path = Path(__file__).parents[1] / config['raw_emg_data']

    data = create_emg_epoch(config['subjects'], config['trials'], read_path, config)

    # Save the dataset
    save_path = Path(__file__).parents[1] / config['epoch_emg_data']
    save_data(str(save_path), data, save=True)
# ------------------ Create Robot data --------------------- #
with skip_run('skip', 'create_PB_data') as check, check():
    data = create_PB_data(config['subjects'], config['trials'], config)
    # save the data
    path = Path(__file__).parents[1] / config['raw_PB_data']
    save_data(str(path), data, True)

with skip_run('skip', 'create_PB_epoch') as check, check():
    data = create_PB_epoch(config['subjects'], config['trials'], config)
    
    # save the data
    path = Path(__file__).parents[1] / config['epoch_PB_data']
    dd.io.save(str(path), data)
    
with skip_run('skip', 'create_IMU_data_epoch_clean') as check, check():
    data = create_IMU_data(config['subjects'], config['trials'], config)
    # save the data
    path = Path(__file__).parents[1] / config['raw_IMU_data']
    save_data(str(path), data, True)
    
    data = create_IMU_epoch(config['subjects'], config['trials'], config)
    # save the data
    path = Path(__file__).parents[1] / config['epoch_IMU_data']
    dd.io.save(str(path), data)
    
    data = clean_epoch_data(config['subjects'], config['trials'], 'IMU', config)
    save_data(str(Path(__file__).parents[1] / config['clean_IMU_data']), data, save=True)

with skip_run('skip', 'clean_emg_epoch') as check, check():
    data = clean_epoch_data(config['subjects'], config['trials'], 'EMG', config)
    
    # Save the dataset
    if config['n_class'] == 3:
        save_path = Path(__file__).parents[1] / config['clean_emg_data_3class']
    elif config['n_class'] == 4:
        save_path = Path(__file__).parents[1] / config['clean_emg_data_4class']

    save_data(str(save_path), data, save=True)

with skip_run('skip', 'clean_PB_epoch') as check, check():
    data = clean_epoch_data(config['subjects'], config['trials'], 'PB', config)

    # Save the dataset
    if config['n_class'] == 3:
        save_path = Path(__file__).parents[1] / config['clean_PB_data_3class']
    elif config['n_class'] == 4:
        save_path = Path(__file__).parents[1] / config['clean_PB_data_4class']
    save_data(str(save_path), data, save=True)

with skip_run('skip', 'save_EMG_PB_IMU_data') as check, check():
    
    subjects = config['subjects']
    features = clean_combined_data(subjects, config['trials'], config['n_class'], config)

    # path to save
    path = str(Path(__file__).parents[1] / config['clean_emg_pb_data'])
    dd.io.save(path, features)

# NOTE: All the above files have to be run if any update to the data parameters are made #


with skip_run('skip', 'save_EMG_PB_data_for_Rakesh') as check, check():
    
    subjects = ['9007_2']
    features1 = clean_combined_data_for_fatigue_study(subjects, ['LowComb'], config['n_class'], config)
    features2 = clean_combined_data_for_fatigue_study(subjects, ['HighComb'], config['n_class'], config)

    # path to save
    path = str(Path(__file__).parents[1] / config['dataset_fatigue_study1'])
    dd.io.save(path, features1)

    path = str(Path(__file__).parents[1] / config['dataset_fatigue_study2'])
    dd.io.save(path, features2)

with skip_run('skip', 'save_EMG_PB_data_for_Dr_Ehsan') as check, check():
    
    subjects = config['subjects']
    features_raw = clean_combined_data(subjects, config['trials'], config['n_class'], config)

    features_cov = collections.defaultdict()
    features_ts  = collections.defaultdict()

    for subject in subjects:
        data1 = collections.defaultdict()
        data2 = collections.defaultdict()

        cov = Covariances().fit_transform(features_raw['subject_'+subject]['EMG'])
        ts = TangentSpace().fit_transform(cov)
        
        data1['EMG']    = cov
        data1['PB']     = features_raw['subject_'+subject]['PB']
        data1['labels'] = features_raw['subject_'+subject]['labels']

        data2['EMG']    = ts
        data2['PB']     = features_raw['subject_'+subject]['PB']
        data2['labels'] = features_raw['subject_'+subject]['labels']

        features_cov['subject_'+subject] = data1
        features_ts['subject_'+subject] = data2
    
    # path to save raw data
    file_path = str(Path(__file__).parents[1] / config['epochs_for_Dr_Ehsan'])
    scipy.io.savemat(file_path, features_raw)

    # path to save covariance data
    file_path = str(Path(__file__).parents[1] / config['covariance_for_Dr_Ehsan'])
    scipy.io.savemat(file_path, features_cov)

    # path to save tangent space features
    file_path = str(Path(__file__).parents[1] / config['tangent_feat_for_Dr_Ehsan'])
    scipy.io.savemat(file_path, features_ts)
    dd.io.save(str(Path(__file__).parents[1] / config['tangent_feat_for_Joe']), features_ts)

### --------------- Pilot analysis --------------------###
with skip_run('skip', 'create_statistics_dataframe') as check, check():
    data = create_robot_dataframe(config)

    # Save the dataset
    save_path = Path(__file__).parents[1] / config['statistics_dataframe']
    save_data(str(save_path), data, save=True)

with skip_run('skip', 'statistical_analysis') as check, check():
    dataframe = matlab_dataframe(config)

    vars = ['task + damping', 'task * damping']

    # # Perform for total force
    # for var in vars:
    #     md_task = mixed_effect_model(dataframe,
    #                                  dependent='total_force',
    #                                  independent=var)
    # Perform for velocity
    for var in vars:
        print(var)
        md_task = mixed_effect_model(dataframe,
                                     dependent='velocity',
                                     independent=var)

with skip_run('skip', 'svm_pooled_data') as check, check():
    subjects = config['subjects']
    if config['n_class'] == 3:
        save_path = Path(__file__).parents[1] / config['clean_emg_data_3class']
    elif config['n_class'] == 4:
        save_path = Path(__file__).parents[1] / config['clean_emg_data_4class']
    save_path = str(save_path)
    # Load main data
    features, labels, leave_tags = subject_pooled_EMG_data(subjects, path, config)

    # # Perform for total force
    # for var in vars:
    #     md_task = mixed_effect_model(dataframe,
    #                                  dependent='total_force',
    #                                  independent=var)
    # Perform for velocity
    for var in vars:
        print(var)
        md_task = mixed_effect_model(dataframe,
                                     dependent='velocity',
                                     independent=var)

with skip_run('skip', 'svm_pooled_riemannian_features') as check, check():
    if config['n_class'] == 3:
        save_path = str(Path(__file__).parents[1] / config['clean_emg_data_3class'])
    elif config['n_class'] == 4:
        save_path = str(Path(__file__).parents[1] / config['clean_emg_data_4class'])
    # Load main data
    features, labels, leave_tags = subject_pooled_EMG_data(config['subjects'], path, config)
    # Get the data
    data = train_test_data(features, labels, leave_tags, config)

    # Train the classifier and predict on test data
    clf = tangent_space_classifier(data['train_x'], data['train_y'], 'svc')
    tangent_space_prediction(clf, data['test_x'], data['test_y'])

with skip_run('skip', 'svm_cross_validated_pooled_data') as check, check():
    if config['n_class'] == 3:
        save_path = str(Path(__file__).parents[1] / config['clean_emg_data_3class'])
    elif config['n_class'] == 4:
        save_path = str(Path(__file__).parents[1] / config['clean_emg_data_4class'])
    # Load main data
    features, labels, leave_tags = subject_pooled_EMG_data(config['subjects'], path, config)

    # Get the data
    data = train_test_data(features, labels, leave_tags, config)
    svm_tangent_space_cross_validate(data)

with skip_run('skip', 'torch_pooled_data') as check, check():
    dataset = pooled_data_iterator(config)
    model, model_info = train_torch_model(ShallowERPNet, config, dataset)
    path = Path(__file__).parents[1] / config['trained_model_path']
    save_path = str(path)
    save_trained_pytorch_model(model, model_info, save_path, save_model=False)

with skip_run('skip', 'average_accuracy') as check, check():
    plot_average_model_accuracy('experiment_0', config)

with skip_run('skip', 'bar_plot') as check, check():
    # Get the data
    dataframe = matlab_dataframe(config)

    plt.subplots(figsize=(7, 4))
    sns.set(font_scale=1.2)

    # Force
    plt.subplot(1, 2, 1)
    dependent = 'task'
    independent = 'total_force'
    plot_bar(config, dataframe, independent, dependent)

    # Velocity
    plt.subplot(1, 2, 2)
    dependent = 'task'
    independent = 'velocity'
    plot_bar(config, dataframe, independent, dependent)

    plt.tight_layout()
    plt.show()
## ----------------------------------------------------------##


# functions added by Sri
## -----Classification with initial feature set (September results)--------##
# SVM and LDA classification of TD-EMG features #

with skip_run('skip', 'sort_order_emg_channels') as check, check():
    #FIXME: not a good idea to sort channels
    data = sort_order_emg_channels(config)

    # save the data in h5 format
    path = str(Path(__file__).parents[1] / config['emg_channel_order'])
    save_data(path,data,save=True)

with skip_run('skip', 'extract_emg_features') as check, check():
    # path to save
    path = str(Path(__file__).parents[1] / config['clean_emg_pb_data'])
    data = dd.io.load(path)

    data = extract_emg_features(data, config, scale=True)

    # save the data in h5 format
    path = str(Path(__file__).parents[1] / config['subject_emg_features'])
    save_data(path,data,save=True)

with skip_run('skip', 'pool_subject_emg_features') as check, check():

    path = str(Path(__file__).parents[1] / config['subject_emg_features'])
    data = dd.io.load(path)
    
    subjects = set(config['subjects']) ^ set(config['test_subjects'])
    X1, X2, Y = pool_subject_emg_features(data, subjects, config)

    data = {}
    data['X1'] = X1
    data['X2'] = X2
    data['Y']  = Y
  
    # save the data in h5 format
    path = str(Path(__file__).parents[1] / config['pooled_emg_features'])
    save_data(path,data,save=True)

with skip_run('skip', 'SVM_RF_cross_validated_balanced_emg_features') as check, check():
    X1_res, X2_res, Y_res = balance_pooled_emg_features(config)

    Y_res = np.argmax(Y_res, axis=1) + 1
    print('Class 1: ', Y_res[Y_res==1].shape, 'Class 2: ', Y_res[Y_res==2].shape, 'Class 3: ', Y_res[Y_res==3].shape, 'Class 4: ', Y_res[Y_res==4].shape)

    print('---Accuracy of SVM for Balanced data for feature set 1---')
    svm_cross_validated_pooled_emg_features(X1_res, Y_res, config)
    RF_cross_validated_pooled_emg_features(X1_res, Y_res, config)

    print('---Accuracy of SVM for Balanced data for feature set 2---')
    svm_cross_validated_pooled_emg_features(X2_res, Y_res, config)
    RF_cross_validated_pooled_emg_features(X2_res, Y_res, config)

with skip_run('skip', 'svm_lda_cross_validated_pooled_emg_features') as check, check():
    path = str(Path(__file__).parents[1] / config['pooled_emg_features'])
    data = dd.io.load(path)

    X1   = data['X1']
    X2   = data['X2']
    Y    = data['Y']

    # SVM for unbalanced data
    print('Class 1: ', Y[Y==1].shape, 'Class 2: ', Y[Y==2].shape, 'Class 3: ', Y[Y==3].shape, 'Class 4: ', Y[Y==4].shape)

    print('---Accuracy for Unbalanced data for feature set 1---')
    svm_cross_validated_pooled_emg_features(X1, Y, config)
    lda_cross_validated_pooled_emg_features(X1, Y, config)

    print('---Accuracy for Unbalanced data for feature set 2---')
    svm_cross_validated_pooled_emg_features(X2, Y, config)
    lda_cross_validated_pooled_emg_features(X2, Y, config)

# classifier transferability
with skip_run('skip', 'inter-session classification using Hudgins features') as check, check():
    train_subjects = list(set(config['subjects']) ^ set(config['test_subjects']))
    test_subjects  = config['test_subjects']
    
    path = str(Path(__file__).parents[1] / config['subject_emg_features'])
    data = dd.io.load(path)
    
    train_X, train_X2, train_y = pool_subject_emg_features(data, train_subjects, config)
    
    rus = RandomUnderSampler()
    train_X, train_y = rus.fit_resample(train_X, train_y)
    
    train_X2 = train_X2[rus.sample_indices_, :]

    test_X, test_X2, test_y = pool_subject_emg_features(data, test_subjects, config)
    
    train_y = np.argmax(train_y, axis=1) + 1
    test_y = np.argmax(test_y, axis=1) + 1
    
    clf1 = SVC(C=1.0, kernel='rbf', gamma='auto', decision_function_shape='ovr')
    clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)
    
    scores = clf1.fit(train_X, train_y).score(test_X, test_y)
    print('SVM Inter-session accuracy Hudgins feature set: %f ' % (scores))
    
    scores = clf2.fit(train_X, train_y).score(test_X, test_y)
    print('RF Inter-session accuracy Hudgins feature set: %f ' % (scores))
    
    scores = clf2.score(train_X, train_y)
    print('Rf training accuracy on feature set 1: %f ' % (scores))

    scores = clf1.fit(train_X2, train_y).score(test_X2, test_y)
    print('SVM Inter-session accuracy feature set2: %f ' % (scores))
    
    scores = clf2.fit(train_X2, train_y).score(test_X2, test_y)
    print('RF Inter-session accuracy feature set2: %f ' % (scores))

    
with skip_run('skip', 'classifier_transferability_across_subjects') as check, check():
    path = str(Path(__file__).parents[1] / config['subject_emg_features'])
    data = dd.io.load(path)

    acc_feat1 = []
    acc_feat2 = []

    for i in range(5):
        sub_list = copy.copy(config['subjects'])
        random.shuffle(sub_list)

        X1_train = np.empty((0,8*4))
        X2_train = np.empty((0,8*3))
        Y_train  = np.empty((0,1))

        X1_test = np.empty((0,8*4))
        X2_test = np.empty((0,8*3))
        Y_test  = np.empty((0,1))

        counter = 0
        for subject in sub_list:
            if counter < 8:
                X1_train = np.concatenate((X1_train, data['subject_'+subject]['features1']),axis=0)
                X2_train = np.concatenate((X2_train, data['subject_'+subject]['features2']),axis=0)

                temp = data['subject_'+subject]['labels']
                Y_train = np.concatenate((Y_train, np.expand_dims(temp, axis=1)),axis=0)

            else :
                X1_test = np.concatenate((X1_test, data['subject_'+subject]['features1']),axis=0)
                X2_test = np.concatenate((X2_test, data['subject_'+subject]['features2']),axis=0)

                temp = data['subject_'+subject]['labels']
                Y_test = np.concatenate((Y_test,np.expand_dims(temp, axis=1)),axis=0)
            counter += 1

        # initializing the classifier
        clf  = SVC(C=1.0, kernel='rbf', gamma='auto', decision_function_shape='ovr')
        # balance the data
        rus = RandomUnderSampler()

        # using feature set 1
        X1_tr, Y_tr = rus.fit_resample(X1_train, Y_train.flatten())
        X1_tst, Y_tst = rus.fit_resample(X1_test, Y_test.flatten())

        clf.fit(X1_tr, Y_tr)
        acc_feat1.append(clf.score(X1_tst, Y_tst))

        # using feature set 2
        X2_tr, Y_tr = rus.fit_resample(X2_train, Y_train.flatten())
        X2_tst, Y_tst = rus.fit_resample(X2_test, Y_test.flatten())

        clf.fit(X2_tr, Y_tr)
        acc_feat2.append(clf.score(X2_tst,Y_tst))

    print('Accuracy for feature set 1: %0.4f +/- %0.4f' %(statistics.mean(acc_feat1), statistics.stdev(acc_feat1)) )
    print('Accuracy for feature set 2: %0.4f +/- %0.4f' %(statistics.mean(acc_feat2), statistics.stdev(acc_feat2)) )

## ----------------------------------------------------------##


##-- Classification with 4 main sets of features RMS, TD, HIST, mDWT (October results)--##
## ----------------------------------------------------------##
# TODO:

# ------------------ Force features ------------------------#
with skip_run('skip', 'extract_force_features') as check, check():
    # Subject information
    subjects = config['subjects']

    if config['n_class'] == 3:
        save_path = str(Path(__file__).parents[1] / config['clean_PB_data_3class'])
    elif config['n_class'] == 4:
        save_path = str(Path(__file__).parents[1] / config['clean_PB_data_4class'])

    # Load main data
    dataset = extract_passivity_index(subjects, config['trials'], 'PB', config)
    features, labels = pool_force_features(dataset, config)
    
    X   = features
    y   = np.dot(labels,np.array(np.arange(1, config['n_class']+1)))
    print(X.shape)
    print('# of samples in Class 1:%d, Class 2:%d, Class 3:%d, Class 4:%d' % (y[y==1].shape[0],y[y==2].shape[0],y[y==3].shape[0], y[y==4].shape[0]))

    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')

    accuracy = cross_val_score(clf1, X, y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using SVM: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))


# -------------------- Classification --------------------- #
##-- Classification using Riemannian features--##
with skip_run('skip', 'classify_using_riemannian_emg_features_cross_validate') as check, check():
    # Subject information
    # subjects = config['train_subjects']
    subjects = set(config['subjects']) ^ set(config['test_subjects'])
    if config['n_class'] == 3:
        save_path = str(Path(__file__).parents[1] / config['clean_emg_data_3class'])
    elif config['n_class'] == 4:
        save_path = str(Path(__file__).parents[1] / config['clean_emg_data_4class'])
    # Load main data
    features, labels, leave_tags = subject_pooled_EMG_data(subjects, save_path, config)

    #FIXME:
    # ------------------Remove this later --------------------
    path = str(Path(__file__).parents[1] / config['clean_emg_pb_data'])    
    features, _ , _, labels= subject_pooled_EMG_PB_IMU_data(subjects, path, config)
    X   = features
    y   = np.dot(labels,np.array(np.arange(1, config['n_class']+1)))
    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(y[:,np.newaxis], y[:,np.newaxis])
    X = X[rus.sample_indices_, :, :]
    y = y[rus.sample_indices_]
    labels = labels[rus.sample_indices_, :]
    # -------------------------------------------------------
    # X   = features
    # y   = np.dot(labels,np.array(np.arange(1, config['n_class']+1)))
    print(X.shape)
    print('# of samples in Class 1:%d, Class 2:%d, Class 3:%d, Class 4:%d' % (y[y==1].shape[0],y[y==2].shape[0],y[y==3].shape[0], y[y==4].shape[0]))

    # estimation of the covariance matrix
    covest = Covariances().fit_transform(X)
    
    # project the covariance into the tangent space
    ts = TangentSpace().fit_transform(covest)

    # Singular Value Decomposition of covest
    # V = np.zeros((covest.shape[0], covest.shape[1] * covest.shape[2]))
    # for i in range(0, covest.shape[0]):
    #     _, _, v = np.linalg.svd(covest[i])
    #     V[i,:] = np.ravel(v, order ='F')

    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr',probability=True)
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)
    
    accuracy = cross_val_score(clf1, ts, y, cv=KFold(10,shuffle=True))
    print("cross validation accuracy using SVM: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

    accuracy = cross_val_score(clf2, ts, y, cv=KFold(10,shuffle=True))
    print("cross validation accuracy using Random Forest: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

    clf1.fit(ts, y)
    clf2.fit(ts, y)

    # print the confusion matrix of the fitted models
    print("confusion matrix of the SVM fitted model:", confusion_matrix(y, clf1.predict(ts)))
    print("confusion matrix of the RF fitted model:", confusion_matrix(y, clf2.predict(ts)))
    
    # save the model to disk
    filename_SVM = str(Path(__file__).parents[1] / config['saved_SVM_classifier'])
    filename_RF  = str(Path(__file__).parents[1] / config['saved_RF_classifier'])
    joblib.dump(clf1, filename_SVM)
    joblib.dump(clf2, filename_RF)

    # Linear discriminant analysis - # does not provide good accuracy
    # clf3 = LinearDiscriminantAnalysis(solver='svd')
    # accuracy = cross_val_score(clf3, ts, y, cv=KFold(5,shuffle=True))
    # print("cross validation accuracy using LDA: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))
    
    # Verify the accuracy using the train-test split
    data = train_test_data(X, labels, leave_tags, config)

    # Train the classifier and predict on test data
    clf = tangent_space_classifier(data['train_x'], data['train_y'], 'rf')
    tangent_space_prediction(clf, data['test_x'], data['test_y'])

with skip_run('skip', 'classify_using_PCA_reduced_riemannian_emg_features_cross_validate') as check, check():
    # Subject information
    subjects = config['subjects']
    if config['n_class'] == 3:
        save_path = str(Path(__file__).parents[1] / config['clean_emg_data_3class'])
    elif config['n_class'] == 4:
        save_path = str(Path(__file__).parents[1] / config['clean_emg_data_4class'])
    # Load main data
    features, labels, _ = subject_pooled_EMG_data(subjects, save_path, config)

    X   = features
    y   = np.dot(labels,np.array(np.arange(1, config['n_class']+1)))
    print(X.shape)
    print('# of samples in Class 1:%d, Class 2:%d, Class 3:%d, Class 4:%d' % (y[y==1].shape[0],y[y==2].shape[0],y[y==3].shape[0], y[y==4].shape[0]))

    # estimation of the covariance matrix
    covest = Covariances().fit_transform(X)

    # project the covariance into the tangent space
    ts = TangentSpace().fit_transform(covest)

    for comp in [36, 10, 8]:
        print('# PCA components: %d' %(comp))
        feat_PCA = PCA(n_components=comp, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None).fit(ts)
        cov = feat_PCA.get_covariance()
        ts = feat_PCA.transform(ts)
        print(np.diagonal(cov))

        # SVM classifier
        clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')
        # Random forest classifier
        clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)
        
        accuracy = cross_val_score(clf1, ts, y, cv=KFold(5,shuffle=True))
        print("cross validation accuracy using SVM: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

        accuracy = cross_val_score(clf2, ts, y, cv=KFold(5,shuffle=True))
        print("cross validation accuracy using Random Forest: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

with skip_run('skip', 'classify_using_riemannian_emg_features_split_train_test') as check, check():
    split_percent = 0.7 # train split percentage
    # Subject information
    subjects = config['subjects']
    if config['n_class'] == 3:
        save_path = str(Path(__file__).parents[1] / config['clean_emg_data_3class'])
    elif config['n_class'] == 4:
        save_path = str(Path(__file__).parents[1] / config['clean_emg_data_4class'])
    # Load main data
    features, labels, _ = subject_pooled_EMG_data(subjects, save_path, config)

    X   = features
    y   = np.dot(labels,np.array(np.arange(1, config['n_class']+1)))
    
    # shuffle the data
    ind = np.random.permutation(np.arange(X.shape[0]))
    X   = X[ind, :, :]
    y   = y[ind]

    print(X.shape)
    print('# of samples in Class 1:%d, Class 2:%d, Class 3:%d, Class 4:%d' % (y[y==1].shape[0],y[y==2].shape[0],y[y==3].shape[0], y[y==4].shape[0]))

    ind = round(split_percent * X.shape[0])
    train_X = X[0:ind, :, :]
    train_y = y[0:ind]

    test_X = X[ind:, :, :]
    test_y = y[ind:]

    print(test_X.shape, test_y.shape)
    # estimation of the covariance matrix and project the covariance into the tangent space
    train_covest = Covariances().fit_transform(train_X)
    train_ts = TangentSpace().fit_transform(train_covest)

    # estimation of the covariance matrix and project the covariance into the tangent space
    test_covest = Covariances().fit_transform(test_X)
    test_ts = TangentSpace().fit_transform(test_covest)

    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)
    
    model    = clf1.fit(train_ts, train_y)
    accuracy = model.score(test_ts, test_y)
    print("Accuracy using SVM: %0.4f " % (accuracy.mean()))

    model    = clf2.fit(train_ts, train_y)
    accuracy = model.score(test_ts, test_y)
    print("Accuracy using Random Forest: %0.4f " % (accuracy.mean()))

with skip_run('skip', 'classify_using_mean_force_features') as check, check():
    # Subject information
    subjects = config['subjects']

    if config['n_class'] == 3:
        save_path = str(Path(__file__).parents[1] / config['clean_PB_data_3class'])
    elif config['n_class'] == 4:
        save_path = str(Path(__file__).parents[1] / config['clean_PB_data_4class'])

    # Load main data
    features, labels, _ = subject_pooled_EMG_data(subjects, save_path, config)
    print(features.shape)

    #FIXME: Remove the following and replace it by the line in the bottom
    # X = np.zeros((features.shape[0],2))
    # for i in range(features.shape[0]):
    #     X[i, :] = np.mean(PCA().fit_transform(np.transpose(features[i,4:6,:])), axis=0)

    # use 0,1 columns for the actual force and 4,5 for the tangential and normal components
    X   = np.mean(features[:,4:6,:], axis=2)
    pos = np.mean(features[:,2:4,:], axis=2)
    y   = np.dot(labels,np.array(np.arange(1, config['n_class']+1)))
    
    print(X.shape)
    print('# of samples in Class 1:%d, Class 2:%d, Class 3:%d, Class 4:%d' % (y[y==1].shape[0],y[y==2].shape[0],y[y==3].shape[0], y[y==4].shape[0]))

    # ts = np.mean(X, axis=2)

    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)

    accuracy = cross_val_score(clf1, X, y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using SVM: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

    accuracy = cross_val_score(clf2, X, y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using Random Forest: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

with skip_run('skip', 'classify_using_riemannian_force_features') as check, check():
    # Subject information
    subjects = config['subjects']
    
    if config['n_class'] == 3:
        save_path = str(Path(__file__).parents[1] / config['clean_PB_data_3class'])
    elif config['n_class'] == 4:
        save_path = str(Path(__file__).parents[1] / config['clean_PB_data_4class'])

    # Load main data
    features, labels, _ = subject_pooled_EMG_data(subjects, save_path, config)

    X   = features[:,4:6,:]
    y   = np.dot(labels,np.array(np.arange(1, config['n_class']+1)))
    print(X.shape)
    print('# of samples in Class 1:%d, Class 2:%d, Class 3:%d' % (y[y==1].shape[0],y[y==2].shape[0],y[y==3].shape[0]))

    # estimation of the covariance matrix
    covest = Covariances().fit_transform(X)

    # project the covariance into the tangent space
    ts = TangentSpace().fit_transform(covest)

    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)

    accuracy = cross_val_score(clf1, ts, y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using SVM: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

    accuracy = cross_val_score(clf2, ts, y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using Random Forest: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

with skip_run('skip', 'classify_using_emg_and_force_features') as check, check():
    # Subject information
    subjects = config['subjects']
    
    # load the data
    path = str(Path(__file__).parents[1] / config['clean_emg_pb_data'])
    emg, pb, _, labels = subject_pooled_EMG_PB_IMU_data(subjects, path, config)

    print(emg.shape, pb.shape)

    # estimation of the force features
    features_pb = pb[:,4:5,:].mean(axis=2)

    # estimation of the EMG features using the covariance matrix projection on the tangent space
    covest = Covariances().fit_transform(emg)
    features_emg = TangentSpace().fit_transform(covest)
    
    X = np.concatenate((features_emg,features_pb), axis=1)
    y = np.dot(labels,np.array(np.arange(1, config['n_class']+1)))

    # # Load EMG data
    # if config['n_class'] == 3:
    #     path1 = str(Path(__file__).parents[1] / config['clean_emg_data_3class'])
    # elif config['n_class'] == 4:
    #     path1 = str(Path(__file__).parents[1] / config['clean_emg_data_4class'])

    # # Load PB data
    # if config['n_class'] == 3:
    #     path2 = str(Path(__file__).parents[1] / config['clean_PB_data_3class'])
    # elif config['n_class'] == 4:
    #     path2 = str(Path(__file__).parents[1] / config['clean_PB_data_4class'])
    
    # # pool the respective EMG and force features from all the subjects
    # features1, labels1, _ = subject_pooled_EMG_data(subjects, path1, config)
    # features2, labels2, _ = subject_pooled_EMG_data(subjects, path2, config)

    # y1   = np.dot(labels1,np.array(np.arange(1, config['n_class']+1)))
    # y2   = np.dot(labels2,np.array(np.arange(1, config['n_class']+1)))

    # # Remove the extra elements to match the dimensions
    # feat_len = features1.shape[0]
    # if features2.shape[0] < features1.shape[0]:
    #     feat_len = features2.shape[0]
    
    # features1   = features1[:feat_len]
    # features2   = features2[:feat_len]
    # y1          = y1[:feat_len]
    # y2          = y2[:feat_len]

    # # Using the label information, check if all the EMG and force features are aligned     
    # temp = y1 - y2
    
    # # Remove the non-matching elements
    # features1   = features1[temp == 0]
    # features2   = features2[temp == 0]
    # y1          = y1[temp == 0]
    # y2          = y2[temp == 0]

    # print('EMG samples: %d, Force samples: %d ' %(features1[:feat_len].shape[0], features2[:feat_len].shape[0])) #[0:features1.shape[0],:].shape[0]))
    # sys.exit()

    # X   = features1[:feat_len] # np.concatenate((features1, features2[0:features1.shape[0],0:2,:]), axis=1)
    # y   = np.dot(labels1,np.array(np.arange(1, config['n_class']+1)))

    # print(X.shape)
    # print('# of samples in Class 1:%d, Class 2:%d, Class 3:%d, Class 4:%d' % (y[y==1].shape[0],y[y==2].shape[0],y[y==3].shape[0],y[y==4].shape[0]))

    # # estimation of the covariance matrix
    # covest = Covariances().fit_transform(X)

    # # project the covariance into the tangent space
    # ts = TangentSpace().fit_transform(covest)

    # Combine the 
    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)

    accuracy = cross_val_score(clf1, X, y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using SVM: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

    accuracy = cross_val_score(clf2, X, y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using Random Forest: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))


# -------------- Classifier transferability ----------------- #
with skip_run('skip', 'inter_subject_transferability_using_riemannian_features') as check, check():
    # Subject information
    subjects = copy.copy(list(set(config['subjects']) ^ set(config['test_subjects'])))
    random.shuffle(subjects)

    # Number of subjects to train the classifier
    N = 15

    if config['n_class'] == 3:
        path = str(Path(__file__).parents[1] / config['clean_emg_data_3class'])
    elif config['n_class'] == 4:
        path = str(Path(__file__).parents[1] / config['clean_emg_data_4class'])

    print('List of subjects chosen for training: ', subjects[0:N])
    print('List of subjects chosen for testing : ', subjects[N:])
    # sys.exit()
    # Load main data
    train_x, train_y, _ = subject_pooled_EMG_data(subjects[0:N], path, config)
    test_x, test_y, _   = subject_pooled_EMG_data(subjects[N:], path, config)

    print('# Training samples: ', train_y.shape, '# Testing samples: ', test_y.shape)

    train_y = np.dot(train_y,np.array(np.arange(1, config['n_class']+1)))
    test_y  = np.dot(test_y,np.array(np.arange(1, config['n_class']+1)))

    # estimation of the covariance matrix and its projection in tangent space
    train_cov = Covariances().fit_transform(train_x)
    train_ts  = TangentSpace().fit_transform(train_cov)

    test_cov  = Covariances().fit_transform(test_x)
    test_ts   = TangentSpace().fit_transform(test_cov)

    # calculate the mean covariance matrix of each class
    class_mean_cov    = np.zeros((config['n_class'], config['n_electrodes'], config['n_electrodes']))
    
    for category in range(config['n_class']):
        # pick all the covariances matrices that belong to corresponding class="category+1"
        covest                          = Covariances().fit_transform(train_x[train_y == category + 1, :, :])
        class_mean_cov[category,:,:]    = mean_riemann(covest)

    train_cov_dists      = np.zeros((train_x.shape[0], config['n_class']))    
    train_cov_dist_feats = np.zeros((train_x.shape[0], config['n_class']))    
    for i in range(train_x.shape[0]):
        for category in range(config['n_class']):
            train_cov_dists[i, category] = distance_riemann(train_cov[i, :, :], class_mean_cov[category,:,:])
        
        # one hot encoding based on the minimum distance between the cov matrix and the class mean cov
        temp_ind = np.argmin(train_cov_dists[i, :])
        train_cov_dist_feats[i, temp_ind] = 1

    test_cov_dists      = np.zeros((test_x.shape[0], config['n_class']))    
    test_cov_dist_feats = np.zeros((test_x.shape[0], config['n_class']))    
    for i in range(test_x.shape[0]):
        for category in range(config['n_class']):
            test_cov_dists[i, category] = distance_riemann(test_cov[i, :, :], class_mean_cov[category,:,:])
        
        # one hot encoding based on the minimum distance between the cov matrix and the class mean cov
        temp_ind = np.argmin(test_cov_dists[i, :])
        test_cov_dist_feats[i, temp_ind] = 1
    print('_________ applying one hot encoding to the distance features _________')

    # concatenate the cov_distance features with the tangent space features
    train_ts = np.concatenate((train_ts, train_cov_dists), axis=1)
    test_ts = np.concatenate((test_ts, test_cov_dists), axis=1)

    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)

    accuracy = clf1.fit(train_ts, train_y).score(test_ts, test_y)
    print("Inter-subject tranfer accuracy using SVM: %0.4f " % accuracy.mean())

    accuracy = clf2.fit(train_ts, train_y).score(test_ts, test_y)
    print("Inter-subject tranfer accuracy using Random Forest: %0.4f " % accuracy.mean())

with skip_run('skip', 'inter_session_transferability_using_riemannian_features') as check, check():
    # Subject information
    subjects_train = list(set(config['subjects']) ^ set(config['test_subjects']))
    # subjects_train = config['train_subjects']
    subjects_test = config['test_subjects']
    print('List of subject for training: ', subjects_train)
    print('List of subject for testing : ', subjects_test)

    # clean_intersession_test_data(subjects_test, config['comb_trials'], config['n_class'], config)

    # load the data
    path = str(Path(__file__).parents[1] / config['clean_emg_pb_data'])
    
    train_emg, train_pos, train_imu, train_y = subject_pooled_EMG_PB_IMU_data(subjects_train, path, config)
    test_emg, test_pos, test_imu, test_y   = subject_pooled_EMG_PB_IMU_data(subjects_test, path, config)

    # convert the labels from one-hot-encoding to int
    train_y = np.dot(train_y,np.array(np.arange(1, config['n_class']+1)))
    test_y = np.dot(test_y,np.array(np.arange(1, config['n_class']+1)))

    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(train_y[:,np.newaxis], train_y[:,np.newaxis])

    train_emg = train_emg[rus.sample_indices_, :, :]
    train_pos = train_pos[rus.sample_indices_, :, :]
    train_imu = train_imu[rus.sample_indices_, :, :]
    train_y = train_y[rus.sample_indices_]

    print('# of samples in Class 1:%d, Class 2:%d, Class 3:%d, Class 4:%d' 
          % (train_y[train_y==1].shape[0], train_y[train_y==2].shape[0],
             train_y[train_y==3].shape[0], train_y[train_y==4].shape[0]))
    print('# of samples in Class 1:%d, Class 2:%d, Class 3:%d, Class 4:%d' 
          % (test_y[test_y==1].shape[0], test_y[test_y==2].shape[0],
             test_y[test_y==3].shape[0], test_y[test_y==4].shape[0]))

    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=100, oob_score=True) # , class_weight={1:2, 2:2, 3:1}) # class weight is not good
   
    #####----- EMG covariance matrix and its projection in tangent space
    # cov = Covariances().fit(train_emg)
    # ts  = TangentSpace().fit(cov.transform(train_emg))

    train_cov = Covariances().fit_transform(train_emg)
    train_X   = TangentSpace().fit_transform(train_cov)
    
    test_cov  = Covariances().fit_transform(test_emg)
    test_X    = TangentSpace().fit_transform(test_cov)

    # Required if you want to include the distance based features
    # # calculate the mean covariance matrix of each class
    # class_mean_cov    = np.zeros((config['n_class'], config['n_electrodes'], config['n_electrodes']))

    # for category in range(config['n_class']):
    #     # pick all the covariances matrices that belong to corresponding class="category+1"
    #     covest                          = Covariances().fit_transform(train_emg[train_y == category + 1, :, :])
    #     class_mean_cov[category,:,:]    = mean_riemann(covest)

    # train_cov_dists      = np.zeros((train_X.shape[0], config['n_class']))    
    # train_cov_dist_feats = np.zeros((train_X.shape[0], config['n_class']))    
    # for i in range(train_X.shape[0]):
    #     for category in range(config['n_class']):
    #         train_cov_dists[i, category] = distance_riemann(train_cov[i, :, :], class_mean_cov[category,:,:])
        
    #     # one hot encoding based on the minimum distance between the cov matrix and the class mean cov
    #     temp_ind = np.argmin(train_cov_dists[i, :])
    #     train_cov_dist_feats[i, temp_ind] = 1

    # test_cov_dists      = np.zeros((test_X.shape[0], config['n_class']))    
    # test_cov_dist_feats = np.zeros((test_X.shape[0], config['n_class']))    
    # for i in range(test_X.shape[0]):
    #     for category in range(config['n_class']):
    #         test_cov_dists[i, category] = distance_riemann(test_cov[i, :, :], class_mean_cov[category,:,:])
        
    #     # one hot encoding based on the minimum distance between the cov matrix and the class mean cov
    #     temp_ind = np.argmin(test_cov_dists[i, :])
    #     test_cov_dist_feats[i, temp_ind] = 1
    # print('_________ applying one hot encoding to the distance features _________')

    # # concatenate the cov_distance features with the tangent space features
    # train_X = np.concatenate((train_X, train_cov_dists), axis=1)
    # test_X = np.concatenate((test_X, test_cov_dists), axis=1)

    # print('*--------Accuracy reported from EMG features after PCA')
    # train_X = PCA(n_components=10).fit_transform(train_X)
    # test_X = PCA(n_components=10).fit_transform(test_X)

    #FIXME: RMS values are not improving the accuracy of the inter-session
    # Calculate RMS values and append it to the ts vectors
    # RMS_train = np.zeros((train_emg.shape[0],8))
    # for i in range(train_emg.shape[0]):
    #     RMS_train[i,:] = np.sqrt(np.mean(np.square(train_emg[i,:,:], dtype=float), axis=1)).T
    # train_X = np.concatenate((train_X, RMS_train), axis=1)

    # RMS_test = np.zeros((test_emg.shape[0],8))
    # for i in range(test_emg.shape[0]):
    #     RMS_test[i,:] = np.sqrt(np.mean(np.square(test_emg[i,:,:], dtype=float), axis=1)).T
    # test_X = np.concatenate((test_X, RMS_test), axis=1)

    #FIXME: Using the jerk extracted from the accelerometer readings
    # jerk_train = np.zeros((train_imu.shape[0],1))
    # for i in range(train_imu.shape[0]):
    #     jerk_train[i,:] = np.sum(np.sum(np.square(train_imu[i,:,:], dtype=float), axis=1).T) * 0.02
    # train_X = np.concatenate((train_X, jerk_train), axis=1)

    # jerk_test = np.zeros((test_imu.shape[0],1))
    # for i in range(test_imu.shape[0]):
    #     jerk_test[i,:] = np.sum(np.sum(np.square(test_imu[i,:,:], dtype=float), axis=1).T) * 0.02 # wrong formulation of jerk
    # test_X = np.concatenate((test_X, jerk_test), axis=1)

    accuracy = clf1.fit(train_X, train_y).score(test_X, test_y)    
    print("Inter-session tranfer accuracy using SVM: %0.4f " % accuracy.mean())

    accuracy = clf2.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using Random Forest: %0.4f " % accuracy.mean())

    # model selection is not yielding good results
    # model = clf2.fit(train_X, train_y).score(test_X, test_y)
    # print("I SelectFromModel(clf2, prefit=True)
    # train_X = model.transform(train_X)
    # test_X  = model.transform(test_X)
    # accuracynter-session tranfer accuracy using Random Forest: %0.4f " % accuracy.mean())

    # print the confusion matrix for the inter-session predictions
    print("Confusion matrix for inter-session classification:", confusion_matrix(test_y, clf2.fit(train_X, train_y).predict(test_X)))

    print('*--------Accuracy reported from TS space EMG features using TreeGrad')
    model = tgd.TGDClassifier(num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, autograd_config={'refit_splits':False})
    model.fit(train_X, train_y)
    print("Inter-session tranfer accuracy using Deep Neural Random Forest: ", accuracy_score(test_y, model.predict(test_X)))

    # Recursive feature elimination takes a lot of time but does not yield good results for inter-session accuracy
    # print('*--------Accuracy reported after Recursive feature elimination')
    # rfe = RFE(estimator=clf2, n_features_to_select=None, step=1)
    # rfe.fit(train_X, train_y)
    # accuracy = rfe.score(test_X, test_y)
    # print("Inter-session tranfer accuracy using RF: %0.4f " % accuracy.mean())

    # estimator = SVR(kernel="linear")
    # rfe = RFE(estimator=estimator, n_features_to_select=None, step=1)
    # rfe.fit(train_X, train_y)
    # accuracy = rfe.score(test_X, test_y)
    # print("Inter-session tranfer accuracy using SVM: %0.4f " % accuracy.mean())
    # ranking = rfe.ranking_
    # print(ranking)
    # Plot pixel ranking
    # plt.matshow(ranking, cmap=plt.cm.Blues)
    # plt.colorbar()
    # plt.title("Ranking of pixels with RFE")    

    #####------- mean Force features
    print('*--------Accuracy reported from the mean Force features:')
    train_X   = train_pos[:,0:2,:].mean(axis=2)
    test_X    = test_pos[:,0:2,:].mean(axis=2)

    accuracy = clf1.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using SVM: %0.4f " % accuracy.mean())

    accuracy = clf2.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using Random Forest: %0.4f " % accuracy.mean())

    #####------- mean of Tangent and Normal force 
    print('*--------Accuracy reported from the mean Normal and Tangent Force features:')
    train_X   = train_pos[:,4:6,:].mean(axis=2)
    test_X    = test_pos[:,4:6,:].mean(axis=2)

    accuracy = clf1.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using SVM: %0.4f " % accuracy.mean())

    accuracy = clf2.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using Random Forest: %0.4f " % accuracy.mean())

    #####------- PCA Force components  
    print('*--------Accuracy reported from the PCA Force features:')
    train_X   = train_pos[:,6:8,:].mean(axis=2)
    test_X    = test_pos[:,6:8,:].mean(axis=2)

    accuracy = clf1.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using SVM: %0.4f " % accuracy.mean())

    accuracy = clf2.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using Random Forest: %0.4f " % accuracy.mean())

    #####------ EMG + force features
    temp1 = np.array(np.linalg.norm(train_pos[:,0:2,:].mean(axis=2), axis=1)).reshape(train_cov.shape[0],1)
    temp2 = np.array(np.linalg.norm(test_pos[:,0:2,:].mean(axis=2), axis=1)).reshape(test_cov.shape[0],1)
    print('*--------Accuracy reported from the EMG + mean Force features:')
    train_X   = np.concatenate((TangentSpace().fit_transform(train_cov), temp1), axis=1)
    test_X    = np.concatenate((TangentSpace().fit_transform(test_cov),  temp2), axis=1)

    accuracy = clf1.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using SVM: %0.4f " % accuracy.mean())

    accuracy = clf2.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using Random Forest: %0.4f " % accuracy.mean())

    print('*--------Accuracy reported from the EMG + tangent Force features:')
    train_X   = np.concatenate((TangentSpace().fit_transform(train_cov), train_pos[:,4:6,:].mean(axis=2)), axis=1)
    test_X    = np.concatenate((TangentSpace().fit_transform(test_cov), test_pos[:,4:6,:].mean(axis=2)), axis=1)

    accuracy = clf1.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using SVM: %0.4f " % accuracy.mean())

    accuracy = clf2.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using Random Forest: %0.4f " % accuracy.mean())

    # plt.show()

with skip_run('skip', 'inter_task_transferability_using_riemannian_features') as check, check():
    # Subject information
    # subjects_train = list(set(config['subjects']) ^ set(config['test_subjects']))
    subjects_train = config['test_subjects'] #config['train_subjects']
    subjects_test = config['test_subjects']
    print('List of subject for training: ', subjects_train)
    print('List of subject for testing : ', subjects_test)
    
    # load the data
    filepath = str(Path(__file__).parents[1] / config['clean_emg_pb_data'])
    # training data
    train_emg, _, _, train_y = subject_pooled_EMG_PB_IMU_data(subjects_train, filepath, config)
    
    # testing data  
    test_data = clean_combined_data_for_fatigue_study(subjects_test, config['comb_trials'], config['n_class'], config)
    # Initialise for each subject
    test_emg = []
    test_y   = []
    for subject in subjects_test:
        test_emg.append(test_data['subject_' + subject]['EMG'])
        test_y.append(  test_data['subject_' + subject]['labels'])

    # Convert to array
    test_emg = np.concatenate(test_emg, axis=0)
    test_y   = np.concatenate(test_y, axis=0)

    # convert the labels from one-hot-encoding to int
    train_y = np.dot(train_y,np.array(np.arange(1, config['n_class']+1)))
    test_y = np.dot(test_y,np.array(np.arange(1, config['n_class']+1)))

    # Balance the dataset
    # rus = RandomUnderSampler()
    # rus.fit_resample(train_y[:,np.newaxis], train_y[:,np.newaxis])

    # train_emg = train_emg[rus.sample_indices_, :, :]
    # train_y = train_y[rus.sample_indices_]

    print('# of samples in Class 1:%d, Class 2:%d, Class 3:%d, Class 4:%d' 
          % (train_y[train_y==1].shape[0], train_y[train_y==2].shape[0],
             train_y[train_y==3].shape[0], train_y[train_y==4].shape[0]))
    print('# of samples in Class 1:%d, Class 2:%d, Class 3:%d, Class 4:%d' 
          % (test_y[test_y==1].shape[0], test_y[test_y==2].shape[0],
             test_y[test_y==3].shape[0], test_y[test_y==4].shape[0]))

    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)

    #####----- EMG covariance matrix and its projection in tangent space
    print('*--------Accuracy reported from the EMG features')
    train_cov = Covariances().fit_transform(train_emg)
    train_X   = TangentSpace().fit_transform(train_cov)
    
    test_cov  = Covariances().fit_transform(test_emg)
    test_X    = TangentSpace().fit_transform(test_cov)
    

    accuracy = clf1.fit(train_X, train_y).score(test_X, test_y)    
    print("Inter-session tranfer accuracy using SVM: %0.4f " % accuracy.mean())

    accuracy = clf2.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using Random Forest: %0.4f " % accuracy.mean())

    # print the confusion matrix for the inter-session predictions
    print("Confusion matrix for inter-session classification:", confusion_matrix(test_y, clf2.fit(train_X, train_y).predict(test_X)))


## ------------Project features on to manifold----------------##
with skip_run('skip', 'project_EMG_features_UMAP') as check, check():
    # Subject information
    # subjects = list(set(config['subjects']) ^ set(config['subjects2']))
    
    # Use this if you want to visualize a single subject's data
    subjects = ['9005_2']
    
    # ---------- This particular part is not used further ---------------------#
    # if config['n_class'] == 3:
    #     path = str(Path(__file__).parents[1] / config['clean_emg_data_3class'])
    # elif config['n_class'] == 4:
        # path = str(Path(__file__).parents[1] / config['clean_emg_data_4class'])
        
    # Load main data
    # features, labels, _ = subject_pooled_EMG_data(subjects, path, config)
    # -------------------------------------------------------------------------#
    
    path = str(Path(__file__).parents[1] / config['clean_emg_pb_data'])    
    features, _ , _, labels= subject_pooled_EMG_PB_IMU_data(subjects, path, config)
    
    X   = features
    y   = np.dot(labels, np.array(np.arange(1, config['n_class']+1)))
    
    # if I want to project the features of only two classes
    # X   = X[(y==1) | (y==4), :, :]
    # y   = y[(y==1) | (y==4)]

    # estimation of the covariance matrix
    covest = Covariances().fit_transform(X)

    # project the covariance into the tangent space
    ts = TangentSpace().fit_transform(covest)

    ts = np.reshape(covest, (covest.shape[0], covest.shape[1] * covest.shape[2]), order='C')
    # print(ts.shape)
    ts = ts[:, 0:36]
    
    temp1 = y == 1
    temp2 = y == 2
    temp3 = y == 3
    temp4 = y == 4
    
    print('Class 1: {}, Class 2: {}, Class 3: {}, Class 4: {}'.format(len(temp1), len(temp2), len(temp3), len(temp4)))
    
    # TSNE based projection
    # print('t-SNE based data visualization')
    # X_embedded = TSNE(n_components=2, perplexity=100, learning_rate=50.0).fit_transform(ts)

    # plt.figure()
    # plt.plot(X_embedded[temp1,0],X_embedded[temp1,1],'bo')
    # plt.plot(X_embedded[temp2,0],X_embedded[temp2,1],'ro')
    # plt.plot(X_embedded[temp3,0],X_embedded[temp3,1],'ko')
    # plt.show()

    # UMAP based projection
    # umap_fit = UMAP(n_neighbors=15, min_dist=0.75, n_components=3, metric='chebyshev')
    umap_fit = UMAP()
    X_embedded = umap_fit.fit_transform(ts)

    fig = plt.figure()
    ax = Axes3D(fig)

    # ax.scatter(X_embedded[:,0], X_embedded[:,1], X_embedded[:,2], c=y.astype(int), cmap='viridis', s=2)

    ax.plot(X_embedded[temp1,0],X_embedded[temp1,1],'b.') # ,X_embedded[temp1,2]
    ax.plot(X_embedded[temp2,0],X_embedded[temp2,1],'r.') # ,X_embedded[temp2,2]
    ax.plot(X_embedded[temp3,0],X_embedded[temp3,1],'g.') # ,X_embedded[temp3,2]
    # ax.plot(X_embedded[temp4,0],X_embedded[temp4,1],'go') # ,X_embedded[temp4,2]
        
    plt.show()

#NOTE: This is not working
with skip_run('skip', 'project_EMG_features_DiffusionMap') as check, check():
    # Subject information
    subjects = list(set(config['subjects']) ^ set(config['subjects2']))
    
    path = str(Path(__file__).parents[1] / config['clean_emg_pb_data'])    
    features, _ , _, labels= subject_pooled_EMG_PB_IMU_data(subjects, path, config)
    
    X   = features
    y   = np.dot(labels, np.array(np.arange(1, config['n_class']+1)))
    
    X = X[0:1000:2000, :, :]
    y = y[0:1000:2000]
    # if I want to project the features of only two classes
    # X   = X[(y==1) | (y==4), :, :]
    # y   = y[(y==1) | (y==4)]

    # estimation of the covariance matrix
    covest = Covariances().fit_transform(X)

    # project the covariance into the tangent space
    ts = TangentSpace().fit_transform(covest)

    ts = np.reshape(covest, (covest.shape[0], covest.shape[1] * covest.shape[2]), order='C')
    # print(ts.shape)
    ts = ts[:, 0:36]
    
    temp1 = y == 1
    temp2 = y == 2
    temp3 = y == 3
    temp4 = y == 4
    
    print('Class 1: {}, Class 2: {}, Class 3: {}, Class 4: {}'.format(len(temp1), len(temp2), len(temp3), len(temp4)))
    
    # Diffusion map based projection
    mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs = 2, epsilon = 'bgh', alpha = 0.5, k=20)
    X_embedded = mydmap.fit_transform(ts)

    print(temp1, temp2, temp3)
    print(X_embedded.shape)
    
    fig = plt.figure()
    ax = Axes3D(fig)

    # ax.scatter(X_embedded[:,0], X_embedded[:,1], X_embedded[:,2], c=y.astype(int), cmap='viridis', s=2)

    ax.plot(X_embedded[temp1,0],X_embedded[temp1,1],'bo') # ,X_embedded[temp1,2]
    ax.plot(X_embedded[temp2,0],X_embedded[temp2,1],'r*') # ,X_embedded[temp2,2]
    ax.plot(X_embedded[temp3,0],X_embedded[temp3,1],'g.') # ,X_embedded[temp3,2]
    # ax.plot(X_embedded[temp4,0],X_embedded[temp4,1],'go') # ,X_embedded[temp4,2]
        
    plt.show()


with skip_run('skip', 'project_Force_data') as check, check():
    # Subject information
    subjects = config['subjects']

    if config['n_class'] == 3:
        path = str(Path(__file__).parents[1] / config['clean_PB_data_3class'])
    elif config['n_class'] == 4:
        path = str(Path(__file__).parents[1] / config['clean_PB_data_4class'])
        
    # Load main data
    features, labels, _ = subject_pooled_EMG_data(subjects, path, config)

    X   = features[:,4:6,:]
    y   = np.dot(labels,np.array(np.arange(1, config['n_class']+1)))
    
    # y   = y[:-1]

    # estimation of the covariance matrix
    # covest = Covariances().fit_transform(X)

    # project the covariance into the tangent space
    # ts = TangentSpace().fit_transform(covest)

    # ! -----Use this only if the tangent space is not used
    # ts = np.reshape(covest, (covest.shape[0], covest.shape[1] * covest.shape[2]), order='C')
    # ts = ts[:,0:3]

    ts = np.mean(X, axis=2)

    temp1 = y == 1
    temp2 = y == 2
    temp3 = y == 3
    temp4 = y == 4

    plt.figure()
    
    plt.plot(ts[temp2,0], ts[temp2,1], 'g.')
    plt.pause(1)
    plt.plot(ts[temp3,0], ts[temp3,1], 'b.')
    plt.pause(1)    
    plt.plot(ts[temp4,0], ts[temp4,1], 'k.')
    plt.pause(1)    
    plt.plot(ts[temp1,0], ts[temp1,1], 'r.')
    
    plt.legend(['LowGross','HighGross','LowFine','HighFine'])
    # plt.show()

    # sys.exit()
    # ax = Axes3D(plt.figure())
    # ax.plot(ts[temp1,0],ts[temp1,1],ts[temp1,2],'b.')
    # ax.plot(ts[temp2,0],ts[temp2,1],ts[temp1,2],'r.')
    # ax.plot(ts[temp3,0],ts[temp3,1],ts[temp1,2],'k.')
    # ax.plot(ts[temp4,0],ts[temp4,1],ts[temp1,2],'g.')
    
    # TSNE based projection
    # print('t-SNE based data visualization')
    # X_embedded = TSNE(n_components=3, perplexity=100, learning_rate=50.0).fit_transform(ts)

    # ax = Axes3D(plt.figure())
    # ax.plot(X_embedded[temp1,0],X_embedded[temp1,1],X_embedded[temp1,2],'b.')
    # ax.plot(X_embedded[temp2,0],X_embedded[temp2,1],X_embedded[temp1,2],'r.')
    # ax.plot(X_embedded[temp3,0],X_embedded[temp3,1],X_embedded[temp1,2],'k.')
    # ax.plot(X_embedded[temp4,0],X_embedded[temp4,1],X_embedded[temp1,2],'g.')
    # plt.show()

    # UMAP based projection
    for neighbor in [10]:
        fit = UMAP(n_neighbors=neighbor, min_dist=0.25, n_components=2, metric='chebyshev')
        X_embedded = fit.fit_transform(ts)
    
        fig = plt.figure()
        # ax = Axes3D(fig)

        # ax.scatter(X_embedded[:,0], X_embedded[:,1], X_embedded[:,2], c=y.astype(int), cmap='viridis', s=2)
        # plt.plot(X_embedded[temp1,0],X_embedded[temp1,1],'bo')
        # plt.plot(X_embedded[temp2,0],X_embedded[temp2,1],'ro')
        # plt.plot(X_embedded[temp3,0],X_embedded[temp3,1],'ko')

        # ax.plot(X_embedded[temp1,0],X_embedded[temp1,1],X_embedded[temp1,2],'b.')
        # ax.plot(X_embedded[temp2,0],X_embedded[temp2,1],X_embedded[temp2,2],'r.')
        # ax.plot(X_embedded[temp3,0],X_embedded[temp3,1],X_embedded[temp3,2],'k.')
        # ax.plot(X_embedded[temp4,0],X_embedded[temp4,1],X_embedded[temp4,2],'g.')

        plt.plot(X_embedded[temp3,0],X_embedded[temp3,1],'k.')
        plt.plot(X_embedded[temp4,0],X_embedded[temp4,1],'g.')

        fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.plot(X_embedded[temp1,0],X_embedded[temp1,1],X_embedded[temp1,2],'b.')
        # ax.plot(X_embedded[temp2,0],X_embedded[temp2,1],X_embedded[temp2,2],'r.')
        plt.plot(X_embedded[temp1,0],X_embedded[temp1,1],'b.')
        plt.plot(X_embedded[temp2,0],X_embedded[temp2,1],'r.')

        fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.plot(X_embedded[temp1,0],X_embedded[temp1,1],X_embedded[temp1,2],'b.')
        # ax.plot(X_embedded[temp2,0],X_embedded[temp2,1],X_embedded[temp2,2],'r.')
        # ax.plot(X_embedded[temp3,0],X_embedded[temp3,1],X_embedded[temp3,2],'r.')
        # ax.plot(X_embedded[temp4,0],X_embedded[temp4,1],X_embedded[temp4,2],'b.')

        plt.plot(X_embedded[temp1,0],X_embedded[temp1,1],'b.')
        plt.plot(X_embedded[temp2,0],X_embedded[temp2,1],'r.')
        plt.plot(X_embedded[temp3,0],X_embedded[temp3,1],'r.')
        plt.plot(X_embedded[temp4,0],X_embedded[temp4,1],'b.')

    plt.show()


## ------------Plot the predicted labels vs position ----------------##
with skip_run('skip', 'split_pooled_subject_EMG_PB_data') as check, check():
    
    subjects = config['subjects']
    # load the data
    path = str(Path(__file__).parents[1] / config['clean_emg_pb_data'])
    Data = split_pooled_EMG_PB_data_train_test(subjects, path, config)

    # path to save the file
    path = str(Path(__file__).parents[1] / config['split_pooled_EMG_PB_data'])
    save_data(path, Data, save=True)

with skip_run('skip', 'store_predicted_and_true_labels_wrt_pos') as check, check():
    # path to save the file
    path = str(Path(__file__).parents[1] / config['split_pooled_EMG_PB_data'])
    Data = dd.io.load(path)
    
    clf = tangent_space_classifier(Data['train']['X'], Data['train']['y'], 'svc')
    predictions = tangent_space_prediction(clf, Data['test']['X'], Data['test']['y'])

    data = {}
    data['true'] = Data['test']['y']
    data['predicted'] = predictions
    data['pos']  = Data['test']['pos']

    # path to save the file
    path = str(Path(__file__).parents[1] / config['true_and_predicted_labels'])
    save_data(path, data, save=True)

with skip_run('skip', 'plot_predicted_vs_true_labels') as check, check():
    # location of the file
    path = str(Path(__file__).parents[1] / config['true_and_predicted_labels'])
    Data = dd.io.load(path)

    true_labels         = Data['true']
    predicted_labels    = Data['predicted']
    pos                 = Data['pos']

    if config['n_class'] == 4:
        classes = config['trials']
    else:
        classes = ['HF', 'LG', 'HG - LF']

    plot_confusion_matrix(true_labels, predicted_labels, classes = classes)

    indices = np.arange(0, len(true_labels))
    
    for label in range(1,config['n_class']+1):
        category = indices[true_labels == label]

        labels = true_labels[category]
        predictions = predicted_labels[category]
        temp_pos   = pos[category]
        
        temp_ind   = np.arange(0, len(labels))
        correct_pred = temp_ind[labels == predictions]
        wrong_pred  = temp_ind[labels != predictions]

        fig = plt.figure()
        ax = Axes3D(fig)             
        # ax.plot(temp_pos[correct_pred,0], temp_pos[correct_pred,1], 0 * temp_pos[correct_pred,1], 'g.')
        # ax.plot(temp_pos[wrong_pred,0], temp_pos[wrong_pred,1], 1 + 0 * temp_pos[wrong_pred,1], 'r.') 
        ax.plot(temp_pos[correct_pred,0], temp_pos[correct_pred,1], predictions[correct_pred], 'g.')
        ax.plot(temp_pos[wrong_pred,0], temp_pos[wrong_pred,1], predictions[wrong_pred], 'r.')
        ax.set_zticklabels(classes)

    plt.show()


## ----------alternative code for reading csv files----------##
with skip_run('skip', 'epoch_raw_emg_data') as check, check():
    # Subject information
    subjects = config['subjects']

    # Load main data
    X, y = epoch_raw_emg(subjects, config['trials'], config)
    Data = collections.defaultdict(dict)
    Data['X'] = X
    Data['y'] = y

    path = str(Path(__file__).parents[1] / config['raw_pooled_emg_data'])
    dd.io.save(path, Data)

with skip_run('skip', 'classify_task_riemann_features') as check, check():
    # Subject information
    subjects = config['subjects']

    path = str(Path(__file__).parents[1] / config['raw_pooled_emg_data'])
    Data = dd.io.load(path)

    X = Data['X']
    y = Data['y']

    # pool the emg epochs into a 2d-array
    # X, y = pool_emg_data(subjects, config['trials'], config)
    # print(X.shape)
    # sys.exit()
    # cohest = Coherences(window=100, overlap=0.75, fmin=None, fmax=None, fs=None).fit_transform(X)
    # for i in range(0,config['n_electrodes']-1):
    #     for j in range(i+1,config['n_electrodes']):
    #         cohest = scipy.signal.coherence(X[:,i], X[:,j], fs=200.0, window='hann', nperseg=200, noverlap=50, nfft=None, detrend='constant', axis=-1)
    #         print(cohest)
    #         sys.exit()

    # estimation of the covariance matrix
    covest = Covariances().fit_transform(X)

    # project the covariance into the tangent space
    ts = TangentSpace().fit_transform(covest)

    # save the riemannian features
    data = collections.defaultdict(dict)
    data['features'] = ts
    data['labels']   = y
    path = str(Path(__file__).parents[1] / config['pooled_riemannian_features'])
    dd.io.save(path, data)

    # balancing the data
    rus = RandomUnderSampler()
    ts, y = rus.fit_resample(ts, y)

    print('# of samples in Class 1:%d, Class 2:%d, Class 3:%d' % (y[y==1].shape[0],y[y==2].shape[0],y[y==3].shape[0]))

    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)

    accuracy = cross_val_score(clf1, ts, y.ravel(), cv=KFold(5,shuffle=True))
    print("cross validation accuracy using SVM: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

    accuracy = cross_val_score(clf2, ts, y.ravel(), cv=KFold(5,shuffle=True))
    print("cross validation accuracy using Random Forest: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))


## -----------------files to test concepts-------------------##
with skip_run('skip', 'calculate_distance_features_using_mean_covariance') as check, check():
    # Subject information
    subjects = config['subjects']
    if config['n_class'] == 3:
        save_path = str(Path(__file__).parents[1] / config['clean_emg_data_3class'])
    elif config['n_class'] == 4:
        save_path = str(Path(__file__).parents[1] / config['clean_emg_data_4class'])
    
    # Load main data
    features, labels, _ = subject_pooled_EMG_data(subjects, save_path, config)

    X   = features
    y   = np.dot(labels,np.array(np.arange(1, config['n_class']+1)))
    print(X.shape)
    print('# of samples in Class 1:%d, Class 2:%d, Class 3:%d, Class 4:%d' % (y[y==1].shape[0],y[y==2].shape[0],y[y==3].shape[0], y[y==4].shape[0]))

    # calculate the mean covariance matrix of each class
    class_mean_cov    = np.zeros((config['n_class'], config['n_electrodes'], config['n_electrodes']))
    
    for category in range(config['n_class']):
        # pick all the covariances matrices that belong to corresponding class="category+1"
        covest                          = Covariances().fit_transform(X[y == category + 1, :, :])
        class_mean_cov[category,:,:]    = mean_riemann(covest)
    
    if config['print_dist_between_mean_covs']:
        if config['n_class'] == 3:
            ind_cat = [[0,1], [0,2], [1,2]]
        else:
            ind_cat = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
        for ind in ind_cat:
            temp_dist = distance_riemann(class_mean_cov[ind[0],:,:], class_mean_cov[ind[1],:,:])
            # temp_dist = distance_logeuclid(class_mean_cov[ind[0],:,:], class_mean_cov[ind[1],:,:])

            print(ind, temp_dist)

    # estimation of the covariance matrix
    covest = Covariances().fit_transform(X)

    print('_________ applying one hot encoding to the distance features _________')
    cov_dists      = np.zeros((X.shape[0], config['n_class']))    
    cov_dist_feats = np.zeros((X.shape[0], config['n_class']))    
    for i in range(X.shape[0]):
        for category in range(config['n_class']):
            cov_dists[i, category] = distance_riemann(covest[i, :, :], class_mean_cov[category,:,:])
            # cov_dists[i, category] = distance_logeuclid(covest[i, :, :], class_mean_cov[category,:,:])

        
        # one hot encoding based on the minimum distance between the cov matrix and the class mean cov
        temp_ind = np.argmin(cov_dists[i, :])
        cov_dist_feats[i, temp_ind] = 1

    # project the covariance into the tangent space
    ts = TangentSpace().fit_transform(covest)
    
    print('#_____________Accuracy without distance features _____________#')
    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)
    # Minimum distance to mean classifier
    clf3 = MDM()

    accuracy_svm = cross_val_score(clf1, ts, y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using SVM: %0.4f (+/- %0.4f)" % (accuracy_svm.mean(), accuracy_svm.std() * 2))
    
    accuracy_rf  = cross_val_score(clf2, ts, y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using Random Forest: %0.4f (+/- %0.4f)" % (accuracy_rf.mean(), accuracy_rf.std() * 2))

    accuracy_mdm = cross_val_score(clf3, covest, y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using Minimum distance to mean (MDM): %0.4f (+/- %0.4f)" % (accuracy_mdm.mean(), accuracy_mdm.std() * 2))

    print('#_____________Accuracy with distance features _______________#')
    # concatenate the cov_distance features with the tangent space features
    ts = np.concatenate((ts, cov_dist_feats), axis=1)   

    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)
    # Minimum distance to mean classifier
    clf3 = MDM()

    accuracy_svm = cross_val_score(clf1, ts, y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using SVM: %0.4f (+/- %0.4f)" % (accuracy_svm.mean(), accuracy_svm.std() * 2))
    
    accuracy_rf  = cross_val_score(clf2, ts, y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using Random Forest: %0.4f (+/- %0.4f)" % (accuracy_rf.mean(), accuracy_rf.std() * 2))

    accuracy_mdm = cross_val_score(clf3, covest, y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using Minimum distance to mean (MDM): %0.4f (+/- %0.4f)" % (accuracy_mdm.mean(), accuracy_mdm.std() * 2))
    


# The channel selection performed for each subject.
with skip_run('skip', 'accuracy_after_channel_selection') as check, check():
    # Subject information
    subjects = config['subjects']
    if config['n_class'] == 3:
        file_path = str(Path(__file__).parents[1] / config['clean_emg_data_3class'])
    elif config['n_class'] == 4:
        file_path = str(Path(__file__).parents[1] / config['clean_emg_data_4class'])
    
    # load the data
    data = dd.io.load(file_path)

    # Subject information
    subjects = config['subjects']

    # Empty array (list)
    X = []
    y = []

    for subject in subjects:
        x_temp = data['subject_' + subject]['features']
        y_temp = data['subject_' + subject]['labels']

        y_temp = np.dot(y_temp,np.array(np.arange(1, config['n_class']+1)))

        # estimation of the covariance matrix
        covest = Covariances().fit_transform(x_temp)

        # Remove the channels using the channel selection algorithm
        selecElecs = ElectrodeSelection(nelec=6)
        mean_cov = selecElecs.fit(covest, y_temp)
        covest = mean_cov.transform(covest)

        # project the covariance into the tangent space
        ts = TangentSpace().fit_transform(covest)

        X.append(ts)
        y.append(y_temp)

    # Convert to array
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    print(y.shape)
    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(y.reshape(len(y) ,-1), y)

    # Store them in dictionary
    X = X[rus.sample_indices_, :]
    y = y[rus.sample_indices_]

    print(X.shape)
    print('# of samples in Class 1:%d, Class 2:%d, Class 3:%d, Class 4:%d' % (y[y==1].shape[0],y[y==2].shape[0],y[y==3].shape[0], y[y==4].shape[0]))

    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)

    accuracy_svm = cross_val_score(clf1, X, y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using SVM: %0.4f (+/- %0.4f)" % (accuracy_svm.mean(), accuracy_svm.std() * 2))
    
    accuracy_rf  = cross_val_score(clf2, X, y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using Random Forest: %0.4f (+/- %0.4f)" % (accuracy_rf.mean(), accuracy_rf.std() * 2))

    # accuracies_dict = collections.defaultdict()

    # for nelec in range(2, config['n_electrodes']+1):
    #     data = collections.defaultdict()

    #     # estimation of the covariance matrix
    #     covest = Covariances().fit_transform(X)

    #     # Remove the channels using the channel selection algorithm
    #     if nelec < config['n_electrodes']:
    #         selecElecs = ElectrodeSelection(nelec=nelec)
    #         mean_cov = selecElecs.fit(covest, y)
    #         covest = mean_cov.transform(covest)
    #         print(covest.shape)

    #     # project the covariance into the tangent space
    #     ts = TangentSpace().fit_transform(covest)
            
    #     # SVM classifier
    #     clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')
    #     # Random forest classifier
    #     clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)
    #     # Minimum distance to mean classifier
    #     clf3 = MDM()


    #     accuracy_svm = cross_val_score(clf1, ts, y, cv=KFold(5,shuffle=True))
    #     print("cross validation accuracy using SVM: %0.4f (+/- %0.4f)" % (accuracy_svm.mean(), accuracy_svm.std() * 2))
        
    #     accuracy_rf  = cross_val_score(clf2, ts, y, cv=KFold(5,shuffle=True))
    #     print("cross validation accuracy using Random Forest: %0.4f (+/- %0.4f)" % (accuracy_rf.mean(), accuracy_rf.std() * 2))

    #     accuracy_mdm = cross_val_score(clf3, covest, y, cv=KFold(5,shuffle=True))
    #     print("cross validation accuracy using Minimum distance to mean (MDM): %0.4f (+/- %0.4f)" % (accuracy_mdm.mean(), accuracy_mdm.std() * 2))

    #     data['SVM'] = accuracy_svm 
    #     data['RF']  = accuracy_rf
    #     data['MDM'] = accuracy_mdm

    #     accuracies_dict[nelec] = data

    # # save the accuracies of the classifiers after removing channels 
    # path = str(Path(__file__).parents[1] / config['accuracies_channel_selection'])
    # dd.io.save(path, accuracies_dict)

with skip_run('skip', 'plot_errorbar_for_accuracies_after_channel_selection') as check, check():

    #load the accuracies file
    path = str(Path(__file__).parents[1] / config['accuracies_channel_selection'])
    accuracies_dict = dd.io.load(path)

    Nelecs       = np.arange(2, config['n_electrodes']+1)
    
    accuracy_svm = np.zeros((len(Nelecs),2))
    accuracy_rf  = np.zeros((len(Nelecs),2))
    accuracy_mdm = np.zeros((len(Nelecs),2))
    # plot the accuracies
    plt.figure()
    i = 0
    while i < len(Nelecs):
        svm = accuracies_dict[Nelecs[i]]['SVM']
        rf  = accuracies_dict[Nelecs[i]]['RF']
        mdm = accuracies_dict[Nelecs[i]]['MDM']

        plt.errorbar(Nelecs[i], svm.mean(), 2 * svm.std(), ecolor='r')
        plt.errorbar(Nelecs[i], rf.mean(),  2 * rf.std(),  ecolor='b')
        plt.errorbar(Nelecs[i], mdm.mean(), 2 * mdm.std(), ecolor='k')

        accuracy_svm[i,:] = [svm.mean(), 2 * svm.std()] 
        accuracy_rf [i,:] = [rf.mean(), 2 * rf.std()]
        accuracy_mdm[i,:] = [mdm.mean(), 2 * mdm.std()]

        i += 1


    plt.plot(Nelecs, accuracy_svm[:,0], label = 'svm', color='r')
    plt.plot(Nelecs, accuracy_rf[:,0], label = 'rf', color='b')
    plt.plot(Nelecs, accuracy_mdm[:,0], label = 'mdm', color='k')

    plt.xlabel('Remaining number of electrodes')
    plt.ylabel('Accuracy')

    plt.legend()
    plt.show()

with skip_run('skip', 'classify_using_riemannian_emg_features_Deep_neural_decision_forest') as check, check():
    # Subject information
    subjects = config['subjects']
    if config['n_class'] == 3:
        save_path = str(Path(__file__).parents[1] / config['clean_emg_data_3class'])
    elif config['n_class'] == 4:
        save_path = str(Path(__file__).parents[1] / config['clean_emg_data_4class'])
    # Load main data
    features, labels, _ = subject_pooled_EMG_data(subjects, save_path, config)

    X   = features
    y   = np.dot(labels,np.array(np.arange(1, config['n_class']+1)))
    print(X.shape)
    print('# of samples in Class 1:%d, Class 2:%d, Class 3:%d, Class 4:%d' % (y[y==1].shape[0],y[y==2].shape[0],y[y==3].shape[0], y[y==4].shape[0]))

    # estimation of the covariance matrix
    covest = Covariances(estimator='oas').fit_transform(X)

    # project the covariance into the tangent space
    ts = TangentSpace().fit_transform(covest)

    mod = tgd.TGDClassifier(num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, autograd_config={'refit_splits':False})

    mod.fit(ts, y)
    print(accuracy_score(y, mod.predict(ts)))
    # accuracy = cross_val_score(clf2, ts, y, cv=KFold(5,shuffle=True))
    # print("cross validation accuracy using Random Forest: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

##-- Classification using Riemannian features--##
with skip_run('skip', 'classify_using_riemannian_emg_features_using_Shrunk_Covariance') as check, check():
    # Subject information
    subjects = config['subjects']
    if config['n_class'] == 3:
        save_path = str(Path(__file__).parents[1] / config['clean_emg_data_3class'])
    elif config['n_class'] == 4:
        save_path = str(Path(__file__).parents[1] / config['clean_emg_data_4class'])
    # Load main data
    features, labels, _ = subject_pooled_EMG_data(subjects, save_path, config)

    X   = features
    y   = np.dot(labels,np.array(np.arange(1, config['n_class']+1)))
    print(X.shape)
    print('# of samples in Class 1:%d, Class 2:%d, Class 3:%d, Class 4:%d' % (y[y==1].shape[0],y[y==2].shape[0],y[y==3].shape[0], y[y==4].shape[0]))

    # Estimating covariance using shrunk covariance approach
    covest = np.zeros((X.shape[0], config['n_electrodes'], config['n_electrodes']))
    for i in range(X.shape[0]):
        cov = ShrunkCovariance().fit(np.transpose(X[i,:,:]))
        covest[i,:,:] = cov.covariance_
    
    # project the covariance into the tangent space
    ts = TangentSpace().fit_transform(covest)

    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)
    
    accuracy = cross_val_score(clf1, ts, y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using SVM: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

    accuracy = cross_val_score(clf2, ts, y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using Random Forest: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

with skip_run('skip', 'inter_session_transferability_using_riemannian_features') as check, check():
    # Subject information
    subjects_train = list(set(config['subjects']) ^ set(config['test_subjects']))
    subjects_test = config['test_subjects']
    print('List of subject for training: ', subjects_train)
    print('List of subject for testing : ', subjects_test)

    # clean_intersession_test_data(subjects_test, config['comb_trials'], config['n_class'], config)

    # load the data
    path = str(Path(__file__).parents[1] / config['clean_emg_pb_data'])
    
    train_emg, train_pos, _, train_y = subject_pooled_EMG_PB_IMU_data(subjects_train, path, config)
    test_emg, test_pos, _, test_y   = subject_pooled_EMG_PB_IMU_data(subjects_test, path, config)

    # convert the labels from one-hot-encoding to int
    train_y = np.dot(train_y,np.array(np.arange(1, config['n_class']+1)))
    test_y = np.dot(test_y,np.array(np.arange(1, config['n_class']+1)))

    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)

    #####----- EMG covariance matrix and its projection in tangent space
    print('*--------Accuracy reported from the EMG features')
    train_cov = Covariances().fit_transform(train_emg)
    train_X   = TangentSpace().fit_transform(train_cov)
    
    test_cov  = Covariances().fit_transform(test_emg)
    test_X    = TangentSpace().fit_transform(test_cov)

    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(train_y[:,np.newaxis], train_y[:,np.newaxis])

    train_cov = train_cov[rus.sample_indices_, :, :]
    train_X = train_X[rus.sample_indices_, :]
    train_pos = train_pos[rus.sample_indices_, :, :]
    train_y = train_y[rus.sample_indices_]

    # print('*--------Accuracy reported from EMG features after PCA')
    # train_X = PCA(n_components=10).fit_transform(train_X)
    # test_X = PCA(n_components=10).fit_transform(test_X)

    # accuracy = clf1.fit(train_X, train_y).score(test_X, test_y)    
    # print("Inter-session tranfer accuracy using SVM: %0.4f " % accuracy.mean())

    # accuracy = clf2.fit(train_X, train_y).score(test_X, test_y)
    # print("Inter-session tranfer accuracy using Random Forest: %0.4f " % accuracy.mean())

    print('*--------Accuracy reported from TS space EMG features using TreeGrad')
    model = tgd.TGDClassifier(num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, autograd_config={'refit_splits':False})
    model.fit(train_X, train_y)
    print("Inter-session tranfer accuracy using Deep Neural Random Forest: ", accuracy_score(test_y, model.predict(test_X)))

    #####------- mean Force features
    print('*--------Accuracy reported from the mean Force features:')
    train_X   = train_pos[:,0:2,:].mean(axis=2)
    test_X    = test_pos[:,0:2,:].mean(axis=2)

    accuracy = clf1.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using SVM: %0.4f " % accuracy.mean())

    accuracy = clf2.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using Random Forest: %0.4f " % accuracy.mean())

    #####------- PCA based projection of forces in major-minor direction
    print('*--------Accuracy reported from the PCA projected force features:')
    train_X   = train_pos[:,0:2,:]
    test_X    = test_pos[:,0:2,:]

    temp_trn = np.zeros((train_X.shape[0],2))
    temp_tst = np.zeros((test_X.shape[0],2))
    for i in range(train_X.shape[0]):
        temp_trn[i, :] = np.mean(PCA().fit_transform(np.transpose(train_X[i,:,:])), axis=0)
    
    for i in range(test_X.shape[0]):
        temp_tst[i, :] = np.mean(PCA().fit_transform(np.transpose(test_X[i,:,:])), axis=0)

    train_X = temp_trn
    test_X  = temp_tst

    accuracy = clf1.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using SVM: %0.4f " % accuracy.mean())

    accuracy = clf2.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using Random Forest: %0.4f " % accuracy.mean())

    #####------- mean of Tangent and Normal force 
    print('*--------Accuracy reported from the mean Normal and Tangent Force features:')
    train_X   = train_pos[:,4:6,:].mean(axis=2)
    test_X    = test_pos[:,4:6,:].mean(axis=2)

    accuracy = clf1.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using SVM: %0.4f " % accuracy.mean())

    accuracy = clf2.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using Random Forest: %0.4f " % accuracy.mean())

    #####------ EMG + force features
    temp1 = np.array(np.linalg.norm(train_pos[:,0:2,:].mean(axis=2), axis=1)).reshape(train_X.shape[0],1)
    temp2 = np.array(np.linalg.norm(test_pos[:,0:2,:].mean(axis=2), axis=1)).reshape(test_X.shape[0],1)
    print('*--------Accuracy reported from the EMG + mean Force features:')
    train_X   = np.concatenate((TangentSpace().fit_transform(train_cov), temp1), axis=1)
    test_X    = np.concatenate((TangentSpace().fit_transform(test_cov),  temp2), axis=1)

    accuracy = clf1.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using SVM: %0.4f " % accuracy.mean())

    accuracy = clf2.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using Random Forest: %0.4f " % accuracy.mean())

    print('*--------Accuracy reported from the EMG + tangent Force features:')
    train_X   = np.concatenate((TangentSpace().fit_transform(train_cov), train_pos[:,4:6,:].mean(axis=2)), axis=1)
    test_X    = np.concatenate((TangentSpace().fit_transform(test_cov), test_pos[:,4:6,:].mean(axis=2)), axis=1)

    accuracy = clf1.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using SVM: %0.4f " % accuracy.mean())

    accuracy = clf2.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using Random Forest: %0.4f " % accuracy.mean())

    print('*--------Accuracy reported from the EMG + PCA projected Force features:')
    train_X   = np.concatenate((TangentSpace().fit_transform(train_cov), temp_trn), axis=1)
    test_X    = np.concatenate((TangentSpace().fit_transform(test_cov), temp_tst), axis=1)

    accuracy = clf1.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using SVM: %0.4f " % accuracy.mean())

    accuracy = clf2.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using Random Forest: %0.4f " % accuracy.mean())
    # plt.show()

# Use normalized covariance for training the classifier
with skip_run('skip', 'normalize_covariance_each_subject') as check, check():
    # define the estimator for covariance
    estimator = 'oas'
    # Subject information
    subjects_train = list(set(config['subjects']) ^ set(config['test_subjects']))
    subjects_test = config['test_subjects']
    print('List of subject for training: ', subjects_train)
    print('List of subject for testing : ', subjects_test)

    # Step 1: Calculate the covariance for each subject
    # load the raw emg data
    load_path1 = Path(__file__).parents[1] / config['raw_emg_data']
    raw_data = dd.io.load(load_path1)

    cov_data = collections.defaultdict()
    for subject in subjects_train:
        temp = np.empty((config['n_electrodes'],0))
        for trial in (set(config['trials'])^set(config['comb_trials'])):
            temp = np.concatenate((temp, raw_data['subject_'+subject]['EMG'][trial].get_data()), axis=1)
        
        cov_data['subject_'+subject] = math.sqrt(np.prod(np.diag(np.cov(temp))))

    # load the epoch EMG and PB data
    load_path2 = str(Path(__file__).parents[1] / config['clean_emg_pb_data'])
    epoch_data = dd.io.load(load_path2)

    train_X = np.empty((0,36))
    train_y = np.empty((0,))
    # Step 2: normalize the test covariances using the subject-wise covariance calculated in step 1
    for subject in subjects_train:
        emg_temp = epoch_data['subject_' + subject]['EMG']
        y_temp   = np.dot(epoch_data['subject_' + subject]['labels'],np.array(np.arange(1, config['n_class']+1)))

        # estimation of the correlation matrix
        covest = (Covariances(estimator=estimator).fit_transform(emg_temp))/cov_data['subject_'+subject]

        # project the covariance into the tangent space
        ts = TangentSpace().fit_transform(covest)

        train_X = np.concatenate((train_X, ts), axis=0)
        train_y = np.concatenate((train_y, y_temp))

    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(train_y[:,np.newaxis], train_y[:,np.newaxis])
    train_X = train_X[rus.sample_indices_, :]
    train_y = train_y[rus.sample_indices_]

    test_X = np.empty((0,36))
    test_y = np.empty((0,))
    for subject in subjects_test:
        # use the covariance calculated during first trial to normalize the test data
        temp_list = subject.split('_')
        temp_sub = temp_list[0] + '_' + '1'

        emg_temp = epoch_data['subject_' + subject]['EMG']
        y_temp   = np.dot(epoch_data['subject_' + subject]['labels'],np.array(np.arange(1, config['n_class']+1)))

        # estimation of the correlation matrix
        covest = (Covariances(estimator=estimator).fit_transform(emg_temp))/cov_data['subject_'+temp_sub]

        # project the covariance into the tangent space
        ts = TangentSpace().fit_transform(covest)

        test_X = np.concatenate((test_X, ts), axis=0)
        test_y = np.concatenate((test_y, y_temp))

    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=200, oob_score=True)
    
    # cross-validation score on the training data
    accuracy = cross_val_score(clf1, train_X, train_y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using SVM: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

    accuracy = cross_val_score(clf2, train_X, train_y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using Random Forest: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

    # inter-session accuracy
    accuracy = clf1.fit(train_X, train_y).score(test_X, test_y)    
    print("Inter-session tranfer accuracy using SVM: %0.4f " % accuracy.mean())

    accuracy = clf2.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using Random Forest: %0.4f " % accuracy.mean())


# use covariance shrinkage only on the testing data
with skip_run('skip', 'cov_shrinkage_testing_data') as check, check():
    # define the estimator for covariance
    estimator = 'oas'
    # Subject information
    subjects_train = list(set(config['subjects']) ^ set(config['test_subjects']))
    subjects_test = config['test_subjects']
    print('List of subject for training: ', subjects_train)
    print('List of subject for testing : ', subjects_test)

    # Step 1: Calculate the covariance for each subject
    # load the raw emg data
    load_path1 = Path(__file__).parents[1] / config['raw_emg_data']
    raw_data = dd.io.load(load_path1)

    # load the epoch EMG and PB data
    load_path2 = str(Path(__file__).parents[1] / config['clean_emg_pb_data'])
    epoch_data = dd.io.load(load_path2)

    train_cov = np.empty((0,8,8))
    train_X = np.empty((0,36))
    train_y = np.empty((0,))
    # Step 2: normalize the test covariances using the subject-wise covariance calculated in step 1
    for subject in subjects_train:
        emg_temp = epoch_data['subject_' + subject]['EMG']
        y_temp   = np.dot(epoch_data['subject_' + subject]['labels'],np.array(np.arange(1, config['n_class']+1)))

        # estimation of the correlation matrix
        covest = (Covariances(estimator=estimator).fit_transform(emg_temp))

        ts = covest[np.triu_indices(8)]
        # apply covariance shrinkage
        covest = Shrinkage().fit(covest).transform(covest)

        # project the covariance into the tangent space
        ts = TangentSpace(tsupdate=True).fit_transform(covest)

        train_cov = np.concatenate((train_cov, covest), axis=0)
        train_X = np.concatenate((train_X, ts), axis=0)
        train_y = np.concatenate((train_y, y_temp))

    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(train_y[:,np.newaxis], train_y[:,np.newaxis])
    train_X = train_X[rus.sample_indices_, :]
    train_y = train_y[rus.sample_indices_]
    train_cov = train_cov[rus.sample_indices_, :, :]

    test_cov = np.empty((0,8,8))
    test_X = np.empty((0,36))
    test_y = np.empty((0,))
    for subject in subjects_test:
        emg_temp = epoch_data['subject_' + subject]['EMG']
        y_temp   = np.dot(epoch_data['subject_' + subject]['labels'],np.array(np.arange(1, config['n_class']+1)))

        # estimation of the correlation matrix
        covest = (Covariances(estimator=estimator).fit_transform(emg_temp))

        # apply covariance shrinkage
        covest = Shrinkage().fit(covest).transform(covest)

        # project the covariance into the tangent space
        ts = TangentSpace(tsupdate=True).fit_transform(covest)

        test_cov = np.concatenate((test_cov, covest), axis=0)
        test_X = np.concatenate((test_X, ts), axis=0)
        test_y = np.concatenate((test_y, y_temp))

    # predicted_y = TSclassifier(tsupdate=True).fit(train_cov, train_y).predict(test_cov)
    # print(accuracy_score(test_y, predicted_y))

    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=200, oob_score=True)
    
    # cross-validation score on the training data
    accuracy = cross_val_score(clf1, train_X, train_y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using SVM: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

    accuracy = cross_val_score(clf2, train_X, train_y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using Random Forest: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

    # inter-session accuracy
    accuracy = clf1.fit(train_X, train_y).score(test_X, test_y)    
    print("Inter-session tranfer accuracy using SVM: %0.4f " % accuracy.mean())

    accuracy = clf2.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using Random Forest: %0.4f " % accuracy.mean())

# Apply Source Power Comodulation filter to the covariance matrix
with skip_run('skip', 'classify_after_SPoC_filter') as check, check():
    
    # Subject information
    subjects_train = list(set(config['subjects']) ^ set(config['test_subjects']))
    subjects_test = config['test_subjects']
    print('List of subject for training: ', subjects_train)
    print('List of subject for testing : ', subjects_test)

    # load the epoch EMG and PB data
    load_path2 = str(Path(__file__).parents[1] / config['clean_emg_pb_data'])
    epoch_data = dd.io.load(load_path2)

    train_X = np.empty((0,4))
    train_y = np.empty((0,))

    for subject in subjects_train:
        emg_temp = epoch_data['subject_' + subject]['EMG']
        y_temp   = np.dot(epoch_data['subject_' + subject]['labels'],np.array(np.arange(1, config['n_class']+1)))

        # estimation of the correlation matrix
        covest = (Covariances().fit_transform(emg_temp))

        # project the covariance into the tangent space
        spoc = SPoC().fit(covest, y_temp)
        ts   = spoc.transform(covest)

        # Common Spatial Patterns
        # csp = CSP().fit(covest, y_temp)
        # ts   = csp.transform(covest)

        train_X = np.concatenate((train_X, ts), axis=0)
        train_y   = np.concatenate((train_y, y_temp))

    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(train_y[:,np.newaxis], train_y[:,np.newaxis])
    train_X = train_X[rus.sample_indices_, :]
    train_y = train_y[rus.sample_indices_]

    test_X = np.empty((0,4))
    test_y = np.empty((0,))
    for subject in subjects_test:
        # use the covariance calculated during first trial to normalize the test data
        temp_list = subject.split('_')
        temp_sub = temp_list[0] + '_' + '1'

        emg_temp = epoch_data['subject_' + subject]['EMG']
        y_temp   = np.dot(epoch_data['subject_' + subject]['labels'],np.array(np.arange(1, config['n_class']+1)))

        # estimation of the correlation matrix
        covest = (Covariances().fit_transform(emg_temp))#/cov_data['subject_'+temp_sub]

        # project the covariance into the tangent space
        spoc = SPoC().fit(covest, y_temp)
        ts   = spoc.transform(covest)

        # Common Spatial Patterns
        # csp = CSP().fit(covest, y_temp)
        # ts   = csp.transform(covest)

        test_X = np.concatenate((test_X, ts), axis=0)
        test_y = np.concatenate((test_y, y_temp))

    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=200, oob_score=True)
    
    # cross-validation score on the training data
    accuracy = cross_val_score(clf1, train_X, train_y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using SVM: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

    accuracy = cross_val_score(clf2, train_X, train_y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using Random Forest: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

    # inter-session accuracy
    accuracy = clf1.fit(train_X, train_y).score(test_X, test_y)    
    print("Inter-session tranfer accuracy using SVM: %0.4f " % accuracy.mean())

    accuracy = clf2.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using Random Forest: %0.4f " % accuracy.mean())
 

# Apply the self-correction algorithm to the classifier output
#-- Save the unpooled data
with skip_run('skip', 'save_subjectwise_data_for_correcting_predictions') as check, check():
    
    data = clean_correction_data(config['subjects'], config['trials'], config['n_class'], config)
    
    # path to save the file
    filepath = str(Path(__file__).parents[1] / config['subject_data_pred_correction'])
    dd.io.save(filepath, data)

#-- Save the weights of the RF classifier--#
with skip_run('skip', 'save_RF_classifier_to_train_correction_NN') as check, check():
    trials = list(set(config['trials']) - set(config['comb_trials']))

    # path to load the file
    filepath = str(Path(__file__).parents[1] / config['subject_data_pred_correction'])
    data = dd.io.load(filepath)

    #TODO: change the balance flag if the data is supposed to be balanced
    data = balance_correction_data(data, trials, config['subjects'], config, balance=True)

    # -----------------------------Training the Random Forest classifier----------------------------------- #
    # TODO: Keep in mind that the subjects used for training the RF classifier don't have the second sessions 
    subjects = list(set(config['subjects']) ^ set(config['test_subjects']))
    # extract the data
    features, _, labels = pool_correction_data(data, subjects, trials, config)

    X   = features
    y   = np.dot(labels,np.array(np.arange(1, config['n_class']+1)))

    print(X.shape)
    print('# of samples in Class 1:%d, Class 2:%d, Class 3:%d, Class 4:%d' % (y[y==1].shape[0],y[y==2].shape[0],y[y==3].shape[0], y[y==4].shape[0]))

    # estimation of the covariance matrix
    covest = Covariances().fit_transform(X)
    
    # project the covariance into the tangent space
    ts = TangentSpace().fit_transform(covest)

    # Random forest classifier
    # clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)
    clf2 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr',probability=True)
    
    
    accuracy = cross_val_score(clf2, ts, y, cv=KFold(10,shuffle=True))
    print("cross validation accuracy using RF: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

    clf2.fit(ts, y)
    accuracy = clf2.score(ts, y)
    print("Training accuracy of  Random Forest: %0.4f" % (accuracy))
    
    # ------------------------------------------------------------------------------------------------------ #

    # -----------------------------Testing the Random Forest classifier----------------------------------- #
    # extract the data
    features, _, labels = pool_correction_data(data, config['test_subjects'], trials, config)

    X   = features
    y   = np.dot(labels,np.array(np.arange(1, config['n_class']+1)))

    print(X.shape)
    print('# of samples in Testing, Class 1:%d, Class 2:%d, Class 3:%d, Class 4:%d' % (y[y==1].shape[0],y[y==2].shape[0],y[y==3].shape[0], y[y==4].shape[0]))

    # estimation of the covariance matrix
    covest = Covariances().fit_transform(X)
    
    # project the covariance into the tangent space
    ts = TangentSpace().fit_transform(covest)

    accuracy = clf2.score(ts, y)
    print("Testing accuracy of Random Forest: %0.4f" % (accuracy))

    # save the model to disk
    filename = str(Path(__file__).parents[1] / config['saved_RF_classifier'])
    joblib.dump(clf2, filename)

with skip_run('skip', 'save_dataset_for_training_NeuralNet') as check, check():
    # load the classifier 
    filename = str(Path(__file__).parents[1] / config['saved_RF_classifier'])
    RF_clf   = joblib.load(filename)

    trials = list(set(config['trials']) - set(config['comb_trials']))

    # path to load the file
    filepath = str(Path(__file__).parents[1] / config['subject_data_pred_correction'])
    data = dd.io.load(filepath)

    #TODO: change the balance flag if the data is supposed to be balanced
    data = balance_correction_data(data, trials, config['subjects'], config, balance=True)

    # create the dataset for NN
    dataset = pooled_data_SelfCorrect_NN(data, RF_clf, config)

    # save the dataset
    filepath = Path(__file__).parents[1] / config['Self_correction_dataset']
    dd.io.save(filepath, dataset)

with skip_run('skip', 'train_correction_classifier_output') as check, check():
    
    # This is required for the code to run in windows
    if __name__ == '__main__':
        # torch.multiprocessing.freeze_support() # this is not requird 
        # load the dataset
        filepath = Path(__file__).parents[1] / config['Self_correction_dataset']
        dataset  = dd.io.load(filepath)

        # train the self-correction network
        model, model_info = train_correction_network(ShallowCorrectionNet, config, dataset)

        # save the model
        path = Path(__file__).parents[1] / config['corrNet_trained_model_path']
        save_path = str(path)
        save_trained_pytorch_model(model, model_info, save_path, save_model=True)

with skip_run('skip', 'Correct_interSession_prediction_using_median') as check, check():

    # path to load the file
    filepath = str(Path(__file__).parents[1] / config['subject_data_pred_correction'])
    data = dd.io.load(filepath)

    trials = list(set(config['trials']) - set(config['comb_trials']))

    # load the classifier 
    filename = str(Path(__file__).parents[1] / config['saved_SVM_classifier']) # SVM
    # filename = str(Path(__file__).parents[1] / config['saved_RF_classifier']) # RF
    RF_clf   = joblib.load(filename)

    #-------Prepare the training data 
    subjects = set(config['subjects']) ^ set(config['test_subjects'])
    data = balance_correction_data(data, trials, subjects, config, balance=True)
    
    # extract the data
    features_train, _, _ = pool_correction_data(data, config['test_subjects'], trials, config)

    # estimation of the covariance matrix using the training data and use it to estimate testing data
    cov = Covariances().fit(features_train)
    covest = cov.transform(features_train)
    
    # project the covariance into the tangent space
    tspace = TangentSpace().fit(covest)
    
    # load the test data
    data = balance_correction_data(data, trials, config['test_subjects'], config, balance=False)
    
    
    plt.figure()
    for wind_len in range(2, 11, 2):

        # wind_len = 6 # config['SELF_CORRECTION_NN']['WIN_LEN']
        act_pred   = []
        corr_pred1 = []
        corr_pred2 = []
        total_test_categ = []

        for subject in config['test_subjects']:
            for trial in list(set(config['trials']) ^ set(config['comb_trials'])):
                #FIXME: A separate covariance instance is not performing good
                # temp_features = TangentSpace().fit_transform(Covariances().fit_transform(data['subject_' + subject][trial]['EMG']))

                # covariance instance fit on the training data is good to be used on the testing data
                temp_features = tspace.transform(cov.transform(data['subject_' + subject][trial]['EMG']))

                temp_labels   = data['subject_' + subject][trial]['labels']
                # print(subject, trial, RF_clf.score(temp_features, np.dot(temp_labels,np.array(np.arange(1, config['n_class']+1)))))

                New_pred = RF_clf.predict(temp_features[:, :]) - 1
                for i in range(wind_len, temp_labels.shape[0]):
                    curr_pred   = RF_clf.predict_log_proba(temp_features[i, :].reshape(1, -1))
                    prob        = RF_clf.predict_log_proba(temp_features[i-wind_len:i+1, :])
                    pred        = int(collections.Counter(RF_clf.predict(temp_features[i-wind_len:i+1, :])).most_common(1)[0][0]) - 1
                    
                    # FIXME: this is not working
                    # pred        = int(collections.Counter(New_pred[i-wind_len:i+1]).most_common(1)[0][0])                    
                    # New_pred[i] = pred
                    
                    act_pred.append(np.argmax(curr_pred))
                    corr_pred1.append(np.argmax(np.median(prob, axis=0)))
                    corr_pred2.append(pred) 
                    total_test_categ.append(np.argmax(temp_labels[i, :]))
                
        # print(confusion_matrix(total_test_categ, act_pred), confusion_matrix(total_test_categ, corr_pred1), confusion_matrix(total_test_categ, corr_pred2))
        print(accuracy_score(total_test_categ, act_pred), accuracy_score(total_test_categ, corr_pred1), accuracy_score(total_test_categ, corr_pred2))
        # print(accuracy_score(total_test_categ, corr_pred1), accuracy_score(total_test_categ, corr_pred2))   

        plt.plot(wind_len, accuracy_score(total_test_categ, corr_pred1), 'r*')
        plt.plot(wind_len, accuracy_score(total_test_categ, corr_pred2), 'b*')
        plt.plot(wind_len, accuracy_score(total_test_categ, act_pred), 'k*')
    
    plt.xlabel('History of epochs used')
    plt.ylabel('Accuracy')
    plt.title('Inter-Session accuracy')
    # plt.show()
    
with skip_run('skip', 'Correct_interTask_prediction_using_median') as check, check():
    
    # path to load the file
    filepath = str(Path(__file__).parents[1] / config['subject_data_pred_correction'])
    data = dd.io.load(filepath)

    #-------Prepare the training data
    subjects = set(config['subjects']) ^ set(config['test_subjects'])
    trials = list(set(config['trials']) - set(config['comb_trials']))
    
    data = balance_correction_data(data, trials, subjects, config, balance=True)
    
    # extract the data
    features_train, _, _ = pool_correction_data(data, config['test_subjects'], trials, config)
    
    # fit the Riemannian predictor using the training data
    cov = Covariances().fit(features_train)
    covest = cov.transform(features_train)    
    # project the covariance into the tangent space
    tspace = TangentSpace().fit(covest)
    
    
    # load the classifier 
    filename = str(Path(__file__).parents[1] / config['saved_RF_classifier'])
    RF_clf   = joblib.load(filename)

    # -------- Prepare the testing data
    test_data = clean_correction_data(config['test_subjects'], config['comb_trials'], config['n_class'], config)
    # test_data, _, _ = pool_correction_data(test_data, config['test_subjects'], config['comb_trials'], config)
    
    plt.figure()
    for wind_len in range(2, 11, 2):

        # wind_len = 6 # config['SELF_CORRECTION_NN']['WIN_LEN']
        act_pred   = []
        corr_pred1 = []
        corr_pred2 = []
        total_test_categ = []

        for subject in config['test_subjects']:
            for trial in config['comb_trials']:
                 #FIXME: A separate covariance instance is performing good for inter-task unlike intersession
                temp_features = TangentSpace().fit_transform(Covariances().fit_transform(test_data['subject_' + subject][trial]['EMG']))

                # covariance instance fit on the training data is good to be used on the testing data
                # temp_features = tspace.transform(cov.transform(test_data['subject_' + subject][trial]['EMG']))

                temp_labels   = test_data['subject_' + subject][trial]['labels']
                # print(subject, trial, RF_clf.score(temp_features, np.dot(temp_labels,np.array(np.arange(1, config['n_class']+1)))))

                New_pred = RF_clf.predict_log_proba(temp_features[:, :]) - 1
                for i in range(wind_len, temp_labels.shape[0]):
                    curr_pred   = RF_clf.predict_log_proba(temp_features[i, :].reshape(1, -1))
                    prob        = RF_clf.predict_log_proba(temp_features[i-wind_len:i+1, :])
                    pred        = int(collections.Counter(RF_clf.predict(temp_features[i-wind_len:i+1, :])).most_common(1)[0][0]) - 1
                    
                    #FIXME: using the corrected prediction for future is not working
                    # pred        = int(collections.Counter(New_pred[i-wind_len:i+1]).most_common(1)[0][0])
                    # New_pred[i] = pred
                    
                    act_pred.append(np.argmax(curr_pred))
                    corr_pred1.append(np.argmax(np.median(prob, axis=0)))
                    corr_pred2.append(pred) 
                    total_test_categ.append(np.argmax(temp_labels[i, :]))

        print(confusion_matrix(total_test_categ, act_pred), confusion_matrix(total_test_categ, corr_pred1), confusion_matrix(total_test_categ, corr_pred2))
        # print(accuracy_score(total_test_categ, corr_pred1), accuracy_score(total_test_categ, corr_pred2))  

        plt.plot(wind_len, accuracy_score(total_test_categ, corr_pred1), 'r*')
        plt.plot(wind_len, accuracy_score(total_test_categ, corr_pred2), 'b*')
        plt.plot(wind_len, accuracy_score(total_test_categ, act_pred), 'k*')
    
    plt.xlabel('History of epochs used')
    plt.ylabel('Accuracy')
    plt.title('Inter-Task accuracy')
    
    # plt.show()

# Hierarchical classification approach
with skip_run('skip', 'Hierarchical classification') as check, check():
    
    # Subject information
    subjects_train = list(set(config['subjects']) ^ set(config['test_subjects']))
    subjects_test = config['test_subjects']
    
    print('List of subject for training: ', subjects_train)
    print('List of subject for testing : ', subjects_test)

    # load the data
    path = str(Path(__file__).parents[1] / config['clean_emg_pb_data'])
    
    train_emg, _, _, train_y = subject_pooled_EMG_PB_IMU_data(subjects_train, path, config)
    test_emg, _, _, test_y   = subject_pooled_EMG_PB_IMU_data(subjects_test, path, config)

    # convert the labels from one-hot-encoding to int
    train_y = np.dot(train_y,np.array(np.arange(1, config['n_class']+1))).astype(int)
    test_y = np.dot(test_y,np.array(np.arange(1, config['n_class']+1))).astype(int)

    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(train_y[:,np.newaxis], train_y[:,np.newaxis])

    train_emg = train_emg[rus.sample_indices_, :, :]
    train_y = train_y[rus.sample_indices_]
    
    #####----- EMG covariance matrix and its projection in tangent space
    train_cov = Covariances().fit_transform(train_emg)
    train_X   = TangentSpace().fit_transform(train_cov)
    
    test_cov  = Covariances().fit_transform(test_emg)
    test_X    = TangentSpace().fit_transform(test_cov)

    print('# of samples in Class 1:%d, Class 2:%d, Class 3:%d, Class 4:%d' 
          % (train_y[train_y==1].shape[0],train_y[train_y==2].shape[0],train_y[train_y==3].shape[0], train_y[train_y==4].shape[0]))

    # convert the labels to strings for compatibility with the hierarchy    
    train_y = train_y.astype(str)
    test_y  = test_y.astype(str) 
    
    print(test_y.shape, type(test_y[0]), test_y)
    #FIXME: uncomment this for task hierarchy Fine, G
    class_hierarchy = {ROOT: ["Fine", "Gross"],
                       "Fine": ["1", "4"],
                       "Gross": ["2", "3"],
                       }
    
    # class_hierarchy = {ROOT: ["Low", "High"],
    #                    "Low": ["2", "4"],
    #                    "High": ["1", "3"],
    #                    }
    
    base_estimator = make_pipeline(
                                    TruncatedSVD(n_components=24),
                                    SVC(gamma=0.001, kernel="rbf", probability=True), # use either this or 
                                    # RandomForestClassifier(n_estimators=100, oob_score=True), # this
                                    )
    
    clf = HierarchicalClassifier(base_estimator=base_estimator, class_hierarchy=class_hierarchy)
    
    
    clf.fit(train_X, train_y)
    pred_y = clf.predict(test_X)

    print("Classification Report:\n", classification_report(test_y, pred_y))

    # Demonstrate using our hierarchical metrics module with MLB wrapper
    with multi_labeled(test_y, pred_y, clf.graph_) as (y_test_, y_pred_, graph_):
        h_fbeta = h_fbeta_score(
            y_test_,
            y_pred_,
            graph_,
        )
        print("h_fbeta_score: ", h_fbeta)


# FIXME: tested correcting the wrongly predicted class 3 predictions using RMS values 
# but the method did not work
with skip_run('skip', 'Some_crazy_little_Idea_hopeit_does_some_good') as check, check():

    # path to load the file
    filepath = str(Path(__file__).parents[1] / config['subject_data_pred_correction'])
    data = dd.io.load(filepath)

    trials = list(set(config['trials']) - set(config['comb_trials']))

    # load the classifier 
    filename = str(Path(__file__).parents[1] / config['saved_RF_classifier'])
    RF_clf   = joblib.load(filename)

    #-------load the testing data
    subjects = set(config['subjects']) ^ set(config['test_subjects'])
    data = balance_correction_data(data, trials, subjects, config, balance=True)
    
    # Extract RMS values
    RMS_array = collections.defaultdict()
    for subject in config['test_subjects']:
        rms_max = 0
        rms_data = collections.defaultdict()
        for trial in trials:
            rms_array = []
            for sample in data['subject_'+subject][trial]['EMG']:
                temp = np.mean(np.sqrt(np.mean(np.square(sample), axis=1)), axis=0)
                rms_array.append(temp)
            rms_array = np.array(rms_array).reshape(-1,1)
            
            if rms_max < np.max(rms_array):
                rms_max = np.max(rms_array)
            rms_data[trial] = rms_array
        
        for trial in trials:
            rms_data[trial] = rms_data[trial] / rms_max
            
        RMS_array['subject_'+subject] = rms_data
         
    
    # extract the data
    features_test, _, _ = pool_correction_data(data, config['test_subjects'], trials, config)

    # estimation of the covariance matrix
    cov = Covariances().fit(features_test)
    covest = cov.transform(features_test)
    
    # project the covariance into the tangent space
    tspace = TangentSpace().fit(covest)
    
    # load the test data
    data = balance_correction_data(data, trials, config['test_subjects'], config, balance=False)
    
    
    plt.figure()
    for wind_len in range(2, 11, 2):

        # wind_len = 6 # config['SELF_CORRECTION_NN']['WIN_LEN']
        act_pred   = []
        corr_pred1 = []
        corr_pred2 = []
        total_test_categ = []

        for subject in config['test_subjects']:
            for trial in list(set(config['trials']) ^ set(config['comb_trials'])):
                temp_features = TangentSpace().fit_transform(Covariances().fit_transform(data['subject_' + subject][trial]['EMG']))
                # temp_features = tspace.transform(cov.transform(data['subject_' + subject][trial]['EMG']))
                temp_labels   = data['subject_' + subject][trial]['labels']
                # print(subject, trial, RF_clf.score(temp_features, np.dot(temp_labels,np.array(np.arange(1, config['n_class']+1)))))
                
                for i in range(wind_len, temp_labels.shape[0]):
                    curr_pred   = int(RF_clf.predict(temp_features[i, :].reshape(1, -1))) - 1
                    # prob        = RF_clf.predict_log_proba(temp_features[i-wind_len:i+1, :])
                    pred        = int(collections.Counter(RF_clf.predict(temp_features[i-wind_len:i+1, :])).most_common(1)[0][0]) - 1
                    
                    
                    # corr_pred1.append(np.argmax(np.median(prob, axis=0)))
                    corr_pred2.append(pred) 
                    total_test_categ.append(np.argmax(temp_labels[i, :]))
                    
                    if (pred == 2) and ( RMS_array['subject_'+subject][trial][i,0] < 0.25):
                        curr_pred = 0
                    
                    act_pred.append(curr_pred)
                    rms_temp = np.sqrt(np.mean(np.square(data['subject_' + subject][trial]['EMG'][i, :, :]), axis=1))
                    
                    # if (curr_pred == 2) and (np.argmax(temp_labels[i, :]) == 0):
                    #     print('Class 1 wrongly predicted as 3, RMS:', RMS_array['subject_'+subject][trial][i,0])
                    # elif (curr_pred == 2) and (np.argmax(temp_labels[i, :]) == 2):
                    #     print('Class 3 and prediction 3, RMS:', RMS_array['subject_'+subject][trial][i,0])
                    

        print(confusion_matrix(total_test_categ, act_pred)) #, confusion_matrix(total_test_categ, corr_pred1), confusion_matrix(total_test_categ, corr_pred2))
        # print(accuracy_score(total_test_categ, corr_pred1), accuracy_score(total_test_categ, corr_pred2))   

        # plt.plot(wind_len, accuracy_score(total_test_categ, corr_pred1), 'r*')
        plt.plot(wind_len, accuracy_score(total_test_categ, corr_pred2), 'b*')
        plt.plot(wind_len, accuracy_score(total_test_categ, act_pred), 'k*')
    
    plt.xlabel('History of epochs used')
    plt.ylabel('Accuracy')
    plt.title('Inter-Session accuracy')



#######################################
#IEEE SMC
#######################################
with skip_run('skip', 'train_test_split_data') as check, check():
    # path to load the data
    path = str(Path(__file__).parents[1] / config['clean_emg_pb_data'])
    data = dd.io.load(path)

    subjects = list(set(config['subjects']) ^ set(config['test_subjects']))
    EMG = []
    PB  = []
    y   = []

    for subject in subjects:
        EMG.append(data['subject_' + subject]['EMG'])
        PB.append( data['subject_' + subject]['PB'])
        y.append(  data['subject_' + subject]['labels'])

    # Convert to array
    EMG = np.concatenate(EMG, axis=0)
    PB  = np.concatenate(PB, axis=0)
    y   = np.concatenate(y, axis=0)

    Data = collections.defaultdict()

    id = np.arange(EMG.shape[0])
    train_id, test_id, _, _ = train_test_split(id, id * 0, test_size=0.25, random_state=42)

    train_X = EMG[train_id, :, :]
    train_pb = PB[train_id, :, :]
    train_y = y[train_id, :]

    rus = RandomUnderSampler(random_state=42)
    rus.fit_resample(train_y, train_y)

    print(len(rus.sample_indices_))

    train_X = train_X[rus.sample_indices_, :, :]
    train_pb = train_pb[rus.sample_indices_, :, :]
    train_y = train_y[rus.sample_indices_,:]

    # Training
    Data['train_x'] = train_X
    Data['train_pb'] = train_pb
    Data['train_y'] = train_y

    # Testing
    Data['test_x'] = EMG[test_id, :, :]
    Data['test_pb'] = PB[test_id, :, :]
    Data['test_y'] = y[test_id, :]


    EMG_sess = []
    PB_sess = []
    y_sess = []

    for subject in config['test_subjects']:
        EMG_sess.append(data['subject_' + subject]['EMG'])
        PB_sess.append( data['subject_' + subject]['PB'])
        y_sess.append(  data['subject_' + subject]['labels'])

    # Convert to array
    EMG_sess = np.concatenate(EMG_sess, axis=0)
    PB_sess  = np.concatenate(PB_sess, axis=0)
    y_sess   = np.concatenate(y_sess, axis=0)

    # session 
    Data['session_x'] = EMG_sess
    Data['session_pb'] = PB_sess
    Data['session_y'] = y_sess

    dd.io.save(Path(__file__).parents[1] / config['train_test_split_dataset'], Data)


with skip_run('skip', 'extract_emg_features_train_test_split') as check, check():
    # save the data in h5 format
    path = str(Path(__file__).parents[1] / config['subject_emg_features'])
    data = dd.io.load(path)

    subjects = list(set(config['subjects']) ^ set(config['test_subjects']))
    TD_feat = []
    y       = []

    for subject in subjects:
        TD_feat.append(data['subject_' + subject]['features1'])
        y.append(data['subject_' + subject]['labels'])

    # Convert to array
    TD_feat = np.concatenate(TD_feat, axis=0)
    y       = np.concatenate(y, axis=0)

    Data = collections.defaultdict()

    id = np.arange(TD_feat.shape[0])
    
    train_id, test_id, _, _ = train_test_split(id, id * 0, test_size=0.25, random_state=42)

    train_X = TD_feat[train_id, :]
    train_y = y[train_id, :]

    rus = RandomUnderSampler(random_state=42)
    rus.fit_resample(train_y, train_y)

    print(len(rus.sample_indices_))
    train_X = train_X[rus.sample_indices_, :]
    train_y = train_y[rus.sample_indices_,:]

    # Training
    Data['train_x'] = train_X
    Data['train_y'] = train_y

    # Testing
    Data['test_x'] = TD_feat[test_id, :]
    Data['test_y'] = y[test_id, :]

    TD_sess = []
    y_sess = []

    for subject in config['test_subjects']:
        TD_sess.append(data['subject_' + subject]['features1'])
        y_sess.append(  data['subject_' + subject]['labels'])

    # Convert to array
    TD_sess = np.concatenate(TD_sess, axis=0)
    y_sess   = np.concatenate(y_sess, axis=0)

    # session 
    Data['session_x'] = TD_sess
    Data['session_y'] = y_sess

    dd.io.save(Path(__file__).parents[1] / config['train_test_split_TD_features'], Data)


with skip_run('skip', 'extract_riemann_features_train_test_split') as check, check():
    # load the data
    data = dd.io.load(Path(__file__).parents[1] / config['train_test_split_dataset'])


    cov = Covariances().fit_transform(data['train_x'])
    train_ts  = TangentSpace().fit_transform(cov)

    cov = Covariances().fit_transform(data['test_x'])
    test_ts  = TangentSpace().fit_transform(cov)

    cov = Covariances().fit_transform(data['session_x'])
    session_ts  = TangentSpace().fit_transform(cov)

    Data = collections.defaultdict()

    # Training
    Data['train_x'] = train_ts
    Data['train_y'] = data['train_y']

    # Testing
    Data['test_x'] = test_ts
    Data['test_y'] = data['test_y']

    # session 
    Data['session_x'] = session_ts
    Data['session_y'] = data['session_y']

    dd.io.save(Path(__file__).parents[1] / config['train_test_split_RM_features'], Data)


with skip_run('skip', 'train_test_classification_first_sessions_SVM_RF') as check, check():
    # load the TD features
    TD_data = dd.io.load(Path(__file__).parents[1] / config['train_test_split_TD_features'])

    # load the RM features
    RM_data = dd.io.load(Path(__file__).parents[1] / config['train_test_split_RM_features'])

    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr', probability=True)
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)

    # train the classifier on TD features 
    TD_svm_y = clf1.fit(TD_data['train_x'], (np.argmax(TD_data['train_y'], axis=1) + 1)).predict(TD_data['test_x'])
    TD_rf_y  = clf2.fit(TD_data['train_x'], (np.argmax(TD_data['train_y'], axis=1) + 1)).predict(TD_data['test_x'])

    # train the classifier on RM features
    RM_svm_y = clf1.fit(RM_data['train_x'], (np.argmax(RM_data['train_y'], axis=1) + 1)).predict(RM_data['test_x'])
    RM_rf_y  = clf2.fit(RM_data['train_x'], (np.argmax(RM_data['train_y'], axis=1) + 1)).predict(RM_data['test_x'])
    
    df = pd.DataFrame({'TD_SVM': TD_svm_y,
                      'TD_RF':  TD_rf_y,
                      'RM_SVM': RM_svm_y,
                      'RM_RF':  RM_rf_y,
                      'true_labels': (np.argmax(RM_data['test_y'], axis=1) + 1)})
    
    # save the predicted labels 
    df.to_csv(str(Path(__file__).parents[1] / config['predicted_labels_train_test']))     


with skip_run('skip', 'train_test_classification_Inter_sessions_SVM_RF') as check, check():
    # load the TD features
    TD_data = dd.io.load(Path(__file__).parents[1] / config['train_test_split_TD_features'])

    # load the RM features
    RM_data = dd.io.load(Path(__file__).parents[1] / config['train_test_split_RM_features'])


    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr', probability=True)
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)


    # train the classifier on TD features 
    TD_svm_y = clf1.fit(TD_data['train_x'], (np.argmax(TD_data['train_y'], axis=1) + 1)).predict(TD_data['session_x'])
    TD_rf_y  = clf2.fit(TD_data['train_x'], (np.argmax(TD_data['train_y'], axis=1) + 1)).predict(TD_data['session_x'])

    # train the classifier on RM features
    RM_svm_y = clf1.fit(RM_data['train_x'], (np.argmax(RM_data['train_y'], axis=1) + 1)).predict(RM_data['session_x'])
    RM_rf_y  = clf2.fit(RM_data['train_x'], (np.argmax(RM_data['train_y'], axis=1) + 1)).predict(RM_data['session_x'])

    df = pd.DataFrame({'TD_SVM': TD_svm_y,
                      'TD_RF':  TD_rf_y,
                      'RM_SVM': RM_svm_y,
                      'RM_RF':  RM_rf_y,
                      'true_labels': (np.argmax(RM_data['session_y'], axis=1) + 1)})
    
    # save the predicted labels 
    df.to_csv(str(Path(__file__).parents[1] / config['predicted_labels_inter_session']))


with skip_run('skip', 'voting_based_classification_train_test_split') as check, check():

    df = pd.read_csv(str(Path(__file__).parents[1] / config['predicted_labels_train_test']), delimiter=',')
    
    temp = df.drop(columns=['Unnamed: 0', 'true_labels'])

    print(accuracy_score(df['true_labels'].to_numpy(), temp.mode(axis=1)[0].to_numpy()))
    print(confusion_matrix(df['true_labels'].to_numpy(), temp.mode(axis=1)[0].to_numpy()))


    # # score
    # print('TD SVM accuracy:', accuracy_score(df['true_labels'].to_numpy(), df['TD_SVM'].to_numpy()))
    # print('TD RF  accuracy:', accuracy_score(df['true_labels'].to_numpy(), df['TD_RF'].to_numpy()))
    # print('RM SVM accuracy:', accuracy_score(df['true_labels'].to_numpy(), df['RM_SVM'].to_numpy()))
    # print('RM RF  accuracy:', accuracy_score(df['true_labels'].to_numpy(), df['RM_RF'].to_numpy()))

    # # confusion matrix
    # print('TD SVM accuracy:', confusion_matrix(df['true_labels'].to_numpy(), df['TD_SVM'].to_numpy()))
    # print('TD RF  accuracy:', confusion_matrix(df['true_labels'].to_numpy(), df['TD_RF'].to_numpy()))
    # print('RM SVM accuracy:', confusion_matrix(df['true_labels'].to_numpy(), df['RM_SVM'].to_numpy()))
    # print('RM RF  accuracy:', confusion_matrix(df['true_labels'].to_numpy(), df['RM_RF'].to_numpy()))


with skip_run('skip', 'voting_based_classification_Inter_session') as check, check():

    df = pd.read_csv(str(Path(__file__).parents[1] / config['predicted_labels_inter_session']), delimiter=',')
    
    temp = df.drop(columns=['Unnamed: 0', 'true_labels'])

    print(accuracy_score(df['true_labels'].to_numpy(), temp.mode(axis=1)[0].to_numpy()))
    print(confusion_matrix(df['true_labels'].to_numpy(), temp.mode(axis=1)[0].to_numpy()))

    # # score
    # print('TD SVM accuracy:', accuracy_score(df['true_labels'].to_numpy(), df['TD_SVM'].to_numpy()))
    # print('TD RF  accuracy:', accuracy_score(df['true_labels'].to_numpy(), df['TD_RF'].to_numpy()))
    # print('RM SVM accuracy:', accuracy_score(df['true_labels'].to_numpy(), df['RM_SVM'].to_numpy()))
    # print('RM RF  accuracy:', accuracy_score(df['true_labels'].to_numpy(), df['RM_RF'].to_numpy()))

    # # confusion matrix
    # print('TD SVM accuracy:', confusion_matrix(df['true_labels'].to_numpy(), df['TD_SVM'].to_numpy()))
    # print('TD RF  accuracy:', confusion_matrix(df['true_labels'].to_numpy(), df['TD_RF'].to_numpy()))
    # print('RM SVM accuracy:', confusion_matrix(df['true_labels'].to_numpy(), df['RM_SVM'].to_numpy()))
    # print('RM RF  accuracy:', confusion_matrix(df['true_labels'].to_numpy(), df['RM_RF'].to_numpy()))


with skip_run('skip', 'Hierarchical_classification_Inter_session') as check, check():
    # FIXME: not the right way of classification: please correct it 
    # classify between balanced class (1+2) and class 3 - get the result
    # then apply class 1 vs class 2 - get the result
    # the classifier is unable to separate 
    # load the TD features
    TD_data = dd.io.load(Path(__file__).parents[1] / config['train_test_split_TD_features'])

    # load the RM features
    RM_data = dd.io.load(Path(__file__).parents[1] / config['train_test_split_RM_features'])


    TD_train = TD_data['train_x']
    RM_train = RM_data['train_x']


    ################# First level Hierarchy #####################
    train_labels = (np.argmax(TD_data['train_y'], axis=1) + 1)
    test_labels = (np.argmax(TD_data['session_y'], axis=1) + 1)

    # club class 1 and 2 as a single class
    train_labels[train_labels != 3] = 0
    test_labels[test_labels != 3] = 0

    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr', probability=True)
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)

    rus = RandomUnderSampler(random_state=32)
    rus.fit_resample(train_labels.reshape(-1, 1), train_labels)

    TD_train = TD_train[rus.sample_indices_, :]
    RM_train = RM_train[rus.sample_indices_, :]
    train_labels = train_labels[rus.sample_indices_]

        
    # train the classifier on TD features 
    # TD_svm_y = clf1.fit(TD_train, train_labels).predict(TD_data['session_x'])
    TD_rf_y  = clf2.fit(TD_train, train_labels).predict(TD_data['session_x'])

    # train the classifier on RM features
    # RM_svm_y = clf1.fit(RM_train, train_labels).predict(RM_data['session_x'])
    RM_rf_y  = clf2.fit(RM_train, train_labels).predict(RM_data['session_x'])


    # print(accuracy_score(test_labels, TD_svm_y))
    print(accuracy_score(test_labels, TD_rf_y))
    # print(accuracy_score(test_labels, RM_svm_y))
    print(accuracy_score(test_labels, RM_rf_y))


    ################# Second level Hierarchy #####################

    train_labels = (np.argmax(TD_data['train_y'], axis=1) + 1)
    test_labels = (np.argmax(TD_data['session_y'], axis=1) + 1)

    # club class 1 and 2 as a single class
    train_labels = train_labels[train_labels != 3]

    TD_train = TD_train[train_labels != 3]
    RM_train = RM_train[train_labels != 3]
    
    
    TD_test  = TD_data['session_x']
    TD_test  = TD_test[test_labels != 3]
    RM_test  = RM_data['session_x']
    RM_test  = RM_test[test_labels != 3]
    test_labels  = test_labels[test_labels != 3]

    # train the classifier on TD features 
    # TD_svm_y = clf1.fit(TD_train, train_labels).predict(TD_test)
    TD_rf_y  = clf2.fit(TD_train, train_labels).predict(TD_test)

    # train the classifier on RM features
    # RM_svm_y = clf1.fit(RM_train, train_labels).predict(RM_test)
    RM_rf_y  = clf2.fit(RM_train, train_labels).predict(RM_test)


    # print(accuracy_score(test_labels, TD_svm_y))
    print(accuracy_score(test_labels, TD_rf_y))
    # print(accuracy_score(test_labels, RM_svm_y))
    print(accuracy_score(test_labels, RM_rf_y))


with skip_run('skip', 'Hierarchical classification two levels pooled data') as check, check():
    # first level Hierarchy class 1, 2, and 3 using TD and Random Forest
    # if class 3 then output is class 3
    # else: Second level Hierarchy
    #       binary  level classification between 1 and 2 using Riemann + Random Forest

    # Random forest classifier
    clf = RandomForestClassifier(n_estimators=100, oob_score=True)

    ################# First level Hierarchy #####################
    # load the TD features
    TD_data = dd.io.load(Path(__file__).parents[1] / config['train_test_split_TD_features'])
    
    TD_train        = TD_data['train_x']
    train_labels    = (np.argmax(TD_data['train_y'], axis=1) + 1)
    test_labels     = (np.argmax(TD_data['test_y'], axis=1) + 1)
    
    # predict the labels
    pred_labels     = clf.fit(TD_train, train_labels).predict(TD_data['test_x'])

    # temp  = pred_labels
    # temp[pred_labels != 3] = 0
    # test_labels[pred_labels != 3] = 0

    # Classification between class 3 vs class 1 & 2 
    print('First level accuracy:', accuracy_score(test_labels, pred_labels))
    

    ################# Second level Hierarchy #####################
    # Binary classification between class 1 and class 2
    # load the RM features
    RM_data = dd.io.load(Path(__file__).parents[1] / config['train_test_split_RM_features'])
    RM_train      = RM_data['train_x']
    train_labels2 = (np.argmax(RM_data['train_y'], axis=1) + 1)

    # make the labels 1 and 2, remove class 3
    RM_train      = RM_train[train_labels2 != 3]
    train_labels2 = train_labels2[train_labels2 != 3]

    RM_test       = RM_data['test_x']
    test_labels2  = (np.argmax(RM_data['test_y'], axis=1) + 1)
    
    # classification between class 1 and class 2
    pred_labels2  = clf.fit(RM_train, train_labels2).predict(RM_test[pred_labels != 3])


    print("Second level accuracy: ", accuracy_score(test_labels2[pred_labels != 3], pred_labels2))


with skip_run('skip', 'Hierarchical classification two binary levels Inter-Session data') as check, check():
    # first level Hierarchy binary classification between (1, 2) and 3 using TD and Random Forest
    # if class 3 then output is class 3
    # else: Second level Hierarchy
    #       binary  level classification between 1 and 2 using Riemann + Random Forest

    # Random forest classifier
    clf = RandomForestClassifier(n_estimators=100, oob_score=True)

    ################# First level Hierarchy #####################
    # load the TD features
    TD_data = dd.io.load(Path(__file__).parents[1] / config['train_test_split_TD_features'])
    
    TD_train        = TD_data['train_x']
    train_labels    = (np.argmax(TD_data['train_y'], axis=1) + 1)
    test_labels     = (np.argmax(TD_data['session_y'], axis=1) + 1)
    
    # predict the labels
    pred_labels     = clf.fit(TD_train, train_labels).predict(TD_data['session_x'])

    temp  = pred_labels
    temp[pred_labels != 3] = 0
    test_labels[pred_labels != 3] = 0

    # Classification between class 3 vs class 1 & 2 
    print('First level accuracy:', accuracy_score(test_labels, temp))
    

    ################# Second level Hierarchy #####################
    # Binary classification between class 1 and class 2
    # load the RM features
    RM_data = dd.io.load(Path(__file__).parents[1] / config['train_test_split_RM_features'])
    RM_train      = RM_data['train_x']
    train_labels2 = (np.argmax(RM_data['train_y'], axis=1) + 1)

    # make the labels 1 and 2, remove class 3
    RM_train      = RM_train[train_labels2 != 3]
    train_labels2 = train_labels2[train_labels2 != 3]

    RM_test       = RM_data['session_x']
    test_labels2  = (np.argmax(RM_data['session_y'], axis=1) + 1)
    
    # classification between class 1 and class 2
    pred_labels2  = clf.fit(RM_train, train_labels2).predict(RM_test[pred_labels != 3])


    print("Second level accuracy: ", accuracy_score(test_labels2[pred_labels != 3], pred_labels2))

#################################
# Extracting the blocks of continuous data and then randomly selecting it for training and testing
#################################
## ------------- Riemannian features------------ ##
with skip_run('skip', 'extract_riemann_features_subjects_and_trial_wise') as check, check():
    
    # path to load the data
    path = str(Path(__file__).parents[1] / config['clean_emg_pb_data'])
    data = dd.io.load(path)

    # prepare the Riemann model using all the training subjects
    subjects = list(set(config['subjects']) ^ set(config['test_subjects']))
    EMG = []

    for subject in subjects:
        EMG.append(data['subject_' + subject]['EMG'])

    # Convert to array
    EMG = np.concatenate(EMG, axis=0)

    train_cov = Covariances(estimator='lwf').fit(EMG)
    train_ts  = TangentSpace().fit(train_cov.transform(EMG))
    
    # path to load the data
    path = str(Path(__file__).parents[1] / config['epoch_emg_data'])
    data = dd.io.load(path)
    # extract riemannian features for each subject
    subjects = config['subjects'] 
    
    Data = collections.defaultdict()
    
    for subject in subjects:
        temp1 = collections.defaultdict()        

        for trial in (set(config['trials']) ^ set(config['comb_trials'])):  
            temp  = collections.defaultdict()
            x = data['subject_' + subject]['EMG'][trial].get_data()    

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

            y = category * np.ones((x.shape[0], 1))

            temp['cov']    = train_cov.transform(x)
            temp['RM']     = train_ts.transform(train_cov.transform(x))
            temp['labels'] = y

            temp1[trial] = temp
        
        Data['subject_' + subject] = temp1
    
    # save the file
    dd.io.save(Path(__file__).parents[1] / config['RM_features_subjectwise'], Data)

with skip_run('skip', 'time_series_split_riemann_features') as check, check():
    # path to load the data
    path = str(Path(__file__).parents[1] / config['RM_features_subjectwise'])
    data = dd.io.load(path)

    subjects = config['subjects'] #list(set(config['subjects']) ^ set(config['test_subjects']))
    train_x , train_y = [], []
    test_x, test_y = [], []

    session_train_x , session_train_y = [], []
    session_test_x,   session_test_y = [], []
     
    for subject in subjects:

        for trial in (set(config['trials']) ^ set(config['comb_trials'])):  
            
            X = data['subject_' + subject][trial]['RM']
            y = data['subject_' + subject][trial]['labels']

            if (trial == "LowFine") or (trial == "HighGross"):
                # 1 sec and 0.5 overlap for 180 sec data = 360 points, use 165 out of them as these two classes are clubbed
                id = np.arange(165)
                id_list = np.split(id, 3) # split the data into 6 equal parts        
                
            else:
                # 1 sec and 0.5 overlap for 180 sec data = 360 points, use 330 out of them
                id = np.arange(330)
                id_list = np.split(id, 6) # split the data into 6 equal parts        
            
            data_blocks = collections.defaultdict()

            for count, ind in enumerate(id_list):
                data_blocks['block_' + str(count)] = {"feat" : X[ind, :], "labels" : y[ind, :]}

            if (trial == "LowFine") or (trial == "HighGross"):
                block_list = ['block_0', 'block_1', 'block_2']
                random.shuffle(block_list)

                for i, block_id in enumerate(block_list):
                    if subject in config['test_subjects']:
                        if i < 1:
                            session_train_x.append(data_blocks[block_id]['feat'])
                            session_train_y.append(data_blocks[block_id]['labels'])
                        else:
                            session_test_x.append(data_blocks[block_id]['feat'])
                            session_test_y.append(data_blocks[block_id]['labels'])
                    else: 
                        if i < 2:
                            # train_x.update(data_blocks[block_id])
                            train_x.append(data_blocks[block_id]['feat'])
                            train_y.append(data_blocks[block_id]['labels'])
                        else:
                            test_x.append(data_blocks[block_id]['feat'])
                            test_y.append(data_blocks[block_id]['labels'])

            else:
                block_list = ['block_0', 'block_1', 'block_2', 'block_3', 'block_4', 'block_5']
                random.shuffle(block_list)

                for i, block_id in enumerate(block_list):
                    if subject in config['test_subjects']:
                        if i < 2:
                            session_train_x.append(data_blocks[block_id]['feat'])
                            session_train_y.append(data_blocks[block_id]['labels'])
                        else:
                            session_test_x.append(data_blocks[block_id]['feat'])
                            session_test_y.append(data_blocks[block_id]['labels'])
                    else: 
                        if i < 4:
                            # train_x.update(data_blocks[block_id])
                            train_x.append(data_blocks[block_id]['feat'])
                            train_y.append(data_blocks[block_id]['labels'])
                        else:
                            test_x.append(data_blocks[block_id]['feat'])
                            test_y.append(data_blocks[block_id]['labels'])

    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    test_x  = np.concatenate(test_x, axis=0)
    test_y  = np.concatenate(test_y, axis=0)

    # second session
    session_train_x = np.concatenate(session_train_x, axis=0)
    session_train_y = np.concatenate(session_train_y, axis=0)
    session_test_x  = np.concatenate(session_test_x, axis=0)
    session_test_y  = np.concatenate(session_test_y, axis=0)

    Data = collections.defaultdict()

    # Training
    Data['train_x'] = train_x
    Data['train_y'] = train_y

    # Testing
    Data['test_x'] = test_x
    Data['test_y'] = test_y

    # Training
    Data['session_train_x'] = session_train_x
    Data['session_train_y'] = session_train_y

    # Testing
    Data['session_test_x'] = session_test_x
    Data['session_test_y'] = session_test_y

    dd.io.save(Path(__file__).parents[1] / config['RM_features_orderly_pool'], Data)

with skip_run('skip', 'RF_classifier_on_RM_orderly_train_test_data') as check, check():
    # this data is orderly RM features obtained for each trial by first n seconds (train) and rest (test)

    # clf = RandomForestClassifier(n_estimators=100, oob_score=True)
    clf = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr',probability=True)
    
    data = dd.io.load(Path(__file__).parents[1] / config['RM_features_orderly_pool'])
    
    train_y = np.argmax(data['train_y'], axis=1) + 1
    test_y  = np.argmax(data['test_y'], axis=1) + 1
    session_test_y = np.argmax(data['session_test_y'], axis=1) + 1

    RF_clf = clf.fit(data['train_x'], train_y)
    score = RF_clf.score(data['test_x'], test_y)
    print("Accuracy of RF on ordered RM features first session : ", score)

    score = RF_clf.score(data['session_test_x'], session_test_y)
    print("Testing accuracy of RF on second session : ", score)

    test_pred = RF_clf.predict(data['test_x'])
    session_pred = RF_clf.predict(data['session_test_x'])

    # test the accuracy by training the classifier on the session2 25% data and test on session2- 75% data
    score = clf.fit(data['session_train_x'], np.argmax(data['session_train_y'], axis=1) + 1).score(data['session_test_x'], session_test_y)
    print("Accuracy only on the session 2 data:", score)
    # plt.figure()
    # plt.plot(test_pred, label="predicted label")
    # plt.plot(test_y, label="true label")
    # plt.title('Session 1 predictions')
    # plt.legend()
    
    # for i in range(config['n_class']):
    #     plt.figure()
    #     plt.plot(session_pred[session_test_y == (i+1)], label="predicted label")
    #     plt.plot(session_test_y[session_test_y == (i+1)], label="true label")
    #     plt.title('Class: ' + str(i+1) + ' - Session 2 predictions with correction')

with skip_run('skip', 'RF_correction_without_session_training_data') as check, check():
    # implement the correction alogrithm without using session data in training
    corr_win = 4
    # clf = RandomForestClassifier(n_estimators=100, oob_score=True)
    clf = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr',probability=True)
    
    data = dd.io.load(Path(__file__).parents[1] / config['RM_features_orderly_pool'])
    
    train_x = np.concatenate((data['train_x'], data['session_train_x']), axis=0)
    train_y = np.concatenate((data['train_y'], data['session_train_y']), axis=0)

    print(train_x.shape, train_y.shape)
    # train_x = data['train_x']
    # train_y = data['train_y']
    test_x  = data['test_x']
    session_test_x = data['session_test_x']

    train_y = np.argmax(train_y, axis=1) + 1
    test_y  = np.argmax(data['test_y'], axis=1) + 1
    session_test_y = np.argmax(data['session_test_y'], axis=1) + 1

    # only using training and second session data
    RF_clf = clf.fit(train_x, train_y)

    score = RF_clf.score(test_x, test_y)
    print("Accuracy of first session without correction: ", score)

    score = RF_clf.score(session_test_x, session_test_y)
    print("Testing accuracy of second session before correction: ", score)

    # plt.figure()
    # plt.plot(RF_clf.predict(test_x), label="predicted label")
    # plt.plot(test_y, label="true label")
    # plt.title('Session 1 predictions without correction')
    # plt.legend()

    session_pred = RF_clf.predict(session_test_x)
    for i in range(config['n_class']):
        plt.figure()
        plt.plot(session_pred[session_test_y == (i+1)], label="predicted label")
        plt.plot(session_test_y[session_test_y == (i+1)], label="true label")
        plt.title('Class: ' + str(i+1) + ' - Session 2 predictions without correction')
        
        # save the predictions to csv
        df = pd.DataFrame({'true': session_test_y[session_test_y == (i+1)], 'predictions': session_pred[session_test_y == (i+1)]})
        savepath  = str(Path(__file__).parents[1]) +  '/data/processed/RM_class_' + str(i+1) +'.csv'
        df.to_csv(savepath)

    test_pred = []
    for i in range(test_x.shape[0]):
        if i < corr_win:
            test_pred.append(RF_clf.predict(test_x[i,:].reshape(1, -1)).item())
        else:
            pred = RF_clf.predict(test_x[i-corr_win:i+1,:])
            test_pred.append(int(collections.Counter(pred).most_common(1)[0][0]))

    test_pred = np.array(test_pred).reshape(-1, 1)
    score = accuracy_score(test_y, test_pred)
    print("Testing accuracy of first session after correction: ", score)

    session_pred = []
    for i in range(session_test_x.shape[0]):
        if i < corr_win:
            session_pred.append(RF_clf.predict(session_test_x[i,:].reshape(1, -1)).item())
        else:
            pred = RF_clf.predict(session_test_x[i-corr_win:i+1,:])
            session_pred.append(int(collections.Counter(pred).most_common(1)[0][0]))

    session_pred = np.array(session_pred).reshape(-1, 1)
    score = accuracy_score(session_test_y, session_pred)
    print("Testing accuracy of second session after correction: ", score)


    # plt.figure()
    # plt.plot(test_pred, label="predicted label")
    # plt.plot(test_y, label="true label")
    # plt.title('Session 1 predictions with correction')
    # plt.legend()

    for i in range(config['n_class']):
        plt.figure()
        plt.plot(session_pred[session_test_y == (i+1)], label="predicted label")
        plt.plot(session_test_y[session_test_y == (i+1)], label="true label")
        plt.title('Class: ' + str(i+1) + ' - Session 2 predictions with correction')

        # save the predictions to csv
        df = pd.DataFrame({'true': session_test_y[session_test_y == (i+1)], 'predictions': session_pred[session_test_y == (i+1)].reshape(-1,)})
        savepath  = str(Path(__file__).parents[1]) +  '/data/processed/RM_corrected_class_' + str(i+1) +'.csv'
        df.to_csv(savepath)

## ------------- Hudgins features------------ ##
with skip_run('skip', 'extract_emg_features') as check, check():
    # path to save
    path = str(Path(__file__).parents[1] / config['clean_emg_pb_data'])
    data = dd.io.load(path)

    data = extract_emg_features(data, config, scale=False)

    # save the data in h5 format
    path = str(Path(__file__).parents[1] / config['subject__unnorm_emg_features'])
    save_data(path,data,save=True)

with skip_run('skip', 'pool_subject_emg_features') as check, check():

    path = str(Path(__file__).parents[1] / config['subject__unnorm_emg_features'])
    data = dd.io.load(path)
    
    subjects = set(config['subjects']) ^ set(config['test_subjects'])
    X1, X2, Y = pool_subject_emg_features(data, subjects, config)

    data = {}
    data['X1'] = X1
    data['X2'] = X2
    data['Y']  = Y
  
    # save the data in h5 format
    path = str(Path(__file__).parents[1] / config['pooled_emg_features'])
    save_data(path,data,save=True)

with skip_run('run', 'extract_hudgins_features_subjects_and_trial_wise') as check, check():
    
    # Normalize the data subject wise
    scale = True
    # Use this data to extract the maxmium and minimum values of the EMG features for each subject
    ############################
    path = str(Path(__file__).parents[1] / config['subject__unnorm_emg_features'])
    TD_data = dd.io.load(path)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))    
    ###########################
    
    # path to load the data
    path = str(Path(__file__).parents[1] / config['epoch_emg_data'])
    data = dd.io.load(path)
    # extract riemannian features for each subject
    subjects = config['subjects'] 
    
    Data = collections.defaultdict()
    
    for subject in subjects:
        temp1 = collections.defaultdict()        

        if scale:
            mm_scaler      = min_max_scaler.fit(TD_data['subject_' + subject]['features1'])

        # mn = np.min(TD_data['subject_' + subject]['features1'],axis=0)
        # mx = np.max(TD_data['subject_' + subject]['features1'],axis=0)
        
        for trial in (set(config['trials']) ^ set(config['comb_trials'])):  
            temp  = collections.defaultdict()
            x = data['subject_' + subject]['EMG'][trial].get_data()    

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

            y = category * np.ones((x.shape[0], 1))

            features = hudgins_features(x, config)
            
            temp['TD'] = mm_scaler.transform(features)
            temp['labels'] = y

            temp1[trial] = temp
        
        Data['subject_' + subject] = temp1
    
    # save the file
    dd.io.save(Path(__file__).parents[1] / config['TD_features_subjectwise'], Data)

with skip_run('skip', 'time_series_split_hudgins_features') as check, check():
    # path to load the data
    path = str(Path(__file__).parents[1] / config['TD_features_subjectwise'])
    data = dd.io.load(path)

    subjects = config['subjects'] #list(set(config['subjects']) ^ set(config['test_subjects']))
    train_x , train_y = [], []
    test_x, test_y = [], []

    session_train_x , session_train_y = [], []
    session_test_x,   session_test_y = [], []
     
    for subject in subjects:

        for trial in (set(config['trials']) ^ set(config['comb_trials'])):  
            
            X = data['subject_' + subject][trial]['TD']
            y = data['subject_' + subject][trial]['labels']

            if (trial == "LowFine") or (trial == "HighGross"):
                # 1 sec and 0.5 overlap for 180 sec data = 360 points, use 165 out of them as these two classes are clubbed
                id = np.arange(165)
                id_list = np.split(id, 3) # split the data into 6 equal parts        
                
            else:
                # 1 sec and 0.5 overlap for 180 sec data = 360 points, use 330 out of them
                id = np.arange(330)
                id_list = np.split(id, 6) # split the data into 6 equal parts        
            
            data_blocks = collections.defaultdict()

            for count, ind in enumerate(id_list):
                data_blocks['block_' + str(count)] = {"feat" : X[ind, :], "labels" : y[ind, :]}

            if (trial == "LowFine") or (trial == "HighGross"):
                block_list = ['block_0', 'block_1', 'block_2']
                random.shuffle(block_list)

                for i, block_id in enumerate(block_list):
                    if subject in config['test_subjects']:
                        if i < 1:
                            session_train_x.append(data_blocks[block_id]['feat'])
                            session_train_y.append(data_blocks[block_id]['labels'])
                        else:
                            session_test_x.append(data_blocks[block_id]['feat'])
                            session_test_y.append(data_blocks[block_id]['labels'])
                    else: 
                        if i < 2:
                            # train_x.update(data_blocks[block_id])
                            train_x.append(data_blocks[block_id]['feat'])
                            train_y.append(data_blocks[block_id]['labels'])
                        else:
                            test_x.append(data_blocks[block_id]['feat'])
                            test_y.append(data_blocks[block_id]['labels'])

            else:
                block_list = ['block_0', 'block_1', 'block_2', 'block_3', 'block_4', 'block_5']
                random.shuffle(block_list)

                for i, block_id in enumerate(block_list):
                    if subject in config['test_subjects']:
                        if i < 2:
                            session_train_x.append(data_blocks[block_id]['feat'])
                            session_train_y.append(data_blocks[block_id]['labels'])
                        else:
                            session_test_x.append(data_blocks[block_id]['feat'])
                            session_test_y.append(data_blocks[block_id]['labels'])
                    else: 
                        if i < 4:
                            # train_x.update(data_blocks[block_id])
                            train_x.append(data_blocks[block_id]['feat'])
                            train_y.append(data_blocks[block_id]['labels'])
                        else:
                            test_x.append(data_blocks[block_id]['feat'])
                            test_y.append(data_blocks[block_id]['labels'])

    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    test_x  = np.concatenate(test_x, axis=0)
    test_y  = np.concatenate(test_y, axis=0)

    # second session
    session_train_x = np.concatenate(session_train_x, axis=0)
    session_train_y = np.concatenate(session_train_y, axis=0)
    session_test_x  = np.concatenate(session_test_x, axis=0)
    session_test_y  = np.concatenate(session_test_y, axis=0)

    Data = collections.defaultdict()

    # Training
    Data['train_x'] = train_x
    Data['train_y'] = train_y

    # Testing
    Data['test_x'] = test_x
    Data['test_y'] = test_y

    # Training
    Data['session_train_x'] = session_train_x
    Data['session_train_y'] = session_train_y

    # Testing
    Data['session_test_x'] = session_test_x
    Data['session_test_y'] = session_test_y

    dd.io.save(Path(__file__).parents[1] / config['TD_features_orderly_pool'], Data)

with skip_run('skip', 'RF_classifier_on_hudgins_orderly_train_test_data') as check, check():
    # this data is orderly RM features obtained for each trial by first n seconds (train) and rest (test)

    # clf = RandomForestClassifier(n_estimators=100, oob_score=True)
    clf = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr',probability=True)

    data = dd.io.load(Path(__file__).parents[1] / config['TD_features_orderly_pool'])
    
    train_y = np.argmax(data['train_y'], axis=1) + 1
    test_y  = np.argmax(data['test_y'], axis=1) + 1
    print(len(train_y[train_y == 1]), len(train_y[train_y == 2]), len(train_y[train_y == 3]))
    
    session_test_y = np.argmax(data['session_test_y'], axis=1) + 1

    RF_clf = clf.fit(data['train_x'], train_y)
    
    score = clf.score(data['train_x'], train_y)
    print('Training accuracy:', score)
    
    score = RF_clf.score(data['test_x'], test_y)
    print("Accuracy of RF on ordered EMG features first session : ", score)

    score = RF_clf.score(data['session_test_x'], session_test_y)
    print("Testing accuracy of EMG on second session : ", score)

    test_pred = RF_clf.predict(data['test_x'])
    session_pred = RF_clf.predict(data['session_test_x'])

    # test the accuracy by training the classifier on the session2 25% data and test on session2- 75% data
    score = clf.fit(data['session_train_x'], np.argmax(data['session_train_y'], axis=1) + 1).score(data['session_test_x'], session_test_y)
    print("Accuracy only on the session 2 data:", score)
    # plt.figure()
    # plt.plot(test_pred, label="predicted label")
    # plt.plot(test_y, label="true label")
    # plt.title('Session 1 predictions')
    # plt.legend()

    for i in range(config['n_class']):
        plt.figure()
        plt.plot(session_pred[session_test_y == (i+1)], 'r.',label="predicted label")
        plt.plot(session_test_y[session_test_y == (i+1)], 'b-',label="true label")
        plt.title('Class: ' + str(i+1) + ' - Session 2 predictions')
        # plt.legend()

with skip_run('skip', 'RF_correction_on_hudgins_without_session_training_data') as check, check():
    # implement the correction alogrithm without using session data in training
    corr_win = 4
    # clf = RandomForestClassifier(n_estimators=100, oob_score=True)
    clf = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr',probability=True)
    
    data = dd.io.load(Path(__file__).parents[1] / config['TD_features_orderly_pool'])
    
    train_x = np.concatenate((data['train_x'], data['session_train_x']), axis=0)
    train_y = np.concatenate((data['train_y'], data['session_train_y']), axis=0)

    print(train_x.shape, train_y.shape)
    # train_x = data['train_x']
    # train_y = data['train_y']
    test_x  = data['test_x']
    session_test_x = data['session_test_x']

    train_y = np.argmax(train_y, axis=1) + 1
    test_y  = np.argmax(data['test_y'], axis=1) + 1
    session_test_y = np.argmax(data['session_test_y'], axis=1) + 1

    # only using training and second session data
    RF_clf = clf.fit(train_x, train_y)

    score = RF_clf.score(test_x, test_y)
    print("Accuracy of first session without correction: ", score)

    score = RF_clf.score(session_test_x, session_test_y)
    print("Testing accuracy of second session before correction: ", score)

    # plt.figure()
    # plt.plot(RF_clf.predict(test_x), label="predicted label")
    # plt.plot(test_y, label="true label")
    # plt.title('Session 1 predictions without correction')
    # plt.legend()

    session_pred = RF_clf.predict(session_test_x)
    for i in range(config['n_class']):
        plt.figure()
        plt.plot(session_pred[session_test_y == (i+1)], label="predicted label")
        plt.plot(session_test_y[session_test_y == (i+1)], label="true label")
        plt.title('Class: ' + str(i+1) + ' - Session 2 predictions without correction')
        
        # save the predictions to csv
        df = pd.DataFrame({'true': session_test_y[session_test_y == (i+1)], 'predictions': session_pred[session_test_y == (i+1)]})
        savepath  = str(Path(__file__).parents[1]) +  '/data/processed/TD_class_' + str(i+1) +'.csv'
        df.to_csv(savepath)

    test_pred = []
    for i in range(test_x.shape[0]):
        if i < corr_win:
            test_pred.append(RF_clf.predict(test_x[i,:].reshape(1, -1)).item())
        else:
            pred = RF_clf.predict(test_x[i-corr_win:i+1,:])
            test_pred.append(int(collections.Counter(pred).most_common(1)[0][0]))

    test_pred = np.array(test_pred).reshape(-1, 1)
    score = accuracy_score(test_y, test_pred)
    print("Testing accuracy of first session after correction: ", score)

    session_pred = []
    for i in range(session_test_x.shape[0]):
        if i < corr_win:
            session_pred.append(RF_clf.predict(session_test_x[i,:].reshape(1, -1)).item())
        else:
            pred = RF_clf.predict(session_test_x[i-corr_win:i+1,:])
            session_pred.append(int(collections.Counter(pred).most_common(1)[0][0]))

    session_pred = np.array(session_pred).reshape(-1, 1)
    score = accuracy_score(session_test_y, session_pred)
    print("Testing accuracy of second session after correction: ", score)


    # plt.figure()
    # plt.plot(test_pred, label="predicted label")
    # plt.plot(test_y, label="true label")
    # plt.title('Session 1 predictions with correction')
    # plt.legend()

    for i in range(config['n_class']):
        plt.figure()
        plt.plot(session_pred[session_test_y == (i+1)], label="predicted label")
        plt.plot(session_test_y[session_test_y == (i+1)], label="true label")
        plt.title('Class: ' + str(i+1) + ' - Session 2 predictions with correction')

        # save the predictions to csv
        df = pd.DataFrame({'true': session_test_y[session_test_y == (i+1)], 'predictions': session_pred[session_test_y == (i+1)].reshape(-1,)})
        savepath  = str(Path(__file__).parents[1]) +  '/data/processed/TD_corrected_class_' + str(i+1) +'.csv'
        df.to_csv(savepath)


#################################
# Transfer learning based on Riemann Procustes Analysis
#################################
# Intersession classification without applying transfer learning 
# The transfer learning code is used from https://github.com/plcrodrigues/PhD-Code  

# number of samples from target to be used for training
N = 20  
with skip_run('skip', 'Prepare_source_target_dataset') as check, check():
    # load RM features
    RM_data = dd.io.load(Path(__file__).parents[1] / config['RM_features_subjectwise'])
    
    # load TD features
    TD_data = dd.io.load(Path(__file__).parents[1] / config['TD_features_subjectwise'])
    
    ### Prepare the source dataset
    source_subjects = list(set(config['subjects']) ^ set(config['test_subjects']))
    # source_subjects = config['train_subjects']
    source_TD       = []
    source_RM       = []
    source_cov      = []
    source_labels   = []
    
    for subject in source_subjects:
        for trial in (set(config['trials']) ^ set(config['comb_trials'])):  
            TD  = TD_data['subject_' + subject][trial]['TD']
            RM  = RM_data['subject_' + subject][trial]['RM']
            cov = RM_data['subject_' + subject][trial]['cov']
            y   = RM_data['subject_' + subject][trial]['labels']

            if (trial == "LowFine") or (trial == "HighGross"):
                # 1 sec and 0.5 overlap for 180 sec data = 360 points, use 165 out of them as these two classes are clubbed
                id = np.arange(165)       
            else:
                # 1 sec and 0.5 overlap for 180 sec data = 360 points, use 330 out of them
                id = np.arange(330)       

            TD  = TD[id, :]
            RM  = RM[id, :]
            cov = cov[id, :, :]
            y   = np.argmax(y[id, :], axis=1) + 1

            source_TD.append(TD)
            source_RM.append(RM)
            source_cov.append(cov)
            source_labels.append( y)
        
    Data = collections.defaultdict()
    
    Data['source_TD']     = np.concatenate(source_TD, axis=0)
    Data['source_RM']     = np.concatenate(source_RM, axis=0)
    Data['source_cov']    = np.concatenate(source_cov, axis=0)
    Data['source_labels'] = np.concatenate(source_labels, axis=0)

    ### Prepare the target dataset
    target_subjects = config['test_subjects']
    target_TD       = []
    target_RM       = []
    target_cov      = []
    target_labels   = []
    
    target_train_TD       = []
    target_train_RM       = []
    target_train_cov      = []
    target_train_labels   = []
    
    for subject in target_subjects:
        for trial in (set(config['trials']) ^ set(config['comb_trials'])):  
            TD  = TD_data['subject_' + subject][trial]['TD']
            RM  = RM_data['subject_' + subject][trial]['RM']
            cov = RM_data['subject_' + subject][trial]['cov']
            y   = RM_data['subject_' + subject][trial]['labels']

            y  = np.argmax(y, axis=1) + 1 
            
            if (trial == "LowFine") or (trial == "HighGross"):
                # 1 sec and 0.5 overlap for 180 sec data = 360 points, use 165 out of them as these two classes are clubbed
                id = np.arange(int(N/2), 165)       
            else:
                # 1 sec and 0.5 overlap for 180 sec data = 360 points, use 330 out of them
                id = np.arange(N, 330)       

            TD  = TD[id, :]
            RM  = RM[id, :]
            cov = cov[id, :, :]
            y   =  y[id]

            target_TD.append(TD)
            target_RM.append(RM)
            target_cov.append(cov)
            target_labels.append( y)
            
            # training data from target
            target_train_TD.append(TD[:N, :])
            target_train_RM.append(RM[:N, :])
            target_train_cov.append(cov[:N, :, :])
            target_train_labels.append(y[:N])
        
    Data['target_TD']     = np.concatenate(target_TD, axis=0)
    Data['target_RM']     = np.concatenate(target_RM, axis=0)
    Data['target_cov']    = np.concatenate(target_cov, axis=0)
    Data['target_labels'] = np.concatenate(target_labels, axis=0)
    
    Data['target_train_TD']     = np.concatenate(target_train_TD, axis=0)
    Data['target_train_RM']     = np.concatenate(target_train_RM, axis=0)
    Data['target_train_cov']    = np.concatenate(target_train_cov, axis=0)
    Data['target_train_labels'] = np.concatenate(target_train_labels, axis=0)
    
    # save the data
    dd.io.save(Path(__file__).parents[1] / config['source_target_dataset'], Data)

with skip_run('skip', 'Inter-session_accuracy_w/o_transfer_learning') as check, check():
    # load the dataset
    data = dd.io.load(Path(__file__).parents[1] / config['source_target_dataset'])
        
    source_RM       = data['source_RM']
    source_TD       = data['source_TD']
    source_labels   = data['source_labels']
    target_RM       = data['target_RM']
    target_TD       = data['target_TD']
    target_labels   = data['target_labels']
    
    source_covest   = data['source_cov']
    target_covest   = data['target_cov']
    
    source_cov = np.reshape(source_covest, (source_covest.shape[0], source_covest.shape[1] * source_covest.shape[2]), order='C')
    source_cov = source_cov[:, 0:36]
    
    target_cov = np.reshape(target_covest, (target_covest.shape[0], target_covest.shape[1] * target_covest.shape[2]), order='C')
    target_cov = target_cov[:, 0:36]
    
    # classifier 
    clf = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr',probability=True)
    
    print("SVM training accuracy using Covariances:{}".format(clf.fit(source_cov, source_labels).score(source_cov, source_labels)))
    print("SVM training accuracy using Tangent Space:{}".format(clf.fit(source_RM, source_labels).score(source_RM, source_labels)))
    print("SVM training accuracy using Hudgins features:{}".format(clf.fit(source_TD, source_labels).score(source_TD, source_labels)))
    
    print("SVM inter-session accuracy using Covariances:{}".format(clf.fit(source_cov, source_labels).score(target_cov, target_labels)))
    print("SVM inter-session accuracy using Tangent Space:{}".format(clf.fit(source_RM, source_labels).score(target_RM, target_labels)))
    print("SVM inter-session accuracy using Hudgins features:{}".format(clf.fit(source_TD, source_labels).score(target_TD, target_labels)))
    
    # use the part of target data for training 
    source_RM       = np.concatenate((source_RM, data['target_train_RM']), axis=0)
    source_TD       = np.concatenate((source_TD, data['target_train_TD']), axis=0)
    source_labels   = np.concatenate((source_labels, data['target_train_labels']), axis=0)
    
    # print("SVM retrain ({} samples) inter-session accuracy using Covariances:{}".format(N, clf.fit(source_cov, source_labels).score(target_cov, target_labels)))
    print("SVM retrain ({} samples) inter-session accuracy using Tangent Space:{}".format(N, clf.fit(source_RM, source_labels).score(target_RM, target_labels)))
    print("SVM retrain ({} samples) inter-session accuracy using Hudgins features:{}".format(N, clf.fit(source_TD, source_labels).score(target_TD, target_labels)))
    
with skip_run('skip', 'Apply_RPA_based_transfer_learning_RM_features') as check, check():
    
    # load the dataset
    data = dd.io.load(Path(__file__).parents[1] / config['source_target_dataset'])
        
    features_source = data['source_cov']
    labels_source   = data['source_labels']
    features_target = data['target_cov']
    labels_target   = data['target_labels']
    
    print('Source Class1: {}, Class2: {}, Class3: {}'.format(len(labels_source[labels_source == 1]), len(labels_source[labels_source == 2]), len(labels_source[labels_source == 3])))
    
    print('Target Class1: {}, Class2: {}, Class3: {}'.format(len(labels_target[labels_target == 1]), len(labels_target[labels_target == 2]), len(labels_target[labels_target == 3])))
    
    # Number of datapoints from session 2 to be considered for training
    ncovs_target_train = 10
    data_source       = {}
    data_target       = {}
    data_target_train = {}
    
    # prepare the source data from session 1
    data_source['covs']   = features_source
    data_source['labels'] = labels_source

    # prepare the target data from session 2
    data_target['covs']   = features_target
    data_target['labels'] = labels_target

    # prepare training dataset from session 2
    data_target_train['covs']  = data['target_train_cov']
    data_target_train['labels']= data['target_train_labels']
    
     # setup the scores dictionary
    scores = {}
    for meth in ['org', 'rct', 'str', 'rot', 'clb']:
        scores[meth] = []

    # run the transfer learning for 5 times 
    for i in tqdm(range(5)):
        # apply RPA to multiple random partitions for the training dataset
        clf = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr',probability=True)
        
        source = {}
        target_train = {}
        target_test = {}

        # source['org'], target_train['org'], target_test['org'] = TL.get_sourcetarget_split(data_source, data_target, ncovs_target_train, paradigm='MI')
        source['org'] = data_source
        target_train['org'] = data_target_train
        target_test['org'] = data_target
        
        # apply RPA 
        source['rct'], target_train['rct'], target_test['rct'] = TL.RPA_recenter(source['org'], target_train['org'], target_test['org'])
        source['str'], target_train['str'], target_test['str'] = TL.RPA_stretch(source['rct'], target_train['rct'], target_test['rct'])
        source['rot'], target_train['rot'], target_test['rot'] = TL.RPA_rotate(source['str'], target_train['str'], target_test['str'])

        # get classification scores
        # scores['clb'].append(TL.get_score_calibration(clf, target_train['org'], target_test['org']))
        # for meth in source.keys():
        #     scores[meth].append(TL.get_score_transferlearning(clf, source[meth], target_train[meth], target_test[meth]))

        
        for meth in source.keys():
            source[meth]['ts']       = TangentSpace().fit_transform(source[meth]['covs'])
            target_train[meth]['ts'] = TangentSpace().fit_transform(target_train[meth]['covs'])
            target_test[meth]['ts']  = TangentSpace().fit_transform(target_test[meth]['covs'])
            
            if meth == 'clb':
                # get the classificaion scores
                scores['clb'].append(TL.get_tangent_space_score_calibration(clf, target_train[meth], target_test[meth]))
            else:
                scores[meth].append(TL.get_tangent_space_score_transferlearning(clf, source[meth], target_train[meth], target_test[meth]))
            
    # print the scores
    for meth in scores.keys():
        print(meth, np.mean(scores[meth]), np.std(scores[meth]))


# Obtain accuracies for increasing number of training samples from the target dataset
with skip_run('run', 'Transfer_learning_accuracy_improvement_by_increasing_training_samples') as check, check():
    # load RM features
    RM_data = dd.io.load(Path(__file__).parents[1] / config['RM_features_subjectwise'])
    
    # load TD features
    TD_data = dd.io.load(Path(__file__).parents[1] / config['TD_features_subjectwise'])
     
    ### Prepare the source dataset
    source_subjects = list(set(config['subjects']) ^ set(config['test_subjects']))
    # source_subjects = config['train_subjects']
    source_TD       = []
    source_RM       = []
    source_cov      = []
    source_labels   = []
    
    for subject in source_subjects:
        for trial in (set(config['trials']) ^ set(config['comb_trials'])):  
            TD  = TD_data['subject_' + subject][trial]['TD']
            RM  = RM_data['subject_' + subject][trial]['RM']
            cov = RM_data['subject_' + subject][trial]['cov']
            y   = RM_data['subject_' + subject][trial]['labels']

            if (trial == "LowFine") or (trial == "HighGross"):
                # 1 sec and 0.5 overlap for 180 sec data = 360 points, use 165 out of them as these two classes are clubbed
                id = np.arange(165)       
            else:
                # 1 sec and 0.5 overlap for 180 sec data = 360 points, use 330 out of them
                id = np.arange(330)       

            TD  = TD[id, :]
            RM  = RM[id, :]
            cov = cov[id, :, :]
            y   = np.argmax(y[id, :], axis=1) + 1

            source_TD.append(TD)
            source_RM.append(RM)
            source_cov.append(cov)
            source_labels.append( y)
        
    Data = collections.defaultdict()
    
    Data['source_TD']     = np.concatenate(source_TD, axis=0)
    Data['source_RM']     = np.concatenate(source_RM, axis=0)
    Data['source_cov']    = np.concatenate(source_cov, axis=0)
    Data['source_labels'] = np.concatenate(source_labels, axis=0)

    ### Prepare the target dataset
    target_subjects = config['test_subjects']
    
    score_coll = collections.defaultdict()
    for N in tqdm(range(10, 51, 10)):
        print(" -------- Accuracies reported for the number of samples N: {} -------- ".format(N))
        target_TD       = []
        target_RM       = []
        target_cov      = []
        target_labels   = []
        
        target_train_TD       = []
        target_train_RM       = []
        target_train_cov      = []
        target_train_labels   = []
        
        for subject in target_subjects:
            for trial in (set(config['trials']) ^ set(config['comb_trials'])):  
                TD  = TD_data['subject_' + subject][trial]['TD']
                RM  = RM_data['subject_' + subject][trial]['RM']
                cov = RM_data['subject_' + subject][trial]['cov']
                y   = RM_data['subject_' + subject][trial]['labels']

                y  = np.argmax(y, axis=1) + 1 
                
                if (trial == "LowFine") or (trial == "HighGross"):
                    # 1 sec and 0.5 overlap for 180 sec data = 360 points, use 165 out of them as these two classes are clubbed
                    id = np.arange(int(N/2), 165)       
                else:
                    # 1 sec and 0.5 overlap for 180 sec data = 360 points, use 330 out of them
                    id = np.arange(N, 330)       

                TD  = TD[id, :]
                RM  = RM[id, :]
                cov = cov[id, :, :]
                y   =  y[id]

                target_TD.append(TD)
                target_RM.append(RM)
                target_cov.append(cov)
                target_labels.append( y)
                
                # training data from target
                target_train_TD.append(TD[:N, :])
                target_train_RM.append(RM[:N, :])
                target_train_cov.append(cov[:N, :, :])
                target_train_labels.append(y[:N])
            
        Data['target_TD']     = np.concatenate(target_TD, axis=0)
        Data['target_RM']     = np.concatenate(target_RM, axis=0)
        Data['target_cov']    = np.concatenate(target_cov, axis=0)
        Data['target_labels'] = np.concatenate(target_labels, axis=0)
        
        Data['target_train_TD']     = np.concatenate(target_train_TD, axis=0)
        Data['target_train_RM']     = np.concatenate(target_train_RM, axis=0)
        Data['target_train_cov']    = np.concatenate(target_train_cov, axis=0)
        Data['target_train_labels'] = np.concatenate(target_train_labels, axis=0)
        
            
        source_RM       = Data['source_RM']
        source_TD       = Data['source_TD']
        source_labels   = Data['source_labels']
        target_RM       = Data['target_RM']
        target_TD       = Data['target_TD']
        target_labels   = Data['target_labels']
        
        source_covest   = Data['source_cov']
        target_covest   = Data['target_cov']
        
        # classifier 
        clf = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr',probability=True)
    
        if N == 0:
            print("SVM inter-session accuracy using Tangent Space:{}".format(clf.fit(source_RM, source_labels).score(target_RM, target_labels)))
            print("SVM inter-session accuracy using Hudgins features:{}".format(clf.fit(source_TD, source_labels).score(target_TD, target_labels)))
        
        else:
            # use the part of target data for training 
            source_RM       = np.concatenate((source_RM, Data['target_train_RM']), axis=0)
            source_TD       = np.concatenate((source_TD, Data['target_train_TD']), axis=0)
            source_labels   = np.concatenate((source_labels, Data['target_train_labels']), axis=0)
            
            print("SVM retrain ({} samples) inter-session accuracy using Tangent Space:{}".format(N, clf.fit(source_RM, source_labels).score(target_RM, target_labels)))
            print("SVM retrain ({} samples) inter-session accuracy using Hudgins features:{}".format(N, clf.fit(source_TD, source_labels).score(target_TD, target_labels)))
                
            features_source = Data['source_cov']
            labels_source   = Data['source_labels']
            features_target = Data['target_cov']
            labels_target   = Data['target_labels']
            
            print('Source Class1: {}, Class2: {}, Class3: {}'.format(len(labels_source[labels_source == 1]), len(labels_source[labels_source == 2]), len(labels_source[labels_source == 3])))
            
            print('Target Class1: {}, Class2: {}, Class3: {}'.format(len(labels_target[labels_target == 1]), len(labels_target[labels_target == 2]), len(labels_target[labels_target == 3])))
            
            # Number of datapoints from session 2 to be considered for training
            ncovs_target_train = 10
            data_source       = {}
            data_target       = {}
            data_target_train = {}
            
            # prepare the source data from session 1
            data_source['covs']   = features_source
            data_source['labels'] = labels_source

            # prepare the target data from session 2
            data_target['covs']   = features_target
            data_target['labels'] = labels_target

            # prepare training dataset from session 2
            data_target_train['covs']  = Data['target_train_cov']
            data_target_train['labels']= Data['target_train_labels']
            
            # setup the scores dictionary
            scores = {}
            for meth in ['org', 'rct', 'str', 'rot', 'clb']:
                scores[meth] = []

            # run the transfer learning for 5 times 
    #         for i in tqdm(range(5)):
    #             # apply RPA to multiple random partitions for the training dataset
    #             clf = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr',probability=True)
                
    #             source = {}
    #             target_train = {}
    #             target_test = {}

    #             # source['org'], target_train['org'], target_test['org'] = TL.get_sourcetarget_split(data_source, data_target, ncovs_target_train, paradigm='MI')
    #             source['org'] = data_source
    #             target_train['org'] = data_target_train
    #             target_test['org'] = data_target
                
    #             # apply RPA 
    #             source['rct'], target_train['rct'], target_test['rct'] = TL.RPA_recenter(source['org'], target_train['org'], target_test['org'])
    #             source['str'], target_train['str'], target_test['str'] = TL.RPA_stretch(source['rct'], target_train['rct'], target_test['rct'])
    #             source['rot'], target_train['rot'], target_test['rot'] = TL.RPA_rotate(source['str'], target_train['str'], target_test['str'])

    #             # get the calibration scores
    #             target_train['org']['ts'] = TangentSpace().fit_transform(target_train['org']['covs'])
    #             target_test['org']['ts']  = TangentSpace().fit_transform(target_test['org']['covs'])
    #             temp_score = TL.get_tangent_space_score_calibration(clf, target_train['org'], target_test['org'])
    #             scores['clb'].append(temp_score)
                
    #             for meth in source.keys():
    #                 source[meth]['ts']       = TangentSpace().fit_transform(source[meth]['covs'])
    #                 target_train[meth]['ts'] = TangentSpace().fit_transform(target_train[meth]['covs'])
    #                 target_test[meth]['ts']  = TangentSpace().fit_transform(target_test[meth]['covs'])
                    
    #                 scores[meth].append(TL.get_tangent_space_score_transferlearning(clf, source[meth], target_train[meth], target_test[meth]))
            
    #         score_coll[str(N)] = scores 
    #         # print the scores
    #         for meth in scores.keys():
    #             print(meth, np.mean(scores[meth]), np.std(scores[meth]))
            
    # dd.io.save(str(Path(__file__).parents[1] / config['scores_RPA']), score_coll)

with skip_run('skip', 'Plot_transfer_learning_accuracies_wrt_#_of_samples_from_file') as check, check():
    scores = dd.io.load(str(Path(__file__).parents[1] / config['scores_RPA']))
    
    clb_scores = {}
    tl_scores  = {}
    
    clb_scores['mean'] = []
    clb_scores['std']  = []
    tl_scores['mean'] = []
    tl_scores['std']  = []
    
    plt.figure()
    for N in range(10, 51, 10):
        print(scores[str(N)]['clb'])
        clb_scores['mean'].append(np.mean(scores[str(N)]['clb']))
        clb_scores['std'].append(np.std(scores[str(N)]['clb']))
        tl_scores['mean'].append(np.mean(scores[str(N)]['rot']))
        tl_scores['std'].append(np.std(scores[str(N)]['rot']))
    
    n_samples = np.arange(10, 51, 10)
    # plt.plot(n_samples, clb_scores['mean'], 'r')
    # plt.plot(n_samples, tl_scores['mean'], 'b')

with skip_run('skip', 'Plot_transfer_learning_accuracies_wrt_#_of_samples') as check, check():

    # reported accuracies 
    score_base    = {}
    score_retrain = {}
    score_tl      = {}
    
    score_base['RM'] = [54.54, 54.54, 54.54, 54.54, 54.54]
    score_base['TD'] = [63.17, 63.17, 63.17, 63.17, 63.17]
    
    score_retrain['RM'] = [56.31, 59.35, 61.46, 62.32, 63.83]
    score_retrain['TD'] = [63.60, 63.97, 64.62, 65.01, 65.02]
    
    score_tl['RM'] = [71.01, 73.54, 75.26, 77.67, 77.73]
    
    sample_keys = np.arange(10, 51, 10)
    
    plt.figure()
    plt.plot(sample_keys, score_base['RM'], 'k-.', label='M1')
    # plt.plot(sample_keys, score_base['TD'], 'b-.',label='M1 TD')
    plt.plot(sample_keys, score_retrain['RM'], 'b--', label='M2')
    # plt.plot(sample_keys, score_retrain['TD'], 'b--', label='M2 TD')
    plt.plot(sample_keys, score_tl['RM'], 'r-', label='M3')
    
    plt.legend()
    # plt.xlabel('# of samples from $T_l$')
    # plt.ylabel('% Accuracy')
    # plt.rc('axes', labelsize=50) 
    plt.rc('legend', fontsize=15)
    plt.grid()
    plt.tight_layout()
    
      
# use it for the later part of the paper
with skip_run('skip', 'Calculate_inter-session_accuracy_of_each_subject_using_transfer_learning') as check, check():
    # number of samples from target to be used for training
    N = 20
    
    # load RM features
    RM_data = dd.io.load(Path(__file__).parents[1] / config['RM_features_subjectwise'])
    
    # load TD features
    TD_data = dd.io.load(Path(__file__).parents[1] / config['TD_features_subjectwise'])
    
    ### Prepare the source dataset
    source_subjects = list(set(config['subjects']) ^ set(config['test_subjects']))
    # source_subjects = config['train_subjects']
    source_TD       = []
    source_RM       = []
    source_cov      = []
    source_labels   = []
    
    for subject in source_subjects:
        for trial in (set(config['trials']) ^ set(config['comb_trials'])):  
            TD  = TD_data['subject_' + subject][trial]['TD']
            RM  = RM_data['subject_' + subject][trial]['RM']
            cov = RM_data['subject_' + subject][trial]['cov']
            y   = RM_data['subject_' + subject][trial]['labels']

            if (trial == "LowFine") or (trial == "HighGross"):
                # 1 sec and 0.5 overlap for 180 sec data = 360 points, use 165 out of them as these two classes are clubbed
                id = np.arange(165)       
            else:
                # 1 sec and 0.5 overlap for 180 sec data = 360 points, use 330 out of them
                id = np.arange(330)       

            TD  = TD[id, :]
            RM  = RM[id, :]
            cov = cov[id, :, :]
            y   = np.argmax(y[id, :], axis=1) + 1

            source_TD.append(TD)
            source_RM.append(RM)
            source_cov.append(cov)
            source_labels.append( y)
        
    Data = collections.defaultdict()
    
    Data['source_TD']     = np.concatenate(source_TD, axis=0)
    Data['source_RM']     = np.concatenate(source_RM, axis=0)
    Data['source_cov']    = np.concatenate(source_cov, axis=0)
    Data['source_labels'] = np.concatenate(source_labels, axis=0)

    ### Prepare the target dataset
    target_subjects = config['test_subjects']
        
    SCORES = collections.defaultdict()
    
    for subject in target_subjects:
        print('-------------Subject id: {}-----------'.format(subject))
        scores_noTL = collections.defaultdict()
        
        target_TD       = []
        target_RM       = []
        target_cov      = []
        target_labels   = []
        
        target_train_TD       = []
        target_train_RM       = []
        target_train_cov      = []
        target_train_labels   = []
    
        for trial in (set(config['trials']) ^ set(config['comb_trials'])):  
            TD  = TD_data['subject_' + subject][trial]['TD']
            RM  = RM_data['subject_' + subject][trial]['RM']
            cov = RM_data['subject_' + subject][trial]['cov']
            y   = RM_data['subject_' + subject][trial]['labels']

            y  = np.argmax(y, axis=1) + 1 
            
            if (trial == "LowFine") or (trial == "HighGross"):
                # 1 sec and 0.5 overlap for 180 sec data = 360 points, use 165 out of them as these two classes are clubbed
                id = np.arange(int(N/2), 165)       
            else:
                # 1 sec and 0.5 overlap for 180 sec data = 360 points, use 330 out of them
                id = np.arange(N, 330)       

            TD  = TD[id, :]
            RM  = RM[id, :]
            cov = cov[id, :, :]
            y   =  y[id]

            target_TD.append(TD)
            target_RM.append(RM)
            target_cov.append(cov)
            target_labels.append( y)
            
            # training data from target
            target_train_TD.append(TD[:N, :])
            target_train_RM.append(RM[:N, :])
            target_train_cov.append(cov[:N, :, :])
            target_train_labels.append(y[:N])
        
        Data['target_TD']     = np.concatenate(target_TD, axis=0)
        Data['target_RM']     = np.concatenate(target_RM, axis=0)
        Data['target_cov']    = np.concatenate(target_cov, axis=0)
        Data['target_labels'] = np.concatenate(target_labels, axis=0)
        
        Data['target_train_TD']     = np.concatenate(target_train_TD, axis=0)
        Data['target_train_RM']     = np.concatenate(target_train_RM, axis=0)
        Data['target_train_cov']    = np.concatenate(target_train_cov, axis=0)
        Data['target_train_labels'] = np.concatenate(target_train_labels, axis=0)
        
                    
        source_RM       = Data['source_RM']
        source_TD       = Data['source_TD']
        source_labels   = Data['source_labels']
        target_RM       = Data['target_RM']
        target_TD       = Data['target_TD']
        target_labels   = Data['target_labels']
        
        source_covest   = Data['source_cov']
        target_covest   = Data['target_cov']
        
        source_cov = np.reshape(source_covest, (source_covest.shape[0], source_covest.shape[1] * source_covest.shape[2]), order='C')
        source_cov = source_cov[:, 0:36]
        
        target_cov = np.reshape(target_covest, (target_covest.shape[0], target_covest.shape[1] * target_covest.shape[2]), order='C')
        target_cov = target_cov[:, 0:36]
    
        # classifier 
        clf = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr',probability=True)
        
        scores_noTL['train_RM'] = clf.fit(source_RM, source_labels).score(source_RM, source_labels)
        scores_noTL['train_TD'] = clf.fit(source_TD, source_labels).score(source_TD, source_labels)
        
        scores_noTL['test_RM'] = clf.fit(source_RM, source_labels).score(target_RM, target_labels)
        scores_noTL['test_TD'] = clf.fit(source_TD, source_labels).score(target_TD, target_labels)
        
        scores_noTL['retest_RM'] = clf.fit(source_RM, source_labels).score(target_RM, target_labels)
        scores_noTL['retest_TD'] = clf.fit(source_TD, source_labels).score(target_TD, target_labels)
        
        print("SVM training accuracy using Tangent Space:{}".format(scores_noTL['train_RM']))
        print("SVM training accuracy using Hudgins features:{}".format(scores_noTL['train_TD']))
        
        print("SVM inter-session accuracy using Tangent Space:{}".format(scores_noTL['test_RM']))
        print("SVM inter-session accuracy using Hudgins features:{}".format(scores_noTL['test_TD']))
    
        # use the part of target Data for training 
        source_RM       = np.concatenate((source_RM, Data['target_train_RM']), axis=0)
        source_TD       = np.concatenate((source_TD, Data['target_train_TD']), axis=0)
        source_labels   = np.concatenate((source_labels, Data['target_train_labels']), axis=0)
        
        # print("SVM retrain ({} samples) inter-session accuracy using Covariances:{}".format(N, clf.fit(source_cov, source_labels).score(target_cov, target_labels)))
        print("SVM retrain ({} samples) inter-session accuracy using Tangent Space:{}".format(N, scores_noTL['retest_RM']))
        print("SVM retrain ({} samples) inter-session accuracy using Hudgins features:{}".format(N, scores_noTL['retest_TD']))
        
        features_source = Data['source_cov']
        labels_source   = Data['source_labels']
        features_target = Data['target_cov']
        labels_target   = Data['target_labels']
        
        print('Source Class1: {}, Class2: {}, Class3: {}'.format(len(labels_source[labels_source == 1]), len(labels_source[labels_source == 2]), len(labels_source[labels_source == 3])))
        
        print('Target Class1: {}, Class2: {}, Class3: {}'.format(len(labels_target[labels_target == 1]), len(labels_target[labels_target == 2]), len(labels_target[labels_target == 3])))
        
        # Number of Datapoints from session 2 to be considered for training
        ncovs_target_train = 10
        data_source       = {}
        data_target       = {}
        data_target_train = {}
    
        # prepare the source data from session 1
        data_source['covs']   = features_source
        data_source['labels'] = labels_source

        # prepare the target data from session 2
        data_target['covs']   = features_target
        data_target['labels'] = labels_target

        # prepare training dataset from session 2
        data_target_train['covs']  = data['target_train_cov']
        data_target_train['labels']= data['target_train_labels']
        
        # setup the scores dictionary
        scores = {}
        for meth in ['org', 'rct', 'str', 'rot']:
            scores[meth] = []

        # run the transfer learning for 5 times 
        for i in tqdm(range(5)):
            # apply RPA to multiple random partitions for the training dataset
            clf = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr',probability=True)
            
            source = {}
            target_train = {}
            target_test = {}

            # source['org'], target_train['org'], target_test['org'] = TL.get_sourcetarget_split(data_source, data_target, ncovs_target_train, paradigm='MI')
            source['org'] = data_source
            target_train['org'] = data_target_train
            target_test['org'] = data_target
            
            # apply RPA 
            source['rct'], target_train['rct'], target_test['rct'] = TL.RPA_recenter(source['org'], target_train['org'], target_test['org'])
            source['str'], target_train['str'], target_test['str'] = TL.RPA_stretch(source['rct'], target_train['rct'], target_test['rct'])
            source['rot'], target_train['rot'], target_test['rot'] = TL.RPA_rotate(source['str'], target_train['str'], target_test['str'])

            for meth in source.keys():
                source[meth]['ts']       = TangentSpace().fit_transform(source[meth]['covs'])
                target_train[meth]['ts'] = TangentSpace().fit_transform(target_train[meth]['covs'])
                target_test[meth]['ts']  = TangentSpace().fit_transform(target_test[meth]['covs'])
                
                scores[meth].append(TL.get_tangent_space_score_transferlearning(clf, source[meth], target_train[meth], target_test[meth]))
                
        # print the scores
        for meth in scores.keys():
            print(meth, np.mean(scores[meth]), np.std(scores[meth]))
        
        SCORES[subject] = {'scores_noTL': scores_noTL, 'scores_TL' : scores} 
    
    dd.io.save(str(Path(__file__).parents[1] / config['scores_RPA']), SCORES)
    
plt.show()