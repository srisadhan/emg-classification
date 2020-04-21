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

from pyriemann.estimation import Covariances, Shrinkage, Coherences
from pyriemann.tangentspace import TangentSpace, FGDA
from pyriemann.utils.distance import distance, distance_riemann, distance_logeuclid
from pyriemann.utils.mean import mean_riemann
from pyriemann.classification import MDM, TSclassifier
from pyriemann.channelselection import ElectrodeSelection
from pyriemann.embedding import Embedding
from pyriemann.spatialfilters import SPoC, CSP
# from pyriemann.utils.viz import plot_confusion_matrix

from pathlib import Path
import collections
from data.clean_data import (clean_epoch_data, clean_combined_data, clean_intersession_test_data, 
                            clean_combined_data_for_fatigue_study, clean_correction_data, balance_correction_data, pool_correction_data)
from data.create_data import (create_emg_data, create_emg_epoch, create_PB_data,
                              create_PB_epoch, create_robot_dataframe,
                              sort_order_emg_channels)
from data.create_data_sri import read_Pos_Force_data, epoch_raw_emg, pool_emg_data

from datasets.riemann_datasets import (subject_pooled_EMG_data, 
                                       train_test_data, 
                                       subject_dependent_data,
                                       subject_pooled_EMG_PB_data,
                                       split_pooled_EMG_PB_data_train_test)

from datasets.torch_datasets import pooled_data_iterator, pooled_data_SelfCorrect_NN
from datasets.statistics_dataset import matlab_dataframe

from models.riemann_models import (tangent_space_classifier,
                                   svm_tangent_space_cross_validate,
                                   tangent_space_prediction)
from models.statistical_models import mixed_effect_model
from models.torch_models import train_torch_model, train_correction_network
from models.torch_networks import ShallowERPNet, ShallowCorrectionNet
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, SVR 
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.covariance import ShrunkCovariance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from features.emg_features import (extract_emg_features, pool_subject_emg_features,
                                svm_cross_validated_pooled_emg_features,
                                balance_pooled_emg_features,
                                lda_cross_validated_pooled_emg_features)

from features.force_features import (extract_passivity_index, pool_force_features)

from visualization.visualise import (plot_average_model_accuracy, plot_bar)
from utils import (skip_run, save_data, save_trained_pytorch_model, plot_confusion_matrix)

from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
import scipy  
from sklearn.manifold import TSNE
from umap import UMAP
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix, accuracy_score
import treegrad as tgd 
import joblib

# The configuration file
config = yaml.load(open('src/config.yml'), Loader=yaml.SafeLoader)

### --------------- Preprocessing data --------------------###
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

with skip_run('skip', 'clean_emg_epoch') as check, check():
    data = clean_epoch_data(config['subjects'], config['trials'], 'EMG', config)
    
    # Save the dataset
    if config['n_class'] == 3:
        save_path = Path(__file__).parents[1] / config['clean_emg_data_3class']
    elif config['n_class'] == 4:
        save_path = Path(__file__).parents[1] / config['clean_emg_data_4class']

    save_data(str(save_path), data, save=True)

with skip_run('skip', 'clean_PB_epoch_data') as check, check():
    data = clean_epoch_data(config['subjects'], config['trials'], 'PB', config)

    # Save the dataset
    if config['n_class'] == 3:
        save_path = Path(__file__).parents[1] / config['clean_PB_data_3class']
    elif config['n_class'] == 4:
        save_path = Path(__file__).parents[1] / config['clean_PB_data_4class']
    save_data(str(save_path), data, save=True)

with skip_run('skip', 'save_EMG_PB_data') as check, check():
    
    subjects = config['subjects']
    features = clean_combined_data(subjects, config['trials'], config['n_class'], config)

    # path to save
    path = str(Path(__file__).parents[1] / config['clean_emg_pb_data'])
    dd.io.save(path, features)

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
    data = sort_order_emg_channels(config)

    # save the data in h5 format
    path = str(Path(__file__).parents[1] / config['emg_channel_order'])
    save_data(path,data,save=True)

with skip_run('skip', 'extract_emg_features') as check, check():
    sort_channels = False
    if sort_channels:
        print("-----Sorting the features based on the decreasing variance of the EMG for each subject-----")
    else:
        print("-----Using the EMG channels AS-IS without sorting them-----")

    data = extract_emg_features(config, sort_channels)

    # save the data in h5 format
    path = str(Path(__file__).parents[1] / config['subject_emg_features'])
    save_data(path,data,save=True)

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

with skip_run('skip', 'pool_subject_emg_features') as check, check():

    X1, X2, Y = pool_subject_emg_features(config)

    data = {}
    data['X1'] = X1
    data['X2'] = X2
    data['Y']  = Y

    # save the data in h5 format
    path = str(Path(__file__).parents[1] / config['pooled_emg_features'])
    save_data(path,data,save=True)

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

with skip_run('skip', 'balance_pooled_emg_features') as check, check():
    X1_res, X2_res, Y_res = balance_pooled_emg_features(config)

    print('Class 1: ', Y_res[Y_res==1].shape, 'Class 2: ', Y_res[Y_res==2].shape, 'Class 3: ', Y_res[Y_res==3].shape, 'Class 4: ', Y_res[Y_res==4].shape)

    print('---Accuracy of SVM for Balanced data for feature set 1---')
    svm_cross_validated_pooled_emg_features(X1_res, Y_res, config)
    lda_cross_validated_pooled_emg_features(X1_res, Y_res, config)

    print('---Accuracy of SVM for Balanced data for feature set 2---')
    svm_cross_validated_pooled_emg_features(X2_res, Y_res, config)
    lda_cross_validated_pooled_emg_features(X2_res, Y_res, config)
## ----------------------------------------------------------##


##-- Classification with 4 main sets of features RMS, TD, HIST, mDWT (October results)--##
## ----------------------------------------------------------##


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
    subjects = set(config['subjects']) ^ set(config['test_subjects'])
    if config['n_class'] == 3:
        save_path = str(Path(__file__).parents[1] / config['clean_emg_data_3class'])
    elif config['n_class'] == 4:
        save_path = str(Path(__file__).parents[1] / config['clean_emg_data_4class'])
    # Load main data
    features, labels, _ = subject_pooled_EMG_data(subjects, save_path, config)

    #FIXME:
    # ------------------Remove this later --------------------
    path = str(Path(__file__).parents[1] / config['clean_emg_pb_data'])    
    features, _ , labels= subject_pooled_EMG_PB_data(subjects, path, config)
    X   = features
    y   = np.dot(labels,np.array(np.arange(1, config['n_class']+1)))
    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(y[:,np.newaxis], y[:,np.newaxis])
    X = X[rus.sample_indices_, :, :]
    y = y[rus.sample_indices_]
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
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)
    
    accuracy = cross_val_score(clf1, ts, y, cv=KFold(10,shuffle=True))
    print("cross validation accuracy using SVM: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

    accuracy = cross_val_score(clf2, ts, y, cv=KFold(10,shuffle=True))
    print("cross validation accuracy using Random Forest: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

    clf2.fit(ts, y)
    # save the model to disk
    filename = str(Path(__file__).parents[1] / config['saved_RF_classifier'])
    joblib.dump(clf2, filename)

    # Linear discriminant analysis - # does not provide good accuracy
    # clf3 = LinearDiscriminantAnalysis(solver='svd')
    # accuracy = cross_val_score(clf3, ts, y, cv=KFold(5,shuffle=True))
    # print("cross validation accuracy using LDA: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

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
    emg, pb, labels = subject_pooled_EMG_PB_data(subjects, path, config)

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
    subjects = copy.copy(config['subjects'])
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
    subjects_test = config['test_subjects']
    print('List of subject for training: ', subjects_train)
    print('List of subject for testing : ', subjects_test)

    # clean_intersession_test_data(subjects_test, config['comb_trials'], config['n_class'], config)

    # load the data
    path = str(Path(__file__).parents[1] / config['clean_emg_pb_data'])
    
    train_emg, train_pos, train_y = subject_pooled_EMG_PB_data(subjects_train, path, config)
    test_emg, test_pos, test_y   = subject_pooled_EMG_PB_data(subjects_test, path, config)

    # convert the labels from one-hot-encoding to int
    train_y = np.dot(train_y,np.array(np.arange(1, config['n_class']+1)))
    test_y = np.dot(test_y,np.array(np.arange(1, config['n_class']+1)))

    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(train_y[:,np.newaxis], train_y[:,np.newaxis])

    train_emg = train_emg[rus.sample_indices_, :, :]
    train_pos = train_pos[rus.sample_indices_, :, :]
    train_y = train_y[rus.sample_indices_]


    # print('# of samples in Class 1:%d, Class 2:%d, Class 3:%d, Class:4%d' % (y[y==1].shape[0],y[y==2].shape[0],y[y==3].shape[0],y[y==4].shape[0]))

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

    accuracy = clf1.fit(train_X, train_y).score(test_X, test_y)    
    print("Inter-session tranfer accuracy using SVM: %0.4f " % accuracy.mean())

    accuracy = clf2.fit(train_X, train_y).score(test_X, test_y)
    print("Inter-session tranfer accuracy using Random Forest: %0.4f " % accuracy.mean())

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

## ------------Project features on to manifold----------------##
with skip_run('skip', 'project_EMG_features') as check, check():
    # Subject information
    subjects = config['subjects']

    if config['n_class'] == 3:
        path = str(Path(__file__).parents[1] / config['clean_emg_data_3class'])
    elif config['n_class'] == 4:
        path = str(Path(__file__).parents[1] / config['clean_emg_data_4class'])
        
    # Load main data
    features, labels, _ = subject_pooled_EMG_data(subjects, path, config)

    X   = features
    y   = np.dot(labels,np.array(np.arange(1, config['n_class']+1)))

    # if I want to project the features of only two classes
    # X   = X[(y==1) | (y==4), :, :]
    # y   = y[(y==1) | (y==4)]

    # estimation of the covariance matrix
    covest = Covariances().fit_transform(X)

    # project the covariance into the tangent space
    ts = TangentSpace().fit_transform(covest)

    # ts = np.reshape(covest, (covest.shape[0], covest.shape[1] * covest.shape[2]), order='C')
    # print(ts.shape)
    # ts = ts[:,0:36]
    
    temp1 = y == 1
    temp2 = y == 2
    temp3 = y == 3

    # TSNE based projection
    # print('t-SNE based data visualization')
    # X_embedded = TSNE(n_components=2, perplexity=100, learning_rate=50.0).fit_transform(ts)

    # plt.figure()
    # plt.plot(X_embedded[temp1,0],X_embedded[temp1,1],'bo')
    # plt.plot(X_embedded[temp2,0],X_embedded[temp2,1],'ro')
    # plt.plot(X_embedded[temp3,0],X_embedded[temp3,1],'ko')
    # plt.show()

    # UMAP based projection
    for neighbor in [10]: #, 30, 60]:
        fit = UMAP(n_neighbors=neighbor, min_dist=0.75, n_components=3, metric='chebyshev')
        X_embedded = fit.fit_transform(ts)
    
        fig = plt.figure()
        ax = Axes3D(fig)

        # ax.scatter(X_embedded[:,0], X_embedded[:,1], X_embedded[:,2], c=y.astype(int), cmap='viridis', s=2)

        ax.plot(X_embedded[temp1,0],X_embedded[temp1,1],X_embedded[temp1,2],'b.')
        ax.plot(X_embedded[temp2,0],X_embedded[temp2,1],X_embedded[temp2,2],'ro')
        ax.plot(X_embedded[temp3,0],X_embedded[temp3,1],X_embedded[temp3,2],'k.')
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
    
    train_emg, train_pos, train_y = subject_pooled_EMG_PB_data(subjects_train, path, config)
    test_emg, test_pos, test_y   = subject_pooled_EMG_PB_data(subjects_test, path, config)

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

    # print probabilities and log-probabilities
    print(clf2.predict_proba(test_X).shape, test_y, clf2.predict_proba(test_X))
    print(clf2.predict_log_proba(test_X).shape, test_y, clf2.predict_log_proba(test_X))
    sys.exit()

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

    # balance the data
    data = balance_correction_data(data, trials, config['subjects'], config)

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
    clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)
    
    accuracy = cross_val_score(clf2, ts, y, cv=KFold(10,shuffle=True))
    print("cross validation accuracy using SVM: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

    clf2.fit(ts, y)
    accuracy = clf2.score(ts, y)
    print("Training accuracy of  Random Forest: %0.4f" % (accuracy))
    
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

    # balance the data
    data = balance_correction_data(data, trials, config['subjects'], config)

    # create the dataset for NN
    dataset = pooled_data_SelfCorrect_NN(data, RF_clf, config)

    # save the dataset
    filepath = Path(__file__).parents[1] / config['Self_correction_dataset']
    dd.io.save(filepath, dataset)


with skip_run('skip', 'self_correction_of_classifier_output') as check, check():
    
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