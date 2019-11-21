import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import deepdish as dd
import sys
import random
import statistics
import copy
import hdf5storage
import h5py
import numpy as np
from pathlib import Path

from pyriemann.estimation import Covariances, Shrinkage, Coherences
from pyriemann.tangentspace import TangentSpace, FGDA

from pathlib import Path
import collections
from data.clean_data import clean_epoch_data
from data.create_data import (create_emg_data, create_emg_epoch, create_PB_data,
                              create_PB_epoch, create_robot_dataframe,
                              sort_order_emg_channels)
from data.create_data_sri import read_Pos_Force_data, epoch_raw_emg, pool_emg_data

from datasets.riemann_datasets import subject_pooled_EMG_data, train_test_data, subject_dependent_data

from datasets.torch_datasets import pooled_data_iterator
from datasets.statistics_dataset import matlab_dataframe

from models.riemann_models import (svm_tangent_space_classifier,
                                   svm_tangent_space_cross_validate,
                                   svm_tangent_space_prediction)
from models.statistical_models import mixed_effect_model
from models.torch_models import train_torch_model
from models.torch_networks import ShallowERPNet
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from models.emg_features import (extract_emg_features, pool_subject_emg_features,
                                svm_cross_validated_pooled_emg_features,
                                balance_pooled_emg_features,
                                lda_cross_validated_pooled_emg_features)

from visualization.visualise import (plot_average_model_accuracy, plot_bar)
from utils import (skip_run, save_data, save_trained_pytorch_model)

from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from scipy import signal
from sklearn.manifold import TSNE
from umap import UMAP
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix

# The configuration file
config = yaml.load(open('src/config.yml'), Loader=yaml.SafeLoader)

with skip_run('skip', 'create_emg_data') as check, check():
    data = create_emg_data(config['subjects'], config['trials'], config)

    # Save the dataset
    save_path = Path(__file__).parents[1] / config['raw_emg_data']
    save_data(str(save_path), data, save=True)

with skip_run('skip', 'create_epoch_data') as check, check():
    data = create_emg_epoch(config['subjects'], config['trials'], config)

    # Save the dataset
    save_path = Path(__file__).parents[1] / config['epoch_emg_data']
    save_data(str(save_path), data, save=True)

with skip_run('skip', 'clean_epoch_data') as check, check():
    data = clean_epoch_data(config['subjects'], config['trials'], 'emg', config)

    # Save the dataset
    save_path = Path(__file__).parents[1] / config['clean_emg_data']
    save_data(str(save_path), data, save=True)

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
    path = str(Path(__file__).parents[1] / config['clean_emg_data'])

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
    path = str(Path(__file__).parents[1] / config['clean_emg_data'])
    # Load main data
    features, labels, leave_tags = subject_pooled_EMG_data(config['subjects'], path, config)
    # Get the data
    data = train_test_data(features, labels, leave_tags, config)

    # Train the classifier and predict on test data
    clf = svm_tangent_space_classifier(data['train_x'], data['train_y'])
    svm_tangent_space_prediction(clf, data['test_x'], data['test_y'])

with skip_run('skip', 'svm_cross_validated_pooled_data') as check, check():
    path = str(Path(__file__).parents[1] / config['clean_emg_data'])
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
    sort_channels = True
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

##-- Classification using Riemannian features--##
with skip_run('skip', 'classify_using_riemannian_emg_features') as check, check():
    # Subject information
    subjects = config['subjects']
    path = str(Path(__file__).parents[1] / config['clean_emg_data'])
    # Load main data
    features, labels, _ = subject_pooled_EMG_data(subjects, path, config)

    X   = features
    y   = np.dot(labels,np.array([1,2,3]))
    print(X.shape)
    print('# of samples in Class 1:%d, Class 2:%d, Class 3:%d' % (y[y==1].shape[0],y[y==2].shape[0],y[y==3].shape[0]))

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

    accuracy = cross_val_score(clf1, ts, y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using SVM: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

    accuracy = cross_val_score(clf2, ts, y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using Random Forest: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

with skip_run('skip', 'inter_subject_transferability_using_riemannian_features') as check, check():
    # Subject information
    subjects = copy.copy(config['subjects'])
    random.shuffle(subjects)

    # Number of subjects to train the classifier
    N = 8

    path = str(Path(__file__).parents[1] / config['clean_emg_data'])
    # Load main data
    train_x, train_y, _ = subject_pooled_EMG_data(subjects[0:N], path, config)
    test_x, test_y, _   = subject_pooled_EMG_data(subjects[N:], path, config)

    train_y = np.dot(train_y,np.array([1,2,3]))
    test_y  = np.dot(test_y,np.array([1,2,3]))

    # print('# of samples in Class 1:%d, Class 2:%d, Class 3:%d' % (y[y==1].shape[0],y[y==2].shape[0],y[y==3].shape[0]))

    # estimation of the covariance matrix and its projection in tangent space
    train_cov = Covariances().fit_transform(train_x)
    train_ts  = TangentSpace().fit_transform(train_cov)

    test_cov  = Covariances().fit_transform(test_x)
    test_ts   = TangentSpace().fit_transform(test_cov)

    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)

    accuracy = clf1.fit(train_ts, train_y).score(test_ts, test_y)
    print("Inter-subject tranfer accuracy using SVM: %0.4f " % accuracy.mean())

    accuracy = clf2.fit(train_ts, train_y).score(test_ts, test_y)
    print("Inter-subject tranfer accuracy using Random Forest: %0.4f " % accuracy.mean())

with skip_run('run', 'create_PB_data') as check, check():
    data = create_PB_data(config['subjects'], config['trials'], config)
    
    # save the data
    path = Path(__file__).parents[1] / config['raw_PB_data']
    dd.io.save(str(path), data)

with skip_run('skip', 'create_PB_epoch') as check, check():
    data = create_PB_epoch(config['subjects'], config['trials'], config)
    
    # save the data
    path = Path(__file__).parents[1] / config['epoch_PB_data']
    dd.io.save(str(path), data)

with skip_run('skip', 'clean_epoch_data') as check, check():
    data = clean_epoch_data(config['subjects'], config['trials'], 'PB', config)

    # Save the dataset
    save_path = Path(__file__).parents[1] / config['clean_PB_data']
    save_data(str(save_path), data, save=True)

with skip_run('skip', 'classify_using_mean_force_features') as check, check():
    # Subject information
    subjects = config['subjects']
    path = str(Path(__file__).parents[1] / config['clean_PB_data'])
    # Load main data
    features, labels, _ = subject_pooled_EMG_data(subjects, path, config)
    
    X   = features[:,0:2,:]
    y   = np.dot(labels,np.array([1,2,3]))
    print(X.shape)
    print('# of samples in Class 1:%d, Class 2:%d, Class 3:%d' % (y[y==1].shape[0],y[y==2].shape[0],y[y==3].shape[0]))

    ts = np.mean(X, axis=2)

    # SVM classifier
    clf1 = SVC(kernel='rbf', gamma='auto', decision_function_shape ='ovr')
    # Random forest classifier
    clf2 = RandomForestClassifier(n_estimators=100, oob_score=True)

    accuracy = cross_val_score(clf1, ts, y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using SVM: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

    accuracy = cross_val_score(clf2, ts, y, cv=KFold(5,shuffle=True))
    print("cross validation accuracy using Random Forest: %0.4f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))

with skip_run('skip', 'classify_using_riemannian_force_features') as check, check():
    # Subject information
    subjects = config['subjects']
    path = str(Path(__file__).parents[1] / config['clean_force_data'])
    # Load main data
    features, labels, _ = subject_pooled_EMG_data(subjects, path, config)

    X   = features[:,0:2,:]
    y   = np.dot(labels,np.array([1,2,3]))
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
    
    # Load main data
    path1 = str(Path(__file__).parents[1] / config['clean_emg_data'])
    features1, labels1, _ = subject_pooled_EMG_data(subjects, path1, config)

    path2 = str(Path(__file__).parents[1] / config['clean_force_data'])
    features2, labels2, _ = subject_pooled_EMG_data(subjects, path2, config)

    print('EMG samples: %d, Force samples: %d ' %(features1.shape[0], features2[0:features1.shape[0],:].shape[0]))
    # sys.exit()

    X   = np.concatenate((features1, features2[0:features1.shape[0],0:2,:]), axis=1)
    y   = np.dot(labels1,np.array([1,2,3]))
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

## ----------------------------------------------------------##
#-- Projecting the EMG data onto manifolds --#
with skip_run('skip', 'project_EMG_Riemannian_features_data') as check, check():
    # Subject information
    subjects = config['subjects']

    path = str(Path(__file__).parents[1] / config['clean_emg_data'])
    # Load main data
    features, labels, _ = subject_pooled_EMG_data(subjects, path, config)

    X   = features
    y   = np.dot(labels,np.array([1,2,3]))

    # estimation of the covariance matrix
    covest = Covariances().fit_transform(X)

    # project the covariance into the tangent space
    # ts = TangentSpace().fit_transform(covest)

    ts = np.reshape(covest, (covest.shape[0], covest.shape[1] * covest.shape[2]), order='C')
    print(ts.shape)
    ts = ts[:,0:36]

    print('t-SNE based data visualization')
    temp1 = y == 1
    temp2 = y == 2
    temp3 = y == 3

    # TSNE based projection
    # X_embedded = TSNE(n_components=2, perplexity=100, learning_rate=50.0).fit_transform(ts)

    # plt.figure()
    # plt.plot(X_embedded[temp1,0],X_embedded[temp1,1],'bo')
    # plt.plot(X_embedded[temp2,0],X_embedded[temp2,1],'ro')
    # plt.plot(X_embedded[temp3,0],X_embedded[temp3,1],'ko')
    # plt.show()

    # UMAP based projection
    for neighbor in [10, 30, 60 ]:
        fit = UMAP(n_neighbors=neighbor, min_dist=0.0, n_components=3,metric='chebyshev')
        X_embedded = fit.fit_transform(ts)
    
        fig = plt.figure()
        ax = Axes3D(fig)
    
        ax.plot(X_embedded[temp1,0],X_embedded[temp1,1],X_embedded[temp1,2],'bo')
        ax.plot(X_embedded[temp2,0],X_embedded[temp2,1],X_embedded[temp2,2],'ro')
        ax.plot(X_embedded[temp3,0],X_embedded[temp3,1],X_embedded[temp3,2],'ko')
    plt.show()

#-- Projecting the force and velocity data onto manifolds --#
with skip_run('skip', 'project_Force_data') as check, check():
    # Subject information
    subjects = config['subjects']

    path = str(Path(__file__).parents[1] / config['clean_force_data'])
    # Load main data
    features, labels, _ = subject_pooled_EMG_data(subjects, path, config)

    X   = features[:,0:2,:]
    y   = np.dot(labels,np.array([1,2,3]))

    # estimation of the covariance matrix
    covest = Covariances().fit_transform(X)

    # project the covariance into the tangent space
    # ts = TangentSpace().fit_transform(covest)
    
    ts = np.reshape(covest, (covest.shape[0], covest.shape[1] * covest.shape[2]), order='C')
    print(ts.shape)
    ts = ts[:,0:36]

    print('t-SNE based data visualization')
    temp1 = y == 1
    temp2 = y == 2
    temp3 = y == 3

    # TSNE based projection
    # X_embedded = TSNE(n_components=2, perplexity=100, learning_rate=50.0).fit_transform(ts)

    # plt.figure()
    # plt.plot(X_embedded[temp1,0],X_embedded[temp1,1],'bo')
    # plt.plot(X_embedded[temp2,0],X_embedded[temp2,1],'ro')
    # plt.plot(X_embedded[temp3,0],X_embedded[temp3,1],'ko')
    # plt.show()

    # UMAP based projection
    for neighbor in [10, 30, 60 ]:
        fit = UMAP(n_neighbors=neighbor, min_dist=0.5, n_components=3)
        X_embedded = fit.fit_transform(ts)
    
        fig = plt.figure()
        ax = Axes3D(fig)

        # plt.plot(X_embedded[temp1,0],X_embedded[temp1,1],'bo')
        # plt.plot(X_embedded[temp2,0],X_embedded[temp2,1],'ro')
        # plt.plot(X_embedded[temp3,0],X_embedded[temp3,1],'ko')

        ax.plot(X_embedded[temp1,0],X_embedded[temp1,1],X_embedded[temp1,2],'bo')
        ax.plot(X_embedded[temp2,0],X_embedded[temp2,1],X_embedded[temp2,2],'ro')
        # ax.plot(X_embedded[temp3,0],X_embedded[temp3,1],X_embedded[temp3,2],'ko')
    plt.show()

## ----------alternative code for reading csv files----------##
##-- Task type classification using Riemannian features --#
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
    #         cohest = signal.coherence(X[:,i], X[:,j], fs=200.0, window='hann', nperseg=200, noverlap=50, nfft=None, detrend='constant', axis=-1)
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
