import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.signal import hilbert
import scipy.stats as stats
from ML_models import conv_nn_2, conv_nn, MachineLearning

from sklearn import svm
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.decomposition import PCA, FastICA

import tensorflow as tf
# from keras.models import Sequential
# from keras import backend as K
# from keras.utils.generic_utils import get_custom_objects

np.random.seed(7)
tf.keras.backend.clear_session()  # For easy reset of notebook State


def remove_ones(data):
    temp_data = pd.DataFrame()
    temp_ones = pd.DataFrame()
    for i in range(len(data)):
        if data.iloc[i, 363] == 0:
            temp_arr = data.iloc[i, :].T
            temp_data = temp_data.append(temp_arr)
        else:
            temp_arr_1 = data.iloc[i, :].T
            temp_ones = temp_ones.append(temp_arr_1)

    return temp_data, temp_ones


def relu_c(x):
    y_r = max(0, x.real)
    y_c = max(0, x.imag)
    return y_r, y_c


def softmax_c(x):
    top = np.exp(np.power(x.real, 2) + np.power(x.imag, 2))
    bottom = np.sum(top)
    return top / bottom


def freq_labels(x):
    temp_one = []
    temp_two = []
    for i in x.values:
        comp = i[362]
        if comp in range(0, 35):
            temp_one.append(0)
        elif comp in range(35, 70):
            temp_one.append(1)
        elif comp in range(70, 106):
            temp_one.append(2)
        elif comp in range(106, 180):
            # print("here")
            temp_one.append(3)

    x[364] = temp_one

    return x


def test_train_format(data):
    training_size = int(len(data) * 0.67)

    train_ = data.iloc[0:training_size, 0:360].values
    test_ = data.iloc[training_size:len(data), 0:360].values

    train_labels_ = data.iloc[0:training_size, 360].values
    test_labels_ = data.iloc[training_size:len(data), 360].values

    return train_, test_, train_labels_, test_labels_


def principle_components(train_data, test_data):
    pca_ = PCA(0.95)
    pca_.fit(train_data)

    x_train_pca = pca_.transform(train_data)
    x_test_pca = pca_.transform(test_data)

    return x_train_pca, x_test_pca


def svm_svc(pca, train_data, test_data, train_labels, test_labels):
    test_case = np.array(len(test_data))
    if not pca:
        clf = svm.SVC()
        clf.fit(train_data, np.ravel(train_labels, order='C'))

        test_case = clf.predict(test_data)
        print("The SVM model accuracy is: ", accuracy_score(test_case, test_labels))

    elif pca:

        x_train_pca, x_test_pca = principle_components(train_data, test_data)

        clf_pca = svm.SVC()
        clf_pca.fit(x_train_pca, np.ravel(train_labels, order='C'))

        test_case = clf_pca.predict(x_test_pca)
        print(type(test_case))
        print("The SVM model accuracy, using principle components, is: ",
              accuracy_score(test_case, test_labels))

    return test_case


df_ = pd.read_csv("Final_dataset\\Filter_specific_data\\ellip_filter_data.csv", header=None, index_col=False)
df_one = pd.read_csv("Final_dataset\\Filter_specific_data\\wavelet_filter_data.csv", header=None, index_col=False)
df_two = pd.read_csv("Final_dataset\\Clean_noise_data_trial\\Clean_signal.csv", header=None, index_col=False)

df_, df_1 = remove_ones(df_)
df_one, df_one_1 = remove_ones(df_one)
df_two, df_two_1 = remove_ones(df_two)

# for the normal SNR values
df_one = df_one.sample(frac=1).reset_index(drop=True)
df_two = df_two.sample(frac=1).reset_index(drop=True)

df_ = df_.append(df_one, ignore_index=True)
df_ = df_.sample(frac=1).reset_index(drop=True)
df_two = df_two.loc[0:len(df_)*0.5, :]

# for the low SNR values
df_one_1 = df_one_1.sample(frac=1).reset_index(drop=True)
df_two_1 = df_two_1.sample(frac=1).reset_index(drop=True)

df_1 = df_1.append(df_one_1, ignore_index=True)
df_1 = df_1.sample(frac=1).reset_index(drop=True)
df_two_1 = df_two_1.loc[0:len(df_1)*0.5, :]
print(len(df_), len(df_two))

# new_df = freq_labels(df_)

train_x, test_x, train_y, test_y = test_train_format(df_)
train_x_, test_x_, train_y_, test_y_ = test_train_format(df_two)

# print(train_y[0:5], len(train_y))
# print("\n")
train_x, train_y = np.vstack([train_x_, train_x]), np.append(train_y_, train_y)
# print(train_y[0:5], len(train_y))

# weights_one = np.ones_like(train_x[0])/len(train_x[0])
# bins = plt.hist(train_x[0], bins=180, weights=weights_one, alpha=0.5, density=True)

# svm_pred = svm_svc(1, train_x, test_x, train_y, test_y)
# rev_svm_pred = svm_svc(1, train_x, test_x, )

# print(train_x[0:5], train_y[0:5])
conv_nn_2(train_x, test_x, train_y, test_y, 2)
# conv_nn(train_x, test_x, train_y, test_y, 2)
# MachineLearning(0, train_x, test_x, train_y, test_y, 3).svm_svc()
