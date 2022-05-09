import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Dropout
from tensorflow.keras.layers import LSTM, MaxPool1D
from tensorflow.keras.utils import to_categorical, normalize

np.random.seed(7)
refactor = 10


def complie_data():
    data_ = []
    labels_ = []
    label_details_ = []

    for i in os.listdir("Completed_dataset"):
        data_file_name = "Completed_dataset\\" + i

        temp = pd.read_csv(data_file_name, header=None, index_col=False)
        data_.append(temp.iloc[:, 3])
        labels_.append(temp.iloc[:, 4])
        label_details_.append(temp.iloc[:, 5])

    return data_, labels_, label_details_


def format_data():
    data, labels, label_details = complie_data()
    data_arr_ = np.array(data).reshape(len(data) * refactor, int(len(data[0]) / refactor))
    labels_arr_ = []
    label_details_arr_ = []

    for i in range(len(labels)):
        labels_arr_.append(labels[i][0::360])
        label_details_arr_.append(label_details[i][0::360])

    labels_arr_ = np.array(labels_arr_).reshape(len(labels_arr_) * refactor, int(len(labels_arr_[0]) / refactor))
    label_details_arr_ = np.array(label_details_arr_).reshape(len(label_details_arr_) * refactor,
                                                              int(len(label_details_arr_[0]) / refactor))

    return data_arr_, labels_arr_, label_details_arr_


def lstm_network(data_full, look_back_1):
    # normalising the dataset (already normalised but require scalar property
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_set = scaler.fit_transform(data_full)

    train_size = int(len(data_full) * 0.67)
    train, test = data_set[0:train_size, :], data_set[train_size:len(data_set), :]
    train_x, train_y = create_dataset(train, look_back_1)
    test_x, test_y = create_dataset(test, look_back_1)

    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back_1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_x, train_y, epochs=50, batch_size=1, verbose=2)

    # make predictions
    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)

    # invert predictions
    train_predict = scaler.inverse_transform(train_predict)
    train_y = scaler.inverse_transform([train_y])

    test_predict = scaler.inverse_transform(test_predict)
    test_y = scaler.inverse_transform([test_y])

    # calculate the mean square error (mse)
    train_score = math.sqrt(mean_squared_error(train_y[0], train_predict[:, 0]))
    print("Train Score: %.2f RMSE" % train_score)

    test_score = math.sqrt(mean_squared_error(test_y[0], test_predict[:, 0]))
    print("Test Score: %.2f RMSE" % test_score)

    # plot baseline and predictions

    # shifting train predictions
    train_predict_plot = np.empty_like(data_full)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back_1:len(train_predict) + look_back_1, :] = train_predict

    # shifting test predictions
    test_predict_plot = np.empty_like(data_full)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (look_back_1 * 2) + 1:len(data_full) - 1, :] = test_predict

    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(data_full), 'b')
    plt.plot(train_predict, 'g')
    plt.plot(test_predict, 'orange')
    plt.show()


def principle_components(train_data, test_data):
    pca_ = PCA(0.98)
    pca_.fit(train_data)

    x_train_pca = pca_.transform(train_data)
    x_test_pca = pca_.transform(test_data)

    return x_train_pca, x_test_pca


def independent_components(train_data, test_data):
    transformer = FastICA(n_components=100, random_state=0)
    x_train_transform = transformer.fit_transform(train_data)
    x_test_transform = transformer.fit_transform(test_data)

    return x_train_transform, x_test_transform


class MachineLearning:

    def __init__(self, pca, train_data, test_data, train_labels, test_labels, classes):
        self.pca = pca
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.classes = classes

    def svm_svc(self):

        if not self.pca:
            clf = svm.SVC()
            clf.fit(self.train_data, np.ravel(self.train_labels, order='C'))

            test_case = clf.predict(self.test_data)
            print("The SVM model accuracy is: ", accuracy_score(test_case, self.test_labels))

        elif self.pca:

            x_train_pca, x_test_pca = principle_components(self.train_data, self.test_data)

            clf_pca = svm.SVC()
            clf_pca.fit(x_train_pca, np.ravel(self.train_labels, order='C'))

            test_case_pca = clf_pca.predict(x_test_pca)
            print("The SVM model accuracy, using principle components, is: ",
                  accuracy_score(test_case_pca, self.test_labels))

    def logistic_regression(self):

        if not self.pca:

            log_reg = LogisticRegression(max_iter=10000)
            log_reg.fit(self.train_data, np.ravel(self.train_labels, order='C'))

            tes_case_log_reg = log_reg.predict(self.test_data)

            print("The Logistic Regression model accuracy is: ", accuracy_score(tes_case_log_reg,
                                                                                np.ravel(self.test_labels)))

        elif self.pca:

            x_train_pca, x_test_pca = principle_components(self.train_data, self.test_data)

            log_reg = LogisticRegression(max_iter=10000)
            log_reg.fit(x_train_pca, np.ravel(self.train_labels, order='C'))

            test_case_log_reg = log_reg.predict(x_test_pca)

            print("The Logistic Regression model accuracy, using principle components, is: ",
                  accuracy_score(test_case_log_reg, np.ravel(self.test_labels)))

    def k_means(self):

        if not self.pca:

            kmeans = KMeans(n_clusters=self.classes, random_state=0).fit(self.train_data)
            test_case_kmeans = kmeans.predict(self.test_data)

            print("The K Means model accuracy score is: ", accuracy_score(test_case_kmeans, np.ravel(self.test_labels)))

        elif self.pca:
            x_train_pca, x_test_pca = principle_components(self.train_data, self.test_data)

            kmeans = KMeans(n_clusters=self.classes, random_state=0).fit(x_train_pca)
            test_case_kmeans = kmeans.predict(x_test_pca)

            print("The K Means model accuracy, using principle components, is: ",
                  accuracy_score(test_case_kmeans, np.ravel(self.test_labels)))

    def k_neighbours(self):

        if not self.pca:

            neigh = KNeighborsClassifier(n_neighbors=self.classes)
            neigh.fit(self.train_data, np.ravel(self.train_labels))

            print("The KNN model accuracy is: ", neigh.score(self.test_data, np.ravel(self.test_labels)))

        elif self.pca:

            x_train_pca, x_test_pca = principle_components(self.train_data, self.test_data)

            neigh = KNeighborsClassifier(n_neighbors=self.classes)
            neigh.fit(x_train_pca, np.ravel(self.train_labels))

            print("The KNN model accuracy, using principle components, is: ", neigh.score(x_test_pca,
                                                                                          np.ravel(self.test_labels)))


class ComplexNetworks:

    def __init__(self, train_x, test_x, train_y, test_y, classes):
        self.train_x_, self.test_x_ = normalize(train_x), normalize(test_x)

        train_x_pca_, test_x_pca_ = principle_components(self.train_x_, self.test_x_)
        self.train_x_pca_, self.test_x_pca_ = normalize(train_x_pca_), normalize(test_x_pca_)

        self.classes = classes

        print(train_y[0], train_y[2], train_y[len(train_y) - 2])

        def neg_1(x):
            new_ = [0.0 if i == 2.0 else i for i in x]
            new_2 = [1.0 if j == 3.0 else j for j in new_]
            return new_2

        # train_y, test_y = list(neg_1(train_y)), list(neg_1(test_y))
        train_y, test_y = list(train_y), list(test_y)

        print(train_y[0], train_y[2], train_y[len(train_y) - 2])
        self.train_y_cat, self.test_y_cat = to_categorical(train_y), to_categorical(test_y)
        print(self.train_y_cat[0], self.train_y_cat[2], self.train_y_cat[len(train_y) - 2])

    def deep_network(self, pca=None):
        if not pca:
            print("Regular")
            model = Sequential()

            model.add(Dense(256, activation='relu',  input_dim=self.train_x_.shape[1]))
            model.add(Dropout(0.25))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.25))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.25))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.25))

            model.add(Dense(self.classes, activation='relu'))
            opt = optimizers.Adadelta(learning_rate=1.0, rho=0.95)

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print("samples: ", self.train_x_.shape, "classes: ", self.train_y_cat.shape)
            model.fit(self.train_x_, self.train_y_cat, validation_split=0.2, batch_size=32, epochs=50)

            model.evaluate(self.test_x_, self.test_y_cat)

            # model_prediction = model.predict(self.test_x_, batch_size=32, verbose=1)
            #
            # return model_prediction

        elif pca:
            print("PCA")
            model_pca = Sequential()

            model_pca.add(Dense(256, activation='relu',  input_dim=self.train_x_pca_.shape[1]))
            model_pca.add(Dropout(0.25))
            model_pca.add(Dense(128, activation='relu'))
            model_pca.add(Dropout(0.25))
            model_pca.add(Dense(64, activation='relu'))
            model_pca.add(Dropout(0.25))
            model_pca.add(Dense(32, activation='relu'))
            model_pca.add(Dropout(0.25))
            model_pca.add(Dense(16, activation='relu'))
            model_pca.add(Dropout(0.25))

            model_pca.add(Dense(self.classes, activation='softmax'))
            opt = optimizers.Adadelta(learning_rate=1.0, rho=0.95)

            model_pca.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            model_pca.fit(self.train_x_pca_, self.train_y_cat, validation_split=0.2, batch_size=32, epochs=20)

            model_pca.evaluate(self.test_x_pca_, self.test_y_cat)

            # model_prediction = model_pca.predict(self.test_x_pca_, batch_size=32, verbose=1)
            # print(model_pca.input_shape)

            # return model_prediction


def create_dataset(data_set, look_back):
    data_x, data_y = [], []
    for i in range(len(data_set) - look_back - 1):
        a = data_set[i:i + look_back, 0]
        data_x.append(a)
        data_y.append(data_set[i + look_back, 0])

    return np.array(data_x), np.array(data_y)


def conv_nn(training_x, testing_x, training_y, testing_y, classes):

    training_x, testing_x = principle_components(training_x, testing_x)

    training_x = training_x.reshape(len(training_x), training_x.shape[1], 1)
    testing_x = testing_x.reshape(len(testing_x), training_x.shape[1], 1)

    def neg_1(x):
        new_ = [0.0 if i == 2.0 else i for i in x]
        new_2 = [1.0 if j == 3.0 else j for j in new_]
        return new_2

    train_y = to_categorical(neg_1(training_y))
    test_y = to_categorical(neg_1(testing_y))

    model_cnn = Sequential()
    model_cnn.add(Conv1D(64, kernel_size=4, activation='relu', input_shape=(training_x.shape[1], 1)))
    model_cnn.add(Dropout(0.25))
    model_cnn.add(MaxPool1D(pool_size=2))
    model_cnn.add(Dense(32, activation='relu'))
    model_cnn.add(MaxPool1D(pool_size=3))
    model_cnn.add(Dropout(0.25))
    # model_cnn.add(Dense(64, activation='relu'))
    # model_cnn.add(Dropout(0.25))
    # model_cnn.add(Dense(32, activation='relu'))
    # model_cnn.add(Dropout(0.25))

    model_cnn.add(Flatten())
    model_cnn.add(Dense(16, activation='relu'))
    model_cnn.add(Dropout(0.5))

    model_cnn.add(Dense(classes, activation='softmax'))

    opt = optimizers.Adam(learning_rate=0.001)
    model_cnn.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model_cnn.fit(training_x, train_y, batch_size=16, epochs=15, verbose=1, validation_split=0.2)

    score = model_cnn.evaluate(testing_x, test_y, verbose=0)
    print("Test Loss: ", score[0])
    print("Test Accuracy: ", score[1])


def test_train_format(data):
    training_size = int(len(data) * 0.67)
    train_, test_ = data.iloc[0:training_size, 0:360].values, \
                    data.iloc[training_size:len(data), 0:360].values
    train_labels_, test_labels_ = data.iloc[0:training_size, 360].values, \
                                  data.iloc[training_size:len(data), 360].values

    return train_, test_, train_labels_, test_labels_


class RunModels:
    def __init__(self, data, mode, classes):
        self.mode = mode
        self.train_, self.test_, self.train_labels_, self.test_labels_ = test_train_format(data)
        self.classes = classes

    def run_models(self):
        if self.mode == 0:
            ml_models = MachineLearning(0, self.train_, self.test_, self.train_labels_, self.test_labels_, self.classes)

            # non pca
            print("Regular data results: ")
            print("\n")
            ml_models.k_means()
            ml_models.svm_svc()
            ml_models.logistic_regression()
            ml_models.k_neighbours()
            print("\n")

        elif self.mode == 1:
            # pca models
            ml_models_pca = MachineLearning(1, self.train_, self.test_, self.train_labels_,
                                            self.test_labels_, self.classes)
            print("PCA data results: ")
            print("\n")
            ml_models_pca.k_means()
            ml_models_pca.svm_svc()
            ml_models_pca.logistic_regression()
            ml_models_pca.k_neighbours()
            print("\n")

        elif self.mode == 2:
            # deep model
            print("Deep model with regular data: ")
            print("\n")
            ComplexNetworks(self.train_, self.test_, self.train_labels_, self.test_labels_,
                            self.classes).deep_network(pca=False)
            print("\n")

        elif self.mode == 3:
            # deep model with pca
            print("Deep model with PCA: ")
            print("\n")
            ComplexNetworks(self.train_, self.test_, self.train_labels_, self.test_labels_,
                            self.classes).deep_network(pca=True)
            print("\n")

        elif self.mode == 4:
            lstm_network(self.test_[0][0:360], 1)

        elif self.mode == 5:
            conv_nn(self.train_, self.test_, self.train_labels_, self.test_labels_, self.classes)

        elif self.mode == 6:
            ml_models = MachineLearning(0, self.train_, self.test_, self.train_labels_,
                                        self.test_labels_, self.classes)

            # # non pca
            print("Regular data results: ")
            print("\n")
            ml_models.k_means()
            ml_models.svm_svc()
            ml_models.logistic_regression()
            ml_models.k_neighbours()
            print("\n")

            # pca models
            ml_models_pca = MachineLearning(1, self.train_, self.test_, self.train_labels_,
                                            self.test_labels_, self.classes)
            print("PCA data results: ")
            print("\n")
            ml_models_pca.k_means()
            ml_models_pca.svm_svc()
            ml_models_pca.logistic_regression()
            ml_models_pca.k_neighbours()
            print("\n")

            # deep model
            print("Deep model with regular data: ")
            print("\n")
            ComplexNetworks(self.train_, self.test_, self.train_labels_,
                            self.test_labels_, self.classes).deep_network(pca=False)
            print("\n")

            print("Deep model with PCA: ")
            print("\n")
            ComplexNetworks(self.train_, self.test_, self.train_labels_,
                            self.test_labels_, self.classes).deep_network(pca=True)
            print("\n")

            # lstm
            lstm_network(self.test_[0][0:360], 1)

            # Conv
            conv_nn(self.train_, self.test_, self.train_labels_, self.test_labels_, self.classes)


class SecondClassifier:

    def __init__(self, full_data):
        self.full_data = full_data

    def second_classifier_format(self):
        data_butter = pd.DataFrame()
        data_cheby = pd.DataFrame()
        data_bessel = pd.DataFrame()
        data_ellip = pd.DataFrame()

        cnt = 0

        for i in range(len(self.full_data)):
            filter_label = round(self.full_data.iloc[i, 360], 1)
            temp_data = self.full_data.iloc[i, :]

            if filter_label == 0:
                data_butter = data_butter.append(temp_data.T, ignore_index=True)
            elif filter_label == 1:
                data_cheby = data_cheby.append(temp_data.T, ignore_index=True)
            elif filter_label == 2:
                data_bessel = data_bessel.append(temp_data.T, ignore_index=True)
            elif filter_label == 3:
                data_ellip = data_ellip.append(temp_data.T, ignore_index=True)
            else:
                cnt += 1

        return data_butter, data_cheby, data_bessel, data_ellip


def save_file(file_str, full_data):
    os.chdir('C:\\Users\\chand\\PycharmProjects\\Project.py\\Final_dataset\\Rand_full_data')

    full_data = pd.DataFrame(full_data)
    full_data.to_csv(file_str, header=False, index=False)


def remove_ones(data):

    temp_data = pd.DataFrame()
    for i in range(len(data)):
        if data.iloc[i, 363] == 0:
            temp_arr = data.iloc[i, :].T
            temp_data = temp_data.append(temp_arr)

    return temp_data


def conv_nn_2(training_x, testing_x, training_y, testing_y, classes):

    # training_x, testing_x = principle_components(training_x, testing_x)

    training_x = training_x.reshape(len(training_x), training_x.shape[1], 1)
    testing_x = testing_x.reshape(len(testing_x), training_x.shape[1], 1)

    def neg_1(x):
        new_ = [1.0 if i == 2.0 else i for i in x]
        new_2 = [0.0 if j == 3.0 else j for j in new_]
        return new_2

    train_y = to_categorical(neg_1(training_y))
    test_y = to_categorical(neg_1(testing_y))
    # train_y = to_categorical(training_y)
    # test_y = to_categorical(testing_y)

    model_cnn = Sequential()
    model_cnn.add(Conv1D(128, kernel_size=4, activation='relu',
                         input_shape=(training_x.shape[1], 1), padding='causal'))
    model_cnn.add(Dropout(0.25))
    model_cnn.add(MaxPool1D(pool_size=2))

    model_cnn.add(Dense(64, activation='relu'))
    # model_cnn.add(MaxPool1D(pool_size=2))
    model_cnn.add(Dropout(0.25))

    model_cnn.add(Dense(32, activation='relu'))
    # model_cnn.add(MaxPool1D(pool_size=2))
    model_cnn.add(Dropout(0.5))

    model_cnn.add(Flatten())
    model_cnn.add(Dense(16, activation='relu'))
    model_cnn.add(Dropout(0.5))

    model_cnn.add(Dense(classes, activation='softmax'))

    opt = optimizers.Adam(learning_rate=0.001)
    model_cnn.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    history = model_cnn.fit(training_x, train_y, batch_size=64, epochs=20, verbose=1, validation_split=0.2)

    score = model_cnn.evaluate(testing_x, test_y, verbose=0)
    print("Test Loss: ", score[0])
    print("Test Accuracy: ", score[1])

    model_cnn.summary()

    if len(training_x[0]) < 359:
        model_cnn.save("model_cnn_pca.h5")
    else:
        model_cnn.save("model_cnn.h5")

    return history


# df_one = pd.read_csv("Final_dataset\\Filter_specific_data\\ellip_filter_data.csv", header=None, index_col=False)
# df_ = pd.read_csv("Final_dataset\\Filter_specific_data\\wavelet_filter_data.csv", header=None, index_col=False)
#
# df_full_ = pd.read_csv("Final_dataset\\Rand_full_data\\Randomised_dataset_fixed.csv", header=None, index_col=False)

# df_one = remove_ones(df_one)
# df_ = remove_ones(df_)
#
# df_one = df_one.sample(frac=1).reset_index(drop=True)
# df_one = df_one.loc[0:len(df_), :]
#
# df_ = df_.append(df_one, ignore_index=True)
#
# df_ = df_.sample(frac=1).reset_index(drop=True)
#
# train_, test_, train_labels_, test_labels_ = test_train_format(df_)
# train_1, test_1 = independent_components(train_, test_)
# train_2, test_2 = principle_components(train_, test_)
# print(len(train_1), len(train_))

# ComplexNetworks(train_1, test_1, train_labels_, test_labels_, 2).deep_network()
# MachineLearning(0, train_, test_, train_labels_, test_labels_, 2).k_neighbours()
# RunModels(df_, 2, 2).run_models()  # not 4 or 6

# hist = conv_nn_2(train_2, test_2, train_labels_, test_labels_, 2)
# conv_nn(train_2, test_2, train_labels_, test_labels_, 2)

# df_full_ = df_full_.sample(frac=1).reset_index(drop=True)
# train_, test_, train_labels_, test_labels_ = test_train_format(df_full_)
# train_1, test_1 = principle_components(train_, test_)

# MachineLearning(0, train_1, test_1, train_labels_, test_labels_, 4).svm_svc()
