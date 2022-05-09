import os
import os.path
from typing import Union, Iterable
import numpy as np
import pandas as pd
import scipy.io as mat
from scipy import signal
from numpy.core._multiarray_umath import ndarray
import random
import csv
import matplotlib.pyplot as plt

path = r"C:\Users\kp826252\Documents\Data\Test 1\ECG signals (1000 fragments)\MLII\1 NSR\\"
dir_path = r"C:\Users\kp826252\PycharmProjects\trial_project\datasets"
data: list = [None] * len(os.listdir(path))


def make_one(way):
    filename: Union[bytes, str]  # initialize filename as a str
    way3: Union[bytes, str] = way  # make a copy of the path to reset the directory
    a: int = 0  # this is an indicator for the number of files in a directory

    for filename in os.listdir(way):  # finding each file in the directory
        way += filename  # concat the finale directory and name
        data[a] = mat.loadmat(way)  # in a variable data, load the file data in columns
        a += 1  # just to keep iterating the column count for the next file
        way = way3  # reset the file name to just the directory

    return data  # return the variable data with columns filled with the file data


def dict2list(way):
    r = make_one(way)  # call the function which creates the data dictionary
    data_tr: ndarray = np.transpose(r)  # transpose to get all files in row by (optional)
    data_hold: Union[Iterable, tuple[int]] = np.shape(data_tr)  # to get a shape of the data
    data_values: list = [None] * data_hold[0]  # create an empty list with the same size of the data
    # this is to then later fill with individual data arrays

    for i in range(data_hold[0]):
        data_values[i] = [v for v in data_tr[i].values()]  # unpack the dictionary and store in new list
        data_values[i] = data_values[i][0][0]  # unpacking the lists in the dict

    return data_values


class Noise:

    def __init__(self, signal_1):
        self.signal = signal_1

    @staticmethod
    def normalise(sig_1, sig_2):  # this function accepts two signals and returns the normalised versions for each

        sig_1 = (sig_1 - np.min(sig_1)) / (np.max(sig_1) - np.min(sig_1))  # normalising of sig_1
        sig_2 = (sig_2 - np.min(sig_2)) / (np.max(sig_2) - np.min(sig_2))  # normalising for sing_2

        return sig_1, sig_2  # returning both normalised sig_1 and sig_2

    def gaussian_noise(self):  # This function produces a noise signal with gaussian distribution and combines
        # it with the original signal

        x_list: Union[Union[int, complex], any] = np.arange(len(self.signal))  # this is creating an x axis vector
        x_list = (x_list - np.min(x_list)) / (np.max(x_list) - np.min(x_list))   # this feeds both the above arrays to
        # be normalised

        sigma, mu = np.sqrt(np.var(self.signal)), np.mean(self.signal)  # getting values for mean and std of the array
        gs_noise = np.random.normal(mu, sigma, len(self.signal))  # filling the dummy signal with random variables
        gs_noise, gs_sig = self.normalise(gs_noise, self.signal)  # normalising both the noise signal and original
        # signal

        gs_sig_1 = gs_noise + gs_sig  # combining the original signal and noise signal by addition
        gs_sig_2 = gs_sig * gs_noise  # multiplying two signals
        gs_sig_1, gs_sig_2 = self.normalise(gs_sig_1, gs_sig_2)  # normalising both the signals

        return gs_sig_1, gs_sig_2, x_list, gs_sig  # returns the two noisy signals along with x axis and y axis
        # arrays

    def poisson_noise(self):  # This function produces a noise signal with poisson distribution and combines
        # it with
        # the original signal
        lmbda: int = 3  # initialising the value for lambda

        ps_noise = np.random.poisson(lmbda, len(self.signal))  # filling the dummy signal
        ps_noise, ps_sig = self.normalise(ps_noise, self.signal)  # normalising the noisy signal and original signal

        ps_sig_1 = ps_noise + ps_sig  # combing the noisy and original signal by addition
        ps_sig_2 = ps_sig * ps_noise  # combing the noisy and original signal by
        # multiplication
        ps_sig_1, ps_sig_2 = self.normalise(ps_sig_1, ps_sig_2)  # normalising both signals

        return ps_sig_1, ps_sig_2, ps_noise  # returning both signals

    def binomial_noise(self):  # function to create a noise signal with binomial distribution

        n, p = len(self.signal), 0.5  # declaring no. of trials and prob. of trial respectively
        bi_noise = np.random.binomial(n, p, len(self.signal))
        bi_noise, bi_sig = self.normalise(bi_noise, self.signal)  # normalising both noise signal and original signal

        bi_sig_1 = bi_noise + bi_sig  # combining noise signal and original signal by addition
        bi_sig_2 = bi_sig * bi_noise  # combining noise signal and original signal
        # by multiplication
        bi_sig_1, bi_sig_2 = self.normalise(bi_sig_1, bi_sig_2)  # normalising both final signals

        return bi_sig_1, bi_sig_2  # returning the two signals


def time_stmp(dats_list, files_no, gs_sig, ps_sig, bi_sig):
    no_of_heads: int = 18
    timestmp = np.empty(len(dats_list[0]), dtype=bytearray)  # create an empty times stamp list
    fin_arr = [[[0 for z in range(0, no_of_heads)] for y in range(0, len(dats_list[x]))] for x in range(0, files_no)]

    return fin_arr


def create_folders(data_array_cf, file_location, hello_uno):

    trial_no_ = pd.Series(range(data_split[0]), dtype='str')
    test_no_ = pd.Series(range(data_split[1]), dtype='str')
    validation_no_ = pd.Series(range(data_split[2]), dtype='str')

    def make_folder_directories():
        if not os.path.exists(trial_path):
            os.makedirs(trial_path)
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        if not os.path.exists(validation_path):
            os.makedirs(validation_path)
        return

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        make_folder_directories()
    else:
        make_folder_directories()

    def csv_name(type_of_data):
        file_dict = {}
        for z in type_of_data:
            file_dict[z] = "NSR_dataset_" + z + ".csv"
        return file_dict

    trial_no_dict = csv_name(trial_no_)
    test_no_dict = csv_name(test_no_)
    validation_no_dict = csv_name(validation_no_)

    def iter_files(files_type, data_array1, files_names):
        if files_names == "Trial":
            os.chdir(trial_path)
        elif files_names == "Test":
            os.chdir(test_path)
        else:
            os.chdir(validation_path)

        with open(files_type, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data_array1)

        return None

    if hello_uno <= data_split[0]:
        if hello_uno == data_split[0]:
            hello_uno -= 1
        iter_files(trial_no_dict[str(hello_uno)], data_array_cf, "Trial")
    elif data_split[0] < hello_uno <= (data_split[0] + data_split[1]):
        hello_uno -= data_split[0]
        if hello_uno == data_split[1]:
            hello_uno -= 1
        iter_files(test_no_dict[str(hello_uno)], data_array_cf, "Test")
    else:
        hello_uno -= (data_split[1] + data_split[0])
        if hello_uno == data_split[2]:
            hello_uno -= 1
        iter_files(validation_no_dict[str(hello_uno)], data_array_cf, "Validation")


def organising_datasets(data_arr_od):

    for i in range(len(rand_list)):
        if i < (data_split[0]):
            dummy_arr = data_arr_od[rand_list[i]]
            create_folders(dummy_arr, data_split[0], i)
        elif (data_split[0]) < i <= (data_split[0] + data_split[1]):
            dummy_arr = data_arr_od[rand_list[i]]
            create_folders(dummy_arr, (data_split[1] + data_split[0]), i)
        else:
            dummy_arr = data_arr_od[rand_list[i]]
            create_folders(dummy_arr, (data_split[2] + data_split[1] + data_split[0]), i)

    return None


def make_noise(data_main):

    data_final = final_arr

    for i in range(len(data_final)):

        signal_noise = Noise(data_main[i])

        gauss_sig_1, gauss_sig_2, x_axis, norm_clean_sig = signal_noise.gaussian_noise()
        poiss_sig_1, poiss_sig_2, poiss_noise = signal_noise.poisson_noise()
        binom_sig_1, binom_sig_2 = signal_noise.binomial_noise()

        for j in range(len(data_final[i])):
            for k in range(len(data_final[i][j])):
                if k == 0:
                    data_final[i][j][k] = norm_clean_sig[j]
                elif k == 3:
                    data_final[i][j][k] = gauss_sig_1[j]
                elif k == 6:
                    data_final[i][j][k] = gauss_sig_2[j]
                elif k == 9:
                    data_final[i][j][k] = poiss_sig_1[j]
                elif k == 12:
                    data_final[i][j][k] = binom_sig_1[j]
                elif k == 15:
                    data_final[i][j][k] = binom_sig_2[j]
                else:
                    continue

        # data_final[i] = pd.DataFrame(data_final[i], columns=cols)

    return data_final


data_list = dict2list(path)
no_of_files: int = len(data_list)
final_arr = time_stmp(data_list, no_of_files, 1, 2, 3)

last_arr = make_noise(data_list)

trial_path: str = dir_path + "\\Train_datasets"
test_path: str = dir_path + "\\Test_datasets"
validation_path: str = dir_path + "\\Validation_datasets"

rand_list = list(np.arange(len(last_arr)))
rand_list = tuple(random.sample(rand_list, len(rand_list)))
data_split = (168, 58, 57)

organising_datasets(last_arr)



