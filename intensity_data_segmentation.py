import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pickle


def get_data(dir_name):  # Returns data from file
    data_dict = {}
    fs_ = 0
    for f_name in os.listdir(dir_name):
        if f_name[-4:] == '.wav':
            fs_, data_ = wavfile.read(dir_name+f_name)
            data_og_ = data_
            data_ = data_.T
            f_name = f_name.replace('.wav', '')
            data_dict[f_name] = data_

    return data_dict, fs_


def t_value(xx, fs_):
    T_ = len(xx) / fs_
    n_samples = int(T_ * fs_)
    tt = np.linspace(0, T_, n_samples, endpoint=False)
    return tt, T_


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(f_name_):
    with (open(f_name_ + '.pkl', "rb")) as openfile:
        while True:
            try:
                data_dict_ = (pickle.load(openfile))
            except EOFError:
                break

    return data_dict_


def check_file(f_name_,  obj_data,  save_=False):
    if not os.path.isfile(f_name_):
        obj_data = {}
        save_obj(obj_data, f_name_)
    elif save_:
        save_obj(obj_data, f_name_)
    else:
        obj_data = load_obj(f_name_)

    return obj_data


def segment_data(data_dict):
    segment_file_name = 'intensity_seg_times'
    segment_times = check_file(data_dir+segment_file_name, {}, save_=False)
    s_keys, total_keys = list(segment_times.keys()), list(data_dict.keys())
    print(s_keys, total_keys)

    for key in data_dict.keys():
        segment_times = check_file(data_dir + segment_file_name)

        if key[0] != 'n':
            vals = data_dict[key]
            sig_a, sig_b = vals
            segment_array = [3.4]
            segment_array = [i * fs for i in segment_array]

            t, T = t_value(sig_a, fs)

            plt.plot(t[fs:], sig_a[fs:])
            plt.show()

            break


data_dir = 'Data/intensity_data/'
file_name = 'test_sig_full.wav'
sigs, fs = get_data(data_dir)
segment_data(sigs)
