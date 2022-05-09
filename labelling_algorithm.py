import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import time
import pywt
import os
import os.path

fs = 360


class SNR:
    def __init__(self, orig_sig, noise_sig):
        self.orig_sig = orig_sig
        self.noise_sig = noise_sig
        self.just_noise = orig_sig - noise_sig

    def signal_to_noise(self):
        original_signal = np.array(self.orig_sig)

        upper = np.mean(original_signal ** 2)
        lower = np.mean(self.just_noise ** 2)
        rms_signal = 10 * np.log10(upper / lower)

        return rms_signal


def butter_filter(data_sig, cutoff_low):
    nyq = fs * 0.5
    cutoff_low = cutoff_low / nyq

    [b, a] = signal.butter(1, cutoff_low, btype='low')
    y = signal.lfilter(b, a, data_sig)

    return y


def cheby_filter(data_sig, cutoff_low):
    nyq = fs * 0.5
    cutoff_low = cutoff_low / nyq

    [b, a] = signal.cheby2(3, 7, cutoff_low, btype='low')
    y = signal.lfilter(b, a, data_sig)

    return y


def bessel_filter(data_sig, cutoff):
    nyq = 0.5 * fs
    cutoff_low = cutoff / nyq

    [b, a] = signal.bessel(2, cutoff_low, 'low')
    y = signal.lfilter(b, a, data_sig)

    return y


def ellip_filter(data_sig, ripple_p, ripple_s, cutoff):
    nyq = fs * 0.5
    cutoff_low = cutoff / nyq

    [b, a] = signal.ellip(7, ripple_p, ripple_s, cutoff_low, 'low')
    y = signal.lfilter(b, a, data_sig)
    return y


def check_zero(to_check):
    if round(np.max(to_check), 2) == 0.0:
        to_check = to_check[to_check != 0]

    return to_check


class ChebyButterFilters:

    def __init__(self, noise_sig, clean_sig, max_cf, last_per, next_per):
        self.noise_sig = noise_sig
        self.clean_sig = clean_sig
        self.max_cf = max_cf
        self.last_per = last_per
        self.next_per = next_per

    def butter_iter(self, data_sig, clean_sig):
        snr_butter = np.zeros(self.max_cf)

        for i in range(1, self.max_cf, 1):
            fin_sig = butter_filter(data_sig, i)
            clt = SNR(clean_sig, fin_sig)
            snr_butter[i] = clt.signal_to_noise()

        snr_butter = check_zero(snr_butter)
        index_val_b = np.unravel_index(np.argmax(snr_butter, axis=None), snr_butter.shape)

        return index_val_b[0], round(snr_butter[index_val_b], 3), snr_butter

    def cheby_iter(self, data_sig, clean_sig):
        snr_cheby = np.zeros(self.max_cf)

        for i in range(1, self.max_cf, 1):
            fin_sig = cheby_filter(data_sig, i)
            clt = SNR(clean_sig, fin_sig)
            snr_cheby[i] = clt.signal_to_noise()

        snr_cheby = check_zero(snr_cheby)
        index_val_c = np.unravel_index(np.argmax(snr_cheby, axis=None), snr_cheby.shape)

        return index_val_c[0], round(snr_cheby[index_val_c], 3), snr_cheby

    def compare_filters(self):
        window_period = self.next_per - self.last_per
        last_per = self.last_per
        fin_arr = pd.DataFrame()
        warning = 0

        for i in range(1, int((len(norm_sig))/window_period)+1):
            next_per = i * window_period
            sig_to_compare = self.noise_sig[last_per:next_per]
            clean_sig = self.clean_sig[last_per:next_per]

            # if i == 10:
            #     sig_to_compare = np.append(sig_to_compare, sig_to_compare[len(sig_to_compare)-1])
            #     clean_sig = np.append(clean_sig, clean_sig[len(clean_sig)-1])

            [index_c, max_c, snr_cheby] = self.cheby_iter(sig_to_compare, clean_sig)
            [index_b, max_b, snr_butter] = self.butter_iter(sig_to_compare, clean_sig)

            if max_b >= max_c:
                temp_var = [0, max_b, index_b, warning]
                temp_var = np.append(sig_to_compare, temp_var)
                temp_var = pd.DataFrame(temp_var, columns=None).T
                fin_arr = fin_arr.append(temp_var, ignore_index=True)
            else:
                temp_var = [1, max_c, index_c, warning]
                temp_var = np.append(sig_to_compare, temp_var)
                temp_var = pd.DataFrame(temp_var, columns=None).T
                fin_arr = fin_arr.append(temp_var, ignore_index=True)

            last_per = next_per

        return fin_arr


class EllipBesselFilters:

    def __init__(self, full_data, clean_data, rp, rs, max_cutoff, last_per, next_per):
        self.data = full_data
        self.clean_data = clean_data
        self.rs = rs
        self.rp = rp
        self.max_cutoff = max_cutoff
        self.last_per = last_per
        self.next_per = next_per

    def ellip_iter(self, data_sig, clean_sig):
        snr_ellip = np.zeros(self.max_cutoff)

        for i in range(1, self.max_cutoff, 1):
            fin_sig = ellip_filter(data_sig, self.rp, self.rs, i)
            clt = SNR(clean_sig, fin_sig)
            snr_ellip[i] = clt.signal_to_noise()

        snr_ellip = check_zero(snr_ellip)
        index_val_e = np.unravel_index(np.argmax(snr_ellip, axis=None), snr_ellip.shape)

        return index_val_e[0], round(snr_ellip[index_val_e], 3), snr_ellip

    def wavelet_tr(self, pywt_sig, pywt_norm_sig):
        wavelet_snr = np.zeros(len(pywt.wavelist(kind='discrete')))
        wavelet_types = pywt.wavelist(kind='discrete')

        for i in range(len(wavelet_types)):
            [cA, cD] = pywt.dwt(pywt_sig, wavelet_types[i])
            # cD = pywt.threshold(cD, np.mean(cD), 'soft')
            y = pywt.idwt(cA, None, wavelet_types[i])
            clt = SNR(pywt_norm_sig, y)
            wavelet_snr[i] = clt.signal_to_noise()

        wavelet_snr = check_zero(wavelet_snr)
        index_val_w = np.unravel_index(np.argmax(wavelet_snr, axis=None), wavelet_snr.shape)

        return index_val_w[0], round(wavelet_snr[index_val_w], 3), wavelet_snr

    def warning(self, x):
        warning = 0
        if x < 3:
            warning = 1
        else:
            warning = 0

        return warning

    def compare_filters(self):
        fin_arr = pd.DataFrame()
        clean_fin_arr = pd.DataFrame()
        window_period = self.next_per - self.last_per
        previous_window = self.last_per

        for i in range(1, int((len(self.clean_data)) / window_period) + 1):
            next_window = i * window_period
            sig_to_compare = self.data[previous_window:next_window]
            clean_sig = self.clean_data[previous_window:next_window]

            previous_window = next_window

            # if len(clean_sig) == 359:
            #     clean_sig = np.append(clean_sig, clean_sig[len(clean_sig)-1])

            [index_e, max_e, snr_ellip] = self.ellip_iter(sig_to_compare, clean_sig)
            [index_pywt, max_pywt, snr_pywt] = self.wavelet_tr(sig_to_compare, clean_sig)

            # print("Max ellip: ", max_e)
            # print("Max wavelet: ", max_pywt)
            # print("Max from before: ", signal_rep.iloc[i, 361])

            # if max_pywt >= max_e:
            #     if max_pywt >= signal_rep.iloc[i, 361]:
            #         signal_rep.iloc[i, 360] = 2
            #         signal_rep.iloc[i, 361] = max_pywt
            #         signal_rep.iloc[i, 362] = index_pywt
            #
            #         if max_pywt < 3:
            #             signal_rep.iloc[i, 363] = 1
            #         else:
            #             signal_rep.iloc[i, 363] = 0
            # else:
            #     if max_e > signal_rep.iloc[i, 361]:
            #         signal_rep.iloc[i, 360] = 3
            #         signal_rep.iloc[i, 361] = max_e
            #         signal_rep.iloc[i, 362] = index_e
            #
            #         if max_e < 3:
            #             signal_rep.iloc[i, 363] = 1
            #         else:
            #             signal_rep.iloc[i, 363] = 0

            if max_pywt >= max_e:
                temp_var = [0, max_pywt, index_pywt, self.warning(max_pywt)]
                temp_var = np.append(sig_to_compare, temp_var)
                temp_var = pd.DataFrame(temp_var, columns=None).T
                fin_arr = fin_arr.append(temp_var, ignore_index=True)

                temp_var_clean = [0, max_pywt, index_pywt, self.warning(max_pywt)]
                temp_var_clean = np.append(clean_sig, temp_var_clean)
                temp_var_clean = pd.DataFrame(temp_var_clean, columns=None).T
                clean_fin_arr = clean_fin_arr.append(temp_var_clean, ignore_index=True)
            else:
                temp_var = [1, max_e, index_e, self.warning(max_e)]
                temp_var = np.append(sig_to_compare, temp_var)
                temp_var = pd.DataFrame(temp_var, columns=None).T
                fin_arr = fin_arr.append(temp_var, ignore_index=True)

                temp_var_clean = [1, max_e, index_e, self.warning(max_e)]
                temp_var_clean = np.append(clean_sig, temp_var_clean)
                temp_var_clean = pd.DataFrame(temp_var_clean, columns=None).T
                clean_fin_arr = clean_fin_arr.append(temp_var_clean, ignore_index=True)

        return fin_arr, clean_fin_arr


def save_file(file_str, full_data):
    os.chdir('C:\\Users\\chand\\PycharmProjects\\Project.py\\Final_dataset\\Clean_noise_data_trial')

    full_data = pd.DataFrame(full_data)
    full_data.to_csv(file_str, header=False, index=False)


def get_file(file_str):
    os.chdir('C:\\Users\\chand\\PycharmProjects\\Project.py\\Train_datasets')
    file_name = file_str

    data_ = pd.read_csv(file_name, header=None, index_col=False)

    return data_


last_, next_ = 0, 360
# col_names = ["Data", "Filter", "Max_snr", "Cutoff"]
data = pd.DataFrame()
data_clean = pd.DataFrame()

os.chdir('C:\\Users\\chand\\PycharmProjects\\Project.py')
for file_ in os.listdir("Train_datasets"):
    df = get_file(file_)
    norm_sig = df.iloc[:, 0].values
    sig = df.iloc[:, 3].values

    # cheby_butter = ChebyButterFilters(sig, norm_sig, 179, last_, next_).compare_filters()
    ellip_wave, ellip_wave_clean = EllipBesselFilters(sig, norm_sig, 7, 8, 179, last_, next_).compare_filters()

    data = data.append(ellip_wave, ignore_index=True)
    data_clean = data_clean.append(ellip_wave_clean, ignore_index=True)

    sig = df.iloc[:, 6].values

    # cheby_butter = ChebyButterFilters(sig, norm_sig, 179, last_, next_).compare_filters()
    ellip_wave, ellip_wave_clean = EllipBesselFilters(sig, norm_sig, 7, 8, 179, last_, next_).compare_filters()

    data = data.append(ellip_wave, ignore_index=True)
    data_clean = data_clean.append(ellip_wave_clean, ignore_index=True)

    # print(ellip_wave, data.shape)
    print(file_)
    print("\n")

save_file("Noisy_signal.csv", data)
save_file("Clean_signal.csv", data_clean)
print(data.head)

# data = pd.DataFrame()
# os.chdir('C:\\Users\\chand\\PycharmProjects\\Project.py')
# for file_ in os.listdir("Train_datasets"):
#     print(file_)
#     print("\n")
#     df = get_file(file_)
#
#     sig = df.iloc[:, 9].values
#     norm_sig = df.iloc[:, 0].values
#
#     cheby_butter = ChebyButterFilters(sig, norm_sig, 179, last_, next_).compare_filters()
#     ellip_bessel = EllipBesselFilters(cheby_butter, norm_sig, 7, 8, 179, last_, next_).compare_filters()
#
#     data = data.append(ellip_bessel, ignore_index=True)
#     print(ellip_bessel, data.shape)
#     print("\n")
#
# save_file("Completed_data_binom.csv", data)
# print(data.head)
#
# data = pd.DataFrame()
# os.chdir('C:\\Users\\chand\\PycharmProjects\\Project.py')
# for file_ in os.listdir("Train_datasets"):
#     print(file_)
#     print("\n")
#     df = get_file(file_)
#
#     sig = df.iloc[:, 12].values
#     norm_sig = df.iloc[:, 0].values
#
#     cheby_butter = ChebyButterFilters(sig, norm_sig, 179, last_, next_).compare_filters()
#     ellip_bessel = EllipBesselFilters(cheby_butter, norm_sig, 7, 8, 179, last_, next_).compare_filters()
#
#     data = data.append(ellip_bessel, ignore_index=True)
#     print(ellip_bessel, data.shape)
#     print("\n")
#
# save_file("Completed_data_binom2.csv", data)
# print(data.head)


######################################################################################################################
# df = pd.read_csv("Train_datasets\\NSR_dataset_0.csv")
# sig = df.iloc[:, 3].values
# norm_sig = df.iloc[:, 0].values
#
#
# def wavelet_tr_temp(pywt_sig, pywt_norm_sig):
#     wavelet_snr = np.zeros(len(pywt.wavelist(kind='discrete')))
#     wavelet_types = pywt.wavelist(kind='discrete')
#
#     for i in range(len(wavelet_types)):
#         [cA, cD] = pywt.dwt(pywt_sig, wavelet_types[i])
#         cAA = pywt.threshold(cA, np.mean(cD))
#         y = pywt.idwt(cAA, None, wavelet_types[i])
#         clt = SNR(pywt_norm_sig, y)
#         wavelet_snr[i] = clt.signal_to_noise()
#
#     print(wavelet_snr)
#     wavelet_snr = check_zero(wavelet_snr)
#     index_val_w = np.unravel_index(np.argmax(wavelet_snr, axis=None), wavelet_snr.shape)
#
#     return index_val_w, wavelet_snr
#
#
# pywt_a, pywt_b = wavelet_tr_temp(sig[360:720], norm_sig[360:720])
# # print(SNR(pywt_a, norm_sig[360:720]).signal_to_noise())
#
# cheby_butter = ChebyButterFilters(sig, norm_sig, 179, last_, next_).compare_filters()
# print(cheby_butter)
#
# ellip_bessel = EllipBesselFilters(cheby_butter, norm_sig, 7, 8, 179, last_, next_).compare_filters()
# print(ellip_bessel)
