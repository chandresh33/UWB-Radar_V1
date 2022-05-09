import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import time
import pywt
import os
import os.path

# loading the data set
# s_1 = pd.read_csv('Train_dataset/NSR_dataset_0.csv', header=None)

window_period: int = 360
frequency = 360


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


class Filter:

    def __init__(self, order, data, cf, nxt, last, clean_sig, cnt):
        self.order = order
        self.data = data
        self.cf = cf
        self.N = nxt
        self.L = last
        self.clean_sig = clean_sig
        self.cnt = cnt

    def cheby_lowpass(self, l_cutoff, h_cutoff, fs, order_1, ripple_2):
        nyq = 0.5 * fs
        high = h_cutoff / nyq
        low = l_cutoff / nyq
        cheby_var1 = signal.cheby2(order_1, ripple_2, low)
        cheby_var2 = signal.cheby2(order_1, ripple_2, high, btype='high')

        cheby_a1 = cheby_var1[1]
        cheby_b1 = cheby_var1[0]
        cheby_bb1 = cheby_var2[0]
        cheby_aa1 = cheby_var2[1]

        return cheby_b1, cheby_a1, cheby_bb1, cheby_aa1

    def cheby_lowpass_filter(self, data_1, lcutoff, hcutoff, fs, order_1, ripple_1):
        hcutoff = hcutoff / 1000
        cheby_b2, cheby_a2, cheby_bb2, cheby_aa2 = self.cheby_lowpass(lcutoff, hcutoff, fs, order_1, ripple_1)
        cheby_w, cheby_h = signal.freqz(cheby_b2, cheby_a2)
        y = signal.lfilter(cheby_b2, cheby_a2, data_1)
        y2 = signal.lfilter(cheby_bb2, cheby_aa2, y)
        return cheby_b2, cheby_a2, cheby_w, cheby_h, y

    @staticmethod
    def signal_cleaning_fft(noisy_signal, z, y, order_1):
        N_1 = len(noisy_signal)
        t_1 = np.array([v for v in range(0, N_1)])
        T_1 = t_1[0] - t_1[1]
        w_1 = np.linspace(0, N_1 * (1 / T_1), N_1)
        fft_1 = np.fft.fft(noisy_signal)

        def butter_lowpass(low_cutoff, high_cutoff, fs, order):
            nyq = 0.5 * fs
            # normal_high_cutoff = high_cutoff / nyq
            normal_low_cutoff = low_cutoff / nyq
            # b, a = signal.butter(order, [normal_low_cutoff, normal_high_cutoff], btype='bandpass')
            butter_var = signal.butter(order, normal_low_cutoff, btype='low')

            bb = butter_var[0]
            aa = butter_var[1]
            return bb, aa

        def butter_lowpass_filter(data, l_cutoff, h_cutoff, fs, order):
            butter_b, butter_a = butter_lowpass(l_cutoff, h_cutoff, fs, order=order)
            butter_w, butter_h = signal.freqz(butter_b, butter_a)
            y_butter = signal.lfilter(butter_b, butter_a, data)
            return butter_b, butter_a, butter_w, butter_h, y_butter

        #  Filter requirements
        fs_1 = frequency

        cutoff_2 = y / 1000
        cutoff_1 = z

        b1, a1, w1, h1, y1 = butter_lowpass_filter(noisy_signal, cutoff_1, cutoff_2, fs_1, order_1)
        fft_2 = np.fft.fft(y1)

        return b1, a1, h1, w1, y1, fft_1, w_1, N_1, fft_2

    @property
    def do_rest(self):
        fs_1 = frequency

        snr_arr_one = np.zeros((self.cf, 50))
        # snr_arr_one = np.zeros(self.cf)
        snr_arr_two = np.zeros(self.cf)

        for j in range(1, self.cf):
            b1, a1, h1, w1, fin_arr2, fft_1, w_1, N_1, fft_2 = self.signal_cleaning_fft(self.data, j,
                                                                                        1, self.order)
            clt = SNR(norm_clean_sig[self.L:self.N], fin_arr2)
            snr_arr_two[j] = clt.signal_to_noise()

        for i in range(1, self.cf):
            for j in range(1, 50):
                b2, a2, w2, h2, fin_arr1 = self.cheby_lowpass_filter(self.data, i, 1, fs_1, self.order, j)
                clt = SNR(norm_clean_sig[self.L:self.N], fin_arr1)
                snr_arr_one[i][j] = clt.signal_to_noise()

        indi1 = np.unravel_index(np.argmax(snr_arr_one, axis=None), snr_arr_one.shape)
        indi2 = np.unravel_index(np.argmax(snr_arr_two, axis=None), snr_arr_two.shape)

        if snr_arr_one[indi1] > snr_arr_two[indi2]:
            b2, a2, w2, h2, fin_arr1 = self.cheby_lowpass_filter(self.data, indi1[0], 1, fs_1, self.order, indi1[1])
            fin_arr = fin_arr1
            s_1.iloc[self.L:self.N, (self.cnt + 1)] = 1
            s_1.iloc[self.L:self.N, (self.cnt + 2)] = "{},{},{}".format(round(snr_arr_one[indi1], 3), indi1[0],
                                                                        indi1[1])

            print("This is for snr_ONE *: ", np.max(snr_arr_one), indi1)
            print("This is for snr_TWO", np.max(snr_arr_two), indi2)
            print("\n")

        else:
            b1, a1, h1, w1, fin_arr2, fft_1, w_1, N_1, fft_2 = self.signal_cleaning_fft(self.data, indi2[0],
                                                                                        1, self.order)
            fin_arr = fin_arr2
            s_1.iloc[self.L:self.N, (self.cnt + 1)] = 2
            s_1.iloc[self.L:self.N, (self.cnt + 2)] = "{}".format(round(snr_arr_two[indi2], 3), indi2[0])

            print("This is for snr_ONE : ", np.max(snr_arr_one), indi1)
            print("This is for snr_TWO *: ", np.max(snr_arr_two), indi2)
            print("\n")

        return fin_arr, snr_arr_one, snr_arr_two


class Elliptical:

    def __init__(self, order, data, cf, lst_period, nxt_period, cnt):
        self.order = order
        self.signal = data
        self.cf = cf
        self.last = lst_period
        self.next = nxt_period
        self.count = cnt

    def ellip_filter(self):
        ellip_snr_1 = np.zeros((13, 14, 180))

        for i in range(1, self.cf):
            ii = i / (frequency * 0.5)
            for j in range(20, 12, -1):
                jj = abs(j - 20)
                for k in range(1, 13):
                    [b1, a1] = signal.ellip(self.order, k, j, ii)
                    y = signal.lfilter(b1, a1, self.signal)
                    clt_1 = SNR(norm_clean_sig[self.last:self.next], y)
                    ellip_snr_1[k][jj][i] = clt_1.signal_to_noise()

        return ellip_snr_1

    def wavelet_tr(self):
        wavelet_snr = np.zeros(len(pywt.wavelist(kind='discrete')))
        wavelet_types = pywt.wavelist(kind='discrete')

        for i in range(len(wavelet_types)):
            [cA, cD] = pywt.dwt(self.signal, wavelet_types[i])
            y = pywt.idwt(cA, None, wavelet_types[i])
            clt = SNR(norm_clean_sig[self.last:self.next], y)
            wavelet_snr[i] = clt.signal_to_noise()

            return wavelet_snr

    def compare_snr(self):
        temp_ellip = self.ellip_filter()
        indi_ellip = np.unravel_index(np.argmax(temp_ellip, axis=None), temp_ellip.shape)

        temp_wavelet = self.wavelet_tr()
        indi_wavelet = np.unravel_index(np.argmax(temp_wavelet, axis=None), temp_wavelet.shape)

        previous_snr = list(s_1.iloc[self.last: self.next, (self.count + 2)])[0].split(',')
        snr_prev = []

        for obj in previous_snr:
            if obj.isdigit():
                snr_prev.append(int(obj))
            else:
                snr_prev.append(float(obj))

        print(temp_ellip[indi_ellip])
        print(temp_wavelet[indi_wavelet])
        print(snr_prev[0])

        if temp_ellip[indi_ellip] >= snr_prev[0]:
            if temp_ellip[indi_ellip] > temp_wavelet[indi_wavelet]:
                s_1.iloc[self.last:self.next, (self.count + 2)] = "{},{},{},{}". \
                    format(round(temp_ellip[indi_ellip], 3), indi_ellip[0], indi_ellip[1], indi_ellip[2])
                s_1.iloc[self.last:self.next, (self.count + 1)] = 3
                print("Elliptical filter")
            else:
                s_1.iloc[self.last:self.next, (self.count + 2)] = "{}".format(round(temp_wavelet[indi_wavelet], 3)
                                                                              , indi_wavelet[0])
                s_1.iloc[self.last:self.next, (self.count + 1)] = 4
                print("Wavelet")
        elif temp_wavelet[indi_wavelet] >= snr_prev[0]:
            if temp_wavelet[indi_wavelet] > temp_ellip[indi_ellip]:
                s_1.iloc[self.last:self.next, (self.count + 2)] = "{}".format(round(temp_wavelet[indi_wavelet], 3)
                                                                              , indi_wavelet[0])
                s_1.iloc[self.last:self.next, (self.count + 1)] = 4
                print("Wavelet")
            else:
                s_1.iloc[self.last:self.next, (self.count + 2)] = "{},{},{},{}". \
                    format(round(temp_ellip[indi_ellip], 3), indi_ellip[0], indi_ellip[1], indi_ellip[2])
                s_1.iloc[self.last:self.next, (self.count + 1)] = 3
                print("Elliptical filter")
        else:
            print("from previous (Cheby2)")


def compare_filters(turn):
    last_per = 0

    for count in range(3, 13, 14):
        if count == 3:
            print(1)
            sig = gauss_sig_1
        elif count == 6:
            print(2)
            sig = gauss_sig_2
        elif count == 9:
            print(3)
            sig = poiss_sig_1
        elif count == 12:
            print(4)
            sig = binom_sig_1
        elif count == 15:
            print(5)
            sig = binom_sig_2
        else:
            continue

        if turn == 0:
            for k in range(1, int(len(s_1) / window_period) + 1):
                next_period = k * window_period
                window_data = sig[last_per:next_period]
                fil = Filter(5, window_data, int(frequency / 2), next_period, last_per, norm_clean_sig, count)
                a, b, c = fil.do_rest
                last_per = next_period
        else:
            for p in range(1, int(len(s_1) / window_period) + 1):
                next_period = p * window_period
                window_data = sig[last_per:next_period]
                fil = Elliptical(5, window_data, int(frequency / 2), last_per, next_period, count)
                fil.compare_snr()
                last_per = next_period
                print(next_period)

        last_per = 0

    return None


def save_file(file_str, full_data):
    os.chdir('C:\\Users\\chand\\PycharmProjects\\Project.py\\Completed_dataset')

    file_name = file_str
    full_data = pd.DataFrame(full_data)
    full_data.to_csv(file_name, index=False)


def get_file(file_str):
    os.chdir('C:\\Users\\chand\\PycharmProjects\\Project.py\\Train_datasets')
    file_name = file_str

    data_ = pd.read_csv(file_name, header=None)

    return data_


start = time.time()
for i in os.listdir('Train_datasets'):
    s_1 = get_file(i)

    gauss_sig_1, gauss_sig_2, norm_clean_sig = s_1.iloc[:, 3], s_1.iloc[:, 6], s_1.iloc[:, 0]
    poiss_sig_1 = s_1.iloc[:, 9]
    binom_sig_1, binom_sig_2 = s_1.iloc[:, 12], s_1.iloc[:, 15]

    compare_filters(0)
    compare_filters(1)
    save_file(i, s_1)

end = time.time()
print(end - start)
