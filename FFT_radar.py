from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


def get_data(file_name):
    samplerate_, data_ = wavfile.read(file_name)
    data_og_ = data_
    data_ = data_.T

    return data_, data_og_, samplerate_


def sos_filter(signal_):
    sos = signal.cheby1(3, 1, 1, 'lp', fs=4410, output='sos')
    filtered = signal.sosfilt(sos, signal_)

    return filtered


def max_len(xx, frac_):
    return round(len(xx)/frac_)


data_sig, data_og_sig, sample_rate_sig = get_data("Data/heart_test_1.wav")

print(f"number of channels = {data_og_sig.shape[1]}")
length = data_og_sig.shape[0] / sample_rate_sig
print(f"length = {length}s")

time = np.linspace(0., length, data_og_sig.shape[0])

sig_noisy_1 = data_og_sig[:, 0]
sig_noisy_2 = data_og_sig[:, 1]
frac = 1

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(sos_filter(sig_noisy_1)[0:max_len(sig_noisy_1, frac)], 'g')
ax1.plot(sos_filter(sig_noisy_2[0:max_len(sig_noisy_1, frac)]), 'b')
# ax1.plot(sos_filter(sig_noisy_2)[0:max_len(sig_noisy_2, frac)], 'b')
# ax1.plot(noise, 'b')
ax1.set_title('original signal')

ax2.plot(sos_filter(sig_noisy_1[0:max_len(sig_noisy_1, frac)]), 'g')
ax2.plot(sos_filter(sig_noisy_2[0:max_len(sig_noisy_2, frac)]), 'b')
# ax2.plot(filtered_noise[0:max_len])
ax2.set_title('After 10 Hz low-pass filter')
ax2.set_xlabel('Time [seconds]')
plt.tight_layout()
plt.show()
