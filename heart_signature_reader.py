from scipy.signal import butter, lfilter
from scipy.io import wavfile


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def run():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 44100
    lowcut = 0.6
    highcut = 3

    # Plot the frequency response for a few different orders.
    # plt.figure(1)
    # plt.clf()
    # for order in [3, 6, 9]:
    #     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    #     w, h = freqz(b, a, worN=2000)
    #     plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
    #
    # plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
    #          '--', label='sqrt(0.5)')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Gain')
    # plt.grid(True)
    # plt.legend(loc='best')

    sample_r, x_og = wavfile.read('Data/heart_test_1.wav')
    x = x_og.T
    x = x[0]

    # Filter a noisy signal.
    T = x_og.shape[0] / sample_r
    nsamples = int(T * fs)
    t = np.arange(len(x))/sample_r
    a = 0.01
    f0 = 600.0

    plt.figure(2)
    plt.clf()
    # plt.plot(t, x, label='Noisy signal')

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=2)

    # plt.plot(t, x, label='Bandpass Filtered signal (0.6 Hz - 3 Hz)')
    plt.plot(t, x, label='Raw signal')
    plt.xlabel('time (seconds)')
    # plt.hlines([-a, a], 0, T)
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()


run()
