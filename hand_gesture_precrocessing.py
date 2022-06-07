from scipy.io import wavfile
import os
from scipy import signal, fftpack
from scipy.ndimage import uniform_filter1d
import numpy as np
import matplotlib.pyplot as plt
import tqdm

font = {'family': 'DejaVu Sans', 'size': 11}  # 22

plt.rc('font', **font)


def get_data(file_name):
    samplerate_, data_ = wavfile.read(file_name)
    data_og_ = data_
    data_ = data_.T

    return data_, data_og_, samplerate_


def hand_peace():
    h_files = []
    p_files = []
    for i in os.listdir('Data'):
        if i.endswith('_10.wav'):
            if i[0] == 'h':
                h_files.append(i)
            else:
                p_files.append(i)

    return h_files, p_files


def remove_high_freq(wave_):
    [b, a] = signal.cheby2(3, 30, 1000, 'low', analog=True)
    filtered = signal.filtfilt(b, a, wave_)
    w_, h_ = signal.freqz(b, a)

    return filtered, w_, h_


def remove_low_freq(wave_):
    [b, a] = signal.cheby1(3, 1, 1, 'hp', fs=44100, output='ba')
    filtered = signal.filtfilt(b, a, wave_)

    return filtered


def split_hand_waves(data_):
    ch_1, ch_2 = data_
    ch_1 = remove_high_freq(ch_1)
    ch_2 = remove_high_freq(ch_2)

    window = 100
    sample_points = np.linspace(1, len(ch_1), window)
    zero_val = np.zeros_like(ch_1)
    prev = 0
    final_list = []

    fourier_signal = np.fft.fft(ch_1)
    freq = np.fft.fftfreq(sample_points.shape[-1])

    plt.plot(sample_points, np.sin(2 * np.pi * 11 * sample_points))
    plt.show()


def normalise(sig_1):  # this function accepts two signals and returns the normalised versions for each
    sig_1 = (sig_1 - np.min(sig_1)) / (np.max(sig_1) - np.min(sig_1))  # normalising of sig_1
    return sig_1


def t_value(xx, fs_):
    T_ = len(xx) / fs_
    nsamples = int(T_ * fs_)
    tt = np.linspace(0, T_, nsamples, endpoint=False)
    return tt, T_


def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    [b, a] = signal.butter(order, normal_cutoff, btype='low')
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order)
    y = signal.lfilter(b, a, data)
    w_, h_ = signal.freqz(b, a, worN=8000)
    return y, (w_, h_)


def mfreqz(b, a, Fs):
    # Compute frequency response of the filter
    # using signal.freqz function
    wz, hz = signal.freqz(b, a)

    # Calculate Magnitude from hz in dB
    Mag = 20 * np.log10(abs(hz))

    # Calculate phase angle in degree from hz
    Phase = np.unwrap(np.arctan2(np.imag(hz), np.real(hz))) * (180 / np.pi)

    # Calculate frequency in Hz from wz
    Freq = wz * Fs / (2 * np.pi)

    # Plot filter magnitude and phase responses using subplot.
    fig = plt.figure(figsize=(10, 6))

    # Plot Magnitude response
    sub1 = plt.subplot(2, 1, 1)
    sub1.plot(Freq, Mag, 'r', linewidth=2)
    sub1.axis([1, Fs / 2, -100, 5])
    sub1.set_title('Magnitude Response', fontsize=20)
    sub1.set_xlabel('Frequency [Hz]', fontsize=20)
    sub1.set_ylabel('Magnitude [dB]', fontsize=20)
    sub1.grid()

    # Plot phase angle
    sub2 = plt.subplot(2, 1, 2)
    sub2.plot(Freq, Phase, 'g', linewidth=2)
    sub2.set_ylabel('Phase (degree)', fontsize=20)
    sub2.set_xlabel(r'Frequency (Hz)', fontsize=20)
    sub2.set_title(r'Phase response', fontsize=20)
    sub2.grid()

    plt.subplots_adjust(hspace=0.5)
    fig.tight_layout()
    plt.show()


# Define impz(b,a) to calculate impulse response
# and step response of a system
# input: b= an array containing numerator coefficients,
# a= an array containing denominator coefficients
def impz(b, a):
    # Define the impulse sequence of length 60
    impulse = np.repeat(0., 60)
    impulse[0] = 1.
    x = np.arange(0, 60)

    # Compute the impulse response
    response = signal.lfilter(b, a, impulse)

    # Plot filter impulse and step response:
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.stem(x, response, 'm', use_line_collection=True)
    plt.ylabel('Amplitude', fontsize=15)
    plt.xlabel(r'n (samples)', fontsize=15)
    plt.title(r'Impulse response', fontsize=15)

    plt.subplot(212)
    step = np.cumsum(response)

    # Compute step response of the system
    plt.stem(x, step, 'g', use_line_collection=True)
    plt.ylabel('Amplitude', fontsize=15)
    plt.xlabel(r'n (samples)', fontsize=15)
    plt.title(r'Step response', fontsize=15)
    plt.subplots_adjust(hspace=0.5)

    fig.tight_layout()
    plt.show()


def plot_fft(sig_, fs_, plot=True):
    N_ = len(sig_)
    yf_ = fftpack.fft(sig_)
    xf_ = fftpack.fftfreq(N_, 1 / fs_)

    if plot:
        plt.plot(xf_, np.abs(yf_))
        plt.grid()
        plt.show()

    return xf_, yf_


def bandpass_bandstop(x, bp, bs, fs_):
    pass_b, Ap = bp[0], bp[1]
    stop_b, As = bs[0], bs[1]

    wp = pass_b / (fs_ / 2)
    ws = stop_b / (fs_ / 2)

    N, wc = signal.cheb1ord(wp, ws, Ap, As)
    [z, p] = signal.cheby1(N, Ap, wc, 'bandpass')
    y = signal.filtfilt(z, p, x)

    return y


def generate_filtermat(x, min_freq, max_freq, windows, fs):
    step = 5
    # freq_x = np.linspace(min_freq+step, max_freq, windows)
    freq_x = [10, 15, 23, 30, 50, 75, 112, 140]

    bandpass_ripple = 0.3
    bp = (np.array([min_freq, max_freq]), bandpass_ripple)
    bs_ripple = 20

    output_mat = []
    counter = 0
    for f in tqdm.tqdm(freq_x):
        if counter == 0:
            bs_min, bs_max = f - 3, f + 3
        else:
            bs_min, bs_max = f-step, f+step

        bs = ([np.array([bs_min, bs_max]), bs_ripple])

        y = bandpass_bandstop(x, bp, bs, fs)

        output_mat.append(y)

    output_mat = np.array(output_mat)
    return output_mat


# hand_f, peace_f = hand_peace()
sample_rate, test_file_og = wavfile.read('Data/star_high_1.wav')

# test_file_both, w, h = remove_high_freq(test_file_og.T)

sig = test_file_og.T[0][5*sample_rate:7*sample_rate]
# plot_fft(sig, sample_rate)
# filtered, (w, h) = butter_lowpass_filter(sig, 10, sample_rate)

bandpass = (np.array([0.01, 500]), 0.2)
bandstop = (np.array([100, 200]), 30)

output = bandpass_bandstop(sig, bandpass, bandstop, sample_rate)
w, h = signal.freqz(output, sample_rate)

# min_f, max_f, w_samples = 2, 175, 15
# generate_filtermat(sig, min_f, max_f, w_samples, sample_rate)
# print("here fin")
#
t, T = t_value(output, sample_rate)

xfa, yfa = plot_fft(sig, sample_rate, plot=False)
xfb, yfb = plot_fft(output, sample_rate, plot=False)

# mfreqz(z, p, Fs)
# impz(z, p)

plt.subplot(221)
plt.plot(xfa, np.abs(yfa))
plt.grid()
plt.title("FFT of Original Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")
plt.ticklabel_format(axis="y", style="sci")

plt.subplot(223)
plt.plot(normalise(sig), 'darkcyan')
plt.title('Input Signal')
plt.xlabel("Time (s)")
plt.ylabel("Signal Amplitude")
plt.ticklabel_format(axis="y", style="sci")

plt.subplot(222)
plt.plot(xfb, np.abs(yfb))
plt.grid()
plt.title("FFT of Filtered Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")
plt.ticklabel_format(axis="y", style="sci")

plt.subplot(224)
plt.plot(t, normalise(output), 'turquoise')
plt.title('Filtered Signal')
plt.xlabel("Time (s)")
plt.ylabel("Signal Amplitude")
plt.ticklabel_format(axis="y", style="sci")

plt.tight_layout()
# plt.savefig("")
plt.show()


# For plotting spectrogram
# spec_sig = test_file_og.T[0][int(2.5*sample_rate):9*sample_rate]
# # spec_sig, w, h = remove_high_freq(spec_sig)
#
# fourier = np.fft.fft(spec_sig)
#
# n = len(spec_sig)
# ll = np.int(np.ceil(n/2))
# fourier = (fourier[0:ll-1])/float(n)
# freq_array = np.arange(0, ll-1, 1.0)*(sample_rate*1.0/n)
#
# plt.rcParams['font.size'] = '26'
# plt.figure(figsize=(10,10))
# Pxx, freqs, bins, im = plt.specgram(spec_sig, Fs=sample_rate, NFFT=2048, cmap=plt.get_cmap('RdPu'))
# cbar = plt.colorbar(im)
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
# cbar.set_label('Intensity (dB)')
# plt.show()

