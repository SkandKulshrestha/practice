import math

from scipy.interpolate import interp1d
from scipy.io import loadmat
from smooth import mean_smooth, gaussian_smooth, gaussian_function, find_nearest
import numpy as np
import matplotlib.pyplot as plt


def generate_time_series(sampling_rate, time_duration, noise_amplitude):
    rng = np.random.default_rng()
    t = np.arange(0, time_duration, 1 / sampling_rate)

    # create a random function y = f(x)
    x = np.arange(15)
    y = rng.standard_normal(15) * 30
    f = interp1d(x, y)

    # interpolate on new data
    amplitude = f(t)

    # generate some noise
    noise = noise_amplitude * rng.standard_normal(len(t))
    signal = amplitude + noise

    return t, amplitude, signal


def generate_spike_time_series(num_spikes):
    rng = np.random.default_rng()
    isi = np.asarray(np.round(np.exp(rng.standard_normal(num_spikes)) * 10), dtype=np.int32)

    # generate spike time series
    spike_time_series = np.zeros((1,))

    for i in range(num_spikes):
        extended_length = len(spike_time_series) + isi[i]
        if len(spike_time_series) <= extended_length:
            spike_time_series = np.append(spike_time_series, np.zeros((extended_length - len(spike_time_series) + 1,)))
        spike_time_series[extended_length] = 1

    # add arbitrary length at end
    spike_time_series = np.append(spike_time_series, np.zeros((100,)))

    return np.arange(len(spike_time_series)), spike_time_series


def generate_time_series_with_spike(num_points, prop_noise):
    rng = np.random.default_rng()
    signal = np.cumsum(rng.standard_normal(num_points))
    _signal = np.copy(signal)

    noise_points = rng.permutation(num_points)
    noise_points = noise_points[0:math.ceil(num_points * prop_noise)]
    noise = 50 + rng.random(len(noise_points)) * 100

    for noise_index, signal_index in enumerate(noise_points):
        signal[signal_index] = noise[noise_index]

    return np.arange(num_points), _signal, signal


def time_series_example():
    sampling_rate = 1000  # Hz
    time_duration = 10  # sec
    noise_amplitude = 5  # noise level, measured in standard deviation
    t, _signal, signal = generate_time_series(sampling_rate, time_duration, noise_amplitude)

    plt.plot(t, signal, '-', label='Noisy signal')
    plt.plot(t, _signal, '-', label='Signal')
    plt.legend()
    plt.ylabel('Signal')
    plt.xlabel('Time (s)')
    plt.show()

    filter_window = 50  # actually filter window will be '2 * filter_window + 1'
    mean_filtered_signal = mean_smooth(signal, filter_window)

    plt.plot(t, signal, '-', label='Noisy signal')
    plt.plot(t, _signal, '-', label='Signal')
    plt.plot(t, mean_filtered_signal, '-', label='Mean Smooth')
    plt.legend()
    plt.ylabel('Signal')
    plt.xlabel('Time (s)')
    plt.show()

    full_width_at_half_maximum = 25  # in ms
    filter_window = 50
    gaussian_window = gaussian_function(full_width_at_half_maximum, filter_window, sampling_rate)

    gaussian_filtered_signal = gaussian_smooth(signal, filter_window, gaussian_window)

    plt.plot(t, signal, '-', label='Noisy signal')
    plt.plot(t, _signal, '-', label='Signal')
    plt.plot(t, mean_filtered_signal, '-', label='Mean Smooth')
    plt.plot(t, gaussian_filtered_signal, '-', label='Gaussian Smooth')
    plt.legend()
    plt.ylabel('Signal')
    plt.xlabel('Time (s)')
    plt.show()


def spike_time_series_example():
    num_spikes = 300
    t, spike_time_series = generate_spike_time_series(num_spikes)
    plt.plot(t, spike_time_series, '-', label='Signal')
    plt.legend()
    plt.ylabel('Signal')
    plt.xlabel('Time (a.u.)')
    plt.show()

    full_width_at_half_maximum = 25  # in ms
    filter_window = 50
    gaussian_window = gaussian_function(full_width_at_half_maximum, filter_window)

    gaussian_filtered_signal = gaussian_smooth(spike_time_series, filter_window, gaussian_window)

    plt.plot(t, spike_time_series, '-', label='Signal')
    plt.plot(t, gaussian_filtered_signal, '-', label='Gaussian Smooth')
    plt.legend()
    plt.ylabel('Signal')
    plt.xlabel('Time (a.u.)')
    plt.show()


def denoise_emg_signal():
    # EMG: Electromyogram
    mat_contents = loadmat('emg4TKEO.mat')
    emg = mat_contents.get('emg')
    emg = np.reshape(emg, emg.shape[1])
    emg_time = mat_contents.get('emgtime')
    emg_time = np.reshape(emg_time, emg_time.shape[1])
    fs = mat_contents.get('fs')

    plt.plot(emg_time, emg)
    plt.show()

    # filter using Teager-Kaiser energy-tracking Operatior(TKEO)
    # y(t) = ((x(t)) ^ 2) - (x(t-1) * x(t+1))
    filtered_emg = np.copy(emg)
    for t in range(1, len(emg) - 1):
        filtered_emg[t] = (emg[t] ** 2) - (emg[t - 1] * emg[t + 1])

    # for faster execution
    # filtered_emg = np.copy(emg)
    # filtered_emg[1:-1] = np.square(emg[1:-1]) - emg[0:-2] * emg[2:]

    plt.plot(emg_time, emg / max(emg), label='EMG')
    plt.plot(emg_time, filtered_emg / max(filtered_emg), label='EMG Energy (TKEO)')
    plt.legend()
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude or Energy')
    plt.show()

    # convert signals to z-score since
    # find time point zero
    time0 = find_nearest(emg_time, 0)

    # convert original EMG to z-score from time-zero
    emg_z = (emg - np.mean(emg[0:time0])) / np.std(emg[0:time0])

    # convert filtered EMG energy to z-score from time-zero
    filtered_emg_z = (filtered_emg - np.mean(filtered_emg[0:time0])) / np.std(filtered_emg[0:time0])

    plt.plot(emg_time, emg_z, label='EMG')
    plt.plot(emg_time, filtered_emg_z, label='EMG Energy (TKEO)')
    plt.legend()
    plt.xlabel('Time (ms)')
    plt.ylabel('Z-score related to pre-stimulus')
    plt.show()


def determine_threshold(signal):
    index = 0
    result = np.zeros(math.ceil(max(signal)))
    for i in signal:
        result[math.floor(i)] += 1

    for i in range(len(result)):
        if result[i] == 0:
            index = i
            break

    return index + 1


def median_filter_example():
    t, _signal, signal = generate_time_series_with_spike(2000, 0.05)

    plt.plot(t, signal, '-', label='Noisy signal')
    plt.plot(t, _signal, '-', label='Signal')
    plt.legend()
    plt.ylabel('Signal')
    plt.xlabel('Time (s)')
    plt.show()

    # use histogram to pick threshold
    threshold = determine_threshold(signal)
    plt.hist(signal, bins=100)
    plt.plot((threshold, threshold), (0, threshold))
    plt.show()

    # super_threshold = [i in ]


def main():
    # time_series_example()
    # spike_time_series_example()
    # denoise_emg_signal()
    median_filter_example()


if __name__ == '__main__':
    main()
