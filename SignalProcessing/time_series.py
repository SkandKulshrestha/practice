import math

from scipy.interpolate import interp1d
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


def mean_smooth(signal, filter_window, include_edge=True):
    duration = len(signal)
    filtered_signal = np.zeros(duration)

    if include_edge:
        filtered_signal = np.copy(signal)

    for i in range(filter_window, duration - filter_window):
        filtered_signal[i] = np.mean(signal[i - filter_window:i + filter_window + 1])

    return filtered_signal


def find_nearest(array, value):
    return (np.abs(array - value)).argmin()


def gaussian_function(full_width_at_half_maximum, filter_window, sampling_rate=1000, plot_function=False):
    # normalize time vector in ms
    gaussian_time = np.arange(-filter_window, filter_window + 1) / sampling_rate * 1000

    # create gaussian window
    gaussian_window = np.exp(-(4 * math.log(2) * (gaussian_time ** 2))
                             / (full_width_at_half_maximum ** 2))

    # compute empirical FWHM
    pre_half_peak = find_nearest(gaussian_window[:filter_window], .5)
    post_half_peak = filter_window + 1 + find_nearest(gaussian_window[filter_window + 1:], .5)

    empirical_f_w_h_m = gaussian_time[post_half_peak] - gaussian_time[pre_half_peak]

    if plot_function:
        plt.plot(gaussian_time, gaussian_window, 'ko-')
        plt.plot([gaussian_time[pre_half_peak], gaussian_time[post_half_peak]],
                 [gaussian_window[pre_half_peak], gaussian_window[post_half_peak]],
                 'm')
        plt.title(f'Gaussian kernel with requested FWHM {full_width_at_half_maximum} ms'
                  f' ({empirical_f_w_h_m} ms achieved)')
        plt.ylabel('Gain')
        plt.xlabel('Time (ms)')
        plt.show()

    return gaussian_window / np.sum(gaussian_window)


def gaussian_smooth(signal, filter_window, gaussian_window, include_edge=True):
    duration = len(signal)
    filtered_signal = np.zeros(duration)

    if include_edge:
        filtered_signal = np.copy(signal)

    for i in range(filter_window, duration - filter_window):
        filtered_signal[i] = sum(signal[i - filter_window:i + filter_window + 1] * gaussian_window)

    return filtered_signal


def main():
    # sampling_rate = 1000  # Hz
    # time_duration = 10  # sec
    # noise_amplitude = 5  # noise level, measured in standard deviation
    # t, _signal, signal = generate_time_series(sampling_rate, time_duration, noise_amplitude)
    #
    # plt.plot(t, _signal, '-')
    # plt.legend(['Signal'])
    # plt.ylabel('Signal')
    # plt.xlabel('Time (s)')
    # plt.show()
    #
    # plt.plot(t, signal, '-', t, _signal, '-')
    # plt.legend(['Noisy signal', 'Signal'])
    # plt.ylabel('Signal')
    # plt.xlabel('Time (s)')
    # plt.show()
    #
    # filter_window = 50  # actually filter window will be '2 * filter_window + 1'
    # mean_filtered_signal = mean_smooth(signal, filter_window)
    #
    # plt.plot(t, signal, '-', t, _signal, '-', t, mean_filtered_signal, '-')
    # plt.legend(['Noisy signal', 'Signal', 'Mean Smooth'])
    # plt.ylabel('Signal')
    # plt.xlabel('Time (s)')
    # plt.show()
    #
    # full_width_at_half_maximum = 25  # in ms
    # filter_window = 50
    # gaussian_window = gaussian_function(full_width_at_half_maximum, filter_window, sampling_rate)
    #
    # gaussian_filtered_signal = gaussian_smooth(signal, filter_window, gaussian_window)
    #
    # plt.plot(t, signal, '-', t, _signal, '-',
    #          t, mean_filtered_signal, '-', t, gaussian_filtered_signal, '-')
    # plt.legend(['Noisy signal', 'Signal', 'Mean Smooth', 'Gaussian Smooth'])
    # plt.ylabel('Signal')
    # plt.xlabel('Time (s)')
    # plt.show()

    num_spikes = 300
    t, spike_time_series = generate_spike_time_series(num_spikes)
    plt.plot(t, spike_time_series, '-')
    plt.legend(['Signal'])
    plt.ylabel('Signal')
    plt.xlabel('Time (a.u.)')
    plt.show()

    full_width_at_half_maximum = 25  # in ms
    filter_window = 50
    gaussian_window = gaussian_function(full_width_at_half_maximum, filter_window)

    gaussian_filtered_signal = gaussian_smooth(spike_time_series, filter_window, gaussian_window)

    plt.plot(t, spike_time_series, '-', t, gaussian_filtered_signal, '-')
    plt.legend(['Signal', 'Gaussian Smooth'])
    plt.ylabel('Signal')
    plt.xlabel('Time (a.u.)')
    plt.show()


if __name__ == '__main__':
    main()
