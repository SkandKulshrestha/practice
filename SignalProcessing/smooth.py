import math
import numpy as np
import matplotlib.pyplot as plt


def mean_smooth(signal, filter_window, include_edge=True):
    duration = len(signal)

    if include_edge:
        filtered_signal = np.copy(signal)
    else:
        filtered_signal = np.zeros(duration)

    for i in range(filter_window, duration - filter_window):
        filtered_signal[i] = np.mean(signal[i - filter_window:i + filter_window + 1])

    return filtered_signal


def gaussian_smooth(signal, filter_window, gaussian_window, include_edge=True):
    duration = len(signal)

    if include_edge:
        filtered_signal = np.copy(signal)
    else:
        filtered_signal = np.zeros(duration)

    for i in range(filter_window, duration - filter_window):
        filtered_signal[i] = sum(signal[i - filter_window:i + filter_window + 1] * gaussian_window)

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
