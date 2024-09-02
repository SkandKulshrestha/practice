import math

import numpy as np

from dataclasses import dataclass, field
from typing import Tuple

from PIL.ImageChops import constant
from scipy import signal, fft
from matplotlib import pyplot as plt


class Utility:
    @staticmethod
    def make_integer_odd_by_incrementing_by_one(x: int) -> int:
        if x % 2 == 0:
            x += 1
        return x


@dataclass
class FIR:
    sampling_rate: int
    nyquist_frequency: int = field(init=False)
    frequency_range: Tuple[int, int]
    transition_width: float
    order: int

    def __post_init__(self):
        self.nyquist_frequency = self.sampling_rate // 2


class FIRLS(FIR):
    SHAPE: Tuple = (0, 0, 1, 1, 0, 0)

    def __init__(self, sampling_rate: int, frequency_range: Tuple[int, int], transition_width: float, order: int):
        super().__init__(
            sampling_rate=sampling_rate,
            frequency_range=frequency_range,
            transition_width=transition_width,
            order=order
        )

        # modify order to make odd since "signal.firls" take odd order value
        self.order = Utility.make_integer_odd_by_incrementing_by_one(self.order)

        # calculate frequency vector using transition width
        self.frequency_vector: np.array = np.array([
            0,
            self.frequency_range[0] - (self.frequency_range[0] * self.transition_width),
            self.frequency_range[0],
            self.frequency_range[1],
            self.frequency_range[1] + (self.frequency_range[1] * self.transition_width),
            self.nyquist_frequency
        ])

        #
        # NOTE: No need to divide by Nyquist frequency. Because "band" parameter of "signal.firls" takes monotonic
        # non-decreasing sequence containing the band edges in Hz. All elements must be non-negative and less than
        # or equal to the Nyquist frequency.
        # refer: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firls.html
        #

        # create filer kernel using "signal.firls"
        self.filter_kernel: np.ndarray = signal.firls(
            numtaps=self.order,
            bands=self.frequency_vector,
            desired=self.SHAPE,
            fs=self.sampling_rate
        )


class FIR1(FIR):
    SHAPE: Tuple = (0, 0, 1, 1, 0, 0)

    def __init__(self, sampling_rate: int, frequency_range: Tuple[int, int], order: int):
        super().__init__(
            sampling_rate=sampling_rate,
            frequency_range=frequency_range,
            transition_width=0,
            order=order
        )

        # modify order to make odd since "signal.firwin" take odd order value
        self.order = Utility.make_integer_odd_by_incrementing_by_one(self.order)

        # calculate frequency vector using transition width
        self.frequency_vector: np.array = np.array([
            0,
            self.frequency_range[0],
            self.frequency_range[0],
            self.frequency_range[1],
            self.frequency_range[1],
            self.nyquist_frequency
        ])

        # [TODO] Query: why "pass_zero=False"?

        # create filer kernel using "signal.firwin"
        self.filter_kernel: np.ndarray = signal.firwin(
            numtaps=self.order,
            cutoff=self.frequency_range,
            fs=self.sampling_rate,
            pass_zero=False
        )


@dataclass
class IIR:
    sampling_rate: int
    nyquist_frequency: int = field(init=False)
    frequency_range: Tuple[int, int]
    transition_width: float
    order: int

    def __post_init__(self):
        self.nyquist_frequency = self.sampling_rate // 2


class IIRButterworth(IIR):
    SHAPE: Tuple = (0, 0, 1, 1, 0, 0)

    def __init__(self, sampling_rate: int, frequency_range: Tuple[int, int], order: int):
        super().__init__(
            sampling_rate=sampling_rate,
            frequency_range=frequency_range,
            transition_width=0,
            order=order
        )

        # calculate frequency vector using transition width
        self.frequency_vector: np.array = np.array([
            0,
            self.frequency_range[0],
            self.frequency_range[0],
            self.frequency_range[1],
            self.frequency_range[1],
            self.nyquist_frequency
        ])

        # create filer kernel using "signal.firls"
        self.filter_kernel_b, self.filter_kernel_a = signal.butter(
            N=self.order,
            Wn=self.frequency_range,
            fs=self.sampling_rate,
            btype='bandpass'
        )


class Evaluation:
    def __init__(self, filter_instance):
        self.sampling_rate = filter_instance.sampling_rate
        self.nyquist_frequency = filter_instance.nyquist_frequency
        self.frequency_vector = filter_instance.frequency_vector
        self.filter_kernel = filter_instance.filter_kernel

        # output
        self.hz = None
        self.filter_power = None
        self.interpolated_hz = None
        self.interpolated_filter_power = None
        self.interpolated_ideal_power = None
        self.pearson_correlation = 0

    def evaluate_kernel(self):
        # compute the power spectrum of the filter kernel
        filter_power = np.square(np.absolute(fft.fft(self.filter_kernel)))

        # compute the frequencies vector and remove negative frequencies
        # @WATCH: "Understanding the Discrete Fourier Transform and the FFT" @09:26
        #           https://www.youtube.com/watch?v=QmgJmh2I3Fw&t=145s
        self.hz = np.linspace(0, self.nyquist_frequency, math.floor(len(self.filter_kernel) / 2) + 1)
        self.filter_power: np.ndarray = filter_power[:len(self.hz)]

    def correlate(self, normalize):
        interpolated_hz = np.arange(int(self.hz[-1] // normalize))
        interpolated_filter_power = np.zeros(interpolated_hz.shape, dtype=self.filter_power.dtype)
        interpolated_ideal_power = np.zeros(interpolated_filter_power.shape, dtype=interpolated_filter_power.dtype)
        frequency_vector = self.frequency_vector

        hz_index = 0
        slope = 0
        _constant = 0
        index_changed = True
        fall_slope = rise_slope = None
        fall_constant = rise_constant = None

        if len(frequency_vector) == 6:
            # calculate slope and constant once for ideal trace
            if frequency_vector[2] != frequency_vector[1]:
                rise_slope = 1 / (frequency_vector[2] - frequency_vector[1])
                rise_constant = 0 - rise_slope * frequency_vector[1]
            if frequency_vector[3] != frequency_vector[4]:
                fall_slope = 1 / (frequency_vector[3] - frequency_vector[4])
                fall_constant = 0 - fall_slope * frequency_vector[4]

            for hz in interpolated_hz:
                original_hz = hz * normalize

                # interpolate ideal power
                if original_hz <= frequency_vector[1]:
                    pass
                elif original_hz < frequency_vector[2]:
                    interpolated_ideal_power[hz] = rise_slope * original_hz + rise_constant
                elif original_hz <= frequency_vector[3]:
                    interpolated_ideal_power[hz] = 1
                elif original_hz < frequency_vector[4]:
                    interpolated_ideal_power[hz] = fall_slope * original_hz + fall_constant
                else:
                    pass

                # interpolate filter power
                if original_hz == self.hz[hz_index]:
                    interpolated_filter_power[hz] = self.filter_power[hz_index]
                    hz_index += 1
                    index_changed = True
                else:
                    if original_hz > self.hz[hz_index]:
                        hz_index += 1
                        index_changed = True

                    # calculate slope and constant only when change in hz index is detected
                    if index_changed:
                        slope = (self.filter_power[hz_index] - self.filter_power[hz_index - 1]) / (
                                self.hz[hz_index] - self.hz[hz_index - 1])
                        _constant = self.filter_power[hz_index] - (slope * self.hz[hz_index])
                        index_changed = False

                    # y = mx + c
                    interpolated_filter_power[hz] = slope * original_hz + _constant

            self.interpolated_filter_power = interpolated_filter_power
            self.interpolated_ideal_power = interpolated_ideal_power
            self.interpolated_hz = interpolated_hz
            self.pearson_correlation = np.corrcoef(interpolated_filter_power, interpolated_ideal_power)[0, 1]


def firls_ex1():
    firls = FIRLS(
        sampling_rate=1024,  # Hz
        frequency_range=(20, 45),
        transition_width=0.1,
        order=round(5 * 1024 / 20)
    )

    evaluation = Evaluation(firls)
    evaluation.evaluate_kernel()
    evaluation.correlate(normalize=1)

    plt.subplot(221)
    plt.plot(firls.filter_kernel, color='blue', linestyle='-', linewidth=2)
    plt.xlabel('Time points')
    plt.title('Filter kernel (firls)')

    plt.subplot(222)
    plt.plot(evaluation.hz, evaluation.filter_power, color='black', linestyle='-', marker='s', linewidth=2,
             label='Actual')
    plt.plot(firls.frequency_vector, firls.SHAPE, color='red', linestyle='-', marker='o',
             linewidth=2, label='Ideal')
    plt.xlim(0, firls.frequency_range[0] * 4)
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain')
    plt.title('Frequency response of filter (firls)')

    plt.subplot(223)
    plt.plot(evaluation.interpolated_hz, evaluation.interpolated_filter_power, color='black', linestyle='-',
             marker='s', linewidth=2, label=f'Actual ({evaluation.pearson_correlation:.3f})')
    plt.plot(evaluation.interpolated_hz, evaluation.interpolated_ideal_power, color='red', linestyle='-',
             marker='o', linewidth=2, label='Ideal')
    plt.xlim(0, firls.frequency_range[0] * 4)
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain')
    plt.title('Interpolated frequency response of filter (firls) for correlation')

    plt.subplot(224)
    plt.plot(evaluation.hz, 10 * np.log10(evaluation.filter_power), color='black', linestyle='-', marker='s',
             linewidth=2)
    plt.xlim(0, firls.frequency_range[0] * 4)
    plt.ylim(-50, 2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain (dB)')
    plt.title('Frequency response of filter (firls)')

    plt.show()


def firls_ex2():
    print('Varying Order')
    sampling_rate = 1024
    frequency_range = (20, 45)
    for i in range(1, 10):
        firls = FIRLS(
            sampling_rate=sampling_rate,  # Hz
            frequency_range=frequency_range,
            transition_width=0.1,
            order=round(i * sampling_rate / frequency_range[0])
        )

        evaluation = Evaluation(firls)
        evaluation.evaluate_kernel()
        evaluation.correlate(1)
        print(f'Order ({firls.order:3}) = {evaluation.pearson_correlation:.4f}')

        plt.subplot(221)
        plt.plot(np.arange(firls.order) - firls.order / 2, firls.filter_kernel + 0.02 * i, linestyle='-')

        plt.subplot(222)
        plt.plot(evaluation.hz, evaluation.filter_power, linestyle='-', label=f'Order = {firls.order}')

        plt.subplot(223)
        plt.plot(evaluation.interpolated_hz, evaluation.interpolated_filter_power, linestyle='-',
                 label=f'Order, Corr = {firls.order}, {evaluation.pearson_correlation:.3f}')

        plt.subplot(224)
        plt.plot(evaluation.hz, 10 * np.log10(evaluation.filter_power), linestyle='-')

    plt.subplot(221)
    plt.xlabel('Time points')
    plt.title('Filter kernel (firls)')

    plt.subplot(222)
    plt.plot(firls.frequency_vector, firls.SHAPE, linestyle='--', label='Ideal')
    plt.xlim(0, firls.frequency_range[0] * 5)
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain')
    plt.title('Frequency response of filter (firls)')

    plt.subplot(223)
    plt.plot(evaluation.interpolated_hz, evaluation.interpolated_ideal_power, linestyle='--', label='Ideal')
    plt.xlim(0, firls.frequency_range[0] * 5)
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain')
    plt.title('Interpolated frequency response of filter (firls) for correlation')

    plt.subplot(224)
    plt.xlim(0, firls.frequency_range[0] * 5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain (dB)')
    plt.title('Frequency response of filter (firls)')

    plt.show()


def firls_ex3():
    print('Varying Transition Width')
    sampling_rate = 1024
    frequency_range = (20, 45)
    transition_widths: np.ndarray = np.linspace(0.01, 0.4, 10)
    for i, transition_width in enumerate(transition_widths):
        firls = FIRLS(
            sampling_rate=sampling_rate,  # Hz
            frequency_range=frequency_range,
            transition_width=transition_width,
            order=round(5 * sampling_rate / frequency_range[0])
        )

        evaluation = Evaluation(firls)
        evaluation.evaluate_kernel()
        evaluation.correlate(normalize=1)
        print(f'T.W. ({firls.transition_width:.2f}) = {evaluation.pearson_correlation:.4f}')

        plt.subplot(221)
        plt.plot(np.arange(firls.order) - firls.order / 2, firls.filter_kernel + 0.02 * i, linestyle='-')

        plt.subplot(222)
        plt.plot(evaluation.hz, evaluation.filter_power, linestyle='-', label=f'T.W. = {firls.transition_width:.2f}')

        plt.subplot(223)
        plt.plot(evaluation.interpolated_hz, evaluation.interpolated_filter_power, linestyle='-',
                 label=f'T.W., Corr = {firls.transition_width:.2f}, {evaluation.pearson_correlation:.3f}')

        plt.subplot(224)
        plt.plot(evaluation.hz, 10 * np.log10(evaluation.filter_power), linestyle='-')

    plt.subplot(221)
    plt.xlabel('Time points')
    plt.title('Filter kernel (firls)')

    plt.subplot(222)
    plt.plot(firls.frequency_vector, firls.SHAPE, linestyle='--', label='Ideal')
    plt.xlim(0, firls.frequency_range[0] * 5)
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain')
    plt.title('Frequency response of filter (firls)')

    plt.subplot(223)
    plt.plot(evaluation.interpolated_hz, evaluation.interpolated_ideal_power, linestyle='--', label='Ideal')
    plt.xlim(0, firls.frequency_range[0] * 5)
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain')
    plt.title('Interpolated frequency response of filter (firls) for correlation')

    plt.subplot(224)
    plt.xlim(0, firls.frequency_range[0] * 5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain (dB)')
    plt.title('Frequency response of filter (firls)')

    plt.show()


def fir1_ex1():
    fir1 = FIR1(
        sampling_rate=1024,  # Hz
        frequency_range=(20, 45),
        order=round(5 * 1024 / 20)
    )

    evaluation = Evaluation(fir1)
    evaluation.evaluate_kernel()
    evaluation.correlate(normalize=1)

    plt.subplot(221)
    plt.plot(fir1.filter_kernel, color='blue', linestyle='-', linewidth=2)
    plt.xlabel('Time points')
    plt.title('Filter kernel (fir1)')

    plt.subplot(222)
    plt.plot(evaluation.hz, evaluation.filter_power, color='black', linestyle='-', marker='s', linewidth=2,
             label='Actual')
    plt.plot(fir1.frequency_vector, fir1.SHAPE, color='red', linestyle='-', marker='o',
             linewidth=2, label='Ideal')
    plt.xlim(0, fir1.frequency_range[0] * 4)
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain')
    plt.title('Frequency response of filter (fir1)')

    plt.subplot(223)
    plt.plot(evaluation.interpolated_hz, evaluation.interpolated_filter_power, color='black', linestyle='-',
             marker='s', linewidth=2, label=f'Actual ({evaluation.pearson_correlation:.3f})')
    plt.plot(evaluation.interpolated_hz, evaluation.interpolated_ideal_power, color='red', linestyle='-',
             marker='o', linewidth=2, label='Ideal')
    plt.xlim(0, fir1.frequency_range[0] * 4)
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain')
    plt.title('Interpolated frequency response of filter (fir1) for correlation')

    plt.subplot(224)
    plt.plot(evaluation.hz, 10 * np.log10(evaluation.filter_power), color='black', linestyle='-', marker='s',
             linewidth=2)
    plt.xlim(0, fir1.frequency_range[0] * 4)
    plt.ylim(-50, 2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain (dB)')
    plt.title('Frequency response of filter (fir1)')

    plt.show()


def fir1_ex2():
    print('Varying Order')
    sampling_rate = 1024
    frequency_range = (20, 45)
    for i in range(1, 10):
        fir1 = FIR1(
            sampling_rate=sampling_rate,  # Hz
            frequency_range=frequency_range,
            order=round(i * sampling_rate / frequency_range[0])
        )

        evaluation = Evaluation(fir1)
        evaluation.evaluate_kernel()
        evaluation.correlate(1)
        print(f'Order ({fir1.order:3}) = {evaluation.pearson_correlation:.4f}')

        plt.subplot(221)
        plt.plot(np.arange(fir1.order) - fir1.order / 2, fir1.filter_kernel + 0.02 * i, linestyle='-')

        plt.subplot(222)
        plt.plot(evaluation.hz, evaluation.filter_power, linestyle='-', label=f'Order = {fir1.order}')

        plt.subplot(223)
        plt.plot(evaluation.interpolated_hz, evaluation.interpolated_filter_power, linestyle='-',
                 label=f'Order, Corr = {fir1.order}, {evaluation.pearson_correlation:.3f}')

        plt.subplot(224)
        plt.plot(evaluation.hz, 10 * np.log10(evaluation.filter_power), linestyle='-')

    plt.subplot(221)
    plt.xlabel('Time points')
    plt.title('Filter kernel (fir1)')

    plt.subplot(222)
    plt.plot(fir1.frequency_vector, fir1.SHAPE, linestyle='--', label='Ideal')
    plt.xlim(0, fir1.frequency_range[0] * 5)
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain')
    plt.title('Frequency response of filter (fir1)')

    plt.subplot(223)
    plt.plot(evaluation.interpolated_hz, evaluation.interpolated_ideal_power, linestyle='--', label='Ideal')
    plt.xlim(0, fir1.frequency_range[0] * 5)
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain')
    plt.title('Interpolated frequency response of filter (fir1) for correlation')

    plt.subplot(224)
    plt.xlim(0, fir1.frequency_range[0] * 5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain (dB)')
    plt.title('Frequency response of filter (fir1)')

    plt.show()


def iir_butterworth_ex1():
    iir_butterworth = IIRButterworth(
        sampling_rate=1024,  # Hz
        frequency_range=(20, 45),
        order=4
    )

    plt.subplot(221)
    plt.plot(iir_butterworth.filter_kernel_b * 1e5, color='black', linestyle='-', linewidth=2, marker='s', label='B')
    plt.plot(iir_butterworth.filter_kernel_a, color='red', linestyle='-', linewidth=2, marker='s', label='A')
    plt.xlabel('Time points')
    plt.ylabel('Filter coefficient')
    plt.title('Time-domain filter coefficients')

    # evaluation using an impulse response
    impulse_response = np.zeros(1001)
    impulse_response[501] = 1

    # apply the filter
    filtered_impulse_response = signal.lfilter(
        b=iir_butterworth.filter_kernel_b,
        a=iir_butterworth.filter_kernel_a,
        x=impulse_response,
        axis=-1
    )

    # compute power spectrum
    filter_power = np.square(np.absolute(fft.fft(filtered_impulse_response)))
    hz = np.linspace(0, iir_butterworth.nyquist_frequency, math.floor(len(impulse_response) / 2) + 1)
    filter_power = filter_power[:len(hz)]

    plt.subplot(222)
    plt.plot(impulse_response, color='black', linestyle='-', linewidth=2, label='Impulse')
    plt.plot(filtered_impulse_response, color='red', linestyle='-', linewidth=2, label='Filtered')
    plt.ylim((-0.06, 0.06))
    plt.legend()
    plt.xlabel('Time points (a.u.)')
    plt.title('Filtering an impulse')

    plt.subplot(223)
    plt.plot(iir_butterworth.frequency_vector, iir_butterworth.SHAPE, color='black', linestyle='-', linewidth=2,
             label='Impulse')
    plt.plot(hz, filter_power, color='red', linestyle='-', linewidth=2, label='Filtered')
    plt.xlim(0, iir_butterworth.frequency_range[0] * 4)
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Attenuation')
    plt.title('Frequency response of filter (Butterwoth)')

    plt.subplot(224)
    plt.plot(hz, 10 * np.log10(filter_power), color='black', linestyle='-', linewidth=2)
    plt.xlim(0, iir_butterworth.frequency_range[0] * 4)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Attenuation')
    plt.title('Frequency response of filter (Butterwoth)')

    plt.show()


if __name__ == '__main__':
    # firls_ex1()
    # firls_ex2()
    # firls_ex3()

    # fir1_ex1()
    # fir1_ex2()

    iir_butterworth_ex1()
