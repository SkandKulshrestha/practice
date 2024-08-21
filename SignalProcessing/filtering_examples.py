import math
from cProfile import label

import numpy as np

from dataclasses import dataclass, field
from typing import Tuple
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

    def __init__(self, sampling_rate: int, frequency_range: Tuple[int, int], transition_width: float,
                 order: int | None = None):
        super().__init__(
            sampling_rate=sampling_rate,
            frequency_range=frequency_range,
            transition_width=transition_width,
            order=round(5 * sampling_rate / frequency_range[0]) if order is None else order
        )

        self.order = Utility.make_integer_odd_by_incrementing_by_one(self.order)
        # print(f'{self.order = }')

        self.frequency_vector = np.array([
            0,
            self.frequency_range[0] - (self.frequency_range[0] * self.transition_width),
            self.frequency_range[0],
            self.frequency_range[1],
            self.frequency_range[1] + (self.frequency_range[1] * self.transition_width),
            self.nyquist_frequency
        ])
        # print(f'{self.frequency_vector = }')

        #
        # NOTE: No need to divide by Nyquist frequency. Because "band" parameter of "signal.firls" takes monotonic
        # non-decreasing sequence containing the band edges in Hz. All elements must be non-negative and less than
        # or equal to the Nyquist frequency.
        # refer: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firls.html
        #

        self.filter_kernel: np.ndarray = signal.firls(
            numtaps=self.order,
            bands=self.frequency_vector,
            desired=self.SHAPE,
            fs=self.sampling_rate
        )


class Evaluation:
    def __init__(self, filter_kernel, nyquist_frequency):
        self.filter_kernel = filter_kernel
        self.nyquist_frequency = nyquist_frequency

        self.hz = None
        self.filter_power = None

    def evaluate_kernel(self):
        # compute the power spectrum of the filter kernel
        filter_power = np.square(np.absolute(fft.fft(self.filter_kernel)))
        # print(f'{filter_power = }')

        # compute the frequencies vector and remove negative frequencies
        self.hz = np.linspace(0, self.nyquist_frequency, math.floor(len(self.filter_kernel) / 2) + 1)
        # print(f'{self.hz = }')

        self.filter_power = filter_power[:len(self.hz)]
        # print(f'{self.filter_power = }')


def firls_ex1():
    order = round(5 * 1024 / 20)
    firls = FIRLS(
        sampling_rate=1024,  # Hz
        frequency_range=(20, 45),
        transition_width=0.1,
        order=order
    )
    # print(firls.frequency_vector)

    evaluation = Evaluation(
        filter_kernel=firls.filter_kernel,
        nyquist_frequency=firls.nyquist_frequency
    )
    evaluation.evaluate_kernel()

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
    plt.title('Frequency response of filter (firls)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain')

    plt.subplot(224)
    plt.plot(evaluation.hz, 10 * np.log10(evaluation.filter_power), color='black', linestyle='-', marker='s',
             linewidth=2)
    plt.xlim(0, firls.frequency_range[0] * 4)
    plt.ylim(-50, 2)
    plt.title('Frequency response of filter (firls)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain (dB)')

    plt.show()


if __name__ == '__main__':
    firls_ex1()
