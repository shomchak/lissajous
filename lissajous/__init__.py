import numpy
import bisect


class Analyzer(object):

    def __init__(self, sample_rate, dtype=numpy.int16):
        assert sample_rate % 2 == 0, 'sample_rate must be even'
        self.sample_rate = sample_rate
        self.sample_freq = 1.0 / sample_rate
        self.dtype = dtype
        self.low_bounds = (0, 50)
        self.mid_bounds = (50, 3000)
        self.high_bounds = (3000, 10000)
        self._bytes = None
        self._fft = None

    def load(self, bytes):
        self._signal = numpy.fromstring(bytes, self.dtype)
        self._fft, self._freqs = self._calculate_fft(self._signal)

    def _calculate_fft(self, signal):
        fft = numpy.abs(numpy.fft.rfft(signal))
        n = len(signal)
        freqs = numpy.fft.fftfreq(n, self.sample_freq)[:n/2]
        return fft, freqs

    @property
    def power(self):
        return numpy.log(numpy.sum(self._fft)/len(self._fft)/1000)

    @property
    def freqs(self):
        return self._freqs

    @property
    def fft(self):
        return self._fft

    @property
    def lows(self):
        return self._filter_fft(self.low_bounds)

    @property
    def mids(self):
        return self._filter_fft(self.mid_bounds)

    @property
    def highs(self):
        return self._filter_fft(self.high_bounds)

    def _filter_fft(self, bounds):
        low = bisect.bisect_left(self.freqs, bounds[0])
        high = bisect.bisect_left(self.freqs, bounds[1])
        return self.fft[low:high], self.freqs[low:high]

    @property
    def low_peak(self):
        lows, low_freqs = self.lows
        return low_freqs[numpy.argmax(lows)]

    def mid_peaks(self, n=1):
        mids, mid_freqs = self.mids
        return mid_freqs[numpy.argpartition(-mids, n)[:n]]

    @property
    def high_peak(self):
        highs, high_freqs = self.highs
        return high_freqs[numpy.argmax(highs)]
