from pyaudio import PyAudio
import numpy as np
from bisect import bisect_left


class Analyzer(object):

    def __init__(self, sample_rate, dtype=np.int16):
        assert sample_rate % 2 == 0, 'sample_rate must be even'
        self.sample_rate = sample_rate
        self.sample_freq = 1.0 / sample_rate
        self.dtype = dtype
        self.low_bounds = (0, 60)
        self.mid_bounds = (60, 3000)
        self.high_bounds = (3000, 10000)
        self._bytes = None
        self._fft = None

    def load(self, bytes):
        self._signal = np.frombuffer(bytes, self.dtype)
        self._fft, self._freqs = self._calculate_fft(self._signal)

    def _calculate_fft(self, signal):
        fft = np.abs(np.fft.rfft(signal))
        n = len(signal)
        freqs = np.fft.fftfreq(n, self.sample_freq)[:n/2]
        return fft, freqs

    @property
    def power(self):
        return self._power(self._fft)

    @property
    def low_power(self):
        return self._power(self.lows[0])

    @property
    def mid_power(self):
        return self._power(self.mids[0])

    @property
    def high_power(self):
        return self._power(self.highs[0])

    def _power(self, freqs):
        return np.log(np.sum(freqs)/len(freqs)/1000)

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
        low = bisect_left(self.freqs, bounds[0])
        high = bisect_left(self.freqs, bounds[1])
        return self.fft[low:high], self.freqs[low:high]

    @property
    def low_peak(self):
        lows, low_freqs = self.lows
        return low_freqs[np.argmax(lows)]

    def mid_peaks(self, n=1):
        mids, mid_freqs = self.mids
        return mid_freqs[np.argpartition(-mids, n)[:n]]

    @property
    def high_peak(self):
        highs, high_freqs = self.highs
        return high_freqs[np.argmax(highs)]


class AudioStream(object):

    def __init__(self, sample_rate=44100, channels=1, width=2, chunk=1024,
                 input_device_index=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.width = width
        self.chunk = chunk
        self.input_device_index = input_device_index

    def __enter__(self):
        self._pa = PyAudio()
        if self.input_device_index is None:
            self.input_device_index = \
                self._pa.get_default_input_device_info()['index']
        self._stream = self._pa.open(
            format=self._pa.get_format_from_width(self.width),
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
            input_device_index=self.input_device_index)
        self._stream.start_stream()
        return self

    def stream(self):
        try:
            while True:
                bytes = self._stream.read(self.chunk)
                self.handle(bytes)
        except (KeyboardInterrupt, SystemExit):
            pass

    def __exit__(self, type, value, traceback):
        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()

    def handle(self, bytes):
        pass
