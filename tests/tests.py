from unittest import TestCase

from lissajous import Analyzer


SAMPLE_RATE = 44100


class AnalyzerTests(TestCase):

    def setUp(self):
        self.a = Analyzer(sample_rate=SAMPLE_RATE)

    def test_mid_frequency(self):
        with open('tests/a220_data', 'rb') as f:
            data = f.read()
        self.a.load(data)
        self.assertEqual(self.a.get_mid_peaks(1)[0], 215.33203125)
