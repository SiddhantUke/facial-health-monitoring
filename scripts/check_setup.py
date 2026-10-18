import sys

import numpy
import scipy
import sklearn

print("Python :", sys.version.split()[0])
print("NumPy  :", numpy.__version__)
print("SciPy  :", scipy.__version__)
print("sklearn:", sklearn.__version__)


print("FFT test:", abs(numpy.fft.rfft([0, 1, 0, -1] * 25)).max().round(1))
print("SETUP OK")



