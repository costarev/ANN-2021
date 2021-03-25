import numpy
from pathlib import Path

path = Path("test.txt")
dims = numpy.genfromtxt(path.absolute(), skip_footer=2, dtype="int")
numChannels = numpy.genfromtxt(path.absolute(), skip_footer=1, skip_header=1)
dataImg = numpy.genfromtxt(path.absolute(), skip_header=2)

dataImg = numpy.reshape(dataImg, dims)
imgOutput = numpy.dot(dataImg, numChannels)

numpy.savetxt("out.txt", imgOutput, fmt='%0.1f')
