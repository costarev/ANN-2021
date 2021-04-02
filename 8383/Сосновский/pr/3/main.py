import numpy as np
import sys


def solve(channelInrement, pictureChannels):
    pictureChannels += channelInrement
    return pictureChannels


sys.stdin = open('input.txt', 'r')
height, width, numChannels = map(int, input().split())
pictureMassive = \
    np.array(list(map(int, input().replace('[', ' ').replace(']', ' ').split()))).reshape(numChannels, height, width)
channelsMassive = np.array(list(map(int, input().split())))

for z in range(numChannels):
    pictureMassive[z] = solve(channelsMassive[z], pictureMassive[z])

outputFile = open('output.txt', 'w')
outputFile.write(str(pictureMassive))
