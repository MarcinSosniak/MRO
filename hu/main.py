import numpy as np
# from skimage import data
import matplotlib.pyplot as plt
from enum import  Enum
import math
from PIL import Image
import random

from os import listdir
from os.path import isfile, join


def m(p,q,image):
    out = 0
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
           out += (x**p) * (y**p)


def open_files(letter):
    out = []
    for file in [f for f in listdir('data_in/'+letter) if isfile(join('data_in/'+letter, f))]:
        image= cv2.imread('data_in/'+letter+'/'+file, cv2.IMREAD_GRAYSCALE)
        out.append((file,image))
    return out

def open_all():
    hue = {}
    hue['r'] = open_files('r')
    hue['w'] = open_files('w')
    hue['p'] = open_files('p')
    return hue



import cv2, sys, os
from math import copysign, log10

def main():
    showLogTransformedHuMoments = True
    hue = open_all()

    for letter in ['p','r','w']:
        print(letter)
        for file_name,img in hue[letter]:

            # Obtain filename from command line argument
            filename = file_name

            # Read image
            im = img

            # Threshold image
            _,im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)

            # Calculate Moments
            moment = cv2.moments(im)

            # Calculate Hu Moments
            huMoments = cv2.HuMoments(moment)

            # Print Hu Moments
            print("  {}: ".format(filename),end='')

            for i in range(0,7):
                if showLogTransformedHuMoments:
                    # Log transform Hu Moments to make
                    # squash the range
                    print("{:.5f}".format(-1*copysign(1.0,\
                            huMoments[i])*log10(abs(huMoments[i]))),\
                            end=' ')
                else:
                    # Hu Moments without log transform
                    print("{:.5f}".format(huMoments[i]),end=' ')
            print()


if __name__=="__main__":
    main()
