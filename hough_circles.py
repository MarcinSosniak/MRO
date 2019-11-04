import numpy as np
# from skimage import data
import matplotlib.pyplot as plt
from enum import  Enum
import math
from PIL import Image
import canny
import random

class Object_params:
    def __init__(self,rmax,rmin=-1):
        self.rmax =rmax
        if rmin < 0:
            self.rmin=rmax
        else:
            self.rmin=rmin

    def rs(self):
        return self.rmax-self.rmin+1

def vote(image,params):
    max_r=math.sqrt((image.shape[0]*image.shape[0])+(image.shape[1]*image.shape[1]))+params.rmax
    out = np.zeros((image.shape[0]+params.rmax+1,image.shape[1]+params.rmax+1),dtype=np.uint64)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if not image[x][y]==255:
                continue
            for r in range(params.rmin,params.rmax+1):
                for t in range(0,360):
                    a = x - r* math.cos((t/180)*math.pi)
                    b = y - r* math.sin((t/180)*math.pi)
                    out[int(a),int(b)] += 1
                    # out[int(a)+1,int(b)] += 1
                    # out[int(a),int(b)+1] += 1
                    # out[int(a)+1,int(b)+1] += 1
    return out


def get_valid_votes(votes,error_v=0.1):
    max_v = 0
    for x in range(votes.shape[0]):
        for y in range(votes.shape[1]):
            if votes[x][y] > max_v:
                max_v = votes[x][y]
    max_elidgible = max_v * (1 - error_v)
    vote_max = np.zeros(votes.shape,np.uint8)
    for x in range(votes.shape[0]):
        for y in range(votes.shape[1]):
            if votes[x][y] >= max_elidgible:
                vote_max[x][y]= 255
    return vote_max


def circles_form_vote_by_revoting(votes,image,params,error_v=0.1):
    vote_max= get_valid_votes(votes,error_v=error_v)
    out_img = np.zeros(image.shape,np.uint8)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if not image[x][y] == 255:
                continue
            for r in range(params.rmin, params.rmax + 1):
                for t in range(0, 360):
                    a = x - r * math.cos((t / 180) * math.pi)
                    b = y - r * math.sin((t / 180) * math.pi)
                    if vote_max[int(a)][int(b)] > 0:
                        out_img[x][y]=255
    return out_img


def apply_noise(image, noise_probabilty= 0.02):
    out= np.zeros(image.shape,np.uint8)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            swap = False if random.uniform(0,1) > noise_probabilty else True
            if swap:
                out[x][y]= 255 if image[x][y]==0 else 0
            else:
                out[x][y]= 255 if image[x][y]==255 else 0
    return out





def main():
    params= Object_params(10)
    img = canny.open_img_as_np_arr_bw("circles_in/circles_small_2.png")
    img = apply_noise(img,noise_probabilty=0.1)
    canny.display(img)
    voted= vote(img,params)
    canny.display(voted)
    voted_max= get_valid_votes(voted,error_v=0.2)
    canny.display(voted_max)
    outcome = circles_form_vote_by_revoting(voted,img,params,error_v=0.2)
    canny.display(outcome)


if __name__=="__main__":
    main()