import numpy as np
# from skimage import data
import matplotlib.pyplot as plt
from enum import  Enum
import math
from PIL import Image

from canny import open_img_as_np_arr_bw


if __name__=="__main__":
    arr= open_img_as_np_arr_bw("test.png")
    print(arr)