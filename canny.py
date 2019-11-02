import numpy as np
# from skimage import data
import matplotlib.pyplot as plt
from enum import  Enum
import math
from PIL import Image

class GradDir(Enum):
    VER = 0
    R_UP = 1
    RIGHT =  2
    R_DOWN = 3
    NULL_STATE = 4




class Point2:
    def __init__(self,x,y):
        self._x=x
        self._y=y

    def x(self):
        return self._x

    def y(self):
        return  self._y


    @staticmethod
    def sdist(p0,p1):
        return math.sqrt( ((p0._x-p1._x)*(p0._x-p1._x))+((p0._y - p1._y)*(p0._y - p1._y)))

    def __or__(self, other):
        return Point2.sdist(self,other)

    def dist(self,other):
        return Point2.sdist(self,other)



def get_gausian_kernel(k):
    center_p= Point2(k,k)
    out = np.zeros((2*k+1,2*k+1))
    # out[center_p.x()][center_p.y()]=1.
    for i in range(2*k+1):
        for j in range(2*k+1):
            dist= center_p.dist(Point2(i,j))
            out[i][j] = math.exp(dist*dist/-2)
    return out

def apply_gaussian_kernel(image_arr,kernel,k):
    cpy = np.zeros(image_arr.shape)
    for i in range(image_arr.shape[0]):
        for j in range(image_arr.shape[1]):
            pixel_val=0.
            for i2 in range(-k, k + 1):
                for j2 in range(-k,k+1):
                    if not ( i+i2 < 0 or j+j2 < 0 or i+i2 >= image_arr.shape[0] or j+j2 >= image_arr.shape[1]):
                        pixel_val+= image_arr[i+i2][j+j2]*kernel[k+i2][k+j2]
            # if pixel_val > 255.:
                # print("hue")
            cpy[i][j]= pixel_val
    return cpy


def normalise(image_arr):
    out = np.zeros(image_arr.shape,dtype='uint8')
    max= 0.
    for i in range(image_arr.shape[0]):
        for j in range(image_arr.shape[1]):
            if image_arr[i][j] > max:
                max= image_arr[i][j]
    if max == 0.:
        return
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i][j]= int( image_arr[i][j]/max * 255)
    return out



class Grad:
    def __init__(self,gradX,gradY):
        self._gradX=gradX
        self._gradY=gradY
        self._dir  = Grad._makegrad_dir(gradX,gradY)
        self._grad= math.sqrt(gradX*gradX + gradY*gradY)

    @staticmethod
    def _makegrad_dir(gradX,gradY):
#          y/x = tg(a). a = arctag(y/x)
        if gradX==gradY==0:
            return GradDir.NULL_STATE
        if gradX==0:
            return GradDir.VER
        a = math.atan(gradY/gradX)
        slice = math.pi/8
        if a <= 1* slice and a >= -1*slice:
            return GradDir.RIGHT
        if 3* slice >= a and a > 1 * slice:
            return GradDir.R_UP
        if a > 3*slice or a < -3*slice:
            return GradDir.VER
        if a < -1*slice and a > -3* slice:
            return GradDir.R_DOWN

    def dir(self):
        return self._dir

    def grad(self):
        return self._grad





def gradient_map(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    gradient_map = []
    for i in range(img.shape[0]):
        row_list=[]
        for j in range(img.shape[1]):
            grad_x=0.
            grad_y=0.
            for i2 in range(-1,2):
                for j2 in range(-1,2):
                    if not (i+i2 < 0 or i+i2 >= img.shape[0] or j+j2<0 or j+j2 >= img.shape[1]):
                        grad_x+=img[i+i2][j+j2] * Kx[i2][j2]
                        grad_y+=img[i+i2][j+j2] * Ky[i2][j2]
            row_list.append(Grad(grad_x,grad_y))
        gradient_map.append(row_list)
    return gradient_map,img.shape


def display(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def non_maximum_supprestion_and_normalization(gradient_map,shape):
    out_array= np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            out_array[i][j]=0
            direc= gradient_map[i][j].dir()
            grad = gradient_map[i][j].grad()
            out_array[i][j]=0
            if direc==GradDir.NULL_STATE:
                continue

            if direc==GradDir.VER:
                if not ((i>0 and gradient_map[i-1][j].grad() > grad ) or ( i<shape[0] -1 and gradient_map[i+1][j].grad() > grad)):
                    out_array[i][j]=grad
            if direc == GradDir.R_UP:
                if not ((i > 0 and j < shape[1]-1 and gradient_map[i - 1][j+1].grad() > grad) or (
                        i < shape[0] -1 and j>0 and gradient_map[i + 1][j-1].grad() > grad)):
                    out_array[i][j] = grad
            if direc == GradDir.R_DOWN:
                if not ((i > 0 and j>0  and gradient_map[i - 1][j-1].grad() > grad) or (
                        i < shape[0] -1 and j < shape[1] -1 and gradient_map[i + 1][j+1].grad() > grad)):
                    out_array[i][j] = grad
            if direc == GradDir.RIGHT:
                if not ((j > 0 and gradient_map[i][j-1].grad() > grad) or (
                        j < shape[0] -1 and gradient_map[i][j+1].grad() > grad)):
                    out_array[i][j] = grad
    max_val=0.
    for i in range(shape[0]):
        for j in range(shape[1]):
            if out_array[i][j] > max_val:
                max_val=out_array[i][j]
    real_out_array= np.zeros(out_array.shape,dtype="uint8")
    for i in range(shape[0]):
        for j in range(shape[1]):
            real_out_array[i][j]=int(255*out_array[i][j]/max_val)
    return real_out_array

def double_treshold(image, lower_limit, upper_limit):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] >= upper_limit:
                continue
            if image[i][j] <= lower_limit:
                image[i][j]=0
                continue
            x_list=[i2 for i2 in range(i-1 if i>0 else i, i+1 if i+1 < image.shape[0] else i)]
            y_list=[j2 for j2 in range(j-1 if j>0 else j, j+1 if j+1 < image.shape[1] else j)]
            x_y_list= [(i3,j3) for i3 in x_list for j3 in y_list]
            if upper_limit <= max(list(map(lambda x: image[x[0]][x[1]],x_y_list))):
                continue
            image[i][j] = 0
    return  image


def max_out(image):
    hue = np.zeros(image.shape,dtype="uint8")
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if hue[i][j] > 0:
                hue[i][j]=255



if __name__=="__main__":
    img = Image.open("ellington-1965-15-2.png", "r")
    img_a= np.array(img)
    # plt.imshow(img_a)
    # plt.show()
    new_img_a= img_a[:,:,1]
    # plt.imshow(new_img_a, cmap='gray')
    # plt.show()
    kern = get_gausian_kernel(1)
    cpy= apply_gaussian_kernel(new_img_a,kern,1)
    final=  normalise(cpy)
    # plt.imshow(final, cmap='gray')
    # plt.show()
    grad_map,shape_ =gradient_map(final)
    hue = np.array(list(map(lambda x : list(map(lambda y : y.grad(),x)) ,grad_map)))
    # display(hue)
    after_non_max= non_maximum_supprestion_and_normalization(grad_map,shape_)
    display(after_non_max)
    after_d_t= double_treshold(after_non_max,140,100)
    display(after_d_t)