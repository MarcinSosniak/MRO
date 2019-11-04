import numpy as np
# from skimage import data
import matplotlib.pyplot as plt
from enum import  Enum
import math
from PIL import Image
import canny
import random
import scipy.misc as scm
import imageio


FI_STEPS = 1

gl_max_x=0
gl_max_y=0
gl_snd_x=0
gl_snd_y=0



def make_r_table(shape_img):
    ref_x= int(shape_img.shape[0]/2)
    ref_y= int(shape_img.shape[1]/2)
    table= [ [] for i in range(FI_STEPS) ]
    for x in range(shape_img.shape[0]):
        for y in range(shape_img.shape[1]):
            if not shape_img[x][y]==255: continue
            a = math.atan2(y-ref_y,x-ref_x)
            r = math.sqrt(((y-ref_y)* (y-ref_y))+ ((x-ref_x) *(x-ref_x)))
            for fi in range(FI_STEPS):
                a_adjusted = a + 2*math.pi*fi / FI_STEPS
                if a_adjusted > 2*math.pi: a_adjusted-=2*math.pi
                table[fi].append((r, a_adjusted))
    print("table.len {}".format(table[0].__len__()))
    return table


def get_ref_point_picture(shape_img):
    out = np.zeros(shape_img.shape)
    ref_x = int(shape_img.shape[0] / 2)
    ref_y = int(shape_img.shape[1] / 2)
    out[ref_x][ref_y]=255
    return out

def count_non_zero(image):
    sum =0
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x][y]==255: sum+=1
    return sum

def count_votes(votes, fi):
    image = votes[fi]
    sum=0
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            sum+=image[x][y]
    return sum


def vote(image, r_table):
    votes= [np.zeros(image.shape) for i in range(FI_STEPS)]

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if not image[x][y] == 255: continue
            for fi in range(FI_STEPS):
                for r,a in r_table[fi]:
                    vote_x =int( x + r* math.cos(a))
                    vote_y = int(y + r* math.sin(a))
                    if vote_x >= image.shape[0] or vote_x < 0: continue
                    if vote_y >= image.shape[1] or vote_y < 0: continue
                    votes[fi][vote_x][vote_y]+=1
    return votes


def max_votes(votes):
    out = []
    snd =[]
    for fi_vote in votes:
        max_v = 0
        max_x = 0
        max_y = 0
        max_last_x=0
        max_last_y=0
        for x in range(fi_vote.shape[0]):
            for y in range(fi_vote.shape[1]):
                if fi_vote[x][y] > max_v:
                    max_v=fi_vote[x][y]
                    max_last_x=max_x
                    max_last_y=max_y
                    max_x=x
                    max_y=y

        global gl_max_x
        gl_max_x=max_x
        global gl_max_y
        gl_max_y=max_y
        global gl_snd_x
        gl_snd_x=max_last_x
        global gl_snd_y
        gl_snd_y=max_last_y

        fi_vote_final = np.zeros(fi_vote.shape,dtype=np.uint8)
        fi_vote_final[max_x][max_y]=255
        fi_vote_snd = np.zeros(fi_vote.shape,dtype=np.uint8)
        fi_vote_snd[max_last_x][max_last_y]=255
        print("({};{})".format(max_x,max_y))
        out.append(fi_vote_final)
        snd.append(fi_vote_snd)
    return out,snd

def re_vote(image,votes,r_table, fi):
    out = np.zeros(image.shape,dtype=np.uint8)
    single_vote = votes[fi]
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if not image[x][y] == 255: continue
            for r,a in r_table[fi]:
                vote_x = int(x + r* math.cos(a))
                vote_y = int(y + r* math.sin(a))
                if vote_x >= image.shape[0] or vote_x < 0: continue
                if vote_y >= image.shape[1] or vote_y < 0: continue
                if single_vote[vote_x][vote_y] == 255:
                    print("haha got one")
                    out[x][y]=255
    return out


def re_draw(shape_img, img_shape,c_x,c_y):
    out = np.zeros(img_shape)
    ref_x= shape_img.shape[0]
    ref_y= shape_img.shape[1]
    v_x= c_x - ref_x
    v_y = c_y -ref_y
    for x in range(shape_img.shape[0]):
        for y in range(shape_img.shape[1]):
            if shape_img[x][y]==255:
                out[x+v_x][y+v_y]=255
    return out





def main():
    # shape  = Image.open("shapes_out/t1.png")
    # shape = np.array(shape)
    shape = canny.open_img_as_np_arr_bw("shapes_in/test_shape_2.png")
    canny.display(get_ref_point_picture(shape))
    r_table = make_r_table(shape)
    print("table made, voting")
    voted= vote(shape, r_table)
    print("Non zero :{} votes:{} r_table_size: ".format(count_non_zero(shape), count_votes(voted,0)),len(r_table[0]))
    canny.display(voted[0])
    max_v,snd= max_votes(voted)
    canny.display(max_v[0])
    re_vote_out= re_vote(shape,max_v,r_table,0)
    canny.display(re_vote_out)
    re_vote_snd= re_vote(shape,snd,r_table,0)
    hue = np.zeros((re_vote_out.shape[0],re_vote_out.shape[1],3))
    # for x in range(re_vote_out.shape[0]):
    #     for y in range(re_vote_out.shape[1]):
    #         hue[x][y][0]=re_vote_out[x][y]
    #         hue[x][y][2]=re_vote_snd[x][y]
    #         hue[x][y][1]=shape[x][y]
    im_max = re_draw(shape,shape.shape,gl_max_x,gl_max_y)
    im_snd = re_draw(shape,shape.shape,gl_snd_x,gl_snd_x)


    for x in range(re_vote_out.shape[0]):
        for y in range(re_vote_out.shape[1]):
            hue[x][y][0] = im_max[x][y]

            hue[x][y][1] = im_snd[x][y]

    plt.imshow(hue)
    plt.show()



    # new_img_a =  canny.open_img_as_np_arr_bw("shapes_in/shape_solo.png")
    # # canny.display(new_img_a)
    # canny_out = canny.canny(new_img_a,  40,40)
    # # canny.display(canny_out)
    # imageio.imwrite("shapes_out/t1.png",canny_out)
    pass

if __name__=="__main__":
    main()
