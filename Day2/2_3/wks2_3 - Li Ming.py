#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# Workshop 2.3
# ------------


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir("/Users/liming/projects/vision/Day2/2_3")

def cv2plt(img):
    plt.axis('off')
    if np.size(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img,cmap='gray',vmin=0,vmax=255)
    plt.show()
    
    
    
# a. Read in the image 'symbols.jpg'. Name the array as 'sym'
#
sym = cv2.imread("symbols.jpg")




##########


# b. Convert the image to grayscale. Name the array as 'symg'
#
symg = cv2.cvtColor(sym,cv2.COLOR_BGR2GRAY)
cv2plt(symg)
symg3 = cv2.merge((symg, symg, symg))
cv2plt(symg3)



##########


# c. Perform canny edge detction on symg. Name the array as 'symc'
#
#symg = cv2.GaussianBlur(symg, (11,11), 0)
symc = cv2.Canny(symg, 70, 100, apertureSize=3)
cv2plt(symc)
symc3 = cv2.merge((symc, symc, symc))
cv2plt(symc3)

cv2.imshow("my window", symc)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)

##########


# d. Create the output 'wks2_3_d.jpg'. Name the final output as 'symb'
#    The colour of the boundaries is (191,191,191)

(_,thr) = cv2.threshold(symg,200,255,cv2.THRESH_BINARY)
cv2plt(thr)
msk = 255-thr
cv2plt(msk)

msk1 = np.round(np.float32(msk)/255)
iso = cv2.bitwise_and(sym, sym,
                      mask=np.uint8(msk))
cv2plt(iso)


ctrs= cv2.findContours(np.uint8(msk),
                       cv2.RETR_EXTERNAL,
                       cv2.CHAIN_APPROX_SIMPLE)
ctrs= ctrs[1]

symb = cv2.drawContours(iso,
                 ctrs,
                 -1,
                 (191,191,191),
                 5)
cv2plt(symb)


##########


# e. Create the output 'wks2_3_e.jpg'. Name the final output as 'symhv'


cv2plt(sym)
np.concatenate((symb, symg3)).shape
symhv0 = np.concatenate((np.concatenate((sym, symb)), np.concatenate((symc3, symg3))), axis=1)
symhv = cv2.copyMakeBorder(symhv0,5,5,5,5,cv2.BORDER_CONSTANT, value=(0,0,0))
cv2plt(symhv)


