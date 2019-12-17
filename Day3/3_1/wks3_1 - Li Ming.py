#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# Workshop 3.1
# ------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import morphsnakes as ms

os.chdir("/Users/liming/projects/vision/Day3/3_1")




def cv2plt(img):
    plt.axis('off')
    if np.size(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img,cmap='gray',vmin=0,vmax=255)



##########
    

# a. Read in the image 'symbols.jpg'. Name the array as 'imgcolor'.
#

imgcolor = cv2.imread("symbols.jpg")
cv2plt(imgcolor)



##########


# b. Use morphsnakes to segment the objects in the image. 
#    Create the mask as shown in 'wks3_1_b.jpg'.

kernel = np.ones((5,5),np.uint8)
noiserem = cv2.morphologyEx(np.uint8(imgcolor),
                            cv2.MORPH_OPEN,
                            kernel)
cv2plt(noiserem)

gray = cv2.cvtColor(noiserem,cv2.COLOR_BGR2GRAY)
cv2plt(gray)
gray.shape

gray01 = gray/255.0
invg = ms.inverse_gaussian_gradient(gray01, alpha=700, sigma=3)
ls0 = ms.circle_level_set(gray01.shape, (200, 250), 250)

callback = ms.visual2d(cv2.cvtColor(imgcolor,cv2.COLOR_BGR2RGB))
lsf = ms.morphological_geodesic_active_contour(invg,
                                               iterations=500,
                                               init_level_set=ls0,
                                               smoothing=1,
                                               threshold=0.5,
                                               balloon=-1,
                                               iter_callback=callback)

cv2plt(lsf*255)
