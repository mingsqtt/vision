#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# Workshop 2.2
# ------------

# The objective of the workshop is to isolate blood vessels from
# the retinal image


import cv2
import numpy as np
import matplotlib.pyplot as plt


def cv2plt(img):
    plt.axis('off')
    if np.size(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img,cmap='gray',vmin=0,vmax=255)  
    plt.show()



# a. Read in the image 'aria.jpg'
#

Ar      = cv2.imread('aria.jpg')


##########



# b. Shift the image up by 50 pixels
#

(row,clm,_)     = Ar.shape
Mt      = np.float32([[1,0,0],[0,1,-50]])   #  MUST use np.float32, else error !!
Ar      = cv2.warpAffine(Ar,Mt,(row,clm))

plt.figure()
cv2plt(Ar)


##########


# c. Rotate the image by 45 degrees, clockwise with
#    respect to the image center
#

Mr      = cv2.getRotationMatrix2D((clm/2,row/2),-45,1)
Ar      = cv2.warpAffine(Ar,Mr,(row,clm))

plt.figure()
cv2plt(Ar)


##########


# d. Create a mask that consists only of the retinal region.
#    (Refer to wks2_2_d.jpg for the output)
#

(_,mask)    = cv2.threshold(Ar[:,:,2],18,255,cv2.THRESH_BINARY)

plt.figure()
cv2plt(mask)


##########


# e. Read in image 'neu.jpg' and create a mask that segments
#    all the elements in the image (Refer to wks2_2_e.jpg for the output)


neu         = cv2.imread('neu.jpg')
neug        = cv2.cvtColor(neu, cv2.COLOR_BGR2GRAY)
neug        = 255-neug

(_,ntr)     = cv2.threshold(neug,31,255,cv2.THRESH_BINARY)

plt.figure()
cv2plt(ntr)


