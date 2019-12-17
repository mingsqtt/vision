#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# Workshop 2.1
# ------------




import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

os.getcwd()
os.chdir("/Users/liming/projects/vision/Day2/2_1")


# a. Read in the 'theWave.jpg' image, store it in a variable named 'wave'
#
img_wave = cv.imread("theWave.jpg")




##########


# b. Check the dimensions of the image and its data type.
#    Print the output in the below fashion:
#
#       "The height of the image is XXX pixels."
#       "The width of the image is XXX pixels."
#       "The image has XXX channels."
#       "The data type is XXX."

print("The height of the image is {} pixels.".format(img_wave.shape[0]))
print("The width of the image is {} pixels.".format(img_wave.shape[1]))
print("The image has {} channels.".format(img_wave.shape[2]))
print("The data type is {}.".format(img_wave.dtype))



##########


# c. Display the 'wave' using opencv function
#
cv.imshow("my window", img_wave)
cv.waitKey(0)
cv.destroyAllWindows()

cv.waitKey(1)
cv.waitKey(1)
cv.waitKey(1)
cv.waitKey(1)




##########


# d. Create a function 'cv2plt' to display a BGR image or gray image 
#    without using cv2.imshow() function.
#    The function needs only an input argument 'img'. Set axis to 'off'.


def cv2plt(img):
    plt.axis('off')
    if (np.size(img.shape) == 3):
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.show()

cv2plt(img_wave)

##########



# e. Create a 2D numpy zero array that shares the same height, width and data type
#    as 'wave'. Name the array as 'zfill'

zfill = np.zeros(img_wave.shape[:2], dtype="uint8")


##########



# f. Create the the output wks2_1_f.jpg.
#


cv2plt(cv.merge((zfill, img_wave[:,:,1], zfill)))



##########


# g. Create the output wks2_1_g.jpg.
#

wavep   = cv.copyMakeBorder(img_wave,0,0,50,50,cv.BORDER_REFLECT)
plt.figure()
cv2plt(wavep)


##########


# h. Create the output wks2_1_t.jpg.
#
#   The colour and the thickness for text and rectangle
#   (63,63,63), 2
#
#   Font size: 1.1


wavet   = img_wave.copy()
cv.rectangle(wavet,
              (510,32),
              (663,75),
              (63,63,63),
              2)
cv.putText(wavet,
            'Hokusai',
            (520,65),
            cv.FONT_HERSHEY_SIMPLEX,
            1.1,
            (63,63,63),
            2,
            cv.LINE_AA)
plt.figure()
cv2plt(wavet)




