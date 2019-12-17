#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# Workshop 2.1
# ------------


import cv2



# a. Read in the 'theWave.jpg' image, store it in a variable named 'wave'
#

wave    = cv2.imread('theWave.jpg')


##########


# b. Check the dimensions of the image and its data type.
#    Print the output in the below fashion:
#
#       "The height of the image is XXX pixels."
#       "The width of the image is XXX pixels."
#       "The image has XXX channels."
#       "The data type is XXX."


print("The height of the image is %d pixels." % (wave.shape[0]))
print("The width of the image is %d pixels." % (wave.shape[1]))
print("The image has %d channels." % (wave.shape[2]))
print("The data type is %s ." % (wave.dtype))



##########


# c. Display the 'wave' using opencv function
#

cv2.imshow('wave',wave)
cv2.waitKey(0)
cv2.destroyAllWindows()
                                    # the below is needed to close the image window
                                    # solution from https://blog.csdn.net/GAN_player/article/details/75098226
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)


##########


# d. Create a function 'cv2plt' to display a BGR image or gray image 
#    without using cv2.imshow() function.
#    The function needs only an input argument 'img'. Set axis to 'off'.


import matplotlib.pyplot as plt

def cv2plt(img):
    plt.axis('off')
    if np.size(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img,cmap='gray',vmin=0,vmax=255)  # To avoid auto-normalization by pyplot                                                     # Must set the vmin and vmax
    plt.show()


##########



# e. Create a 2D numpy zero array that shares the same height, width and data type
#    as 'wave'. Name the array as 'zfill'

import numpy as np

zfill   = np.zeros((wave.shape[0],wave.shape[1]),dtype=wave.dtype)


##########



# f. Create the the output wks2_1_f.jpg.
#

waveg   = cv2.merge((zfill,wave[:,:,1],zfill))
plt.figure()
cv2plt(waveg)


##########


# g. Create the output wks2_1_g.jpg.
#

wavep   = cv2.copyMakeBorder(wave,0,0,50,50,cv2.BORDER_REFLECT)
plt.figure()
cv2plt(wavep)


##########


# h. Create the output wks2_1_t.jpg.
#
#   The colour and the thickness for text and rectangle
#   (63,63,63), 2
#
#   Font size: 1.1

wavet   = wave.copy()
cv2.rectangle(wavet,
              (510,32),
              (663,75),
              (63,63,63),
              2)
cv2.putText(wavet,
            'Hokusai',
            (520,65),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (63,63,63),
            2,
            cv2.LINE_AA)
plt.figure()
cv2plt(wavet)




# the code to write image
# cv2.imwrite('wks2_1_t.jpg',wavet)




