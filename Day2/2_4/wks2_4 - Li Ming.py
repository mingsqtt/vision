#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# Workshop 2.4
# ------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir("/Users/liming/projects/vision/Day2/2_4")

def cv2plt(img):
    plt.axis('off')
    if np.size(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img,cmap='gray',vmin=0,vmax=255)
    plt.show()
    
    
    
# a. Read in the image 'ajbp.jpg'. Name the array as 'ajbp'
#
ajbp = cv2.imread('ajbp.jpg')
cv2plt(ajbp)



##########


# b. Write a function that can detect faces, eyes and smiles with the below signature:
#
#       def faceDetection(img,scaleFct=1.3,faceNbr=5,eyeNbr=None,smileNbr=None):
#        
#           .....
#        
#           return [img,faces,eyes,smiles]
#
#    
#    The above function by default detect faces in an image. 
#    faceNbr, eyeNbr, smileNbr denote the minNeighbors for face, eye, and smile
#    respectively. When no value is specified for eyeNbr or smileNbr, no detection 
#    will be done for eye or smile.

#    The returned 'img' shows the detected faces, eyes or smiles.
#    See 'wks2_4_c.jpg' for example.

#    The returned 'faces', 'eyes', 'smiles' contain the x, y, w, h
#    of each identified boxes

#    The colour of the box for faces: (255,255,255)
#    The colour of the box for eyes: (191,191,191)
#    The colour of the box for smiles: (127,127,127)
#    The line thickness is 2 for all types of boxes

def faceDetection(img,scaleFct=1.3,faceNbr=5,eyeNbr=None,smileNbr=None):
    img = ajbp
    img_copy = img.copy()
    img_g= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    fce_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_model = cv2.CascadeClassifier('haarcascade_eye.xml')
    smile_model = cv2.CascadeClassifier('haarcascade_smile.xml')
    faces = fce_model.detectMultiScale(img_g,
                               scaleFactor=scaleFct,
                               minNeighbors=faceNbr)
    
    all_eyes = list()
    all_smiles = list()
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img_copy, 
                      (x, y),
                      (x+w, y+h),
                      (255,255,255), 2)
    
        face_img_copy = img_copy[y:y+h,x:x+w]
        face_img_g = img_g[y:y+h,x:x+w]
        
        if eyeNbr is not None:
            eyes = eye_model.detectMultiScale(face_img_g,
                                          scaleFactor=scaleFct,
                                          minNeighbors=eyeNbr)
            for (px,py,pw,ph) in eyes:
                cv2.rectangle(face_img_copy,(px,py),(px+pw,py+ph),(191,191,191),2)
            all_eyes = eyes
            
            
        if smileNbr is not None:
            smiles = smile_model.detectMultiScale(face_img_g,
                                          scaleFactor=scaleFct,
                                          minNeighbors=smileNbr)
            for (px,py,pw,ph) in smiles:
                cv2.rectangle(face_img_copy,(px,py),(px+pw,py+ph),(127,127,127),2)
            all_smiles = smiles
    
    return [img_copy,faces,all_eyes,all_smiles]
##########


# c. Perform the detection of faces, eyes and faces on
#    ajbp.jpg
    
cv2plt(faceDetection(ajbp, eyeNbr=20, smileNbr=30)[0])

