#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def cv2plt(img):
    plt.axis('off')
    if np.size(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img,cmap='gray',vmin=0,vmax=255)
    plt.show()
    
os.chdir("/Users/liming/projects/vision/Day3/3_2")


##########
    

# a. create a function that receives an image and returns the top 5 prediction
#       the function should have the below signature:
#
#       def leNetPredict(img,scFactor=1,nrMean=(104,117,123),RBSwap=True)
#           ...
#
#           return [Top5Classes,Top5Probability]
#
#
#   Read in the image '380.jpg' and use leNet to make top 5 prediction
    


def leNetPredict(img,scFactor=1,nrMean=(104,117,123),RBSwap=True):
    lbl_file = 'synset_words.txt'
    labels = open(lbl_file).read().strip().split("\n")
    classes = [r[r.find(" ") + 1:].split(",")[0] for r in labels]
    
    prototxt = 'bvlc_googlenet.prototxt'
    caffemodel= 'bvlc_googlenet.caffemodel'
    net = cv2.dnn.readNetFromCaffe(prototxt,caffemodel)
    
    blob = cv2.dnn.blobFromImage(image=img,
                                 scalefactor=scFactor,
                                 size=(224, 224),
                                 mean=nrMean,
                                 swapRB=RBSwap,
                                 crop=True)
    
    net.setInput(blob)
    preds = net.forward()
    pr_idx = np.argsort(preds[0])
    pr_idx = pr_idx[::-1]
    
    top5class = [classes[pr_idx[idx]] for idx in range(0, 5)]
    top5prob = [preds[0,pr_idx[idx]] for idx in range(0, 5)]
    
    return([top5class,top5prob])
    

img = cv2.imread("380.jpg")    
leNetPredict(img)

img1 = cv2.imread("mask.jpg")
leNetPredict(img1)

img2 = cv2.imread("wheelchair.jpg")
leNetPredict(img2)

img3 = cv2.imread("desk.jpg")
leNetPredict(img3)
i
img4 = cv2.imread("truck.png")
leNetPredict(img4)


##########



# b. Create a function that receives an image, Top5Classes and Top5Probability
#       and produce the output 'wks3_2_b.jpg'
#
#   The name of the function should be 'pltPredict'

def appendText(img,top5class,top5prob):
    img_copy = img.copy()
    for line in range(0, 5):
        cv2.putText(img_copy,
                ("{} ({}%)" if line == 0 else "{}").format(top5class[line], np.round(top5prob[line] * 100.0, 2)),
                (10,20+line*20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,255,255),
                1,
                cv2.LINE_AA)
    return(img_copy)
    
pred = leNetPredict(img1)
cv2plt(appendText(img1, pred[0], pred[1]))


                      