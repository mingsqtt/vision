#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import cv2
import numpy as np
import os

import matplotlib.pyplot as plt

def cv2plt(img):
    plt.axis('off')
    if np.size(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img,cmap='gray',vmin=0,vmax=255)
    plt.show()


os.chdir("/Users/liming/projects/vision/Day3/3_3")
##########
    

# a. create a function that receives an image and returns bounding boxes and their
#       corresponding class labels
#
#       the function should have the below signature:
#
#       def yoloV3Detect(img,scFactor=1/255,nrMean=(0,0,0),RBSwap=True,
#                        scoreThres=0.5,nmsThres=0.4)
#           ...
#
#           return [fboxes,fclasses]
#
#
#   Try to perform object detections on image 'sr1.jpg', 'sr2.jpg' and etc.
    
def getOutputLayers(net):
    layers = net.getLayerNames()
    outLayers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return outLayers




def yoloV3Detect(img,scFactor=1/255,nrMean=(0,0,0),RBSwap=True,confidence_level=0.5,scoreThres=0.5,nmsThres=0.4):
    yoloconfig = 'yolov3.cfg'
    yoloweights= 'yolov3.weights'
    net = cv2.dnn.readNet(yoloweights,yoloconfig)
    
    blob = cv2.dnn.blobFromImage(image=img,
                                 scalefactor=scFactor,
                                 size=(416, 416),
                                 mean=nrMean,
                                 swapRB=RBSwap,
                                 crop=False)
    
    imgHeight = img.shape[0]
    imgWidth = img.shape[1]
    
    net.setInput(blob)
    outLyrs = getOutputLayers(net)
    preds = net.forward(outLyrs)
    
    classId = []
    confidences = []
    boxes = []
    
    for scale in preds:
        for pred in scale:
            scores = pred[5:]
            clss = np.argmax(scores)
            confidence = scores[clss]
            
            if confidence > confidence_level:
                xc = int(pred[0]*imgWidth)
                yc = int(pred[1]*imgHeight)
                w = int(pred[2]*imgWidth)
                h = int(pred[3]*imgHeight)
                x = xc - w/2
                y = yc - h/2
                
                classId.append(clss)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                
    selected = cv2.dnn.NMSBoxes(bboxes=boxes,
                                            scores=confidences,
                                            score_threshold=scoreThres,
                                            nms_threshold=nmsThres)

    if (len(selected) > 0):
        selected = selected.flatten()
        return([
            [boxes[i] for i in selected], 
            [classId[i] for i in selected]])
    else:
        return([])




##########


# b. Create a function that receives an image, fboxes, fclasses and classes
#       and produce the output that looks like 'wks3_3_b.jpg'
#
#   The name of the function should be 'pltDetect'

def drawBoundingBoxes(img,fboxes,fclasses,classes):
    img_copy = img.copy()
                
    colorset = np.random.uniform(0,255,size=(len(classes),3))
                
    for j in range(len(fboxes)):
        box = fboxes[j]
        color = colorset[fclasses[j]]
        txtlbl = str(classes[fclasses[j]])
        x = int(box[0])
        y = int(box[1])
        w = int(box[2])
        h = int(box[3])
        
        cv2.rectangle(img_copy,
                      (x,y),
                      (x+w,y+h),
                      color,
                      2)
        cv2.putText(img_copy,
                    txtlbl,
                    (x,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA)
    return(img_copy)



lbl_file = 'yolov3.txt'
classes = open(lbl_file).read().strip().split("\n")
for i in range(1, 7):
    img = cv2.imread("sr{}.jpg".format(i))
    info = yoloV3Detect(img,scoreThres=0.5,nmsThres=0.4)
    cv2plt(drawBoundingBoxes(img, info[0], info[1], classes))

img = cv2.imread("/Users/liming/projects/imgproc/data/container/container_h1.JPG")
info = yoloV3Detect(img,scoreThres=0.1, nmsThres=0.4, confidence_level=0.1)
if (len(info) > 0):
    print("{} objects found".format(len(info)))
    cv2plt(drawBoundingBoxes(img, info[0], info[1], classes))
else:
    print("No object detected")