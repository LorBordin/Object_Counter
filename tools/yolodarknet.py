# import the necessary packages
import numpy as np
import cv2

def get_img_size_from_cgf(cfgPath):
    f = open(cfgPath, "r")
    for line in f.readlines():
        if "width" in line:
            imgSize = int(line.split("=")[-1].strip())
            break
    f.close()
    return imgSize


class YoloDarknet:
    
    def __init__(self, labelsPath,  weightPath, configPath, 
                 confidence, threshold, n_classes, use_cuda=True):
        
        self.labels = open(labelsPath).read().strip().split("\n")
        self.confidence = confidence
        self.threshold = threshold
        self.n_classes = n_classes
        self.img_size = get_img_size_from_cgf(configPath)
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightPath)
        
        if use_cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        # get the output layers for inference
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.ln = ln
    
    def predict(self, img):
        '''
        make inference and returns 3 arrays:
        - boxes: b_boxes in format (Xt, Yt, W, H)
        - classIDs: class the object belongs to
        - confidences: detection confidences
        '''
        
        # grab img dimensions
        H, W = img.shape[:2]
        
        # load input frame and preprocess for YOLOs
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, 
                                     (self.img_size, self.img_size), 
                                     swapRB=True, crop=False)
        
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)
        boxes, confidences, classIDs = [], [], []
        
        # loop over each of the layer outputs
        for output in layerOutputs:
            
            # filter out weak predictions
            if self.n_classes == 1:     #persons
                output = output[output[:,5]>self.confidence]
            
            elif self.n_classes == 2:   # weapons
                output = output[(output[:,5]>self.confidence) |
                                (output[:,6]>self.confidence)]
            
            elif self.n_classes == 3:   # faces
                output = output[(output[:,5]>self.confidence) |
                                (output[:,6]>self.confidence) |
                                (output[:,7]>self.confidence)]
            
            else:                       # n classes
                output = output[(output[:,5:]>self.confidence).any(axis=1)]
            
            # loop over each of the detections
            for detection in output:
                scores = detection[5:]          # probabilities
                classID = np.argmax(scores)     # classes
                confidence = scores[classID]
                
                # get center coords, width and height of the bboxes
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                #convert into top-left, bottom right coords
                Xt = int(centerX - (width / 2))
                Yt = int(centerY - (height / 2))
                
                # update boxes, confidences and classIDs
                boxes.append([Xt, Yt, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
        
        # apply non-maxima suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 
                                self.confidence, self.threshold)
            
        boxes = np.array(boxes)
        classIDs = np.array(classIDs)
        confidences = np.array(confidences)
    
        # return bboxes coords and classId
        if len(idxs) > 0:
            return boxes[idxs.flatten()], classIDs[idxs.flatten()], confidences[idxs.flatten()]
        else:
            return np.empty((0,4), dtype="int"), np.empty((0,1), dtype="int"), np.empty((0,1))
