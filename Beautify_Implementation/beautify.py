from imutils import face_utils
import numpy as np
import dlib
import cv2
import sys
shape_predictor= "shape_predictor_68_face_landmarks.dat" 
img = cv2.imread('Face.jpg')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)
# loop over the face detections
a=[]
for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    a=list(face_utils.FACIAL_LANDMARKS_IDXS.items())

def apply_lipstick(img)  :    
    if(len(a)):
        name, (i, j)=a[0]
        points= []
        
        for (x, y) in shape[i:j]:
            points.append([x,y])
        points= np.reshape(points, (-1, 1, 2))    
        cv2.fillPoly(img, [points], (255, 153, 255), 8)        
    
    return img

def view(img):
    cv2.imshow('Lipstick', img)
    cv2.waitKey(0) 

view(apply_lipstick(img))