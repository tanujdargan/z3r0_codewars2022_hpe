import cv2
import numpy as np
import dlib
import math
from math import hypot

font= cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks1 = predictor(gray, face)
        left_point1 = (landmarks1.part(36).x, landmarks1.part(36).y)
        right_point1 = (landmarks1.part(39).x, landmarks1.part(39).y)
        center_top1 = midpoint(landmarks1.part(37), landmarks1.part(38))
        center_bottom1 = midpoint(landmarks1.part(41), landmarks1.part(40))
        

        #hor_line1 = cv2.line(frame, left_point1, right_point1, (0, 255, 0), 2)
        #ver_line1 = cv2.line(frame, center_top1, center_bottom1, (0, 255, 0), 2)
        
        landmarks2 = predictor(gray, face)
        left_point2 = (landmarks2.part(42).x, landmarks2.part(42).y)
        right_point2 = (landmarks2.part(45).x, landmarks2.part(45).y)
        center_top2 = midpoint(landmarks2.part(43), landmarks2.part(44))
        center_bottom2 = midpoint(landmarks2.part(47), landmarks2.part(46))

        #hor_line2 = cv2.line(frame, left_point2, right_point2, (0, 255, 0), 2)
        #ver_line2 = cv2.line(frame, center_top2, center_bottom2, (0, 255, 0), 2)
        
        hor_line_length = hypot((left_point1[0] - right_point1[0]), (left_point1[1] - right_point1[1]))
    
        ver_line_length = hypot((center_top1[0] - center_bottom1[0]), (center_top1[1] - center_bottom1[1]))
        
        ratio= hor_line_length//ver_line_length
        
        if ratio > 5.5:
            cv2.putText(frame, "DROWSINESS ALERT!!!", (275,30), font, 1, (0, 0, 255),2)
            
        left_eye_region = np.array([(landmarks1.part(36).x, landmarks1.part(36).y),
                                    (landmarks1.part(37).x, landmarks1.part(37).y),
                                    (landmarks1.part(38).x, landmarks1.part(38).y),
                                    (landmarks1.part(39).x, landmarks1.part(39).y),
                                    (landmarks1.part(40).x, landmarks1.part(40).y),
                                    (landmarks1.part(41).x, landmarks1.part(41).y)], np.int32)
        
        cv2.polylines(frame, [left_eye_region], True, (0, 255, 0), 2)
        
        right_eye_region = np.array([(landmarks2.part(42).x, landmarks2.part(42).y),
                                     (landmarks2.part(43).x, landmarks2.part(43).y),
                                     (landmarks2.part(44).x, landmarks2.part(44).y),
                                     (landmarks2.part(45).x, landmarks2.part(45).y),
                                     (landmarks2.part(46).x, landmarks2.part(46).y),
                                     (landmarks2.part(47).x, landmarks2.part(47).y)], np.int32)
        
        cv2.polylines(frame, [right_eye_region], True, (0, 255, 0), 2)
        
        min1_x = np.min(left_eye_region[:, 0])
        max1_x = np.max(left_eye_region[:, 0])
        min1_y = np.min(left_eye_region[:, 1])
        max1_y = np.max(left_eye_region[:, 1])
        
        eye1 = frame[min1_y: max1_y, min1_x: max1_x]
        
        min2_x = np.min(right_eye_region[:, 0])
        max2_x = np.max(right_eye_region[:, 0])
        min2_y = np.min(right_eye_region[:, 1])
        max2_y = np.max(right_eye_region[:, 1])
        
        eye2 = frame[min2_y: max2_y, min2_x: max2_x]
        
        #eye1 = cv2.resize(eye1, None, fx=5, fy=5)
        
        #eye2 = cv2.resize(eye2, None, fx=5, fy=5)
        
        #cv2.imshow("Eye1", eye1)
        
        #cv2.imshow("Eye2", eye2)

        print(ratio)
        
        cv2.putText(frame, "EAR: " + str(ratio), (10,30), font, 1, (0, 0, 255), 2)
                       
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()