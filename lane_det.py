import cv2
import numpy as np
import matplotlib.pyplot as plt
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),0)
    canny = cv2.Canny(blur,50,150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    triangle = np.array([
        [(150, height), (1100,height), (550,250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(lines, image):
    line_image = np.zeros_like(image)
   # print(type(lines))
    if lines is not None:
        points = []
        for line in lines:
            x1,y1, x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
            points.append([x1, y1])
            points.append([x2, y2])
        points = np.array(points)
        #print(type(points))
        a,b,c,d  = points[0][0], points[0][1], points[1][0], points[1][1]
        e,f,g,h = points[2][0], points[2][1], points[3][0], points[3][1]
        pts = np.array([[a,b], [c,d], [g,h], [e,f]], np.int32)
        pts = pts.reshape((-1,1,2))
        #cv2.polylines(line_image,[pts],True,(0,255,255))
        cv2.fillPoly(line_image, [pts], (255,0,0), lineType=8, shift=0)
        #cv2.polylines(line_image,[[a,b], [c,d], [g,h], [e,f]],True,(0,255, 0), -1)
        #cv2.ellipse(line_image,(256,256),(100,50),0,0,180,255,-1)
    return line_image 


def make_coordinate(image, line_parameters):
        line_parameters = np.array(line_parameters)
    #print(line_parameters)
    #if not all(np.isfinite(line_parameters)):
     #   pass
    #else:
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1*(2.7/5))
        x1 = int((y1-intercept)/slope)
        x2 = int((y2-intercept)/slope)
        return np.array([x1,y1,x2,y2])


def average_slope_intercept(image, lines):
    left_fit = []
    global b
    right_fit = []
    #print(lines)
    for line in lines:
        x1,y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope<0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))        
    left_avg = np.average(left_fit, axis = 0)
    right_avg = np.average(right_fit, axis = 0)
    if not np.any(np.isnan(left_avg)):
        b = np.array(left_avg)
    if np.any(np.isnan(left_avg)):
        left_line = make_coordinate(image, b)
    else:
        left_line = make_coordinate(image, left_avg)
    
    
    right_line = make_coordinate(image, right_avg)
    return np.array([left_line, right_line])



cap = cv2.VideoCapture(r'C:\Users\PRAVEEN\Desktop\test2.mp4')
while(cap.isOpened()):
    _, frame =  cap.read()
    lane_image= np.copy(frame)
    canny_image = canny(lane_image)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength= 40, maxLineGap = 5)
    average_line = average_slope_intercept(lane_image, lines)
    line_image = display_lines(average_line, lane_image)
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    cv2.imshow('result',combo_image)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break;
cap.release()
cv2.destroyAllWindows()
