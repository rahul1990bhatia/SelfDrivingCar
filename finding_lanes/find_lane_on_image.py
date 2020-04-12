
import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_function(input_image):
    gray = cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height,length = image.shape[0], image.shape[1]
    #print(height,length)
    polygon = np.array([[(200,height),(1100,height),(600,300)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygon,255)
    masked_image = cv2.bitwise_and(image,mask)
    #plt.imshow(masked_image)
    #plt.show()
    return masked_image

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if ( lines is not None ):
        for line in lines:
        #for x1,y1,x2,y2 in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1) ,(x2,y2) ,(255,0,0), 10)
    return line_image

def make_coordinates(image,line_parameters):
    slope,intercept = line_parameters
    print(slope,intercept)
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    #print(np.array([x1, y1, x2, y2]))
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        #print(x1,x2,y1,y2)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
#        print(parameters)
        slope = parameters[0]
        intercept = parameters[1]
        if slope > 0 :
            right_fit.append((slope,intercept))
        else:
            left_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)
    #print([left_line,right_line])
    return np.array([left_line,right_line])


image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny = canny_function(lane_image)
masked_image = region_of_interest(canny)
minLineLength = 40
maxLineGap = 5
lines = cv2.HoughLinesP(masked_image,2,np.pi/180,100,np.array([]),minLineLength=minLineLength,maxLineGap=maxLineGap)
#average_slope_intercept(lane_image,lines)
average_line = average_slope_intercept(lane_image,lines)
line_image = display_lines(lane_image,average_line)
final_image = cv2.addWeighted(lane_image,0.8,line_image,1.0,1)
cv2.imshow("final_image",final_image)
#plt.show()
cv2.waitKey(0)
