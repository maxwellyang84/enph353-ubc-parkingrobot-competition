
# Python program for Detection of a  
# specific color(blue here) using OpenCV with Python 
import cv2 
import numpy as np  
import imutils
  
# Webcamera no 0 is used to capture the frames 

  
# This drives the program into an infinite loop. 

    # Captures the live stream frame-by-frame 
frame = cv2.imread('./image_test.png')
frame = cv2.medianBlur(frame,5)
frame = frame[750:,0:1279]
    # Converts images from BGR to HSV 
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
lower_red = np.array([110,50,50]) 
upper_red = np.array([130,255,255]) 
  
# Here we are defining range of bluecolor in HSV 
# This creates a mask of blue coloured  
# objects found in the frame. 
mask = cv2.inRange(hsv, lower_red, upper_red) 
  
# The bitwise and of the frame and mask is done so  
# that only the blue coloured objects are highlighted  
# and stored in res 
res = cv2.bitwise_and(frame,frame, mask= mask) 
cv2.imshow('frame',frame) 
cv2.imshow('mask',mask) 
cv2.imshow('res',res) 

# ret, thresh = cv2.threshold(mask, 200, 255, 0)
# im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = [c for c in contours if cv2.contourArea(c) > 1000]
# approx = []
# for cnt in contours:
#     epsilon = 0.01*cv2.arcLength(cnt,True)
#     approx.append(cv2.approxPolyDP(cnt,epsilon,True))

# cntx1 = cnt[0]
# cntx = cnt[1]
# contours[0]
# #convert coords to points
# pt1 = (cntx1[0][0],cntx1[0][1])
# pt2 = (cntx[0][0],cntx[0][1])

# #draw circles on coordinates
# cv2.circle(frame,pt1,5,(0,255,0),-1)
# cv2.circle(frame,pt2, 5, (0,255,0),-1)
thresh = cv2.threshold(mask, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)
 
# find contours in thresholded image, then grab the largest
# one
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)  
cnts = imutils.grab_contours(cnts)
cnts = [c for c in cnts if cv2.contourArea(c) > 1000]
c = cnts[0]
c2 = cnts[1]

# determine the most extreme points along the contour
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

extLeft2 = tuple(c2[c2[:, :, 0].argmin()][0])
extRight2 = tuple(c2[c2[:, :, 0].argmax()][0])
extTop2 = tuple(c2[c2[:, :, 1].argmin()][0])
extBot2 = tuple(c2[c2[:, :, 1].argmax()][0])

x,y = extTop2
x2,y2 = extBot2
x3,y3 = extRight
x4,y4 = extLeft2 

cropped = frame[y: y2+ 20, x3:x4]
hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV) 
lower_red = np.array([110,50,50]) 
upper_red = np.array([130,255,255]) 
  
# Here we are defining range of bluecolor in HSV 
# This creates a mask of blue coloured  
# objects found in the frame. 
mask = cv2.inRange(hsv, lower_red, upper_red) 
  
# The bitwise and of the frame and mask is done so  
# that only the blue coloured objects are highlighted  
# and stored in res 
res = cv2.bitwise_and(cropped,cropped, mask= mask) 
cv2.imshow('frame2',cropped) 
cv2.imshow('mask2',mask) 
cv2.imshow('res2',res) 
cv2.imshow("Comparison", cropped)





img = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

ret, thresh = cv2.threshold(mask, 200, 255, 0)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = [c for c in contours if cv2.contourArea(c) > 100]

for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(th3,(x-5,y-5),(x+w+5,y+h+5),(0,255,0),2)

cv2.imshow("Hello", th3)
cv2.imshow("K", th2)
cropped = cv2.GaussianBlur(cropped, (5, 5), 0)
cv2.imshow("Show",cropped)

cv2.drawContours(frame, [c], -1, (0, 255, 255), 2)
cv2.circle(frame, extLeft, 8, (0, 0, 255), -1)
cv2.circle(frame, extRight, 8, (0, 255, 0), -1)
cv2.circle(frame, extTop, 8, (255, 0, 0), -1)
cv2.circle(frame, extBot, 8, (255, 255, 0), -1)

cv2.imshow("One Contour", frame)

cv2.drawContours(frame, [c2], -1, (0, 255, 255), 2)
cv2.circle(frame, extLeft2, 8, (0, 0, 255), -1)
cv2.circle(frame, extRight2, 8, (0, 255, 0), -1)
cv2.circle(frame, extTop2, 8, (255, 0, 0), -1)
cv2.circle(frame, extBot2, 8, (255, 255, 0), -1)

print(type(extLeft2))
# cv2.drawContours(frame,approx,-1,(0,255,0),3)
# for cnt in contours:
#     x,y,w,h = cv2.boundingRect(cnt)
#     cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

 
print(str(y) + " " + str(y2) + " " + str(x3)+ " " + str(x4))
cv2.imshow("Sup", frame)



# This displays the frame, mask  
# and res which we created in 3 separate windows. 
k = cv2.waitKey(5) & 0xFF

while(1):
    pass
# Destroys all of the HighGUI windows. 
