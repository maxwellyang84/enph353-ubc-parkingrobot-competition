
# Python program for Detection of a  
# specific color(blue here) using OpenCV with Python 
import cv2 
import numpy as np  
import imutils
import keras 
from keras.models import load_model
from keras import layers
from keras import models
from keras import optimizers

from keras.utils import plot_model
from keras import backend
from matplotlib import pyplot as plt

MIN_ASPECT_RATIO = 0.45
  
# Webcamera no 0 is used to capture the frames 

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

# This drives the program into an infinite loop. 

    # Captures the live stream frame-by-frame 
frame = cv2.imread('./incorrectly_interpreted/6244.png')
# frame = cv2.medianBlur(frame,5)
#frame = frame[750:,0:1279]
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

lower_grey = np.array([0,0,93]) 
upper_grey = np.array([0,0,210])

mask = cv2.inRange(hsv, lower_grey, upper_grey)

cv2.imshow('maskkkk', mask)
cv2.imshow("Mask", mask)

thresh = cv2.threshold(mask, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)
 
# find contours in thresholded image, then grab the largest
# one
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)  
cnts = imutils.grab_contours(cnts)
cnts = [c for c in cnts if cv2.contourArea(c) > 1000]

c = cnts[-1]
c2 = cnts[-2]
if cv2.contourArea(c) > cv2.contourArea(c2):
	c = cnts[-2]
	c2 = cnts[-1]
# cv2.drawContours(frame, c2,-1, (0,255,255), 3)
# cv2.drawContours(frame, c,-1, (0,255,255), 3)

cv2.imshow("SUP", frame)
# determine the most extreme points along the contour
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

extLeft2 = tuple(c2[c2[:, :, 0].argmin()][0])
extRight2 = tuple(c2[c2[:, :, 0].argmax()][0])
extTop2 = tuple(c2[c2[:, :, 1].argmin()][0])
extBot2 = tuple(c2[c2[:, :, 1].argmax()][0])

cv2.imshow("MM",frame)

x,y = extTop2
x2,y2 = extTop
x3,y3 = extRight
x4,y4 = extLeft2 

# cv2.circle(frame, extLeft, 8, (0, 0, 255), -1)
# cv2.circle(frame, extRight, 8, (0, 255, 0), -1)
# cv2.circle(frame, extTop, 8, (255, 0, 0), -1)
# cv2.circle(frame, extBot, 8, (255, 255, 0), -1)

# cv2.circle(frame, extLeft2, 8, (0, 0, 255), -1)
# cv2.circle(frame, extRight2, 8, (0, 255, 0), -1)
# cv2.circle(frame, extTop2, 8, (255, 0, 0), -1)
# cv2.circle(frame, extBot2, 8, (255, 255, 0), -1)

cv2.imshow("mm", frame)
cropped = frame[y+50: y2+10, x4:x3]

cv2.imshow("again", cropped)
cv2.imshow("MMFS", frame)

height, width, channels = cropped.shape

hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV) 

lower_gray = np.array([0, 2, 0], np.uint8)
upper_gray = np.array([255, 255, 255], np.uint8)
mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
img_res = cv2.bitwise_and(cropped, cropped, mask = mask_gray)


thresh = cv2.threshold(mask_gray, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)
 
# find contours in thresholded image, then grab the largest
# one
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)  
cnts = imutils.grab_contours(cnts)

cv2.imshow("mask_gray", mask_gray)
cv2.imshow("img_res", img_res)

# mask_gray = cv2.bitwise_not(mask_gray)
# ret, thresh = cv2.threshold(mask_gray, 200, 255, 0)
# __, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = [c for c in cnts if cv2.contourArea(c) > 100]
#cv2.drawContours(cropped, contours,-1, (0,255,255), 3)

c = contours[0]

# determine the most extreme points along the contour
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

# cv2.circle(cropped, extLeft, 8, (0, 0, 255), -1)
# cv2.circle(cropped, extRight, 8, (0, 255, 0), -1)
# cv2.circle(cropped, extTop, 8, (255, 0, 0), -1)
# cv2.circle(cropped, extBot, 8, (255, 255, 0), -1)

x,y = extRight
x2,y2 = extLeft
cv2.imshow("SUP", cropped)

#cropped = four_point_transform(cropped, np.array([(0,0),(width,0),extRight, (x2, y)], dtype="float32"))
cv2.imshow("gray", cropped)

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

mask = cv2.bitwise_not(mask)
cv2.imshow("bitwsie_not", mask)
lower_black = np.array([0,0,0])
upper_black = np.array([180,255,60])
imgThreshold = cv2.inRange(hsv, lower_black, upper_black)

cv2.imshow('lol', imgThreshold)

 
# img = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
# th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY,11,2)
# cv2.imshow("SDF", th3)


ret, thresh = cv2.threshold(mask, 200, 255, 0)
__, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(cv2.contourArea(contours[0]))
contours = [c for c in contours if cv2.contourArea(c) > 50 and cv2.contourArea(c) < 5000]

gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

cv2.drawContours(cropped, contours,-1, (0,255,255), 3)

for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    aspect_ratio = float(h)/w
    print(aspect_ratio)
    if aspect_ratio < MIN_ASPECT_RATIO:
        
        cv2.imwrite(str(x+y+w/2) + ".png", gray[y:y+h, x: x+int(w/2)])
        cv2.imwrite(str(x+y+w) + ".png", gray[y:y+h, x+int(w/2):x+w])
    #cv2.rectangle(th3,(x-5,y-5),(x+w+5,y+h+5),(0,255,0),2)
    else:
        cv2.imwrite(str(x+y) + ".png", gray[y:y+h,x:x+w])

cv2.imshow("Hello", mask)

# imgThreshold = imgThreshold[50:, :]
cv2.imshow("SAA", imgThreshold)
ret, thresh = cv2.threshold(imgThreshold, 200, 255, 0)
__,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = [c for c in contours if cv2.contourArea(c) > 100 and cv2.contourArea(c) < 5000]

imgThreshold = cv2.bitwise_not(imgThreshold)
cv2.drawContours(cropped, contours,-1, (0,255,255), 3)
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    #cv2.rectangle(th3,(x-5,y-5),(x+w+5,y+h+5),(0,255,0),2)
    cv2.imwrite("./cropped_letters/" + str(x+y) + ".png", gray[y:y+h,x:x+w])

# ret, thresh = cv2.threshold(th3, 70, 255, 0)
# __,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours = [c for c in contours if cv2.contourArea(c) > 100]
# cv2.drawContours(th3, contours,-1, (0,255,255), 3)
gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
cv2.imshow("grayss", gray)
cv2.imshow("final", imgThreshold)
cv2.imshow("contours", cropped)


# cv2.drawContours(frame, [c], -1, (0, 255, 255), 2)
# cv2.circle(frame, extLeft, 8, (0, 0, 255), -1)
# cv2.circle(frame, extRight, 8, (0, 255, 0), -1)
# cv2.circle(frame, extTop, 8, (255, 0, 0), -1)
# cv2.circle(frame, extBot, 8, (255, 255, 0), -1)

# cv2.drawContours(frame, [c2], -1, (0, 255, 255), 2)
# cv2.circle(frame, extLeft2, 8, (0, 0, 255), -1)
# cv2.circle(frame, extRight2, 8, (0, 255, 0), -1)
# cv2.circle(frame, extTop2, 8, (255, 0, 0), -1)
# cv2.circle(frame, extBot2, 8, (255, 255, 0), -1)



# cv2.imshow("Sup", frame)

image = cv2.imread('./112.png')
print(image.shape)
img = cv2.resize(image, (100, 100))
cv2.imshow("big", img)
image = cv2.resize(image,(64,64))
cv2.imshow("IMAGE", image)
model = load_model('number_neural_network3.h5')
img_aug = np.expand_dims(image, axis=0)
print(img_aug.shape)
y_predict = model.predict(img_aug)[0]
print(y_predict)
plt.imshow(image)  
caption = (str(y_predict))
plt.text(0.5, 0.5, caption, 
        color='orange', fontsize = 16,
        horizontalalignment='left', verticalalignment='bottom')



# This displays the frame, mask  
# and res which we created in 3 separate windows. 
k = cv2.waitKey(5) & 0xFF

while(1):
    pass
# Destroys all of the HighGUI windows. 
