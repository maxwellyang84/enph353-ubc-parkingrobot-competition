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

class license_plate_processor:

    def __init__(self, license_plate_pub):
        self.license_plate_image = None
        self.location_image = None
        self.license_plate_model = load_model('prototype.h5')
        self.license_plate_pub = license_plate_pub
        self.character_map = self.init_character_map()

    def init_character_map(self):
        self.character_map = {}
        for i in range(10):
            self.character_map[i] = i
        for i in range(10, 36):
            self.character_map[i] = str(chr(i+55))
        return self.character_map
        
    def image_cropper(self, image):
        image = image[750:,0:1279]
        # Converts images from BGR to HSV 
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
        
        #Range of blue
        lower_red = np.array([110,50,50]) 
        upper_red = np.array([130,255,255]) 
        
        # Here we are defining range of bluecolor in HSV 
        # This creates a mask of blue coloured  
        # objects found in the image. 
        mask = cv2.inRange(hsv, lower_red, upper_red) 
        
        # The bitwise and of the image and mask is done so  
        # that only the blue coloured objects are highlighted  
        # and stored in res 
        res = cv2.bitwise_and(image,image, mask= mask)  

        thresh = cv2.threshold(mask, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # find contours in thresholded image, then grab the largest
        # one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)  
        cnts = imutils.grab_contours(cnts)
        cnts = [c for c in cnts if cv2.contourArea(c) > 1000] #filter out small contours
        left_blue_contour = cnts[0]
        right_blue_contour = cnts[1]

        # determine the most extreme points along the contour
        extLeft = tuple(left_blue_contour[left_blue_contour[:, :, 0].argmin()][0])
        extRight = tuple(left_blue_contour[left_blue_contour[:, :, 0].argmax()][0])
        extTop = tuple(left_blue_contour[left_blue_contour[:, :, 1].argmin()][0])
        extBot = tuple(left_blue_contour[left_blue_contour[:, :, 1].argmax()][0])

        extLeft2 = tuple(right_blue_contour[right_blue_contour[:, :, 0].argmin()][0])
        extRight2 = tuple(right_blue_contour[right_blue_contour[:, :, 0].argmax()][0])
        extTop2 = tuple(right_blue_contour[right_blue_contour[:, :, 1].argmin()][0])
        extBot2 = tuple(right_blue_contour[right_blue_contour[:, :, 1].argmax()][0])

        x,y = extTop2
        x2,y2 = extBot2
        x3,y3 = extRight
        x4,y4 = extLeft2 

        cropped = image[y+20: y2+ 20, x3:x4] #get license_plate image

        #perspective transform for license_plate image
        height, width, channels = cropped.shape

        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV) 

        #creates a gray range to get bottom two points of license plate
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

        contours = [c for c in cnts if cv2.contourArea(c) > 100]

        c = contours[0]

        # determine the most extreme points along the contour
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        #creates perspective transformed license plate
        cropped = license_plate_processor.four_point_transform(cropped, np.array([(0,0),(width-30,0),extRight, extBot]))
        return cropped
    
    def split_characters(self, cropped):
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

        mask = cv2.bitwise_not(mask)

        lower_black = np.array([0,0,0])
        upper_black = np.array([180,255,30])
        imgThreshold = cv2.inRange(hsv, lower_black, upper_black)

        ret, thresh = cv2.threshold(imgThreshold, 200, 255, 0)
        __,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 500]

        plate_characters = []

        imgThreshold = cv2.bitwise_not(imgThreshold)
        cv2.drawContours(cropped, contours,-1, (0,255,255), 3)
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            plate_characters.append(imgThreshold[y:y+h, x-5, x+w+5])

        ret, thresh = cv2.threshold(mask, 200, 255, 0)
        __, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 500 and cv2.contourArea(c) < 12000]

        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            plate_characters.append(mask[y:y+h, x-5, x+w+5])

        
        return plate_characters

    def neural_network(self, plate_characters):
        plate_string = ''
        for index, characters in enumerate(plate_characters):
            if(index == 2):
                plate_string = plate_string + "_"
            img_aug = np.expand_dims(characters, axis=0)
            y_predict = self.license_plate_model.predict(img_aug)[0]
            order = [i for i, j in enumerate(y_predict) if j == 1]
            plate_string = plate_string + order[0]
        
        return plate_string
    
    def publish_license_plates(self, plate_string):
        self.license_plate_pub.publish(plate_string)

    def license_plate_model_process(self, image):
        license_plate_image = self.image_cropper(image)
        plate_characters = self.split_characters(license_plate_image)
        plate_string = self.neural_network(plate_characters)
        self.publish_license_plates(plate_string)
    
    def location_model_process(self):
        pass

    @staticmethod
    def four_point_transform(image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = license_plate_processor.order_points(pts)
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

    @staticmethod
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






# Want this to process to crop the license plate image using contours, then use contours to crop the license plate out of that
# image, then you want to apply the learning model onto the license plate and for each index, create a map associated with each
# character
# 
# 340-719 , 0-1279