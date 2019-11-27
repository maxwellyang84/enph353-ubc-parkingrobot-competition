#!/usr/bin/env python

import cv2 
import numpy as np  
import imutils
import keras 
from keras.models import load_model
from keras import layers
from keras import models
from keras import optimizers

import tensorflow as tf

from keras.utils import plot_model
from keras import backend
from matplotlib import pyplot as plt
from tensorflow.python.keras.backend import set_session


import rospy
import sys
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from random import randint

MIN_ASPECT_RATIO = 0.51

config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6


class license_plate_processor:

    def __init__(self):
        self.license_plate_image = None
        self.location_image = None

        self.session = tf.Session(config=config)

        keras.backend.set_session(self.session)
        self.license_plate_number_model = load_model('testnumbernn2.h5')#'number_neural_network11_less_blur_no_rotation.h5')
        self.license_plate_number_model._make_predict_function()
        self.license_plate_letter_model = load_model('testletternnaddedletterstomostblurred.h5')#'letter_neural_network4.h5')
        self.license_plate_letter_model._make_predict_function()
        self.license_plate_location_model = load_model('location_model.h5')
        self.license_plate_location_model._make_predict_function()

        self.license_plate_number_model_backup = load_model("number_neural_network5_no_rotation.h5")
        self.license_plate_number_model_backup._make_predict_function()
        self.license_plate_letter_model_backup = load_model("testletternnaddedmoreletterstomostblurred6.h5")
        self.license_plate_letter_model_backup._make_predict_function()

        self.license_plate_pub = rospy.Publisher("/license_plate", String, queue_size=30)
       
        self.character_map = self.init_character_map()
        self.number_map = self.init_number_map()
        self.location_map = self.init_location_map()

        self.richards_mac = False


    def init_character_map(self):
        self.character_map = {}
        for i in range(0, 26):
            self.character_map[i] = str(chr(i+65))
        return self.character_map
    
    def init_number_map(self):
        self.number_map = {}
        for i in range(0,10):
            self.number_map[i] = str(i)
        return self.number_map
    
    def init_location_map(self):
        self.location_map = {}
        for i in range (1,9):
            self.location_map[i-1] = str(i)
        return self.location_map

    def get_contour_coords(self, contour):
        x,y,w,h = cv2.boundingRect(contour)
        return x
  
    def image_cropper(self, image):
        #image = image[:,0:600] #alter if needed
        # Converts images from BGR to HSV 
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
        
        lower_grey = np.array([0,0,93]) 
        upper_grey = np.array([0,0,210])

        mask = cv2.inRange(hsv, lower_grey, upper_grey)

        thresh = cv2.threshold(mask, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # find contours in thresholded image, then grab the largest
        # one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)  
        cnts = imutils.grab_contours(cnts)
        cnts.sort(key=self.get_contour_coords)
        cnts = [c for c in cnts if cv2.contourArea(c) > 1000] #filter out small contours
        # print(len(cnts))
        # print(cv2.contourArea(cnts[0]))
        # print(cv2.contourArea(cnts[1]))
        # print(cv2.contourArea(cnts[2]))
        # print(cv2.contourArea(cnts[-1]))
        # print(cv2.contourArea(cnts[-2]))
        bottom_white_contour = cnts[-1]
        top_white_contour = cnts[-2]
        if cv2.contourArea(bottom_white_contour) > cv2.contourArea(top_white_contour):
            bottom_white_contour = cnts[-2]
            top_white_contour = cnts[-1]
        #cv2.drawContours(image, cnts,-1, (0,255,255), 3)
        
        cv2.imshow("Plate to Process", image) #used to be S

        if not self.richards_mac:
            cv2.imshow("<MM", image)
        
       
       

        # determine the most extreme points along the contour
        extLeft = tuple(bottom_white_contour[bottom_white_contour[:, :, 0].argmin()][0])
        extRight = tuple(bottom_white_contour[bottom_white_contour[:, :, 0].argmax()][0])
        extTop = tuple(bottom_white_contour[bottom_white_contour[:, :, 1].argmin()][0])
        extBot = tuple(bottom_white_contour[bottom_white_contour[:, :, 1].argmax()][0])

        extLeft2 = tuple(top_white_contour[top_white_contour[:, :, 0].argmin()][0])
        extRight2 = tuple(top_white_contour[top_white_contour[:, :, 0].argmax()][0])
        extTop2 = tuple(top_white_contour[top_white_contour[:, :, 1].argmin()][0])
        extBot2 = tuple(top_white_contour[top_white_contour[:, :, 1].argmax()][0])

        x,y = extTop2
        x2,y2 = extTop
        x3,y3 = extRight
        x4,y4 = extLeft2 

       
        cropped = image[y+50: y2+10, x4:x3]

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

        x,y = extRight
        x2,y2 = extLeft

        #creates perspective transformed license plate
        #cropped = license_plate_processor.four_point_transform(cropped, np.array([(0,0),(width,0),extRight, (x2, y)]))   
        return cropped
    
    def split_characters(self, cropped):
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV) 
        lower_red = np.array([110,50,50]) 
        upper_red = np.array([130,255,255])

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) 
        
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
        upper_black = np.array([180,255,60])
        imgThreshold = cv2.inRange(hsv, lower_black, upper_black)

    
        
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        #cv2.drawContours(cropped, contours,-1, (0,255,255), 3)

        plate_characters = []
        if not self.richards_mac:
            cv2.imshow("gray", gray)
        

        ret, thresh = cv2.threshold(imgThreshold, 200, 255, 0)
        __,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print(len(contours))
        contours = [c for c in contours if cv2.contourArea(c) > 100 and cv2.contourArea(c) < 5000]
        #print(len(contours))
        contours.sort(key=self.get_contour_coords)

        imgThreshold = cv2.bitwise_not(imgThreshold)
        cv2.drawContours(cropped, contours,-1, (0,255,255), 3)

        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            #cv2.rectangle(th3,(x-5,y-5),(x+w+5,y+h+5),(0,255,0),2)
            plate_characters.append(gray[y:y+h,x:x+w])

        ret, thresh = cv2.threshold(mask, 200, 255, 0)
        __, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print(cv2.contourArea(contours[0]))
        contours = [c for c in contours if cv2.contourArea(c) > 79 and cv2.contourArea(c) < 5000] #used to be 50
        
        contours.sort(key=self.get_contour_coords)
        #print(len(contours))
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            aspect_ratio = float(h)/w
            #print(cv2.contourArea(cnt))
            print(aspect_ratio)
            if aspect_ratio < MIN_ASPECT_RATIO:
                plate_characters.append(gray[y:y+h, x: x+int(w/2)])
                plate_characters.append(gray[y:y+h, x+int(w/2):x+w])
            #cv2.rectangle(th3,(x-5,y-5),(x+w+5,y+h+5),(0,255,0),2)
            else:
                plate_characters.append(gray[y:y+h,x:x+w])
        
        count = 0
        for characters in plate_characters:
            #cv2.imwrite("./cnn_cropped_letters/" + str(randint(0,10000)) + ".png", characters)
            if not self.richards_mac:
                cv2.imshow(str(count), characters)
            count = count + 1
        
        cv2.drawContours(cropped, contours,-1, (0,255,255), 3)
        if not self.richards_mac:
            cv2.imshow("plates", cropped)
            cv2.imshow("License Plate", mask)

        # imgThreshold = imgThreshold[50:, :]
        if not self.richards_mac:
            cv2.imshow("Location", imgThreshold)
       
        # plate_characters.reverse()
        return plate_characters

    def neural_network(self, plate_characters):
        #print(len(plate_characters))
        if len(plate_characters) != 6:
            return "BAD"
        plate_string = ''
        for index, character in enumerate(plate_characters):
            if index == 0:
                continue
            character = cv2.cvtColor(character, cv2.COLOR_GRAY2BGR)
            #cv2.imwrite(str(randint(0,1000)) + ".png", character)
            if(index == 2):
                plate_string = plate_string + ","
            if not self.richards_mac:
                if index == 4 or index == 5:
                    character = cv2.resize(character,(64,64))
                    img_aug = np.expand_dims(character, axis=0)
                    y_predict = self.license_plate_number_model.predict(img_aug)[0]
                    order = [i for i, j in enumerate(y_predict) if j > 0.5]
                    #print(order)
                    if self.number_map[order[0]] == '4':
                        print(self.number_map[order[0]])
                        y_predict = self.license_plate_number_model_backup.predict(img_aug)[0]
                        order = [i for i, j in enumerate(y_predict) if j > 0.5]
                        print(self.number_map[order[0]])
                    plate_string = plate_string + str(self.number_map[order[0]])
                elif index == 1:
                    character = cv2.resize(character,(64,64))
                    img_aug = np.expand_dims(character, axis=0)
                    y_predict = self.license_plate_location_model.predict(img_aug)[0]
                    order = [i for i, j in enumerate(y_predict) if j > 0.5]
                    #print(order)
                    plate_string = plate_string + str(self.location_map[order[0]])
                else:
                    character = cv2.resize(character,(64,64))
                    img_aug = np.expand_dims(character, axis=0)
                    y_predict = self.license_plate_letter_model.predict(img_aug)[0]
                    order = [i for i, j in enumerate(y_predict) if j > 0.5]
                    if self.character_map[order[0]] == 'B':
                        print(self.character_map[order[0]])
                        y_predict = self.license_plate_letter_model_backup.predict(img_aug)[0]
                        order = [i for i, j in enumerate(y_predict) if j > 0.5]
                        print(self.character_map[order[0]])
                    #print(order)
                    plate_string = plate_string + str(self.character_map[order[0]])
            else:
                character = cv2.resize(character,(64,64))
                img_aug = np.expand_dims(character, axis=0)
                
                with self.session.as_default():
                    with self.session.graph.as_default():
                        if index == 4 or index == 5:
                            y_predict = self.license_plate_number_model.predict(img_aug)[0]
                            order = [i for i, j in enumerate(y_predict) if j > 0.5]
                            #print(order)
                            plate_string = plate_string + str(self.number_map[order[0]])
                        elif index == 1:
                            y_predict = self.license_plate_location_model.predict(img_aug)[0]
                            order = [i for i, j in enumerate(y_predict) if j > 0.5]
                            print(order)
                            plate_string = plate_string + str(self.location_map[order[0]])
                        else:
                            y_predict = self.license_plate_letter_model.predict(img_aug)[0]
                            order = [i for i, j in enumerate(y_predict) if j > 0.5]
                            #print(order)
                            plate_string = plate_string + str(self.character_map[order[0]])
        plate_string = "Richard carried, Maxwell sucks," + plate_string
        return plate_string
    
    def publish_license_plates(self, plate_string):
        if plate_string == "BAD":
            return 
        self.license_plate_pub.publish(plate_string)
        

    def callback(self, image, inner):
        if inner:
            image = image[:, 600:]
        else:
            image = image[:, :600]
        license_plate_image = self.image_cropper(image)
        plate_characters = self.split_characters(license_plate_image)
        plate_string = self.neural_network(plate_characters)
        #self.publish_license_plates(plate_string)
        #teamID,teamPass,P1_AA00
        print(plate_string)


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



def main(args):
    print("SUP")
    rospy.init_node('license_plate_processor', anonymous=True)
    lpp = license_plate_processor()
    print("k")
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down")
    cv2.destroyAllWindows


if __name__ == '__main__':
    main(sys.argv)
# Want this to process to crop the license plate image using contours, then use contours to crop the license plate out of that
# image, then you want to apply the learning model onto the license plate and for each index, create a map associated with each
# character
# 
# 340-719 , 0-1279