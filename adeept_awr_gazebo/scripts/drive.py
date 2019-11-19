import cv2 as cv
import sys
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import rospy
import time

KP = 0.001
KD = 0.0005
NUM_PIXELS_X = 1280
NUM_PIXELS_Y = 379
READING_GAP = 215
Y_READING_LEVEL = 345

class state_machine:

    def __init__(self):
        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback)
        self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=30)
        self.bridge = CvBridge()
        self.last_err = 0
        self.last_pos = 0
   
    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") # gets from camera
        except CvBridgeError as e:
            print(e)
        
        frame = state_machine.image_converter(cv_image) # crops 
        
        position = state_machine.get_position(frame, self.last_pos)
        self.last_pos = position
        #self.speed_controller(position)
        
        if (self.check_crosswalk(frame)):
        #    self.stop()
            print("Stopped!")
            #self.watch()
        if (self.check_blue_car(frame)):
            print("Sign on the left!")
        
        #      DEBUGGING TOOLS TO SEE BLACK AND WHITE
        #light_grey = (70, 70, 70)
        #dark_grey = (135, 135, 135)
        #frame = cv.inRange(frame, light_grey, dark_grey) #road is white and majority of other stuff is black
        #frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        #cv.circle(frame, (READING_GAP, 345), 15, (255,205,195), -1)
        #cv.circle(frame, (NUM_PIXELS_X-READING_GAP,345), 15, (255,205,195), -1)

        #light_grey = (95, 0, 0)
        #dark_grey = (110, 5, 5)
        #frame = cv.inRange(frame, light_grey, dark_grey) #road is white and majority of other stuff is black
        cv.imshow("Robot", frame)
        cv.waitKey(3) 


    @staticmethod
    def get_position(frame, last_pos):
        light_grey = (65, 65, 65)
        dark_grey = (140, 140, 140)

        frame = cv.inRange(frame, light_grey, dark_grey) #road is white and majority of other stuff is black

        weighted_sum = 0
        pixel_count = 0

        for i in range(NUM_PIXELS_X-2*READING_GAP): #length of pixels we wanna look at
            val = frame[Y_READING_LEVEL,i+READING_GAP]
            if (val != 0):
                pixel_count = pixel_count + 1
                weighted_sum = weighted_sum + i + READING_GAP
        
        if (pixel_count != 0):
            weighted_sum = weighted_sum/pixel_count
            pos = NUM_PIXELS_X/2-weighted_sum-1
            print(pos)
            return pos#if positive, turn left
        else:
            if (last_pos > 0):
                pos = 500          
            elif (last_pos < 0):
                pos = -500
            else:
                pos = 0
            print(pos)
            return pos
        
        #count grey from: x:200 1150 , y: 345

    @staticmethod
    def image_converter(cv_image):
        #cropping irrelevant pixels out: x:0,1278 ; y:0,378 (0 at top)
        return cv_image[340:719,0:1279]
    
    @staticmethod
    def check_crosswalk(frame):
        light_red = (0, 0, 245) # BGR
        dark_red = (10, 10, 255)
        red_pixels = 0

        check_frame = cv.inRange(frame, light_red, dark_red) #only red appears, all else black
        
        for i in range(NUM_PIXELS_X-2*READING_GAP): #length of pixels we wanna look at
            val = check_frame[Y_READING_LEVEL,i+READING_GAP]
            if (val != 0):
                red_pixels = red_pixels + 1

        if (red_pixels > NUM_PIXELS_X/3):
            return 1
        else:
            return 0
        #when seeing red go till zebra, 5 white strips in grey road (not including white border)

    @staticmethod
    def check_blue_car(frame):
        light_blue = (95, 0, 0) # BGR
        dark_blue = (110, 5, 5)
        blue_pixels = 0
        first_blue = -1
        last_blue = 0

        check_frame = cv.inRange(frame, light_blue, dark_blue) #only red appears, all else black
        
        for i in range(NUM_PIXELS_Y): #length of pixels we wanna look at
            val = check_frame[i,0]    # checking all values on left edge of screen
            if (val != 0):
                blue_pixels = blue_pixels + 1
                if (first_blue == -1):
                    first_blue = i
        last_blue = first_blue + blue_pixels
        
        if (blue_pixels < 100): #we expect that on average there should be 150 blue pixels
            return 0
        else:
            for i in range(125,300): #got these magic numbers from seeing on average where the right hand side of the blue sign will be
                if (check_frame[(int)(last_blue-first_blue)/2,i] != 0):
                    return 1
            return 0
       # v val 125 to 300

    #def watch(self):  #y pixel at 85 is where we wanna look to see movement
    #    for i in range(500):
    #        try:
    #            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") # gets from camera
    #        except CvBridgeError as e:
    #            print(e)
    #     
    #        #offset = 0
    #        frame = state_machine.image_converter(cv_image) # crops 

    def stop(self):
        velocity = Twist()
        velocity.linear.x = 0
        self.vel_pub.publish(velocity)


    def speed_controller(self, position):
        velocity = Twist()
        if (abs(position) < 50): #worked well w 60
            velocity.linear.x = 1 #0.005
        else:
            p = KP * position
            d = KD * (position - self.last_err)
            velocity.angular.z = p+d #turns left when positive

        self.last_err = position
        self.vel_pub.publish(velocity)

def main(args):
    
    rospy.init_node('image_processor', anonymous=True)
    sm = state_machine()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down")
    cv.destroyAllWindows

if __name__ == '__main__':
    main(sys.argv)