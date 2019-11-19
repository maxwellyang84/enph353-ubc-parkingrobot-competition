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
READING_GAP = 215 #205
Y_READING_LEVEL = 345
Y_READ_PLATES = 230 #270

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
        self.speed_controller(position)
        
        if (self.check_crosswalk(frame)):
        #    self.stop()
            print("Stopped!")
            #self.watch()
        if (self.check_blue_car(frame)):
            self.stop()
            time.sleep(1)
            self.go(0.1)

        
        #      DEBUGGING TOOLS TO SEE BLACK AND WHITE
        #light_grey = (70, 70, 70)
        #dark_grey = (135, 135, 135)
        #frame = cv.inRange(frame, light_grey, dark_grey) #road is white and majority of other stuff is black
        #frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        cv.circle(frame, (READING_GAP, Y_READING_LEVEL), 15, (255,205,195), -1)
        cv.circle(frame, (NUM_PIXELS_X-READING_GAP,Y_READING_LEVEL), 15, (255,205,195), -1)        
        cv.circle(frame, (READING_GAP,Y_READ_PLATES), 15, (255,205,195), -1)

        #light_test = (245, 245, 245)
        #dark_test = (255, 255, 255)
        #frame = cv.inRange(frame, light_test, dark_test) #road is white and majority of other stuff is black
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
            #print(pos)
            return pos #if positive, turn left
        else:
            if (last_pos > 0):
                pos = 500          
            elif (last_pos < 0):
                pos = -500
            else:
                pos = 0
            #print(pos)
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
        light_blue = (85, 0, 0) # BGR 
        light_blue2 = (190,90,90)
        dark_blue = (135, 35, 35)
        dark_blue2 = (210,110,110)
        light_white = (245,245,245)        
        dark_white = (255,255,255)        
        
        blue_pixels = 0
        white_pixels = 0

        blue_mask = cv.inRange(frame, light_blue, dark_blue) #only red appears, all else black     
        blue2_mask = cv.inRange(frame, light_blue2, dark_blue2) #only red appears, all else black           
        white_mask = cv.inRange(frame, light_white, dark_white) #only red appears, all else black
        
        # inRange is x,y
        for i in range(READING_GAP): #ReadGap
             if (white_mask[Y_READ_PLATES,i] != 0):
                 white_pixels = white_pixels + 1
        print("Num:" + " " +  str(white_pixels))
        
        if (white_pixels < READING_GAP/4): #READING_GAP/3
            return 0

        if ((blue_mask[150,0] != 0) or (blue2_mask[150,0] != 0)): #make sure edge isnt blue
            return 0

        else:
            print("SAW WHITE, 2/3")
            check_stripes = 0
            for i in range(2*READING_GAP):
                if (((blue_mask[150,i] == 0) or (blue2_mask[150,i] == 0)) and check_stripes == 0): #first not blue
                    check_stripes = 1
                if (((blue_mask[150,i] != 0) or (blue2_mask[150,i] != 0)) and check_stripes == 1): #first blue
                    check_stripes = 2
                if (((blue_mask[150,i] == 0) or (blue2_mask[150,i] == 0)) and check_stripes == 2): #second not blue
                    check_stripes = 3
                if (((blue_mask[150,i] != 0) or (blue2_mask[150,i] != 0)) and check_stripes == 3): #second blue
                    check_stripes = 4
                
                if (check_stripes == 4):
                    print("passed stripes test")
                    return 1
            return 0
            # for i in range(READING_GAP):
            #     if (blue_mask[150,i] != 0):
            #         print("Sign TYPE 1!")
            #         return 1
            #     if (blue2_mask[150,i] != 0):
            #         print("Sign TYPE 2!")
            #         return 2        
        
        #for i in range(NUM_PIXELS_Y): #length of pixels we wanna look at
        #    val = blue_mask[i,0]    # checking all values on left edge of screen
        #    if (val != 0):
        #        blue_pixels = blue_pixels + 1
        
        #if (blue_pixels < 100): #we expect that on average there should be 150 blue pixels
        #    return 0
        #else:
        #    for i in range(NUM_PIXELS_Y):
        #        if (white_mask[150,i] != 0):
        #            white_side_pixels = white_side_pixels + 1
           
        #if (white_side_pixels > 5):
        #    return 1
        #else:
        #    return 0

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

    def go(self, sec):
        velocity = Twist()
        velocity.linear.x = 1
        self.vel_pub.publish(velocity)
        time.sleep(sec)

    def speed_controller(self, position):
        velocity = Twist()
        if (abs(position) < 25): #30 was bang bang but pretty good for white line, og was 50,20 fire
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