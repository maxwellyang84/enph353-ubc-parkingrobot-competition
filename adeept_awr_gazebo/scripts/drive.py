import cv2 as cv
import sys
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import rospy
import time
from random import randint
from license_plate_processor import license_plate_processor

KP = 0.001
KD = 0.0005

NUM_PIXELS_X = 1280
NUM_PIXELS_Y = 379
READING_GAP = 215 
Y_READING_LEVEL = 345
Y_READ_PLATES = 230 
X_READ_INNER_PLATES = 1000
Y_READ_PED = 90 
Y_CHECK_TRUCK = 70
X_CHECK_TRUCK_START = 430
X_CHECK_TRUCK_END = 600
X_CHECK_INNER_BLUE = 950
Y_CHECK_INNER_BLUE = 100
Y_CHECK_OUTER_BLUE = 120


### STATES ### 
INITIALIZE = 0
DRIVING_INNER = 1
TRANSITION = 2
DRIVING = 3
WATCHING = 4
CROSS_THE_WALK = 5

class state_machine:

    def __init__(self):
        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback)
        self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=30)
        self.image_pub = rospy.Publisher('license_plate_processor', Image, queue_size=30)
        self.bridge = CvBridge()
        self.last_err = 0
        self.last_pos = 0
        self.current_state = INITIALIZE #INITIALIZE
        self.starting_time = time.time()
        while not rospy.get_time():
            self.ros_starting_time = rospy.get_time()
        self.ros_starting_time = rospy.get_time()
        self.num_of_plates_snapped = 0
        self.at_top_crosswalk = True
        self.initalized = False
        self.truck_spotted = False
        self.first_inner_car = False
        self.first_inner_pic_snapped = False
        print("Let's start with the inner plates...")
        self.lpp = license_plate_processor()
        
        
   
    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") # gets from camera
        except CvBridgeError as e:
            print(e)
        frame = state_machine.image_converter(cv_image) # crops 
        
    
        if self.current_state == INITIALIZE: #for turning into initial loop first
            ros_time_elapsed = rospy.get_time() - self.ros_starting_time
            #print("elapsed time: " + str(ros_time_elapsed))
            if ros_time_elapsed < 2.4 and not self.initalized:
                # print("rotating...")
                self.speed_controller(500)
            else:
                if not self.initalized:
                    print("We're gonna let the truck pass first...")
                self.initalized = True
                if not self.truck_spotted:
                    if self.watch_truck(frame):
                        self.truck_spotted = True
                        print("There it is! Now let's give it a head start...")
                    # print("looking for truck: " + str(ros_time_elapsed))
                    self.stop()
                    self.ros_starting_time = rospy.get_time()
                elif ros_time_elapsed < 5.6:
                    # print("waiting 5.6 sec: " + str(ros_time_elapsed))
                    self.stop()
                elif ros_time_elapsed < 6:
                    # print("going forward a bit: " + str(ros_time_elapsed))
                    self.speed_controller(0)
                elif ros_time_elapsed < 6.5: #7.75
                    # print("rotating a bit:" + str(ros_time_elapsed))
                    self.speed_controller(500)
                else:
                    # print("driving!")
                    self.ros_starting_time = rospy.get_time()
                    self.current_state = DRIVING_INNER


        elif self.current_state == DRIVING_INNER:
            ros_time_elapsed = rospy.get_time() - self.ros_starting_time
      
            position = state_machine.get_position(frame, self.last_pos)
            self.last_pos = position
            self.speed_controller(position)

            if not self.first_inner_car:
                if self.check_blue_car(frame, True) and not self.first_inner_pic_snapped:
                    ros_time_elapsed = 0
                    self.stop()
                    self.first_inner_pic_snapped = True
                    self.ros_starting_time = rospy.get_time()
                    print("Inner License plate snapped")
                    # print(time_elapsed)
                    #cv.imshow("License Plate Frame", frame)
                    cv.imwrite("./license_plates/" + str(randint(0,10000)) + ".png", frame)
                    self.lpp.callback(frame, True)
                
                if ros_time_elapsed < 5 and self.first_inner_pic_snapped:
                    # print("waiting: " + str(time_elapsed))
                    self.stop()
                elif ros_time_elapsed > 5 and self.first_inner_pic_snapped: 
                    self.first_inner_car = True
                    self.ros_starting_time = rospy.get_time()
                    # print("onto second car")
            else:
                # print("second car")
                if self.check_blue_car(frame, True) and ros_time_elapsed > 1.5:
                    print("Inner License plate snapped")
                    #cv.imshow("License Plate Frame", frame)
                    cv.imwrite("./license_plates/" + str(randint(0,10000)) + ".png", frame)
                    self.stop()
                    self.current_state = TRANSITION    
                    self.ros_starting_time = rospy.get_time()
        #         #self.stop()
                    cv.imwrite("./license_plates/" + str(randint(0,10000)) + ".png", frame)
                    self.lpp.callback(frame, True)

        elif self.current_state == TRANSITION:
            ros_time_elapsed = rospy.get_time() - self.ros_starting_time
            if ros_time_elapsed < 1.2:
                self.speed_controller(0)
            elif ros_time_elapsed < 2.4:
                self.speed_controller(500)
            elif ros_time_elapsed < 4.4:
                self.speed_controller(0)
            elif ros_time_elapsed < 4.8:
                self.speed_controller(500)
            else:
                self.current_state = DRIVING
                print("Onto the outer plates!")

        
        elif self.current_state == DRIVING:
            ros_time_elapsed = rospy.get_time() - self.ros_starting_time
            position = state_machine.get_position(frame, self.last_pos)
            self.last_pos = position
            self.speed_controller(position)

            if self.check_crosswalk(frame):
                self.stop()
                self.current_state = WATCHING
                print("Stop! Looking for pedestrians...")

            if self.check_blue_car(frame, False) and ros_time_elapsed > 0.4: # 0.4
                self.ros_starting_time = rospy.get_time()
                print("Outer License plate snapped")
                #cv.imshow("License Plate Frame", frame)
                #self.stop()
                cv.imwrite("./license_plates/" + str(randint(0,10000)) + ".png", frame)
                self.lpp.callback(frame, False)

        elif self.current_state == WATCHING:
            if self.at_top_crosswalk:
                if self.watch_people(frame, True): #at top crosswalk
                    self.current_state = CROSS_THE_WALK
                    self.at_top_crosswalk = False # future crosswalk will be bottom one
                    self.ros_starting_time = rospy.get_time()
            else:
                if self.watch_people(frame, False): #at bottom crosswalk
                    self.current_state = CROSS_THE_WALK
                    self.at_top_crosswalk = True # future crosswalk will be top one
                    self.ros_starting_time = rospy.get_time()
        
        elif self.current_state == CROSS_THE_WALK:
            ros_time_elapsed = rospy.get_time() - self.ros_starting_time
            #print(time_elapsed)
            if ros_time_elapsed < 0.2:
                self.speed_controller(0)
            elif ros_time_elapsed < 3.4 : # 2.4, then 2.6, then 2.8, then 3.4
                if self.check_crosswalk(frame):
                    self.speed_controller(0)
                else:
                    position = state_machine.get_position(frame, self.last_pos)
                    self.last_pos = position
                    self.speed_controller(position)
            else:
                print("Back to driving state...")
                self.current_state = DRIVING

        
        #      DEBUGGING TOOLS TO SEE BLACK AND WHITE
        # cv.circle(frame, (READING_GAP, Y_READING_LEVEL), 15, (255,205,195), -1) #top two cicles are for linefollowing
        # cv.circle(frame, (NUM_PIXELS_X-READING_GAP,Y_READING_LEVEL), 15, (255,205,195), -1)        
        # cv.circle(frame, (READING_GAP,Y_READ_PLATES), 15, (255,205,195), -1)  #reading for white here  (outer loop)
        # cv.circle(frame, (NUM_PIXELS_X/6,Y_READ_PED), 15, (255,205,195), -1) #checking for ped on left, top crosswalk
        # cv.circle(frame, (2*NUM_PIXELS_X/6+15,Y_READ_PED), 15, (255,205,195), -1)
        cv.circle(frame, (750,Y_READ_PED-10), 15, (255,205,195), -1) #checking for ped on right, bottom crosswalk
        cv.circle(frame, (815,Y_READ_PED-10), 15, (255,205,195), -1)
        # cv.circle(frame, (X_CHECK_TRUCK_START, Y_CHECK_TRUCK), 15, (255,205,195), -1) #looking for the ford
        # cv.circle(frame, (X_CHECK_TRUCK_END, Y_CHECK_TRUCK), 15, (255,205,195), -1) #looking for the ford
        # cv.circle(frame, (X_READ_INNER_PLATES, Y_READ_PLATES), 15, (255,205,195), -1)  #inner reading plate
        # cv.circle(frame, (NUM_PIXELS_X-1, Y_READ_PLATES), 15, (255,205,195), -1) #inner reading plate
        # cv.circle(frame, (X_CHECK_INNER_BLUE, Y_CHECK_INNER_BLUE), 15, (255,205,195), -1) #inner check blue
  
        # cv.circle(frame, (X_CHECK_INNER_BLUE, Y_CHECK_INNER_BLUE), 15, (255,205,195), -1) #looking for ped
        # cv.circle(frame, (X_CHECK_INNER_BLUE, Y_CHECK_INNER_BLUE), 15, (255,205,195), -1) #looking for ped
        #         for i in range(NUM_PIXELS_X/6+15): #we want 10 pixels
        #     if jeans_mask[Y_READ_PED, NUM_PIXELS_X/6+i] != 0:
        #         jean_pixels = jean_pixels + 1

        # light_test = (20, 20, 20)
        # dark_test = (110, 70, 70)
        # frame = cv.inRange(frame, light_test, dark_test) #road is white and majority of other stuff is black
        # cv.circle(frame, (750,Y_READ_PED-10), 15, (255,205,195), -1) #checking for ped on right, bottom crosswalk
        # cv.circle(frame, (815,Y_READ_PED-10), 15, (255,205,195), -1)
        cv.imshow("Robot's view :3", frame)
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
            return pos #if positive, turn left
        else:
            if (last_pos > 0):
                pos = 500          
            elif (last_pos < 0):
                pos = -500
            else:
                pos = 0
            return pos
        
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
            val = check_frame[NUM_PIXELS_Y-1,i+READING_GAP] # Y_READING_LEVEL
            if (val != 0):
                red_pixels = red_pixels + 1

        if (red_pixels > NUM_PIXELS_X/5):  #num_pixels/4
            return 1
        else:
            return 0


    # def adjust_crosswalk(frame):
    #     light_red = (0, 0, 245) # BGR
    #     dark_red = (10, 10, 255)
    #     red_pixels = 0

    #     check_frame = cv.inRange(frame, light_red, dark_red) #only red appears, all else black
        
    #     if check_frame[NUM_PIXELS_Y-1,NUM_PIXELS_X/3] != 0 and check_frame[NUM_PIXELS_Y-1,2*NUM_PIXELS_X/3] != 0:
    #         return 1 # in line
    #     elif check_frame[NUM_PIXELS_Y-1,NUM_PIXELS_X/3] == 0:

    @staticmethod
    def check_blue_car(frame, inner_ring):
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

        if not inner_ring: 
            # inRange is x,y
            for i in range(READING_GAP): #ReadGap
                if (white_mask[Y_READ_PLATES,i] != 0):
                    white_pixels = white_pixels + 1
        #print("Num:" + " " +  str(white_pixels))
        
            if (white_pixels < 60): #50 <--- if its 50 you will get corner pics!!!
                return 0

            if ((blue_mask[150,0] != 0) or (blue2_mask[150,0] != 0)): #make sure edge isnt blue
                return 0
            else:
                #print("SAW WHITE")
                check_stripes = 0
                for i in range(2*READING_GAP+10): #ADDED THE +10, changed the y value from 150 to 60,
                    if (((blue_mask[Y_CHECK_OUTER_BLUE,i] == 0) or (blue2_mask[Y_CHECK_OUTER_BLUE,i] == 0)) and check_stripes == 0): #first not blue
                        check_stripes = 1
                    if (((blue_mask[Y_CHECK_OUTER_BLUE,i] != 0) or (blue2_mask[Y_CHECK_OUTER_BLUE,i] != 0)) and check_stripes == 1): #first blue
                        check_stripes = 2
                    if (((blue_mask[Y_CHECK_OUTER_BLUE,i] == 0) or (blue2_mask[Y_CHECK_OUTER_BLUE,i] == 0)) and check_stripes == 2): #second not blue
                        check_stripes = 3
                    if (((blue_mask[Y_CHECK_OUTER_BLUE,i] != 0) or (blue2_mask[Y_CHECK_OUTER_BLUE,i] != 0)) and check_stripes == 3): #second blue
                        check_stripes = 4
                    if (check_stripes == 4):
                        #print("passed stripes test")
                        #print("Passed, num white = " + str(white_pixels))
                        return 1
                return 0
        
        elif inner_ring:
            for i in range(NUM_PIXELS_X-X_READ_INNER_PLATES-1):
                if (white_mask[Y_READ_PLATES,i+X_READ_INNER_PLATES] != 0):
                    white_pixels = white_pixels + 1
            
            #print(white_pixels)
            if (white_pixels < 80):
                return 0           

            # print(white_pixels)

            for i in range(NUM_PIXELS_X-X_CHECK_INNER_BLUE-1):
                if blue_mask[Y_CHECK_INNER_BLUE,i+X_CHECK_INNER_BLUE] != 0 or blue2_mask[Y_CHECK_INNER_BLUE,i+X_CHECK_INNER_BLUE] != 0: #first blue
                    #print("FOUND INNER PLATE AYY")
                    return 1

            return 0


    def watch_people(self, frame, top_crosswalk):  #y pixel at 85 is where we wanna look to see movement
        #Y_READ_PED = 90             (NUM_PIXELS_X/3,Y_READ_PED)
        light_jeans = (20, 20, 20)
        dark_jeans = (110, 70, 70)
        jeans_mask = cv.inRange(frame, light_jeans, dark_jeans) #road is white and majority of other stuff is black
        jean_pixels = 0
        mid_pixels = 0

        # cv.circle(frame, (NUM_PIXELS_X/6,Y_READ_PED), 15, (255,205,195), -1) #checking for ped on left
        # cv.circle(frame, (2*NUM_PIXELS_X/6+15,Y_READ_PED), 15, (255,205,195), -1)
        # cv.circle(frame, (X_READ_PED_INNER_RIGHT,Y_READ_PED), 15, (255,205,195), -1)
        
        if top_crosswalk:
                           # this x value iss 440
            for i in range(NUM_PIXELS_X/6+15): #we want 10 pixels, counting pixels to the left...
                if jeans_mask[Y_READ_PED, NUM_PIXELS_X/6+i] != 0:
                    jean_pixels = jean_pixels + 1

            #print("(top) num of pixels between these balls: " + str(jean_pixels))

            if jean_pixels > 10:
                print("Top crosswalk: He's on the left!!! Go")
                return 1
            else:
                return 0
        else:
            for i in range(815-750): #we want 10 pixels, counting pixels to the right...
                if jeans_mask[Y_READ_PED-10, 750+i] != 0:
                    jean_pixels = jean_pixels + 1

            #print("(bot) num of jean pixels between these balls: " + str(jean_pixels))

            if jean_pixels > 10:
                print("Bottom crosswalk: He's on the right!!! Go")
                return 1
            else:
                return 0    


    def watch_truck(self, frame):  #gets like over 40 pixels pretty often
        light_truck = (100, 100, 100)
        dark_truck = (130, 130, 130)
        truck_mask = cv.inRange(frame, light_truck, dark_truck) #road is white and majority of other stuff is black
        truck_pixels = 0

        for i in range(X_CHECK_TRUCK_END-X_CHECK_TRUCK_START): #   #430,70 to 600,70
            if truck_mask[Y_CHECK_TRUCK, X_CHECK_TRUCK_START+i] != 0:
                truck_pixels = truck_pixels + 1

        # print("Num of truck pixels = " + str(truck_pixels))
        if truck_pixels > 40:
            # print("Truck spotted!")
            return 1
        else:
            return 0

    def stop(self):
        velocity = Twist()
        velocity.linear.x = 0
        self.vel_pub.publish(velocity)


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