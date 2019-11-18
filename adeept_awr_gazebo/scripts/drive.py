import cv2 as cv
import sys
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import rospy

KP = 0.001
KD = 0.0005
NUM_PIXELS_X = 1280
READING_GAP = 215

class state_machine:

    def __init__(self):
        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback)
        self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=30)
        self.bridge = CvBridge()
        self.last_err = 0
        self.last_pos = 0
   
    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        offset = 0
        frame = state_machine.image_converter(cv_image)
        position = state_machine.get_position(frame, self.last_pos)
        self.last_pos = position
     
        light_grey = (70, 70, 70)
        dark_grey = (135, 135, 135)
        frame = cv.inRange(frame, light_grey, dark_grey) #road is white and majority of other stuff is black
        
        self.speed_controller(position)

        #frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)


        cv.circle(frame, (READING_GAP, 345), 15, (255,205,195), -1)
        cv.circle(frame, (NUM_PIXELS_X-READING_GAP,345), 15, (255,205,195), -1)
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
            val = frame[345,i+READING_GAP]
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
        #cv.COLOR_BGR2RGB  cv.COLOR_BGR2GRAY
        
        frame = cv_image[340:719,0:1279] #cropping irrelevant pixels out: x:0,1278 ; y:0,378 (0 at top)
        #frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        #ret, frame = cv.threshold(frame, 85, 255, cv.THRESH_BINARY)
                                  # threshold, 255 is white

        return frame

    def speed_controller(self, position):
        velocity = Twist()
        if (abs(position) < 60):
            velocity.linear.x = 0.005 #0.3
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