import cv2 as cv
import sys
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import rospy


class state_machine:

    def __init__(self):
        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback)
        self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=30)
        self.bridge = CvBridge()
        self.last_err = 0
   
    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        offset = 0
        frame = state_machine.image_converter(cv_image)
       # centre = state_machine.get_centre(frame)
     
        # self.speed_controller(centre)
        #cv.circle(frame, (int(centre), 700), 20, (255, 0, 0), -1)

        cv.imshow("Robot", frame)
        cv.waitKey(3) 
    
    @staticmethod
    def image_converter(cv_image):

        frame = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)
        frame = frame[340:719,0:1279]
        #crop_img = img[y:y+h, x:x+w]
        #x: 0,1279 ; y: 340,719
        ret, frame = cv.threshold(frame, 85, 255, cv.THRESH_BINARY)
                                  # threshold, 255 is white
        return frame

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