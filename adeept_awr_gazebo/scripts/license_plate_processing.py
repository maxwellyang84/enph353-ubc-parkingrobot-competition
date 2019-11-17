import cv2 as cv
import keras 
from keras.models import load_model
from keras import layers
from keras import models
from keras import optimizers

from keras.utils import plot_model
from keras import backend

class license_plate_processor:

    def __init__(self, license_plate_pub):
        self.license_plate_image = None
        self.location_image = None
        self.license_plate_model = load_model('prototype.h5')
        self.license_plate_pub = license_plate_pub

    def image_cropper(self, image):
        pass

    def license_plate_model_process(self):
        pass
    
    def location_model_process(self):
        pass






# Want this to process to crop the license plate image using contours, then use contours to crop the license plate out of that
# image, then you want to apply the learning model onto the license plate and for each index, create a map associated with each
# character
# 
# 340-719 , 0-1279