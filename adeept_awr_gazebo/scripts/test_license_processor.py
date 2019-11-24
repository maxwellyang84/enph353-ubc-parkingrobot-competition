from license_plate_processor import license_plate_processor
import cv2

lpp = license_plate_processor()
img = cv2.imread("./license_plates/7677.png") #5249, 7677
lpp.callback(img)