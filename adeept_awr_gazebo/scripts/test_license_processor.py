from license_plate_processor import license_plate_processor
import cv2

lpp = license_plate_processor()
img = cv2.imread("./license_plates/4692.png") #5249, 7677, 4692 filter license contour by area more
lpp.callback(img)