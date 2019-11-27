from license_plate_processor import license_plate_processor
import cv2

lpp = license_plate_processor()
#img = cv2.imread("./license_plates/4692.png") #5249, 7677, 4692 filter license contour by area more
img = cv2.imread("./incorrectly_interpreted/8181.png") #5249, 7677, 4692 filter license contour by area more, 966

lpp.callback(img, False)

#RESULTS:
#NN 4 thought 6244 had a 4 when it was a 0, 3776 had a 4 when it was an 8
#NN5 thought 9735 had a C when it was a L, 8181 had an R when it had an E, 6244 was a 6 when it was a 0
#NN6 thought 6244 had 2 4's when it was a 3 and a 0, more blur ain't it
#NN9 thought 6244 had a F when it was a K and a 4 when it was a 0