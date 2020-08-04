from darkflow.net.build import TFNet
import cv2
import sys
import numpy as np
import pytesseract
from PIL import Image


options = {"model": "yolov2-tiny-voc.cfg", "load": "yolov2-tiny-voc.weights", "threshold": 0.01, "load": 3000}

tfnet = TFNet(options)

imgcv = cv2.imread("./sample_img/test/test2.jpg")

result = tfnet.return_predict(imgcv)	
print(result)
config = ('-l eng --oem 1 --psm 3')
for i in range(len(result)):
	print(result[i]['topleft']['y'])
	print(result[i]['bottomright']['y'])
	print(result[i]['topleft']['x'])
	print(result[i]['bottomright']['x'])
	roi=imgcv[result[i]['topleft']['y']:result[i]['bottomright']['y'],result[i]['topleft']['x']:result[i]['bottomright']['x']]
	cv2.imshow('roi',roi)
	cv2.waitKey(2000)
	cv2.destroyAllWindows()
	print(pytesseract.image_to_string(roi,config=config))

