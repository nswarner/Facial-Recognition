import cv2, os
import numpy as np
from PIL import Image

hCascadeDir = "N:\\Python27\\haarcascades\\"
cascadePath = hCascadeDir + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

recognizer = cv2.createLBPHFaceRecognizer()



