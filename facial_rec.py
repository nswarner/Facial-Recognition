import cv2, os
import numpy as np
from PIL import Image

'''
    Much of the facial detection/recognition code was provided by Bikramjot Singh Hanzra
    Contact: bikz.05@gmail.com
    URL: http://hanzratech.in/2015/02/03/face-recognition-using-opencv.html

    Revised Todo:
     1. Download a personal training dataset from FB
        - Set up as training_set folder
     2. Validate the training set via display-verify
        - Display a found face, ask who's face this is, discard ""
     3. Incorporate RPi's camera module to constantly scan for faces
     4. If recognized face, load personalized document (browser?)
'''

import cv2, os
#import numpy as np
from PIL import Image
import face_recognizer

# Test if we already have a recognizer set

load_previous = "./last_recognizer.xml"
create_new = "./training_set"
#create_new = "./yalefaces"
recog = face_recognizer.FaceRecognizer(load_previous, create_new)
#recog.save_recognizer("./last_recognizer.xml")

path = "./testing_set"

# Append the images with the extension .sad into image_paths
#image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.sad')]
image_paths = [os.path.join(path, f) for f in os.listdir(path)]
for image_path in image_paths:
    predict_image_pil = Image.open(image_path).convert('L')
    predict_image = np.array(predict_image_pil, 'uint8')
    cascade_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(predict_image)
    for (x, y, w, h) in faces:
        #nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
        nbr_predicted, conf = recog.predict2(predict_image[y: y + h, x: x + w])
        #nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        # Let's fix this instead of leaving it broken...
        nbr_actual = 1
        if nbr_actual == nbr_predicted:
            print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf)
            cv2.waitKey(1000)
        else:
            print "{} is Incorrect Recognized as {}".format(nbr_actual, nbr_predicted)
            cv2.waitKey(1000)
        cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
        cv2.waitKey(200)