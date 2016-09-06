import cv2, os
import numpy as np
from PIL import Image


class FaceRecognizer:

    def __init__(self, load_previous=None, create_new=None):
        # For face recognition we will the the LBPH Face Recognizer
        self.recognizer = cv2.createLBPHFaceRecognizer()

        if self.old(load_previous):
            self.load_recognizer(load_previous)

        else:
            images, labels = self.prepare_images_and_labels(create_new)
            self.train_from_images(images, labels)

    def load_recognizer(self, r_path):
        self.recognizer = cv2.createLBPHFaceRecognizer()
        self.recognizer.load(r_path)

    def new_recognizer(self, path):
        # For face recognition we will the the LBPH Face Recognizer
        self.recognizer = cv2.createLBPHFaceRecognizer()
        images, labels = self.prepare_images_and_labels(path)
        self.train_from_images(images, labels)
        self.save_recognizer("./last_recognizer.xml")

    def old(self, path):
        if os.path.isfile(path):
            return True
        else:
            return False

    def save_recognizer(self, save_file):
        self.recognizer.save(save_file)

    def predict2(self, image):
        nbr_predicted, conf = self.recognizer.predict(image)
        return nbr_predicted, conf

    def prepare_images_and_labels(self, path):
        # For face detection we will use the Haar Cascade provided by OpenCV.
        cascade_path = "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        # Append all the absolute image paths in a list image_paths
        # We will not read the image with the .sad extension in the training set
        # Rather, we will use them to test our accuracy of the training
        #image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        # images will contains face images
        images = []
        # labels will contains the label that is assigned to the image
        labels = []
        for image_path in image_paths:
            # Read the image and convert to grayscale
            image_pil = Image.open(image_path).convert('L')
            # Convert the image format into numpy array
            image = np.array(image_pil, 'uint8')
            # Get the label of the image
            #nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
            #nbr = os.path.split(image_path)[1].split(".")[0]
            if (image_path.startswith("./training_set\\g")):
                nbr = 1
            else:
                nbr = 2
            # Detect the face in the image
            faces = face_cascade.detectMultiScale(image)
            # If face is detected, append the face to images and the label to labels
            for (x, y, w, h) in faces:
                # Let's verify that each photo is in the training set
                cv2.imshow("Adding faces to training set...", image[y: y + h, x: x + w])
                cv2.waitKey(200)
                var = raw_input("[" + image_path + "] 'n' if this isn't a face, [enter] otherwise: ").lower()
                # Verify it's someone
                if var == "":
                    images.append(image[y: y + h, x: x + w])
                    labels.append(nbr)
                    if (nbr == 1):
                        print("Image added, recognized [Grant]")
                    else:
                        print("Image added, recognized [Ashley]")
        # return the images list and labels list
        return images, labels

    def train_from_images(self, images, labels):
        # Path to the Yale Dataset
        #path = './yalefaces'
        # Call the get_images_and_labels function and get the face images and the
        # corresponding labels
        cv2.destroyAllWindows()
        # Perform the tranining
        self.recognizer.train(images, np.array(labels))
