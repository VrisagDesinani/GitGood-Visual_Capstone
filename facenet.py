from database import Profile
from facenet_models import FacenetModel
import numpy as np
import cv2

model = FacenetModel()


def setup_database(image_filepath, name):
    bgr_image = cv2.imread(image_filepath)
    rgb_image = cv2.cvtColor(bgr_image, cv2.BGRTORGB)

    boxes, probabilities, landmarks = model.detect(rgb_image)
    descriptors = model.compute_descriptors(rgb_image, boxes)

    profile = Profile(name)
    profile.add_descriptor(descriptors)

setup_database("faces/Amanda.jpg", "Amanda")
setup_database("faces/Aryaman.jpg", "Aryaman")
setup_database("faces/gwyneth.jpg", "Gwyneth")
setup_database("faces/Kritik.jpg", "Kritik")
setup_database("faces/Lily.jpg", "Lily")
setup_database("faces/sofie.jpg", "Sofie")
setup_database("faces/steven.jpg", "Steven")
setup_database("faces/Vrisag.jpg", "Vrisag")

def display_image(image, box_coords, names):
    img = image # update later based on image parameter data type
    for (x1, y1, x2, y2,), name in zip(box_coords, names):
        color = (240,128,128)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Faces w/ Boxes", img)