from database import Database
from facenet_models import FacenetModel
import numpy as np
import cv2

model = FacenetModel()

def setup_database(name, image_filepath, database):

    bgr_image = cv2.imread(image_filepath)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    boxes, probabilities, landmarks = model.detect(rgb_image)
    descriptors = model.compute_descriptors(rgb_image, boxes)

    if boxes is not None and len(boxes) > 0:
        index = probabilities >= 0.4
        boxes = boxes[index]
        
    database.add_image(name, descriptors)
    
    
def display_image(image, box_coords, names):
    img = image.copy() # update later based on image parameter data type
    for (x1, y1, x2, y2,), name in zip(box_coords, names):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = (240,128,128)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Faces w/ Boxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()