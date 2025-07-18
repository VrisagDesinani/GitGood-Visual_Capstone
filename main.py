from facenet import setup_database, display_image
from facenet_models import FacenetModel
from database import Database
from match import has_match
import cv2

db = Database()
model = FacenetModel()

THRESHOLD = 1.5

setup_database("Amanda", "faces/Amanda.jpg", db)
setup_database("Aryaman", "faces/Aryaman.jpg", db)
setup_database("Gwyneth", "faces/gwyneth.jpg", db)
setup_database("Kritik", "faces/Kritik.jpg", db)
setup_database("Lily", "faces/Lily.jpg", db)
setup_database("Sofie", "faces/sofie.jpg", db)
setup_database("Steven", "faces/steven.jpg", db)
setup_database("Vrisag", "faces/Vrisag.jpg", db)

test_photo = "faces/Amanda.jpg"

bgr_image = cv2.imread(test_photo)
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

boxes, probabilities, landmarks = model.detect(rgb_image)
if boxes is not None:   
    descriptors = model.compute_descriptors(rgb_image, boxes)

names = [has_match(descriptor, db.db, THRESHOLD) for descriptor in descriptors]
display_image(bgr_image, boxes, names)

print(descriptors.shape)
