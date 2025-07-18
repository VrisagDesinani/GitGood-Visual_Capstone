from facenet import setup_database, display_image
from facenet_models import FacenetModel
from database import Profile, Database
from match import has_match
db = Database()
import cv2

profile = Profile()
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

test_photo = "faces/group_photo.jpg"

bgr_image = cv2.imread(image_filepath)
rgb_image = cv2.cvtColor(bgr_image, cv2.BGRTORGB)

boxes, probabilities, landmarks = model.detect(rgb_image)
descriptors = model.compute_descriptors(rgb_image, boxes)

names = []
for descriptor in descriptors:
    name = has_match(descriptor, db, THRESHOLD)
    names.append(name)

display_image(bgr_image, boxes, names)