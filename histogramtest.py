from facenet import setup_database, display_image
from facenet_models import FacenetModel
from database import Profile, Database
from match import has_match
db = Database()
import cv2
from pathlib import Path
from cos import cos_distances

model = FacenetModel()
DuncanPath = "../faces/TimDuncan"
ParkerPath = "../faces/TonyParker"
TimDuncan = [item for item in DuncanPath.iterdir() if item.is_file()]
TonyParker = [item for item in ParkerPath.iterdir() if item.is_file()]

tim_descriptors = []
for path in TimDuncan:
    setup_database("Tim Duncan", path, db)
    bgr_image = cv2.imread(str(path))
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    descriptor = model.compute_descriptors(rgb_image) 
    if descriptor is not None:
        tim_descriptors.append(descriptor)

tony_descriptors = []
for path in TonyParker:
    setup_database("Tony Parker", path, db)
    bgr_image = cv2.imread(str(path))
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    descriptor = model.compute_descriptors(rgb_image)
    if descriptor is not None:
        tony_descriptors.append(descriptor)

all_distances = []

for i in range(len(tim_descriptors)):
    for j in range(i+1, len(tim_descriptors)):
        distance = cos_distances(tim_descriptors[i], tim_descriptors[j])
        all_distances.append(distance)

for i in range(len(tony_descriptors)):
    for j in range(i+1, len(tony_descriptors)):
        distance = cos_distances(tony_descriptors[i], tony_descriptors[j])
        all_distances.append(distance)

# Tim Duncan vs Tony Parker (should have large distances - different people)
for tim_desc in tim_descriptors:
    for tony_desc in tony_descriptors:
        distance = cos_distances(tim_desc, tony_desc)
        all_distances.append(distance)


def plot_historgram(cos_distances, bins=30):
    '''
    cos_distances = array for cos distances (M, N)
    '''
    cos_distances = cos_distances.flatten()
    plt.hist(cos_distances, bins=bins, color = 'Black', edgecolor = 'Green')
    plt.xlabel("cos distance between 2 descriptors")
    plt.ylabel("count")
    plt.grid(axis='y', alpha=0.75)
    plt.title("histogram of cos distances")
    plt.show()

plot_histogram(all_distances)