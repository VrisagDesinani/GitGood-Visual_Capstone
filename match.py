import numpy as np
from cos import cos_distances

def has_match(descriptor, database, threshold):
    '''
    Functionality to see if a new descriptor has a match in your database, given the aforementioned cutoff threshold.

    Inputs: descriptor (shape (1, D) array), database, threshold
    Outputs: return a string of the profile

    '''
    means = np.array([profile.get_descriptor_average() for profile in database.values()])    #shape (N,D)
    keys = list(database.keys())

    distances = cos_distances(descriptor, means)    #shape (1,N)
    best_match_index = np.argmin(distances) #index out of N
    best_match_distance = distances[0, best_match_index]

    if best_match_distance < threshold:
        return keys[best_match_index]
    else:
        return "Unknown"