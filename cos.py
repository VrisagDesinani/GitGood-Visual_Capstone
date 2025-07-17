# Function to measure cosine distance between face descriptors. 
import numpy as np

Threshold = 1.5 

def cos_distances(descriptorM: np.ndarray, 
                  descriptorN: np.ndarray) -> np.ndarray:

    #do we have to check inputs are 2d arrays
    #normalize vectors
    descriptorM_norm = descriptorM/np.linalg.norm(descriptorM, axis = 1)
    descriptorN_norm = descriptorN/np.linalg.norm(descriptorN, axis = 1)

    #find similarity using dot product 
    cos_distances = 1.0 - np.dot(descriptorM_norm, descriptorN_norm.T)

    return cos_distances

class Node:
    def __init__(self, ID, neighbors, descriptor, ):
        self.id = ID
