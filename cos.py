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
    def __init__(self, ID, neighbors, descriptor, truth=None, file_path=None):
        self.id = ID
        self.label = ID
        self.neighbors = tuple(neighbors)
        self.descriptor = descriptor
        self.truth = truth
        self.file_path = file_path

    
def create_adj_matrix(descriptors):
    #first create an empty matrix with size [M, M]
    #where M is size of descriptors 
    adj_matrix = np.zeros((len(descriptors), len(descriptors)))

    #access every element of the array
    for row in range(len(descriptors)):
        for col in range(len(descriptors)):
            distance = cos_distances(descriptors[row], descriptors[col]) #This "should" return an np.array with one element
            adj_matrix[row, col] = 1/(distance[0,0])
            
            #add functionality so the exact same images arent read as the same
            if (row == col):
                adj_matrix[row, col] = 0.0
    
    #return the final weighted matrix
    return adj_matrix

def compute_neighbors(descriptor, descriptors, adj_matrix):
    #computing the neighors for the descriptor that you sepecify
    #first we have to find the descriptor that is in the adj_matrix 
    index = 0 
    for i in range(len(descriptors):
        if (descriptor == descriptors[i]):
            index = 




            
