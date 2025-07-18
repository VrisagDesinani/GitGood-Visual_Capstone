import numpy as np

class Profile:
    
    def __init__(self, name):
        self.name = name
        self.descriptors = np.empty((0, 512))
    
    def add_descriptor(self, descriptor):
        # Shape of descriptors: (N, 512)
        # Shape of each descriptor: (512)
        
        self.descriptors = np.vstack([self.descriptors, descriptor])

    def get_descriptor_average(self):
        # Returns 0s if there are no descriptors yet, and the average of the descriptors if descriptors is not empty
        if len(self.descriptors) == 0:
            return np.zeroes(512)
        return self.descriptors.mean(axis=0)
    
    def get_name(self):
        return self.name 


class Database:    
    
    def __init__(self):
        self.db = {}
        
    def add_profile(self, name, profile):
        #pass in a name and a profile object 
        #every name is mapped to a profile 
        self.db[name] = profile
        
    def remove_profile(self, name):
        #removes the profile object by passing in name 
        self.db.pop(name)
    
    
    def add_image(self, name, descriptor):
        # if name exists 
        if name in self.db:
            self.db[name].add_descriptor(descriptor)
        else:
            profile = Profile(name)
            profile.add_descriptor(descriptor)
            self.add_profile(name, profile)
