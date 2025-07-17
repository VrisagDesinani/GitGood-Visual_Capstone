import numpy as np

class Profile:
    
    def __init__(self, name):
        self.name = name
        self.descriptors = np.empty()
    
    def add_descriptor(self, descriptor):
        # Shape of each descriptor should be (512,)
        # We want descriptors to be shape (N, 512)
        
        self.descriptors = np.vstack([self.descriptors, descriptor])

    def get_descriptor_average(self):
        # our descriptors is a np array already
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
        del self.db.pop(name)
    
    
    def add_image(self, name, descriptor):
        # if name exists 
        if name in self.db:
            self.db[name].add_descriptor(descriptor)
        else:
            profile = Profile(name)
            profile.add_descriptor(descriptor)
            self.add_profile(name, profile)
        
