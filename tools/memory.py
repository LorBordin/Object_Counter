import numpy as np

class Memory():
    """ 
    Stores the objects appeared in the previous [length] frames to be compared with the object
    in the current frame. 
    """
    def __init__(self, length=10):
        self.storage = [None]*length
    
    def update(self, objs):
        self.storage[1:] = self.storage[:-1]
        self.storage[0] = objs
    