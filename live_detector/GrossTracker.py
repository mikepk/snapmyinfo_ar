#!/usr/bin/python
#
import sys
from square_detector import SquareDetector


class GrossTracker():
    '''A Class to implement a tracking object. Accesses shared buffer and does some operations on it.'''
    def __init__(self,size=(1280,720), scale=4.0):
        
        self.sd = SquareDetector()

        self.sd.init_image_buffers((int(size[0]/scale + 0.5),int(size[1]/scale + 0.5)))
        
    
    def analyze_frame(self):
        '''Analyze the frame for squares.'''
        # get the square regions we think there are QR codes in
        return self.sd.find_collapsed_squares(self.small_frame)
    
    