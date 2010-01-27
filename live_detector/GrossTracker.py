#!/usr/bin/python
#
import sys
from square_detector import SquareDetector


class GrossTracker():
    '''A Class to implement a tracking object. Accesses shared buffer and does some operations on it.'''
    def __init__(self,image_buffer):
        self.ib = image_buffer
        self.sd = SquareDetector()
        self.sd.init_image_buffers(self.ib.scaled_size)
        
    
    def analyze_frame(self):
        '''Analyze the frame for squares.'''
        # get the square regions we think there are QR codes in
        return self.sd.find_collapsed_squares(self.ib.small_frame)
    
    