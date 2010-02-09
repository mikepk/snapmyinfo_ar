#!/usr/bin/python
#
import sys
from ctypes_opencv import *
import pygame
# import pygame.camera
from pygame.locals import *

class ImageBuffer(object):
    '''An object to manage several trackers, run the tracking functions, deal with idle and working states.'''

    # have a small buffer of previous tracked bounding squares
    # needs a thread that operates on the object changing the state of the CardUser
    # states are "TooFar", "Scanning", "ID"

    def __init__(self, video_size=(1280,720), display_size=(1280,720), scale=4.0):
        # An object to hold the image buffers for the application
        self.scale = scale
        self.video_size = video_size
        self.display_size = display_size

        self.display_scale = (self.scale * display_size[0] / video_size[0], self.scale * display_size[1] / video_size[1])

        self.scaled_size = (int(self.video_size[0] / scale + 0.5),int(self.video_size[1] / scale + 0.5))

        self.frame_buffer = None
        self.small_frame = cvCreateImage(self.scaled_size, 8, 3)

        self.paint_buffer = pygame.Surface(self.display_size).convert()
        self.paint_buffer.set_colorkey((0,0,0))


        self.hud_buffer = pygame.Surface(self.display_size).convert_alpha()
        self.hud_buffer.fill((0,0,0,0),pygame.Rect(0,0,*self.display_size))
        #self.hud_buffer.set_alpha(190)
        
    def update(self):
        '''Update the image buffer and scaled image buffer.'''
        # create a small version of the frame
        cvResize(self.frame_buffer, self.small_frame)
        
