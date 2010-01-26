#!/usr/bin/python
#
import sys

# #from opencv.cv import *
# #from opencv.highgui import *
# 
# from ctypes_opencv import *
# 
# from math import sqrt
# 
# import cPickle
# 
# import random
# import math
# 
# import threading
# import time
# from optparse import OptionParser
# 
# from qrcode import qrcode
# 
# import Image, ImageEnhance, ImageFont, ImageDraw, ImageOps
# 
# # remember to setup the ssh tunnel first
# from SnapRemoteCard import SnapRemoteCard
# from SnapCommand import SnapCommand
# 
# import re
# 
# 
# import pygame
# import pygame.camera
# from pygame.locals import *
# 
# from pygame import gfxdraw, image, Rect, transform
# from pygame import surfarray
# import numpy
import pygame
from pygame.locals import *

from square_detector import SquareDetector
from qrcode import qrcode

class TrackedCard(object):
    '''An object to track users in the frame.'''

    # have a small buffer of previous tracked bounding squares
    # needs a thread that operates on the object changing the state of the CardUser
    # states are "TooFar", "Scanning", "ID"

    def __init__(self, my_id):
        print "id! %d" % my_id
        self.id = my_id
        self.frames = []
        self.last_frame = None
        self.buffer = None
        self.idle = True
        self.colors = [Color(0,255,0), Color(0,0,255),Color(255,255,0),Color(255,0,0)]
        self.color = self.colors[self.id]

    def add(self,square):
        self.buffer = square

    def update(self):
        ''' update the tracker, adding the buffer to the historical data. '''
        if self.buffer:
            self.buffer["window"] = pygame.Rect(self.buffer["center"].x  - 25, self.buffer["center"].y  - 25, 50, 50)
            self.frames.insert(0,self.buffer)
            self.buffer = None
            self.idle = False
        elif self.frames:
            self.frames.pop()

        if len(self.frames) > 30:
            self.frames.pop()

        # not tracking anything, we're now idle
        if not self.idle and len(self.frames) == 0:
            self.idle = True


    def get_bound_rect(self):
        if len(self.frames) > 0:
            return pygame.Rect(self.frames[0]["center"].x  - 25, self.frames[0]["center"].y - 25, 50, 50)
        else:
            return pygame.Rect(0,0,0,0)

    def draw(self,surface):
        '''Draw the state / representation of the tracker.'''

        return None

        if len(self.frames) > 1:
        #if self.last_frame:
            for i in range(0,len(self.frames)):
                rm_sq = self.frames[i]
                #self.last_frame = None
                pyg_r = pygame.Rect(rm_sq["center"].x - 25, rm_sq["center"].y - 25, 50, 50)
                pygame.gfxdraw.rectangle(surface, pyg_r, Color(255,0,0))

                # if self.frames[0]:
                sq = self.frames[0]
                pyg_r = pygame.Rect(sq["center"].x  - 25, sq["center"].y  - 25, 50, 50)
                pygame.gfxdraw.rectangle(surface, pyg_r, self.colors[self.id])

        elif len(self.frames) == 1:
            rm_sq = self.frames[0]
            #self.last_frame = None
            pyg_r = pygame.Rect(rm_sq["center"].x  - 25, rm_sq["center"].y  - 25, 50, 50)
            pygame.gfxdraw.rectangle(surface, pyg_r, Color(128,0,0))
            
            
    
    def check(self,square):
        '''Check if this square is part of the tracker pattern.'''
        check = pygame.Rect(square["center"].x  - 25, square["center"].y  - 25, 50, 50)
        if check.collidelist([f["window"] for f in self.frames]) >= 0:
            self.add(square)
            return True
        else:
            return False



