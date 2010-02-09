#!/usr/bin/python
#
import sys

from ctypes_opencv import *
# 
from math import sqrt
# 
# import cPickle
# 
# import random
# import math
# 
import threading
import time
# from optparse import OptionParser
# 
from qrcode import qrcode
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

from SnapRemoteCard import SnapRemoteCard

from GraphicCard import GraphicCard

# use collection.dequeue for FIFO operation
# http://docs.python.org/library/collections.html#collections.deque

class TrackedCard(object):
    '''An object to track users in the frame.'''

    # have a small buffer of previous tracked bounding squares
    # needs a thread that operates on the object changing the state of the CardUser
    # states are "TooFar", "Scanning", "ID"

    def __init__(self, my_id, buffers, win_size=(200,200), color=Color(255,255,255), track_depth=6):
        print "Starting %d" % my_id

        self.image_buffer = buffers

        self.id = my_id
        self.frames = []
        self.last_frame = None
        self.buffer = None

        self.win_size = win_size        
        self.color = color
        
        self.track_depth = track_depth
        
        self.sd = SquareDetector()
        self.sd.init_image_buffers(self.win_size)
        
        self.idle = True        
        self.user_id = None
        self.sprite = None

        self.state = ''

        self.src = SnapRemoteCard()
        self.gc = GraphicCard()
        
        #self.lock = threading.Lock()
        # self.pool = pool


    def add(self,square):
        '''Add a candidate square to this trackers buffer. Will be updated on the update cycle.'''
        self.buffer = square

    def update(self):
        ''' update the tracker, adding the buffer to the historical data. '''
        if self.buffer:
            self.frames.insert(0,self.buffer)
            self.buffer = None
            self.idle = False
        elif len(self.frames) > 0:
            self.frames.pop()

        if len(self.frames) > self.track_depth:
            self.frames.pop()

        # not tracking anything, we're now idle
        if not self.idle and len(self.frames) == 0:
            self.set_idle()

        # self.processed = False


    def set_idle(self):
        self.user_id = None
        self.idle = True
        self.sprite = None
        self.state = ''
        

    def gen_window(self,square,scale,size):
        return pygame.Rect(square["center"].x * scale  - (size/2), square["center"].y * scale - (size/2), size, size)

    def get_window(self,scale,size):
        '''Get the bounding rectangle'''
        if len(self.frames) > 0:
            return self.gen_window(self.frames[0],scale,size)
        else:
            return pygame.Rect(0,0,0,0)

    def get_bound_rect(self):
        '''Get the scaled bounding rectangle'''
        # print self.frames[0]["bound"]
        
        # compute the bounding rectangle with the x and y scale factors computed separately
        return pygame.Rect(self.frames[0]["bound"][0] * self.image_buffer.display_scale[0],
        self.frames[0]["bound"][1] * self.image_buffer.display_scale[1],
        self.frames[0]["bound"][2] * self.image_buffer.display_scale[0],
        self.frames[0]["bound"][3] * self.image_buffer.display_scale[1])

    def get_bounding_points(self):
        return [(point.x * self.image_buffer.display_scale[0], point.y * self.image_buffer.display_scale[1]) for point in self.frames[0]['points']]

    def get_avg_perimeter(self, count=0):
        '''Find the average perimeter for all matched boundaries.'''
        return sum([frame["perim"] for frame in self.frames ]) / len(self.frames)
        

    def get_avg_center(self, count=2):
        '''find the average center point for the last [count] frames.'''
        f = self.frames[:count]
        x_sum = sum([frame["center"].x for frame in f ])
        y_sum = sum([frame["center"].y for frame in f ])
        
        x_avg = x_sum / len(f)
        y_avg = y_sum / len(f)
        
        return cvPoint(int(x_avg),int(y_avg))
        
    
    def is_similar(self,square):
        '''Check for similarity in perimeter and center position.'''
        # if len(self.frames) < 2:
        #     return True        
        c = square["center"]
        a_c = self.frames[0]["center"]#self.get_avg_center(4)
        
        distance = sqrt((c.y - a_c.y)**2 + (c.x - a_c.x)**2)
        print distance
        # if distance > 10.0:
        #     return False
        # else:
        return True
        # if square["center"]
        # avg_perim = sum([sq["perim"] for sq in self.frames]) / len(self.frames)
        # delta = avg_perim * 0.30
        # if abs(square["perim"] - avg_perim) > delta:
        #     print "Average: %f   This Square: %f    Delta: %f" % (avg_perim, square["perim"], delta)
        #     return False
        # else:
        #     return True
    
    def check(self,square):
        '''Check if this square is part of the tracker pattern.'''
        #check = self.gen_bound_rect(square,self.image_buffer.scale,200) #["bound"] #pygame.Rect(square["center"].x  - 25, square["center"].y  - 25, 50, 50)
        bound_check = square["bound"]
        # window_check = pygame.Rect(*[i * self.image_buffer.scale for i in square["bound"]]) #self.gen_bound_rect(square,self.image_buffer.scale,200)
        # if check.collidelist([f["window"] for f in self.frames]) >= 0:

        if bound_check.collidelist([f["bound"] for f in self.frames]) >= 0:
            #if self.is_similar(square):
            self.add(square)
            return True
        #else:
        #     window_check = pygame.Rect(*[i * self.image_buffer.scale for i in square["bound"]]) #self.gen_bound_rect(square,self.image_buffer.scale,200)
        #     if window_check.collidelist([f["window"] for f in self.frames]) >= 0:
        #         print "."
        #         self.add(square)
        #         return True

        return False


    def constrain_rect(self,rect,image):
        rect.x = max(rect.x, 0)
        rect.y = max(rect.y, 0)
        rect.x = min(rect.x, image.video_size[0] - rect.width)
        rect.y = min(rect.y, image.video_size[1] - rect.height)


    def analyze(self):
        '''Thread to analyze the data in the tracker window.'''
        self.running = True
                
        # hard code for now
        work_buffer = cvCreateImage(self.win_size,8,3) # (int(self.image_buffer.scale * 50),int(self.image_buffer.scale * 50)),8,3)
        win_name = 'thread-%d' % self.id
        
        warped = cvCreateImage(self.win_size,8,3)
        qc = qrcode()
        
        big_size = cvCreateImage((self.win_size[0]*2,self.win_size[1]*2),8,3)

        win_name = "thread-%d" % self.id

        while self.running:
            # update 2 times a second
            time.sleep(0.5)
            # nothing to do, sleep the thread some more
            if self.idle or self.user_id:
                time.sleep(0.1)
                continue

            # extract the working image from the tracking window
            # this creates a rectangle of scale,size based on the current
            # tracking position
            rect = cvRect(*self.get_window(self.image_buffer.scale,200))

            # if the rectangle exits the available pixels, an exception is thrown that
            # kills openCV. So here we make sure the tracking window can't poke outside
            # the available pixels of the image
            self.constrain_rect(rect,self.image_buffer)

            # copy the subrectangle defined by the window + image_buffer into a working
            # buffer
            cvCopy(cvGetSubRect(self.image_buffer.frame_buffer,None,rect),work_buffer)

            # inside the working window, find the squares
            squares = self.sd.find_collapsed_squares(work_buffer)
            
            # if there are no squares (no codes) don't bother trying to decode
            if not squares:
                continue

            # if the code size is too small, we won't be able to decode it
            # 65px on a side seems to be reasonable after testing. Still depends
            # on lighting conditions and other variables.

            # add a "too far away" state to the tracked card.
            # include some hysterisis to keep it from flipping between
            # states
            if self.state == 'too_far':
                if squares[0]["perim"] < 280.0:
                    continue
                else:
                    self.sprite = None
                    self.state = ''
            else:
                if squares[0]["perim"] < 260.0:
                    self.sprite = self.gc.gen_closer_sprite()
                    self.state = 'too_far'
                    continue

            # compute a perspective warp that takes the corners of the found 
            # square quadrilateral, and warps it into a perfect square.
            p_mat, dest = self.sd.compute_perspective_warp(squares[0])

            if not p_mat:
                continue
            
            # apply the persepctive warp to the image and flip it for
            # decoding
            cvWarpPerspective(work_buffer,warped,p_mat)
            cvFlip(warped,warped,1)

            # does this help? Resizing it doesn't add any information, but it seems
            # to help the decoder
            cvResize(warped,big_size,CV_INTER_CUBIC)
            # cvShowImage(win_name,work_buffer)

            # create a pil image to pass to the decoder
            decode_img = ipl_to_pil(big_size)
            data = qc.decode(decode_img)
            if data != "NO BARCODE" and data != "java.lang.IllegalArgumentException Data Error":
                try:
                    card = self.src.get_card(data)
                except Exception, e:
                    print str(e)
                    print "Something went wrong on the server."
                    self.src.socket = None
                    self.src.start()

                self.user_id = card
                self.sprite = self.gc.gen_sprite(card)
             
            
            
            