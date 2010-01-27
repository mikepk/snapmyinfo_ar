#!/usr/bin/python
#
import sys

from ctypes_opencv import *
# 
# from math import sqrt
# 
# import cPickle
# 
# import random
# import math
# 
# import threading
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

class TrackedCard(object):
    '''An object to track users in the frame.'''

    # have a small buffer of previous tracked bounding squares
    # needs a thread that operates on the object changing the state of the CardUser
    # states are "TooFar", "Scanning", "ID"

    def __init__(self, my_id, buffers):
        print "id! %d" % my_id

        self.image_buffer = buffers

        self.id = my_id
        self.frames = []
        self.last_frame = None
        self.buffer = None

        self.colors = [Color(0,255,0), Color(0,0,255),Color(255,255,0),Color(255,0,0)]
        self.color = self.colors[self.id]
        
        self.sd = SquareDetector()
        self.sd.init_image_buffers((int(50 * self.image_buffer.scale + 0.5),int(50 * self.image_buffer.scale + 0.5)))
        
        self.idle = True        
        self.user_id = None
        
        # self.pool = pool


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
            self.user_id = None
            self.idle = True
        
        self.processed = False


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
            
            
    def get_avg_center(self, count=2):
        '''find the average center point for the last several frames.'''
        f = self.frames[:count]
        x_sum = sum([frame["center"].x for frame in f ])
        y_sum = sum([frame["center"].y for frame in f ])
        
        x_avg = x_sum / len(f)
        y_avg = y_sum / len(f)
        
        return cvPoint(int(x_avg),int(y_avg))
        
    
    def check(self,square):
        '''Check if this square is part of the tracker pattern.'''
        check = pygame.Rect(square["center"].x  - 25, square["center"].y  - 25, 50, 50)
        if check.collidelist([f["window"] for f in self.frames]) >= 0:
            self.add(square)
            return True
        else:
            return False


    def analyze(self):
        '''Thread to analyze the data in the tracker window.'''
        self.running = True
        # hard code for now
        work_buffer = cvCreateImage((int(self.image_buffer.scale * 50),int(self.image_buffer.scale * 50)),8,3)
        win_name = 'thread-%d' % self.id
        
        warped = cvCreateImage((int(self.image_buffer.scale * 50),int(self.image_buffer.scale * 50)),8,3)
        qc = qrcode()
        
        big_size = cvCreateImage((int(self.image_buffer.scale * 50) * 3,int(self.image_buffer.scale * 50) * 3),8,3)

        while self.running:
            # update 2 times a second
            time.sleep(0.5)
            # nothing to do, sleep the thread some more
            if self.idle or self.user_id:
                time.sleep(0.1)
                continue

            # extract the working image from the tracking window
            #print str(tuple(*self.get_bound_rect()))
            # args = []
            # for i in self.get_bound_rect():
            #     args.append(int(i*2.0 + 0.5))
            rect = cvRect(*[int(i*self.image_buffer.scale + 0.5) for i in self.get_bound_rect()])
            #print str(rect)
            # # rect = Rect(rect.x * scale, rect.y * scale, rect.width * scale, rect.height * scale)
            # work_buffer = cvGetSubRect(self.image_buffer.frame_buffer,None,rect)
            cvCopy(cvGetSubRect(self.image_buffer.frame_buffer,None,rect),work_buffer)
            #square = self.sd.get_collapsed_square
            squares = self.sd.find_collapsed_squares(work_buffer)
            
            # if there are no squares (no codes) don't bother trying to decode
            if not squares:
                continue

            # if the code size is too small, we won't be able to decode it
            if squares[0]["perim"] < 276.0:
                print "too small"
                continue

            p_mat, dest = self.sd.compute_perspective_warp(squares[0])

            if not p_mat:
                continue

            cvWarpPerspective(work_buffer,warped,p_mat)
            cvFlip(warped,warped,1)

            # clip = cvRect(dest[0].x,dest[0].y,dest[2].x-dest[0].x,dest[2].y-dest[0].y)
            # chop = cvGetSubRect(warped,None,clip)

            cvResize(warped,big_size)

            # cvShowImage(win_name,big_size)

            decode_img = ipl_to_pil(big_size)
            data = qc.decode(decode_img)
            print data
            if data != "NO BARCODE":
                self.user_id = data

            #     print data

                
            # print str(square)
            # cvShowImage(win_name,self.image_buffer.frame_buffer)
            # cvShowImage(win_name,work_buffer)
            # print rect

            
            
            
            