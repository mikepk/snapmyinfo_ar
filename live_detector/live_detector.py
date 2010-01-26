#!/usr/bin/python
#
import sys

#from opencv.cv import *
#from opencv.highgui import *

from ctypes_opencv import *

#from math import sqrt

#import cPickle

#import random
#import math

import threading
import time
from optparse import OptionParser

# from qrcode import qrcode

# import Image, ImageEnhance, ImageFont, ImageDraw, ImageOps

# remember to setup the ssh tunnel first
# from SnapRemoteCard import SnapRemoteCard
# from SnapCommand import SnapCommand

import re


import pygame
# import pygame.camera
from pygame.locals import *

from pygame import gfxdraw, image, Rect, transform
from pygame import surfarray
import numpy

from square_detector import SquareDetector
from GrossTracker import GrossTracker
from TrackedCard import TrackedCard
from TrackerPool import TrackerPool




def main():
    # import os
    # os.environ['SDL_VIDEODRIVER'] = 'windib'
    # os.environ['SDL_VIDEODRIVER'] = 'directx'

    # initialize pygame
    pygame.init()

    parser = OptionParser(usage="""\
    Detect SnapMyinfo QRcodes in a live video stream

    Usage: %prog [options] camera_index
    """)

    opts, args = parser.parse_args()

    pg_size = (1280,720)

    scale = 4.0
    
    names =  [args[0]]
    name = names[0]

    # connect to web camera and set the webcam options
    capture = cvCreateCameraCapture( int(name) )
    cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH, pg_size[0] )
    cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT, pg_size[1] )
    
    throwaway_frame = cvQueryFrame( capture )

    # query one frame to see what the actual resolution turned out to be
    (o_width,o_height) = [cvGetCaptureProperty(capture, prop) for prop in [CV_CAP_PROP_FRAME_WIDTH,CV_CAP_PROP_FRAME_HEIGHT]]
    
    pg_size = (int(o_width),int(o_height))

    worker = GrossTracker(pg_size, scale)
    worker.small_frame = cvCreateImage((int(pg_size[0] / scale + 0.5), int(pg_size[1] / scale + 0.5)), 8, 3)
    # frame = cvCreateImage((int(pg_size[0]),int(pg_size[1])), 8, 3)


    # create the pygame display
    pg_display = pygame.display.set_mode( pg_size, 0) 
    # option for fullscreen HW accelerated
    # pg_display = pygame.display.set_mode( pg_size, pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.FULLSCREEN )

    # create a painting surface
    worker.paint_buffer = pygame.Surface(pg_size).convert()

    # set the painting surface color (0,0,0) or black to transparent
    worker.paint_buffer.set_colorkey((0,0,0))
    
    # set the window name
    pygame.display.set_caption('Live Detector') 

    # some debug information
    # print the current driver being used
    print 'Driver %s  Resolution: %s\n' % (pygame.display.get_driver(), pg_size)
        
    # small = "small window"
    # cvNamedWindow(small, CV_WINDOW_AUTOSIZE)
    
    # pool of tracker objects
    pool = TrackerPool(4)
        
    pyg_clock = pygame.time.Clock()
    update_count = 0
    max_frame_rate = 0
    last_rects = []
    
    running = True
    while running:
        pyg_clock.tick(max_frame_rate)
        if update_count > 100:
            print '''%.3f frames per second''' % pyg_clock.get_fps()
            update_count = 0
        update_count += 1

        # get the pygame events
        events = pygame.event.get()
        for e in events:
            # 'quit' event key
            if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                running = False
            
        # take a frame from the web camera
        worker.frame_buffer = cvQueryFrame( capture )

        # create a small version of the frame
        cvResize(worker.frame_buffer, worker.small_frame)

        # analyze the small frame to find collapsed candidates
        squares = worker.analyze_frame()        

        # check squares and assign a tracker if new
        pool.check(squares)
        # update all trackers
        pool.update()


        # clear the paint buffer
        for rect in last_rects:
            pygame.gfxdraw.rectangle(worker.paint_buffer, rect, Color(0,0,0))
        last_rects = []

        # draw the tracker boundaries
        for t_id in pool.active_trackers:
            rect = pool.trackers[t_id].get_bound_rect()
            rect = pygame.Rect(rect.x * scale, rect.y * scale, rect.width * scale, rect.height * scale)
            pygame.gfxdraw.rectangle(worker.paint_buffer, rect, pool.trackers[t_id].color)
            last_rects.append(rect)

        # draw the orphans
        for orphans in pool.orphan_frames:
            for orphan in orphans:
                orphan = pygame.Rect(orphan.x * scale, orphan.y * scale, orphan.width * scale, orphan.height * scale)
                pygame.gfxdraw.rectangle(worker.paint_buffer, orphan, Color(190,190,190))
                last_rects.append(orphan)

        # no data in the frame buffer means we're done
        if not worker.frame_buffer:
            break

        # the surface RGB values are not the same as the cameras
        # this means that two of the channels have to be swapped in the numpy
        # array before being blit'ted onto the array
        surf_dat = worker.frame_buffer.as_numpy_array().transpose(1,0,2)[...,...,::-1]
        
        # blit_array zaps anything on the surface and completely replaces it with the array, much
        # faster than converting the bufer to a surface and bliting it
        surfarray.blit_array(pg_display,surf_dat)

        # blit the paint buffer onto the surface. With a chromakey, all black values will show through
        # pg_display.blit(pygame.transform.flip(worker.paint_buffer, True, False),(0,0))
        pg_display.blit(worker.paint_buffer,(0,0))

        # flip() actually displays the surface
        pygame.display.flip()
        

    # print worker.stop()
    #cvReleaseVideoWriter(result_vid)
    #cvReleaseMemStorage( storage )
    # cvDestroyWindow(wndname)
    #cvDestroyWindow(adapt)
    # cvDestroyWindow(canny)
    #cvDestroyWindow(test)

    print 'exiting'
    sys.exit(0)
    

if __name__ == "__main__":
    main()
