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

# import threading
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

# from square_detector import SquareDetector
from GrossTracker import GrossTracker
from TrackedCard import TrackedCard
from TrackerPool import TrackerPool

from ImageBuffer import ImageBuffer
# import Image, ImageDraw, ImageFont

from Connector import Connector
from Status import Status

from DebugDisplay import DebugDisplay

thread_objects = []

def run_detector():
    # import os
    # os.environ['SDL_VIDEODRIVER'] = 'windib'
    # os.environ['SDL_VIDEODRIVER'] = 'directx'


    parser = OptionParser(usage="""\
    Detect SnapMyinfo QRcodes in a live video stream

    Usage: %prog [options] camera_index
    """)

    parser.add_option('-f','--fs','--fullscreen',
                      dest="fullscreen", default=False,
                      action='store_true',
                      help="""Run the Live Decoder full screen.""")

    parser.add_option('--hw','--hw_accel',
                    dest="hw_accel", default=False,
                    action='store_true',
                    help="""Runs pygame with the directx hw driver (if avail). Automatically assumes fullscreen.""")

    parser.add_option('-s','--scale',
                    dest="scale", default=4.0,
                    action='store', type="float",
                    help="""Sets the precision of code tracking. Valid values are >1.0 and less than 8.0. Lower values represent more accurate tracking, but consume more CPU.""")

    parser.add_option('--flip',
                    dest="flip", default=False,
                    action='store_true',
                    help="""Flip the video image horizontally before processing.""")

    parser.add_option('-m','--max_cards',
                    dest="tracker_count", default=3, type="int",
                    action="store",
                    help="""The number of simultaneous snap codes that can be tracked in the video stream.""")


    parser.add_option('--nv','--no_video',
                    dest="no_video", default=False, 
                    action="store_true",
                    help="""A debugging option, turns off the video stream from the web cam.""")

    parser.add_option('-d','--debug',
                    dest="debug", default=False, 
                    action="store_true",
                    help="""Debugging option, turns on fps display and additional tracking data on the display.""")


    opts, args = parser.parse_args()

    import os
    os.environ['SDL_VIDEODRIVER'] = 'windib'
    if opts.hw_accel:
        os.environ['SDL_VIDEODRIVER'] = 'directx'
        opts.fullscreen = True

    # initialize pygame
    pygame.init()

    display_size = (800,600)
    video_size = (1280,720)
    
    # scale is roughly equivalent to tracking precision
    # the higher the scale, the less precise, but faster, the gross
    # tracking will be scale of 1 means 1:1 tracking, but can be slow
    # recommend 2-4 as a good scale/precision factor. higher res images
    # usually benefit from higher scale
    scale = opts.scale
    
    
    # this can be used to throttle the max framerate processed. 0 means no throttle
    max_frame_rate = 30
    
    names =  [args[0]]
    name = names[0]

    if not opts.no_video:
        # connect to web camera and set the webcam options
        capture = cvCreateCameraCapture( int(name) )
        cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH, video_size[0] )
        cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT, video_size[1] )
    
        # query the camera once to get the camera properties
        # the frame is just a throwaway
        cvQueryFrame( capture )

        # get the camera properties
        (o_width,o_height) = [cvGetCaptureProperty(capture, prop) for prop in [CV_CAP_PROP_FRAME_WIDTH,CV_CAP_PROP_FRAME_HEIGHT]]
        video_size = (int(o_width),int(o_height))
    else:
        blank = cvCreateImage(video_size,8,3)
        cvZero(blank)

    # create the pygame display
    flags = 0
    if opts.fullscreen:
        flags = pygame.FULLSCREEN
    if opts.hw_accel:
        flags = flags|pygame.HWSURFACE|pygame.DOUBLEBUF

    display_layer = pygame.display.set_mode( display_size,  flags ) 

    video = pygame.Surface(video_size).convert()

    # set the window name
    pygame.display.set_caption('Live Detector') 

    # some debug information
    # print the current driver being used
    print 'Driver %s\nVideo Input Resolution: %s\nDisplay Resolution: %s\n' % (pygame.display.get_driver(), video_size, display_size)

    # for convert to work, pygame video mode has to be set
    image_buffer = ImageBuffer(video_size,display_size,scale)
    if opts.no_video:
        image_buffer.frame_buffer = cvCreateImage(video_size,8,3)
        # blank = cvCreateImage(video_size,8,3)
        # cvZero(blank)
    worker = GrossTracker(image_buffer) 
    
    # pool of tracker objects
    pool = TrackerPool(image_buffer, opts.tracker_count)
    thread_objects.append(pool)


    status = Status()
    
    connector = Connector(pool,status)
    connector.start()
    thread_objects.append(connector)
    
    # for i in range(4):
    #     win_name = "thread-%d" % i
    #     cvNamedWindow(win_name, CV_WINDOW_AUTOSIZE)
    
    pyg_clock = pygame.time.Clock()
    update_count = 0
    last_rects = []
    last_fills = []
    hud_last_fills = []

    if opts.debug:
        dbg = DebugDisplay()

    snap_logo = pygame.image.load('./images/snap_logo.png').convert_alpha()
    
    still = False
    running = True
    fps = 0.0
    while running:
        pyg_clock.tick(max_frame_rate)
        if update_count > 20:
            fps = pyg_clock.get_fps()
            update_count = 0
        update_count += 1

        # get the pygame events
        events = pygame.event.get()
        for e in events:
            # 'quit' event key
            if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                running = False
            elif e.type == KEYDOWN and e.unicode == u't':
                still = True
            
            
        # take a frame from the web camera
        if opts.no_video:
            cvCopy(blank,image_buffer.frame_buffer)
        else:
            image_buffer.frame_buffer = cvQueryFrame( capture )

        if opts.flip:
            cvFlip(image_buffer.frame_buffer,image_buffer.frame_buffer,1)
            
        # update the image buffer with the latest frame
        image_buffer.update()

        # analyze the small frame to find collapsed candidates
        squares = worker.analyze_frame()        

        # check squares and assign a tracker if new
        pool.check(squares)
        # update all trackers
        pool.update()

        status.update()
        
        # clear the paint buffer
        for rect in last_rects:
            pygame.gfxdraw.rectangle(image_buffer.paint_buffer, rect, Color(0,0,0))
        last_rects = []
        
        for blank in last_fills:
            image_buffer.paint_buffer.fill((0,0,0),blank)
        last_fills = []


        for blank in hud_last_fills:
            image_buffer.hud_buffer.fill((0,0,0,0),blank)
        hud_last_fills = []



        # draw the sprite and tracker boundaries
        # boundaries will be replaced (or turned off)
        pool.sort_active()
        for t_id in pool.active_trackers:
            #rect = pool.trackers[t_id].get_bound_rect()
            center = pool.trackers[t_id].get_avg_center(2)

            frame_color = Color(128,255,128)
            if pool.trackers[t_id].sprite:
                # print str(dir(pool.trackers[t_id].sprite))
                sprite_size = pool.trackers[t_id].sprite.get_size()
                # print str(sprite_size)
                x_diff = sprite_size[0] / 2.0
                y_diff = sprite_size[1] / 2.0
                
                # frame_color = Color(250,250,255)
                
                rect = pygame.Rect(center.x * image_buffer.display_scale[0] - x_diff, center.y * image_buffer.display_scale[1] - y_diff, sprite_size[0],sprite_size[1])
                # pygame.gfxdraw.rectangle(image_buffer.paint_buffer, rect, pool.trackers[t_id].color)
                # last_rects.append(rect)
                #if pool.trackers[t_id].user_id:
                image_buffer.paint_buffer.blit(pool.trackers[t_id].sprite,(rect.x ,rect.y ))
                last_fills.append(rect) #pygame.Rect(rect.x ,rect.y ,closer_img.size[0],closer_img.size[1]))
            else:
                # rect = pygame.Rect(center.x * scale - 100, center.y * scale - 100, 200,200)
                # 
                # 
                # pygame.gfxdraw.rectangle(image_buffer.paint_buffer, rect, frame_color)
                # last_rects.append(rect)

                #c = pygame.Color(164,229,135,150)
                #c1 = pygame.Color(164,229,135,255)
                c = pygame.Color(229,229,135,200)
                # c1 = pygame.Color(229,229,135,255)

                pygame.gfxdraw.filled_polygon(image_buffer.hud_buffer, pool.trackers[t_id].get_bounding_points(),c)
                # pygame.gfxdraw.polygon(image_buffer.hud_buffer, pool.trackers[t_id].get_bounding_points(),c1)
                # pygame.gfxdraw.rectangle(image_buffer.hud_buffer, rect, frame_color)
                # pygame.gfxdraw.filled_polygon(image_buffer.hud_buffer, pool.trackers[t_id].get_bounding_points(),c)
                hud_last_fills.append(pool.trackers[t_id].get_bound_rect())
                


        # draw the orphans and frame rate display
        # debug for now, lets me know when it's trying to lock onto something
        if opts.debug:
            fps_sprite = dbg.gen_sprite('''%.2f fps''' % fps)
            image_buffer.hud_buffer.blit(fps_sprite,(image_buffer.display_size[0]-100,10))
            fps_rect = pygame.Rect(video_size[0]-100,10, dbg.size[0], dbg.size[1])
            hud_last_fills.append(fps_rect)
            
            for orphans in pool.orphan_frames:
                for orphan in orphans:
                    orphan = pygame.Rect(orphan.x * image_buffer.display_scale[0], orphan.y * image_buffer.display_scale[1], orphan.width * image_buffer.display_scale[0], orphan.height * image_buffer.display_scale[1])
                    pygame.gfxdraw.rectangle(image_buffer.paint_buffer, orphan, Color(190,255,190))
                    last_rects.append(orphan)

        # no data in the frame buffer means we're done
        if not image_buffer.frame_buffer:
            break

        # Surf_dat array is not oriented the same way as the pyame display so
        # first it's transposed. Then the
        # the surface RGB values are not the same as the cameras
        # this means that two of the channels have to be swapped in the numpy
        # array before being blit'ted onto the array
        surf_dat = image_buffer.frame_buffer.as_numpy_array().transpose(1,0,2)[...,...,::-1]
        
        # blit_array zaps anything on the surface and completely replaces it with the array, much
        # faster than converting the bufer to a surface and bliting it
        # replaces video with the data in surf_dat
        surfarray.blit_array(video,surf_dat)
        
        # this resizes the video surface and stores it in display_layer. Completely
        # overwrites whatever is in display_layer
        pygame.transform.scale(video, image_buffer.display_size, display_layer)
        
        # blit the paint buffer onto the surface. Paint buffer has a chromakey so all black values will show through
        display_layer.blit(image_buffer.paint_buffer,(0,0))

        # the "HUD" is added next. 
        display_layer.blit(image_buffer.hud_buffer,(0,0))

        # Write out the status sprite to the display_layer
        if status.sprite:
            display_layer.blit(status.sprite,(10,image_buffer.display_size[1] - status.height - 10 ))


        # finally watermark the screen with the company logo and name of the product
        # putting it here means it always shows above the other layers
        logo_size = snap_logo.get_size()
        display_layer.blit(snap_logo,(image_buffer.display_size[0]-logo_size[0]-10 , image_buffer.display_size[1]-logo_size[1]-10))


        if still == True:
            pygame.image.save(display_layer, 'test.jpg')
            still = False

        # flip() actually displays the surface
        pygame.display.flip()

    # we've left the loop
    # exit
    print 'exiting...'
    

def main():
    try:
        run_detector()
    except Exception, e:
        raise
    finally:
        # clean up all dangling threads
        # to avoid locking up the python interpreter
        for obj in thread_objects:
            obj.stop()
    time.sleep(1)
    sys.exit(0)
    

if __name__ == "__main__":
    main()
    