#!/usr/bin/python
#
import sys

#from opencv.cv import *
#from opencv.highgui import *

from ctypes_opencv import *

from math import sqrt

import cPickle

import random
import math

import threading
import time
from optparse import OptionParser

from qrcode import qrcode

import Image, ImageEnhance, ImageFont, ImageDraw, ImageOps

# remember to setup the ssh tunnel first
from SnapRemoteCard import SnapRemoteCard
from SnapCommand import SnapCommand

import re


import pygame
import pygame.camera
from pygame.locals import *

from pygame import gfxdraw, image, Rect, transform
from pygame import surfarray
import numpy


from square_detector import SquareDetector

pygame.init()
# pygame.camera.init()

# camlist = pygame.camera.list_cameras()
# if camlist:
#     cam = pygame.caemra.Camera(camlist[0],(640,480))
# 
# sys.exit(0)

wndname = "Live Detector"
adapt = "Threshold"
canny = "Canny"
test = "Test Window"
lk = "Optical Flow"
zone1 = "Left User Card"
zone2 = "Right User Card"
paint_name = "Paint Display"
sub_win = "Sub Window"
small_win = "Half Size"

def print_timing(func):
    def wrapper(*arg):
        t1 = time.time()
        res = func(*arg)
        t2 = time.time()
        print '%s took %0.3f ms, roughly %0.3f fps' % (func.func_name, (t2-t1)*1000.0, 1.0/((t2-t1)+0.001))
        return res
    return wrapper



class Code():
    '''Object to represent the model of a QRcode in the display'''

    def __init__(self):
        self.sample_window = []
        
    def scan_samples(self,samples):
        '''Check all square samples, collect ones that may be relevant to this code.'''
        pass



class Circle():
    '''a circle object in the display'''
    def __init__(self,x,y,r):
        self.x = x
        self.y = y
        self.r = r

class Square():
    '''a circle object in the display'''
    def __init__(self,center,perim,points):
        self.center = center
        self.perim = perim
        self.points = points



class WorkerTest():
    '''A Class to implement a worker object. Accesses shared buffer and does some operations on it.'''
    def __init__(self,size=(1280,720)):
        self.frame_buffer = None
        self.paint_buffer = None

        self.small_frame = None
        self.oflow_test = []

        # self.roi_buffer = []
        # self.of_points_buffer = []

        self.running = True
        # self.adapt_history = []
        self.lock = threading.Lock()
        self.zonecolor = {}
        self.decoded = {}
        self.decode_window = {}

        self.visible_user = {}

        self.display_scale = 1.0
        # self.src = SnapRemoteCard()
        
        
        self.thresh_bot = 230
        self.thresh_top = 340        
        
        self.avg_mode = False
        
        self.thresh_samples = []
        self.last_gray = None
        
        self.bounds = [100,100,300,300]
        
        self.sd = SquareDetector()

        self.sd.init_image_buffers((int(size[0]/4.0 + 0.5),int(size[1]/4.0 + 0.5)))
        
    def stop(self):
        '''Kill the worker threads?'''
        self.running = False
        time.sleep(1)
        return "Done!"    
    
    def connect(self):
        com = SnapCommand()
        while self.running:
            self.lock.acquire()
            users = [self.visible_user[key] for key in self.visible_user.keys()]
            self.lock.release()
            if len(users) > 1:
                # print str(users)
                if self.src.connect(users):
                    # take a photo of the connectee's
                    # if self.frame_buffer:
                    #     # flip the image first
                    
                    save_img = ipl_to_pil(cvCloneImage(self.frame_buffer))
                    users_data = [str(user_id) for user_id in users]
                    user_string = '_'.join(users_data)
                    save_img.save('''%s_connection.jpg''' % (user_string))
                    # cvSaveImage('connection.jpg',self.frame_buffer)
                    print "CONNECTED!"
                else:
                    print "Not Connected"

                for key in self.visible_user.keys():
                    del self.visible_user[key]
                # print str(users)
            time.sleep(0.25)
    
    def detect(self, window_name):
        '''Detect and decode barcodes in the image. If rect is passed, only process that subregion of the image.'''
        # self.zonecolor[window_name] = CV_RGB(255,255,255)
        # self.decoded[window_name] = False
        # self.decode_window[window_name] = 0

        # qc = qrcode()
        # storage = cvCreateMemStorage(0)

        # half_height = int(rect.height / 2.0)
        # half_width = int(rect.width / 2.0)
        # top_left = cvRect(rect.x,rect.y,half_width,half_height)
        # top_right = cvRect(half_width,rect.y,half_width,half_height)
        # 
        # bot_left = cvRect(rect.x,half_height,half_width,half_height)
        # bot_right = cvRect(half_width,half_height,half_width,half_height)


        # cpy = cvCloneImage(self.frame_buffer)
        orig_size = cvGetSize(self.frame_buffer)
        small_size = cvGetSize(self.small_frame)

        # rects = [rect]        
        last_frame = []
        last_rect = None
        last_gx = None
        last_squares = []
        
        
        warped = cvCreateImage( orig_size, 8, 3)
        cpy = cvCreateImage( small_size, 8, 3)

        large_code = cvCreateImage( (290,290), 8, 3)

        while self.running:
            # sleep to avoid monopolizing CPU
            # time.sleep(0.5)

            # cpy = cvCloneImage(self.small_frame)
            
            cvCopy(self.small_frame,cpy)
            cvCopy(self.frame_buffer,warped)
            # orig_copy = cvCloneImage(self.frame_buffer)
            
            if last_rect:
                gfxdraw.rectangle(self.paint_buffer, last_rect, Color(255,255,0))

            squares = self.sd.find_collapsed_squares(cpy)
            
            # if len(squares) >= 3:
            #     print "====================="
            #     for sq in squares:
            #         print sq["perim"]
            #         print sq["bound"]
            #         print sq["points"]
            #         print "--------"
            #         
            #     # print str(squares)
            
            # cvShowImage(adapt,self.sd.threshold_buffer)

            if squares:


                if last_squares:
                    for sq in last_squares:
                        # pyg_r = pygame.Rect(sq["bound"].x * 4.0, sq["bound"].y * 4.0, sq["bound"].width * 4.0, sq["bound"].height * 4.0)
                        # gx = [(pt.x *4.0, pt.y  * 4.0) for pt in sq["points"]]
                        gfxdraw.rectangle(self.paint_buffer, sq, Color(0,0,0))
                    last_squares = []
                    

                for sq in squares:
                    pyg_r = pygame.Rect(sq["center"].x * 4.0 - 100, sq["center"].y * 4.0 - 100, 200, 200)
                    # print str(pyg_r)
                    # gx = [(pt.x *4.0, pt.y  * 4.0) for pt in sq["points"]]
                    gfxdraw.rectangle(self.paint_buffer, pyg_r, Color(0,255,0))
                    last_squares.append(pyg_r)
                # last_squares = squares[:]
                
                bounds = None
                # bounds = self.sd.get_bounding_box(squares,4.0,0.15)
                # print str(bounds)
                
                if bounds and bounds[0] > 0 and bounds[1] > 0 and bounds[0]+bounds[2] < orig_size.width and bounds[1]+bounds[3] < orig_size.height:

                    if last_rect:
                        gfxdraw.rectangle(self.paint_buffer, last_rect, Color(0,0,0))
                        last_rect = None

                    draw_rect = pygame.Rect(bounds[0],bounds[1],bounds[2],bounds[3])
                    last_rect = draw_rect
                    gfxdraw.rectangle(self.paint_buffer, draw_rect, Color("red"))
                    # gfxdraw.circle(self.paint_buffer, bounds[0]+100,bounds[1]+100,5, Color(0,128,200))

                    
                    # # if p_mat:
                    # #     cvWarpPerspective(small,warped,p_mat)
                    # 
                    # small = cvGetSubRect(warped, None, bounds)
                    # # cvShowImage(sub_win,small)
                    # 
                    # 
                    # small_size = (200,200) #cvGetSize(small)
                    # sq2 = SquareDetector()
                    # sq2.init_image_buffers(small_size)
                    # 
                    # 
                    # b_squares = sq2.find_squares(small)
                    # # # cvShowImage(adapt,sq2.threshold_buffer)
                    # # for sq in b_squares:
                    # #     gx = [(pt.x + bounds[0], pt.y  + bounds[1]) for pt in sq["points"]]
                    # #     gfxdraw.polygon(self.paint_buffer, gx, Color(255,255,0))
                    # # last_gx = b_squares[:]
                    #     
                    # hires_square = sq2.get_max_square(b_squares)
                    #             
                    # 
                    # if hires_square:
                    #     # other_points = [(pt.x,pt.y) for pt in hires_square["points"]]
                    #     # pygame.gfxdraw.polygon(self.paint_buffer, other_points, (255,0,255))
                    #     p_mat, dest = sq2.compute_perspective_warp(hires_square)
                    #     new_warped = cvCreateImage(small_size, 8, 3)
                    #     if p_mat:
                    #         cvWarpPerspective(small,new_warped,p_mat)
                    #         
                    #         # cut_out = (dest[0].x,dest[0].y,dest[2].x-dest[0].x,dest[2].y-dest[0].y)
                    #         # clipped = cvGetSubRect(new_warped, None, cut_out)
                    #         #                     
                    #         # cvResize(clipped,large_code)
                    #         cvShowImage(sub_win,new_warped)

            # #if self.frame_buffer:
            # rect = rects[rect_idx]
            # rect_idx += 1
            # if rect_idx > len(rects)-1:
            #     rect_idx=0

            # scale = 4.0
            # # if orig_size.width > 640:
            # #     small_image =  cvCreateImage( cvSize(orig_size.width / 2, orig_size.height/2), 8, 3 )
            # #     cvPyrDown( cpy, small_image, CV_GAUSSIAN_5x5 );
            # #     cpy = small_image
            # #     scale = 2.0

            # cpy = cvGetSubRect(cpy, None, rect)
            # cvFlip(cpy,cpy,1)

            # new_width = 960.0
            # scale = new_width / self.frame_buffer.width
            # small_rect = cvRect(int(rect.x*scale),int(rect.y*scale),int(rect.height*scale),int(rect.width*scale))
            # cpy = cvCreateImage(cvSize(int(new_width),int(self.frame_buffer.height * scale)),8,3)
            # cvResize(self.frame_buffer,cpy)
            # cpy = cvGetSubRect(cpy, None, small_rect)
            
            
            
            # #cvSetImageROI(cpy,rect)
            # # work = cvGetSubRect(cpy, None, rect)
            # # work = cvCloneImage(self.frame_buffer)
            # if not self.frame_buffer:
            #     self.running = False
            #     break
            # cnt = self.find_contours(cpy, storage)
            # 
            # # has_code, bounds = self.draw_contours(cpy, cnt, scale)
            # bounds, p_mat = self.draw_contours(cpy, cnt, scale)
            # # bounds = None
            # has_code = False
            # 
            # 
            # if bounds:
            #     # pass
            #     self.lock.acquire()
            #     self.bounds = bounds
            #     self.lock.release()
            #     # if bounds[2] > 25 and bounds[3] > 25:
            #     #     self.need_to_init = True
            #     if last_rect:
            #         gfxdraw.rectangle(self.paint_buffer, last_rect, (0,0,0))                    
            #     draw_rect = pygame.Rect(bounds[0],bounds[1],bounds[2],bounds[3])
            #     
            #     # gfxdraw.rectangle(self.paint_buffer, draw_rect, Color("red"))
            #     # last_rect = draw_rect
            # 
            # 
            # if has_code and self.decoded[window_name]:
            #     self.decode_window[window_name] = 0
            # elif has_code and not self.decoded[window_name]:
            #     self.zonecolor[window_name] = CV_RGB(255,255,0)
            #     # draw.text((int(self.display_scale*rect.x), int(self.display_scale * rect.y-20)), "Scanning...", font=screen_font, fill=(200,200,0,0))
            #     
            # else:
            #     if self.decode_window[window_name] > 6:
            #         # draw.rectangle([int(self.display_scale*rect.x-100), int(self.display_scale * rect.y-100), int(self.display_scale * rect.x+100+rect.width), int(self.display_scale * rect.y+100+rect.height)], fill=(0,0,0,255))
            #         self.zonecolor[window_name] = CV_RGB(255,255,255)
            #         self.decode_window[window_name] = 0
            #         self.decoded[window_name] = False
            #         try:
            #             if self.visible_user[window_name]:
            #                 self.lock.acquire()
            #                 del self.visible_user[window_name]
            #                 self.lock.release()
            #         except KeyError:
            #             pass
            #             
            #     else:
            #         self.decode_window[window_name] += 1
            # 
            #         # cpy = cvGetSubRect(cpy, None, rect)
            #         # cvFlip(cpy,cpy,1)
            # 
            # if last_rect:
            #     gfxdraw.rectangle(self.paint_buffer, last_rect, (0,0,0))
            #     last_rect = None
            # 
            # if p_mat:
            #     cvWarpPerspective(cpy,warped,p_mat)
            # 
            # # cvShowImage(adapt,warped)
            # 
            # out_of_bounds = False
            # if bounds:
            #     if bounds[0] < 0 or bounds[1] < 0:
            #         out_of_bounds = True
            #     if bounds[0] + bounds[2] > warped.width:
            #         out_of_bounds = True
            #     if bounds[1] + bounds[3] > warped.height:
            #         out_of_bounds = True
            # 
            # 
            # if bounds and not out_of_bounds:
            #     work = cvGetImage(cvGetSubRect(warped, None, bounds)) # cvGetImage(cpy)
            # 
            #     # try:
            #     #     self.last_frame
            #     # except AttributeError:
            #     #     self.last_frame = None
            #     #     self.second_to_last_frame = None
            #     # 
            #     # if self.last_frame:
            #     #     cvAdd(work,self.last_frame,work)
            #     # self.last_frame = work
            #     sz = cvSize( work.width, work.height )
            # 
            # 
            #     # # create grayscale version of the image
            #     # gray = cvCreateImage( sz, 8, 1 )
            #     # cvCvtColor(work ,gray, CV_RGB2GRAY)
            #     # 
            #     # # equalize the grayscale
            #     # cvEqualizeHist( gray, gray );
            #     # 
            #     # cvThreshold(gray,gray,128,255,CV_THRESH_BINARY) 
            #  
            # 
            #     wd = 200
            #     scale = (wd / 1.0) / sz.width
            #     ht = int(sz.height * scale + 0.5)
            # 
            #     big_work = cvCreateImage((wd,ht),8,3)
            #     # new_work = cvCreateImage((wd,ht),8,3)
            #     cvResize(work,big_work,CV_INTER_CUBIC)
            #     cvFlip(big_work,big_work,1)
            #     
            #     if last_frame and self.avg_mode:
            #         # cvAdd(big_work,last_frame,big_work)
            #         cvZero(new_work) # = cvCloneImage(big_work)
            #         for i in range(len(last_frame)):
            #             cvAddWeighted(last_frame[i],1.0/(len(last_frame)),new_work,1.0,0,new_work)
            #     last_frame.append(big_work)
            #     last_frame = last_frame[-5:]
            # 
            #     if not self.avg_mode:
            #         new_work = cvCloneImage(big_work)
            #         # new_work = cvCreateImage((wd,ht),8,1)
            #         # cvCvtColor(new_work, big_work, CV_GRAY2RGB)
            # 
            #     cvShowImage(sub_win,new_work)
            #     
            #         
            #     # Turn the image to a PIL image and sharpen for decode
            #     decode_img = ipl_to_pil(new_work)
            # 
            #     # # decode_img = ImageOps.equalize(decode_img)
            #     # decode_img = ImageEnhance.Sharpness(decode_img).enhance(1.5)
            #     # decode_img = ImageOps.autocontrast(decode_img)
            #     # 
            #     # # 
            #     # 
            #     # # ImageEnhance.Contrast(            
            #     # # ImageEnhance.Brightness(decode_img).enhance(1.2)   
            #     # # ).enhance(0.75)
            #     # 
            #     # cvShowImage(sub_win,pil_to_ipl(decode_img))
            #     
            #     # data = qc.decode(decode_img)
            #     # if data != "NO BARCODE":
            #     #     color = (0,255,0)
            #     #     # print data
            #     # else:
            #     #     color = (255,0,0)
            #     color = (199,199,199)
            #             
            #     # pass
            #     self.lock.acquire()
            #     self.bounds = bounds
            #     self.lock.release()
            #     # if bounds[2] > 25 and bounds[3] > 25:
            #     #     self.need_to_init = True
            #     draw_rect = pygame.Rect(bounds[0],bounds[1],bounds[2],bounds[3])
            #     gfxdraw.rectangle(self.paint_buffer, draw_rect, color)
            #     last_rect = draw_rect
            #     
            # 
            # 
            # # if has_code and not self.decoded[window_name]:
            # #     data = qc.decode(decode_img)
            # #     if data != "NO BARCODE":
            # #         # code decoded
            # #         self.zonecolor[window_name] = CV_RGB(0,255,0)
            # #         self.decoded[window_name] = True
            # #         self.decode_window[window_name] = 0
            # # 
            # #         match = re.search(r'\/([^\/]*)$',data)
            # #         if match:
            # #             code = match.group(1)
            # #             #print str(code)
            # #             user_card = self.src.get_card(code)
            # #             if user_card:
            # #                 # draw.text((int(self.display_scale*rect.x), int(self.display_scale * rect.y-40)), user_card['full_name'], font=data_font, fill=(0,200,0,0))
            # #                 print '''%s decoded %s''' % (window_name,user_card['full_name'])
            # #                 self.lock.acquire()
            # #                 self.visible_user[window_name] = user_card['user_id']
            # #                 self.lock.release()
            # #             else:
            # #                 self.zonecolor[window_name] = CV_RGB(180,0,0)
            # #                 print '''%s : %s''' % (window_name,'No card found')
            # #         else:
            # #             self.zonecolor[window_name] = CV_RGB(180,0,0)
            # #             print '''QR code decoded, is it not a SnapMyInfo code?'''
            # #     else:
            # #         pass
            # #         # print '''%s failed to decode''' % (window_name)
            #         
            # # cvShowImage(window_name, pil_to_ipl(decode_img))
            # cvClearMemStorage( storage )
            # # (window_name, work) #pil_to_ipl(decode_img))

    def oflow_points(self):
        '''Adding the lkdemo code, to track optical flow points'''
        self.need_to_init = False
        # self.of_points = [[], []]

        points_to_find = []
        found_points = []

        night_mode = False
        image = None
        pt = None
        add_remove_pt = False
        flags = 0
        
        mask = None
        
        win_size = 10
        MAX_COUNT = 50
        small_bounds = [int(point / 2.0) for point in self.bounds]
        small_rect = cvRect(*small_bounds)
        
        while self.running:
            # time.sleep(0.25)
            frame = cvCloneImage(self.small_frame)
            
            # # frame = down_sample(frame,1)
            # # # new_width = 960.0
            # # # scale = new_width / self.frame_buffer.width
            # # if not small_rect:
            # self.lock.acquire()
            # small_rect = cvRect(*self.bounds)
            # self.lock.release()
            # # if not mask:
            # #      size = cvGetSize(frame)
            # #      mask = cvCreateImage (cvSize(size.width,size.height), 8, 1)
            # # cvSetZero(mask)
            # self.lock.acquire()
            # # 
            # small_bounds = [int(point / 2.0) for point in self.bounds]
            # # cvFillPoly(mask,[[
            # # cvPoint(small_bounds[0],small_bounds[1]),
            # # cvPoint(small_bounds[0]+small_bounds[2],small_bounds[1]),
            # # cvPoint(small_bounds[0]+small_bounds[2],small_bounds[1]+small_bounds[3]),
            # # cvPoint(small_bounds[0],small_bounds[1]+small_bounds[3])
            # # ]],CV_RGB(255,255,255))
            # 
            # 
            # # cvCvtColor (mymask, mask, CV_BGR2GRAY)
            # # # cvShowImage(adapt,mask)
            # # #small_rect = cvRect(*self.bounds) #int(200),int(200),int(100),int(100))
            # self.lock.release()
            # mask = cvGetArray(mask)

            # 
            # # cpy = cvCreateImage(cvSize(int(new_width),int(self.frame_buffer.height * scale)),8,3)
            # # cvResize(self.frame_buffer,cpy)
            # cvSetImageROI(frame,small_rect)
            # print str(small_rect)
            # cpy = cvGetSubRect(frame, None, small_rect)
            # frame = cvGetImage(cpy)
            # frame = cpy
            # frame = down_sample(self.frame_buffer,1)

            if image is None:
                # create the images we need
                sz = cvGetSize (self.small_frame)
                image = cvCreateImage (sz, 8, 3)
                image.origin = 0
                grey = cvCreateImage (sz, 8, 1)
                prev_grey = cvCreateImage (sz, 8, 1)
                pyramid = cvCreateImage (sz, 8, 1)
                prev_pyramid = cvCreateImage (sz, 8, 1)
                # self.of_points = [[], []]

            # copy the frame, so we can draw on it
            # if frame.origin:
            #     cvFlip(frame, image)
            # else:
            cvCopy (self.small_frame, image)

            # create a grey version of the image
            cvCvtColor (image, grey, CV_BGR2GRAY)

            if night_mode:
                # night mode: only display the points
                cvSetZero (image)


            # cvSetImageROI(grey,small_rect)
            # cvResetImageROI(grey)

            if self.need_to_init:
                # we want to search all the good points
                # self.of_points[1] 
                found_points = []
                for point in self.oflow_test:
                    found_points.append (cvPointTo32f (point))

                    # refine the corner locations
                    # self.of_points [1][-1] = cvFindCornerSubPix (
                    found_points[-1] = cvFindCornerSubPix (
                        grey,
                        [found_points[-1]],
                        cvSize (win_size, win_size), cvSize (-1, -1),
                        cvTermCriteria (CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,
                                           20, 0.03))[0]
                    
                # mask = None
                # found_points = cvGoodFeaturesToTrack(grey, None, None, None, MAX_COUNT, 0.01, 10, mask)
                # 
                # # refine the corner locations
                # cvFindCornerSubPix (
                #     grey,
                #     found_points,
                #     cvSize (win_size, win_size), cvSize (-1, -1),
                #     cvTermCriteria (CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,
                #                        20, 0.03))
            # elif len (self.of_points [0]) > 0:
            elif len (points_to_find) > 0:
                # we have points, so display them

                # calculate the optical flow
                found_points, status = cvCalcOpticalFlowPyrLK (
                    prev_grey, grey, prev_pyramid, pyramid,
                    points_to_find, None, None, 
                    cvSize (win_size, win_size), 3,
                    None, None,
                    cvTermCriteria (CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,
                                       20, 0.03),
                    flags)


                # initializations
                point_counter = -1
                new_points = []
            
                for the_point in found_points:
                    # go trough all the points

                    # increment the counter
                    point_counter += 1
                
                    if add_remove_pt:
                        # we have a point to add, so see if it is close to
                        # another one. If yes, don't use it
                        dx = pt.x - the_point.x
                        dy = pt.y - the_point.y
                        if dx * dx + dy * dy <= 25:
                            # too close
                            add_remove_pt = 0
                            continue

                    if not status [point_counter]:
                        # we will disable this point
                        continue

                    # this point is a correct point
                    new_points.append (the_point)
                

                
                    # draw the current point
                    cvCircle (image, cvPointFrom32f(the_point),
                                 1, cvScalar (0, 255, 0, 0),
                                 -1, 8, 0)

                
                # # print str(range(len(found_points)))
                # if len (points_to_find) > 0:
                #     for p in range(len(found_points)):
                #         # print str(points_to_find[p])
                #         # print str(found_points[p])
                #         if points_to_find[p] and found_points[p]:
                #             try:
                #                 cvLine(image, cvPoint(points_to_find[p].x, points_to_find[p].y), cvPoint(found_points[p].x, found_points[p].y), CV_RGB(255,0,0))
                #             except:
                #                 print 'to: %s   next: %s' % (str(points_to_find[p]),str(found_points[p]))
                                
                                
                # set back the points we keep
                # self.of_points [1] = new_points
                found_points = new_points
        
            if add_remove_pt:
                # we want to add a point
                found_points.append (cvPointTo32f (pt))

                # refine the corner locations
                # self.of_points [1][-1] = cvFindCornerSubPix (
                found_points[-1] = cvFindCornerSubPix (
                    grey,
                    [found_points[-1]],
                    cvSize (win_size, win_size), cvSize (-1, -1),
                    cvTermCriteria (CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,
                                       20, 0.03))[0]

                # we are no more in "add_remove_pt" mode
                add_remove_pt = False

            # swapping
            # prev_grey = grey
            # prev_pyramid = pyramid

            # swapping the buffers
            # rather than copying
            prev_grey, grey = grey, prev_grey
            prev_pyramid, pyramid = pyramid, prev_pyramid

            # look for the found points in the next frame
            points_to_find = found_points
            # points_to_find, found_points = found_points, points_to_find
            self.need_to_init = False
        
            # we can now display the image
            cvShowImage (lk, image)


    def angle(self, pt0, pt1, pt2 ):
        '''Measure the angle between three points.'''
        dx1 = pt1.x - pt0.x;
        dy1 = pt1.y - pt0.y;
        dx2 = pt2.x - pt1.x;
        dy2 = pt2.y - pt1.y;

        # determine the slopes of the two lines created
        m1 = dy1 / (dx1 + 1e-10)
        m2 = dy2 / (dx2 + 1e-10)

        try:
            return (abs(math.atan( ((m2 - m1) + 1e-10)  / (1.0 + m1 * m2)) ))
        # sometimes the slopes are the same or roughly the same (maybe different signs)
        # in this case the angle between them is 0 and the calculation blows up
        except ZeroDivisionError:
            return 0

    def right_angle_check(self, points):
        '''Checks the opposite angles in a closed shape created by four points and makes sure they're nearly 90 degrees.'''
        ang1 = self.angle(points[0],points[1],points[2])
        ang2 = self.angle(points[2],points[3],points[0])
        # print "opposite angles for this countour: %f and %f" % (ang1, ang2)
        # 1.5 radians ~ 86 deg
        # 1.2 radians ~ 70 deg
        # 1 radian ~ 57 deg
        # print "angle 1 %f   angle 2 %f" % ((ang1 * 180 / math.pi), (ang2 * 180 / math.pi))
        if ang1 > 1.1 and ang2 > 1.1:
            return True
        else:
            return False


    # check if the distance between all points is roughly the same
    # in other words, a square
    def equal_length_check( self, points, max_diff=15 ):
        '''Check to see if the four line segments of a four point poly are roughly the same.'''
        last_l = 0
        a = [0,1,2,3]
        b = [1,2,3,0]
        for i in range(4):
            # determine the length of the polygon segment
            # sqrt of delt x ^2 + delt y ^ 2
            l = math.sqrt(abs(points[a[i]].x - points[b[i]].x)**2 + abs(points[a[i]].y - points[b[i]].y)**2)
            if last_l:
                # compare the delta of the last segment to the current segment
                # if it's more than max_diff pixels different, then return
                if (last_l - l) > max_diff:
                    return False
            last_l = l
        return True


    # Get the center coordinate of a square
    def center_point( self, points ):
        '''Average the corner coordinates to find the center of the square.'''
        last_l = 0
        max_diff = 0
        a = [0,1,2,3]
        cx1 = abs(points[0].x + points[2].x) / 2.0
        cy1 = abs(points[0].y + points[2].y) / 2.0

        cx2 = abs(points[1].x + points[3].x) / 2.0
        cy2 = abs(points[1].y + points[3].y) / 2.0

        return {'x':(cx1 + cx2) / 2.0, 'y':(cy1 + cy2) / 2.0}


    # @print_timing
    def find_contours( self, img, my_storage ):
        sz = cvSize( img.width, img.height )
        # half_sz = cvSize( img.width / 2, img.height / 2 )


        # # Code to operate on the separate color channels independantly
        # red = cvCreateImage( sz, 8, 1 )
        # green = cvCreateImage( sz, 8, 1 )
        # blue = cvCreateImage( sz, 8, 1 )
        # 
        # # # Split the color planes
        # # # extract the c-th color plane
        # channels = [red, green, blue]
        # # channels[c] = tgray
        # cvSplit( img, channels[0], channels[1], channels[2], None ) 
        # last = None
        # for chan in channels:
        #     # equalize the grayscale
        #     # cvEqualizeHist( chan, chan );
        # 
        #     
        # 
        #     # compute the adaptive threshold
        #     # tgray = cvCreateImage( sz, 8, 1 );
        #     # cvAdaptiveThreshold(gray,tgray,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY_INV,13,16)
        #     cvThreshold(chan,chan,50,255,CV_THRESH_BINARY) 
        # 
        #     if last:
        #         cvAnd(last,chan,chan)
        #     
        #     last = chan
        #     
        #     # test_images.append(tgray)
        #     # cvShowImage(adapt,gray)
        # 
        # cvAnd(channels[0],channels[1],channels[0])
        # cvAnd(channels[0],channels[2],channels[0])
        # cvAnd(channels[1],channels[0],channels[1])
        # cvAnd(channels[2],channels[0],channels[2])
        # 
        # cvMerge(channels[0], channels[1], channels[2], None, img)


        test_images = []

        # create grayscale version of the image
        gray = cvCreateImage( sz, 8, 1 )
        cvCvtColor(img ,gray, CV_RGB2GRAY)

        # equalize the grayscale

        # cvEqualizeHist( gray, gray )
        # cvSmooth( gray, gray )

        # avg_gray = cvCreateImage( sz, 8, 1 )        
        # if self.last_gray:
        #     cvAddWeighted(gray,0.5,self.last_gray,0.5,0,avg_gray)                
        # else:
        #     avg_gray = cvCloneImage(gray)
        # 
        # self.last_gray = cvCloneImage(gray)
        


        # compute the adaptive threshold
        tgray = cvCreateImage( sz, 8, 1 );
        cvAdaptiveThreshold(gray,tgray,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY_INV,13,16)
        # cvAdaptiveThreshold(gray,tgray,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,17,3)
        # cvThreshold(gray,tgray,190,255,CV_THRESH_BINARY_INV) 
        
        # self.thresh_samples.append(tgray)
        # agray = cvCreateImage( sz, 8, 1 );
        # for thresh_sample in self.thresh_samples:
        #     cvAdd(thresh_sample,agray,agray)
        # 
        # self.thresh_samples = self.thresh_samples[-5:5]
        # test_images.append(agray)
        
        
        test_images.append(tgray)
        
        # cvShowImage(adapt,tgray)

        # # compute the canny edge detector version of the image
        cgray = cvCreateImage( sz, 8, 1 );
        cvCanny( gray, cgray, self.thresh_bot, self.thresh_top, 3 );
        test_images.append(cgray)
        # cvShowImage(adapt,cgray)

        contour_list = []

        # generate a square contour from a synthetic image
        # template_img = cvCreateImage( cvSize(120, 120), 8, 1 )
        # cvRectangle(template_img, cvPoint(0,0),cvPoint(120,120), 255, CV_FILLED)
        # cvRectangle(template_img, cvPoint(10,10),cvPoint(110,110), 0, CV_FILLED)
        # count, square = cvFindContours(template_img,storage,sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0) )

        for image in test_images:
            # original:
            # count, contours = cvFindContours( gray, storage, first_contour, sizeof(CvContour),
            #     CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0) );
            count, contours = cvFindContours( image, my_storage, mode=CV_RETR_LIST, method=CV_CHAIN_APPROX_SIMPLE) 

            # test each contour
            if contours:
                for contour in contours.hrange():            
                    # TODO: Minimum and maximum perimeter is a function
                    # of image size + % of image the code will likely take up
                    perim = cvContourPerimeter(contour)
                    # contour must be at least 10 pixels on a side
                    if perim > 30: #perim > 30 and perim < 300:
                        # if cvMatchShapes(square,contour,2) < 0.001:
                            # print str(cvMatchShapes(square,contour,2))        
                        result = cvApproxPoly( contour, sizeof(CvContour), my_storage,
                            CV_POLY_APPROX_DP, perim*0.03, 0 )
                        # Make sure the result is roughly a perfect square. Check side length
                        # and cross angles
                        # number of verticies == 4
                        if result.total == 4 and cvCheckContourConvexity(result):
                            points = result.asarray(CvPoint)

                            # gfx_points = [(int(point.x),int(point.y)) for point in points]
                            # #draw all contours
                            # pygame.gfxdraw.polygon(self.paint_buffer, gfx_points, (255,255,0))

                            # mult = 1.0
                            # scale the points to the multiplier
                            # points = [CvPoint(int(point.x*mult),int(point.y*mult)) for point in points]

                            # check that the sides are roughly equal in length
                            if self.equal_length_check(points,int(perim/4.0*0.25)):
                                # check that opposite angles are roughly 90 degrees
                                if self.right_angle_check(points):
                                    center = self.center_point(points)
                                    contour_list.append({"perim":perim,"points":points,"center":center})
                    try:
                        contour = contour.h_next
                    except AttributeError:
                        contour = None
        return contour_list



    # @print_timing
    def draw_contours(self, img, square_list, mult=1.0):
        max_x = 0
        max_y = 0
        min_x = int(img.width * mult)
        min_y = int(img.height * mult)
        # squares = False
        bounds = ()
        # shape = []

        circles = []
        # squares = []
        poly_lines = []

        distance_ratio = 0.76
        max_perim = 40
        max_square = None

        p_mat = None

        for square in square_list:
            circles.append(Circle(int(square['center']['x']), int(square['center']['y']), 10))
            
            if square['perim'] > max_perim:
                max_perim = square['perim']
                max_square = square

        if max_square:
            size = max_square['perim'] / 4.0
            window_size = int(math.sqrt((2*size**2)) / 2.0)            

            # reorder the points to maintain a consistent orientation
            # in this case, reorders the points from top to bottom (y axis)
            # then the top two are sorted to be left than right, the bottom two
            # are sorted to bt right then left. This creates a closed path
            # of top left, top right, bottom right, bottom left
            pts = sorted(max_square['points'],lambda x,y:x.y - y.y)

            top = pts[0:2]
            bot = pts[2:4]

            top = sorted(top, lambda x,y:x.x - y.x)
            bot = sorted(bot, lambda x,y:y.x - x.x)
            source_points = top+bot

            # the projected square is a perfect square surrounding the center point. It's created of 
            # equal length sides that are the perimeter / 4
            new_square_top_left = cvPoint(int(max_square['center']['x'] - size / 2.0), int(max_square['center']['y'] - size / 2.0))
            dest_points = [new_square_top_left,cvPoint(new_square_top_left.x + size, new_square_top_left.y),cvPoint(new_square_top_left.x + size, new_square_top_left.y + size), cvPoint(new_square_top_left.x, new_square_top_left.y + size)]            
            # the outer bounds of the newly remapped square image
            bounds = (dest_points[0].x,dest_points[0].y,int(size+0.5),int(size+0.5))

            self.oflow_test = max_square['points'][:]

            # compute distortion matrix
            pt_array = CvPoint2D32f * 4
            p_mat = cvCreateMat(3, 3, CV_32F)
            p_src = pt_array(*((p.x, p.y) for p in source_points))
            p_dst = pt_array(*((p.x, p.y) for p in dest_points))
            
            cvGetPerspectiveTransform(p_src, p_dst, p_mat)

        try:
            for circle in self.last_circles:
                pygame.gfxdraw.circle(self.paint_buffer, circle.x, circle.y, circle.r, (0,0,0))
            self.last_circles = []
        except AttributeError:
            self.last_circles = []


        if square_list:
            for circle in circles:
                pygame.gfxdraw.circle(self.paint_buffer, circle.x, circle.y, circle.r, (128,255,255))

            self.last_circles = [x for x in circles]

        
        return bounds,p_mat

def down_sample(oimg,multiple):
    '''Use the cvPyr algorithm and down sample an image to 1/2.'''
    working_img = cvCloneImage( oimg )
    for x in range(1,multiple+1):
        #print "Step. h:%d w:%d" % (oimg.height, oimg.width)
        working_img = cvCreateImage( cvSize(oimg.width/2, oimg.height/2), 8, 3 );
        cvPyrDown( oimg, working_img, CV_GAUSSIAN_5x5 );
        oimg = cvCloneImage( working_img )
    return working_img

def up_sample(oimg,multiple):
    '''USe the cvPyr function to upsample an image to 2x.'''
    working_img = cvCloneImage( oimg )
    for x in range(1,multiple+1):
        #print "Step. h:%d w:%d" % (oimg.height, oimg.width)
        working_img = cvCreateImage( cvSize(oimg.width*2, oimg.height*2), 8, 3 );
        cvPyrUp( oimg, working_img, CV_GAUSSIAN_5x5 );
        oimg = cvCloneImage( working_img )
    return working_img


def main():
    # import os
    # os.environ['SDL_VIDEODRIVER'] = 'windib'

    parser = OptionParser(usage="""\
    Detect SnapMyinfo QRcodes in a live video stream

    Usage: %prog [options] camera_index
    """)
    # parser.add_option('-d', '--directory',
    #                   type='string', action='store',
    #                   help="""Unpack the MIME message into the named
    #                   directory, which will be created if it doesn't already
    #                   exist.""")
    opts, args = parser.parse_args()

    pg_size = (1280,720)

    worker = WorkerTest(pg_size)

    cvWindows = []

    cvNamedWindow(lk, CV_WINDOW_AUTOSIZE)
    # cvNamedWindow(canny, CV_WINDOW_AUTOSIZE)
    cvNamedWindow(adapt, CV_WINDOW_AUTOSIZE)
    cvNamedWindow(sub_win, CV_WINDOW_AUTOSIZE)
    cvNamedWindow(small_win, CV_WINDOW_AUTOSIZE)


    names =  [args[0]]
    name = names[0]
    #for name in names:


    take_sequence = False
    seq_count = 10
    seq_num = 0
    sequence = []

    synth = False
    synth_image = []
    if synth:
        for i in range(seq_count):
            synth_image.append(cvLoadImage('''image_%.2d.jpg''' % i))
        o_width = pg_size[0]
        o_height = pg_size[1]
        # worker = WorkerTest(o_width,o_height)
        worker.frame_buffer = cvCloneImage(synth_image[0])
    else:
        capture = cvCreateCameraCapture( int(name) )
        cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH, pg_size[0] )
        cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT, pg_size[1] )
        # cvSetCaptureProperty( capture, CV_CAP_PROP_FPS, 10 )
        # worker = WorkerTest(o_width,o_height)
        worker.frame_buffer = cvQueryFrame( capture )

        (o_width,o_height) = [cvGetCaptureProperty(capture, prop) for prop in [CV_CAP_PROP_FRAME_WIDTH,CV_CAP_PROP_FRAME_HEIGHT]]



    half_size = (int(o_width / 4.0), int(o_height / 4.0))
    worker.small_frame = cvCreateImage(half_size, 8, 3)

    cpy = cvCreateImage((int(o_width),int(o_height)), 8, 3)

    # diff = 0
    # if o_height == 1200:
    #     diff = 0
    # elif o_height == 600:
    #     diff = 60
    # elif o_width == 960 and o_height == 720:
    #     diff = 100
    # elif o_width == 1280 and o_height == 720:
    #     diff = 0
    # # elif o_height == 480:
    # #     diff = 50
    # else:
    #     diff = 0

    # o_height = o_height - (2 * diff)
    # 
    # final_width = o_width
    # worker.display_scale = final_width / o_width
    # 
    # width = final_width
    # height = o_height * worker.display_scale

    pg_size = (int(o_width),int(o_height))

    # print "%d x %d" % (width,height)
    # 
    # # scan size is % of the height
    # scan_size = 0.25
    # # display square is a % of the scan size
    # disp_size = scan_size * 0.80
    # 
    # display_dim = int((disp_size * height))
    # disp_left_x = int(0.25 * width - int(display_dim / 2.0))
    # disp_right_x = int(0.75 * width - int(display_dim / 2.0))
    # disp_both_y = int(0.85 * height - int(display_dim / 2.0))
    # 
    # scan_dim = int(scan_size * o_height)
    # left_x = int(0.25 * o_width - int(scan_dim / 2.0))
    # right_x = int(0.75 * o_width - int(scan_dim / 2.0))
    # both_y = int(0.85 * o_height - int(scan_dim / 2.0))


    pg_display = pygame.display.set_mode( pg_size, 0) 
    # pg_display = pygame.display.set_mode( pg_size, pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.FULLSCREEN )

    worker.paint_buffer = pygame.Surface(pg_size).convert()
    # black is transparent
    worker.paint_buffer.set_colorkey((0,0,0))
    
    # set the window name
    pygame.display.set_caption('Live Detector') 
    # pg_clock = pygame.time.Clock()
    # pg_processClock = pygame.time.Clock()

    # some debug information
    print 'Driver %s\n' % pygame.display.get_driver()
    # print '''left x:%d right x:%d  scan dimension: %d''' % (left_x, right_x, scan_dim)
    
    # left_rect = cvRect(0,0,int(width),int(height)) # left_x,both_y,scan_dim,scan_dim)
    worker_thread = threading.Thread(target=worker.detect, args=("Square Scanner",))
    worker_thread.start()


    # worker_oflow = threading.Thread(target=worker.oflow_points)
    # worker_oflow.start()



    # right_rect = cvRect(right_x,both_y,scan_dim,scan_dim)
    # right_worker_thread = threading.Thread(target=worker.detect,args=(right_rect,zone2))
    # right_worker_thread.start()

    # connect_thread = threading.Thread(target=worker.connect)
    # connect_thread.start()    

    running = True
    while running:
        # get the pygame events
        events = pygame.event.get()
        for e in events:
            if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                running = False

            elif e.type == KEYDOWN and e.unicode == u'x':
                if worker.avg_mode:
                    worker.avg_mode = False
                else:
                    worker.avg_mode = True
                print "Averaging mode: %s" % (str(worker.avg_mode))
            elif e.type == KEYDOWN and e.unicode == u'p':
                take_sequence = True
                print "Take Sequence"
            # elif e.type == KEYDOWN and e.unicode == u'r':
            #     worker.thresh_top -= 10
            #     print str(worker.thresh_top)
            # elif e.type == KEYDOWN and e.unicode == u'b':
            #     worker.thresh_bot += 10
            #     print str(worker.thresh_bot)
            # elif e.type == KEYDOWN and e.unicode == u'v':
            #     worker.thresh_bot -= 10
            #     print str(worker.thresh_bot)
            elif e.type == KEYDOWN and e.unicode == u'l':
                worker.need_to_init = True
            # else:
            #     print str(e)
            
        # # copy the capture device frame to the worker frame_buffer
        # if synth:
        #     worker.frame_buffer = cvCloneImage(synth_image[seq_num])
        #     seq_num += 1
        #     if seq_num >= seq_count:
        #         seq_num = 0
        #     time.sleep(0.25)
        #     
        # else:
        worker.frame_buffer = cvQueryFrame( capture )


        cvResize(worker.frame_buffer, worker.small_frame)
        cvShowImage(small_win, worker.small_frame)

        # no data in the frame buffer means we're done
        if not worker.frame_buffer:
            break


        # if take_sequence:
        #     print "Snap Frame"
        #     sequence.append(cvCloneImage(worker.frame_buffer))
        #     
        #     if len(sequence) == seq_count:
        #         for frame in sequence:
        #             cvSaveImage(('''image_%.2d.jpg''' % sequence.index(frame)), frame)
        #         take_sequence = False
        #         sequence = []
            
        # convert the frame_buffer into a numpy array
        cvCopy(worker.frame_buffer, cpy) #CloneImage(worker.frame_buffer)
        
        # if resize:
        #     cvResize(cpy,resiz)
        
        # the surface RGB values are not the same as the cameras
        # this means that two of the channels have to be swapped in the numpy
        # array before being blit'ted ont the array
        surf_dat = cpy.as_numpy_array().transpose(1,0,2)[...,...,::-1] #,0,2) # ipl_to_numpy(disp_img
        
        # blit_array zaps anything on the surface and completely replaces it with the array, much
        # faster than converting the bufer to a surface and bliting it
        surfarray.blit_array(pg_display,surf_dat)

        # blit the paint buffer onto the surface. With a chromakey, all black values will show through
        # pg_display.blit(pygame.transform.flip(worker.paint_buffer, True, False),(0,0))
        pg_display.blit(worker.paint_buffer,(0,0))

        # flip() actually displays the surface
        pygame.display.flip()

    print worker.stop()
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
