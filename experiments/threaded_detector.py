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

import Image, ImageEnhance, ImageFont, ImageDraw

# remember to setup the ssh tunnel first
from SnapRemoteCard import SnapRemoteCard
from SnapCommand import SnapCommand

import re

# # use a bitmap font
# font = ImageFont.load("arial.pil")
# 
# draw.text((10, 10), "hello", font=font)

# use a truetype font


# # Some constants for video codecs
# H263 = 0x33363255
# H263I = 0x33363249
# MSMPEG4V3 = 0x33564944
# MPEG4 = 0x58564944
# MSMPEG4V2 = 0x3234504D
# MJPEG = 0x47504A4D
# MPEG1VIDEO = 0x314D4950
# AC3 = 0x2000
# MP2 = 0x50
# FLV1 = 0x31564C46
# FAAD = 0x31637661


wndname = "SnapCode Detector"
adapt = "Threshold"
canny = "Canny"
test = "Test Window"
zone1 = "Left User Card"
zone2 = "Right User Card"
paint_name = "Paint Display"


def print_timing(func):
    def wrapper(*arg):
        t1 = time.time()
        res = func(*arg)
        t2 = time.time()
        print '%s took %0.3f ms, roughly %0.3f fps' % (func.func_name, (t2-t1)*1000.0, 1.0/((t2-t1)+0.001))
        return res
    return wrapper



class WorkerTest():
    '''A Class to implement a worker object. Accesses shared buffer and does some operations on it.'''
    def __init__(self):
        self.frame_buffer = None
        self.paint_buffer = None
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
        self.src = SnapRemoteCard()
        
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
    
    def detect(self, rect, window_name):
        '''Detect and decode barcodes in the image. If rect is passed, only process that subregion of the image.'''
        self.zonecolor[window_name] = CV_RGB(255,255,255)
        self.decoded[window_name] = False
        self.decode_window[window_name] = 0
        
        screen_font = ImageFont.truetype("arial.ttf", 24)
        data_font = ImageFont.truetype("arial.ttf", 14)
        qc = qrcode()
        storage = cvCreateMemStorage(0)
        draw = ImageDraw.Draw(self.paint_buffer)
        while self.running:
            #if self.frame_buffer:
            cpy = cvCloneImage(self.frame_buffer)
            cpy = cvGetSubRect(cpy, None, rect)
            cvFlip(cpy,cpy,1)

            # new_width = 960.0
            # scale = new_width / self.frame_buffer.width
            # small_rect = cvRect(int(rect.x*scale),int(rect.y*scale),int(rect.height*scale),int(rect.width*scale))
            # cpy = cvCreateImage(cvSize(int(new_width),int(self.frame_buffer.height * scale)),8,3)
            # cvResize(self.frame_buffer,cpy)
            # cpy = cvGetSubRect(cpy, None, small_rect)
            
            
            
            #cvSetImageROI(cpy,rect)
            time.sleep(0.15)
            # work = cvGetSubRect(cpy, None, rect)
            # work = cvCloneImage(self.frame_buffer)
            if not self.frame_buffer:
                self.running = False
                break
            cnt = self.find_contours(cpy, storage)
            has_code = self.draw_contours(cpy, cnt, 1)
            if has_code and self.decoded[window_name]:
                self.decode_window[window_name] = 0
            elif has_code and not self.decoded[window_name]:
                self.zonecolor[window_name] = CV_RGB(255,255,0)
                draw.text((int(self.display_scale*rect.x), int(self.display_scale * rect.y-20)), "Scanning...", font=screen_font, fill=(200,200,0,0))
                
            else:
                if self.decode_window[window_name] > 6:
                    draw.rectangle([int(self.display_scale*rect.x-100), int(self.display_scale * rect.y-100), int(self.display_scale * rect.x+100+rect.width), int(self.display_scale * rect.y+100+rect.height)], fill=(0,0,0,255))
                    self.zonecolor[window_name] = CV_RGB(255,255,255)
                    self.decode_window[window_name] = 0
                    self.decoded[window_name] = False
                    try:
                        if self.visible_user[window_name]:
                            self.lock.acquire()
                            del self.visible_user[window_name]
                            self.lock.release()
                    except KeyError:
                        pass
                        
                else:
                    self.decode_window[window_name] += 1

            work = cvGetImage(cpy)
            
            # work = cvCloneImage(self.frame_buffer)
            # work = cvGetSubRect(work, None, rect)
            # work = cvGetImage(work)

            # cvFlip(work,work,1)
            # work = up_sample(work,1)
            
            # gray = cvCreateImage( cvGetSize(work), 8, 1 )
            # 
            # cvCvtColor(work ,gray, CV_RGB2GRAY)
            # 
            # tgray = cvCreateImage( cvGetSize(work), 8, 1 );
            # cvAdaptiveThreshold(gray,tgray,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,55,15)
            # 
            # cvCvtColor(tgray , work, CV_GRAY2RGB)

            decode_img = ipl_to_pil(work)
            
            decode_img = ImageEnhance.Sharpness(decode_img).enhance(1.5)
            # ImageEnhance.Contrast(            
            # ImageEnhance.Brightness(decode_img).enhance(1.2)   
            # ).enhance(0.75)
            

            
            
            if has_code and not self.decoded[window_name]:
                data = qc.decode(decode_img)
                if data != "NO BARCODE":
                    # code decoded
                    self.zonecolor[window_name] = CV_RGB(0,255,0)
                    self.decoded[window_name] = True
                    self.decode_window[window_name] = 0

                    match = re.search(r'\/([^\/]*)$',data)
                    if match:
                        code = match.group(1)
                        #print str(code)
                        user_card = self.src.get_card(code)
                        if user_card:
                            draw.text((int(self.display_scale*rect.x), int(self.display_scale * rect.y-40)), user_card['full_name'], font=data_font, fill=(0,200,0,0))
                            print '''%s decoded %s''' % (window_name,user_card['full_name'])
                            self.lock.acquire()
                            self.visible_user[window_name] = user_card['user_id']
                            self.lock.release()
                        else:
                            self.zonecolor[window_name] = CV_RGB(180,0,0)
                            print '''%s : %s''' % (window_name,'No card found')
                    else:
                        self.zonecolor[window_name] = CV_RGB(180,0,0)
                        print '''QR code decoded, is it not a SnapMyInfo code?'''
                else:
                    pass
                    # print '''%s failed to decode''' % (window_name)
        
            # cvShowImage(window_name, pil_to_ipl(decode_img))
            cvClearMemStorage( storage )
            # (window_name, work) #pil_to_ipl(decode_img))

    def oflow_points(self):
        '''Adding the lkdemo code, to track optical flow points'''
        self.need_to_init = True
        self.of_points = [[], []]
        night_mode = False
        image = None
        pt = None
        add_remove_pt = False
        flags = 0
        
        win_size = 10
        MAX_COUNT = 200
        
        while self.running:
            frame = cvCloneImage(self.frame_buffer)
            #frame = down_sample(self.frame_buffer,2)

            if image is None:
                # create the images we need
                image = cvCreateImage (cvGetSize (frame), 8, 3)
                image.origin = 0
                grey = cvCreateImage (cvGetSize (frame), 8, 1)
                prev_grey = cvCreateImage (cvGetSize (frame), 8, 1)
                pyramid = cvCreateImage (cvGetSize (frame), 8, 1)
                prev_pyramid = cvCreateImage (cvGetSize (frame), 8, 1)
                self.of_points = [[], []]

            # copy the frame, so we can draw on it
            if frame.origin:
                cvFlip(frame, image)
            else:
                cvCopy (frame, image)

            # create a grey version of the image
            cvCvtColor (image, grey, CV_BGR2GRAY)

            if night_mode:
                # night mode: only display the points
                cvSetZero (image)

            if self.need_to_init:
                # we want to search all the good points
                self.of_points[1] = cvGoodFeaturesToTrack(grey, None, None, None, MAX_COUNT, 0.01, 10)

                # refine the corner locations
                cvFindCornerSubPix (
                    grey,
                    self.of_points [1],
                    cvSize (win_size, win_size), cvSize (-1, -1),
                    cvTermCriteria (CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,
                                       20, 0.03))
                                               
            elif len (self.of_points [0]) > 0:
                # we have points, so display them

                # calculate the optical flow
                self.of_points [1], status = cvCalcOpticalFlowPyrLK (
                    prev_grey, grey, prev_pyramid, pyramid,
                    self.of_points [0], None, None, 
                    cvSize (win_size, win_size), 3,
                    None, None,
                    cvTermCriteria (CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,
                                       20, 0.03),
                    flags)

                # initializations
                point_counter = -1
                new_points = []
            
                for the_point in self.of_points [1]:
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

                # set back the points we keep
                self.of_points [1] = new_points
            
            if add_remove_pt:
                # we want to add a point
                self.of_points [1].append (cvPointTo32f (pt))

                # refine the corner locations
                self.of_points [1][-1] = cvFindCornerSubPix (
                    grey,
                    [self.of_points [1][-1]],
                    cvSize (win_size, win_size), cvSize (-1, -1),
                    cvTermCriteria (CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,
                                       20, 0.03))[0]

                # we are no more in "add_remove_pt" mode
                add_remove_pt = False

            # swapping
            prev_grey, grey = grey, prev_grey
            prev_pyramid, pyramid = pyramid, prev_pyramid
            self.of_points [0], self.of_points [1] = self.of_points [1], self.of_points [0]
            self.need_to_init = False
        
            # we can now display the image
            cvShowImage ('LkDemo', image)


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
        if ang1 > 1.4 and ang2 > 1.4:
            return True
        else:
            return False


    # check if the distance between all points is roughly the same
    # in other words, a square
    def equal_length_check( self, points ):
        '''Check to see if the four line segments of a four point poly are roughly the same.'''
        last_l = 0
        max_diff = 0
        a = [0,1,2,3]
        b = [1,2,3,0]
        for i in range(4):
            # determine the length of the polygon segment
            l = math.sqrt(abs(points[a[i]].x - points[b[i]].x)**2 + abs(points[a[i]].y - points[b[i]].y)**2)
            if last_l:
                # compare the delta of the last segment to the current segment
                # if it's more than 7 pixels different, then return
                if (last_l - l) > 3:
                    return False
            last_l = l
        return True

    def find_contours( self, img, my_storage ):
        sz = cvSize( img.width, img.height )
        # half_sz = cvSize( img.width / 2, img.height / 2 )

        test_images = []
    
        # create grayscale version of the image
        gray = cvCreateImage( sz, 8, 1 )
        cvCvtColor(img ,gray, CV_RGB2GRAY)

        # # compute the adaptive threshold
        # tgray = cvCreateImage( sz, 8, 1 );
        # #cvAdaptiveThreshold(gray,tgray,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,15,15)
        # #cvAdaptiveThreshold(gray,tgray,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,45,15)
        # cvAdaptiveThreshold(gray,tgray,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,41,16)
        # test_images.append(tgray)

        # compute the canny edge detector version of the image
        cgray = cvCreateImage( sz, 8, 1 );
        cvCanny( gray, cgray, 1000, 2000, 5 );
        test_images.append(cgray)
        # cvShowImage(canny, cgray)
        # cvSaveImage('canny_gray.jpg',cgray)

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
            # first_contour = CvContour()
            count, contours = cvFindContours( image, my_storage, mode=CV_RETR_LIST, method=CV_CHAIN_APPROX_SIMPLE) 

            #first_contour, sizeof(CvContour),
            #    CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0) );

            # test each contour
            if contours:
                for contour in contours.hrange():            
                    # TODO: Minimum and maximum perimeter is a function
                    # of image size + % of image the code will likely take up
                    perim = cvContourPerimeter(contour)
                    # contour must be at least 10 pixels on a side
                    if perim > 20: #perim > 30 and perim < 300:
                        # if cvMatchShapes(square,contour,2) < 0.001:
                            # print str(cvMatchShapes(square,contour,2))        
                            result = cvApproxPoly( contour, sizeof(CvContour), my_storage,
                                CV_POLY_APPROX_DP, perim*0.02, 0 )
                            if result.total == 4 and cvCheckContourConvexity(result):
                                contour_list.append(result)
                    try:
                        contour = contour.h_next
                    except AttributeError:
                        contour = None
        return contour_list

    #@print_timing
    def draw_contours(self, img, contour_list, mult=1.0):
        # cpy = cvCloneImage( img );
        max_x = 0
        max_y = 0
        min_x = img.width
        min_y = img.height
        squares = False
        for contour in contour_list:
            # cvDrawContours(cpy,contour,CV_RGB(255,0,0),CV_RGB(0,255,255),0,1)
            # points = []
            # perim = cvContourPerimeter(contour)
            # # if perim > 30 and perim < 300:
            # result = cvApproxPoly( contour, sizeof(CvContour), storage,
            #     CV_POLY_APPROX_DP, perim*0.02, 0 )
            points = contour.asarray(CvPoint)

            #for i in range(len(points)):
            #    points[i].x = points[i].x * mult
            #    points[i].y = points[i].y * mult

            points = [CvPoint(int(point.x*mult),int(point.y*mult)) for point in points]

            # check that the sides are roughly equal in length
            if self.equal_length_check(points):
                # check that opposite angles are roughly 90 degrees
                if self.right_angle_check(points):
                    squares = squares or True
                    # find the outer extremes for a bounding box
                    for pt in points:
                        max_x = max(max_x,pt.x + 15) 
                        max_y = max(max_y,pt.y + 15)
                        min_x = min(min_x,pt.x - 15)
                        min_y = min(min_y,pt.y - 15)

                    # cvPolyLine( img, [points], 1, CV_RGB(0,255,255), 1, CV_AA, 0 )
        if squares:
            bounds = [cvPoint(min_x,min_y),cvPoint(max_x,min_y),cvPoint(max_x,max_y),cvPoint(min_x,max_y)]
            # print '''x:%d y:%d height:%d width:%d''' % (min_x,min_y,max_x-min_x,max_y-min_y)
            # cvPolyLine( img, [bounds], 1, CV_RGB(255,255,0), 3, CV_AA, 0 )
        
        #subimg = cvGetSubRect(cpy, None, cvRect(0,0,100,100))
        # subimg = cvGetSubRect(cpy, None, cvRect(0,0,100,100))
        
        #cvShowImage(test,subimg)

        # cvSaveImage( 'test.jpg', img)
        return squares
        #cvRect(min_x,min_y,max_x-min_x,max_y-min_y)



# def angle( pt0, pt1, pt2 ):
#     '''Measure the angle between three points.'''
#     dx1 = pt1.x - pt0.x;
#     dy1 = pt1.y - pt0.y;
#     dx2 = pt2.x - pt1.x;
#     dy2 = pt2.y - pt1.y;
# 
#     # determine the slopes of the two lines created
#     m1 = dy1 / (dx1 + 1e-10)
#     m2 = dy2 / (dx2 + 1e-10)
# 
#     try:
#         return (abs(math.atan( ((m2 - m1) + 1e-10)  / (1.0 + m1 * m2)) ))
#     # sometimes the slopes are the same or roughly the same (maybe different signs)
#     # in this case the angle between them is 0 and the calculation blows up
#     except ZeroDivisionError:
#         return 0
# 
# def right_angle_check(points):
#     '''Checks the opposite angles in a closed shape created by four points and makes sure they're nearly 90 degrees.'''
#     ang1 = angle(points[0],points[1],points[2])
#     ang2 = angle(points[2],points[3],points[0])
#     # print "opposite angles for this countour: %f and %f" % (ang1, ang2)
#     # 1.5 radians ~ 86 deg
#     if ang1 > 1.4 and ang2 > 1.4:
#         return True
#     else:
#         return False
# 
# 
# # check if the distance between all points is roughly the same
# # in other words, a square
# def equal_length_check( points ):
#     '''Check to see if the four line segments of a four point poly are roughly the same.'''
#     last_l = 0
#     max_diff = 0
#     a = [0,1,2,3]
#     b = [1,2,3,0]
#     for i in range(4):
#         # determine the length of the polygon segment
#         l = math.sqrt(abs(points[a[i]].x - points[b[i]].x)**2 + abs(points[a[i]].y - points[b[i]].y)**2)
#         if last_l:
#             # compare the delta of the last segment to the current segment
#             # if it's more than 7 pixels different, then return
#             if (last_l - l) > 3:
#                 return False
#         last_l = l
#     return True

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

if __name__ == "__main__":
    # create memory storage that will contain all the dynamic data
    parser = OptionParser(usage="""\
    Detect squares with a threshold of X

    Usage: %prog [options] msgfile
    """)
    parser.add_option('-d', '--directory',
                      type='string', action='store',
                      help="""Unpack the MIME message into the named
                      directory, which will be created if it doesn't already
                      exist.""")
    opts, args = parser.parse_args()

    cvNamedWindow(wndname, CV_WINDOW_AUTOSIZE)
    # cvNamedWindow(zone1, CV_WINDOW_AUTOSIZE)
    # cvNamedWindow(zone2, CV_WINDOW_AUTOSIZE)
    # cvNamedWindow(paint_name, CV_WINDOW_AUTOSIZE)

    # cvNamedWindow(adapt, CV_WINDOW_AUTOSIZE)
    # cvNamedWindow(canny, CV_WINDOW_AUTOSIZE)
    # 
    # cvNamedWindow(test, CV_WINDOW_AUTOSIZE)
    # cvNamedWindow('LkDemo', CV_WINDOW_AUTOSIZE)

    worker = WorkerTest()


    # storage = cvCreateMemStorage(0);
    
    names =  [args[0]]
    name = names[0]
    #for name in names:
      
    capture = cvCreateCameraCapture( int(name) )
    cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH, 1280 )
    cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT, 720 )

    worker.frame_buffer = cvQueryFrame( capture )
    #worker_thread = threading.Thread(target=worker.detect)
    #worker_thread.start()

    # worker_oflow = threading.Thread(target=worker.oflow_points)
    # worker_oflow.start()

    # worker_display_thread = threading.Thread(target=worker.display)
    # worker_display_thread.start()

    # props = [CV_CAP_PROP_FOURCC]
    # for prop in props:    
    #     print str(cvGetCaptureProperty(capture, prop))

    (o_width,o_height) = [cvGetCaptureProperty(capture, prop) for prop in [CV_CAP_PROP_FRAME_WIDTH,CV_CAP_PROP_FRAME_HEIGHT]]
    if o_height == 1200:
        diff = 125
    elif o_height == 600:
        diff = 60
    elif o_width == 960 and o_height == 720:
        diff = 100
    elif o_width == 1280 and o_height == 720:
        diff = 0
    elif o_height == 480:
        diff = 50
    else:
        diff = 0

    o_height = o_height - (2 * diff)

    final_width = o_width
    worker.display_scale = final_width / o_width

    width = final_width
    height = o_height * worker.display_scale

    print "%d x %d" % (width,height)

    worker.paint_buffer = Image.new('RGBA',(int(width),int(height)),(0,0,0,255))

    # scan size is % of the height
    scan_size = 0.25
    # display square is a % of the scan size
    disp_size = scan_size * 0.80
    
    display_dim = int((disp_size * height))
    disp_left_x = int(0.25 * width - int(display_dim / 2.0))
    disp_right_x = int(0.75 * width - int(display_dim / 2.0))
    disp_both_y = int(0.85 * height - int(display_dim / 2.0))

    scan_dim = int(scan_size * o_height)
    left_x = int(0.25 * o_width - int(scan_dim / 2.0))
    right_x = int(0.75 * o_width - int(scan_dim / 2.0))
    both_y = int(0.85 * o_height - int(scan_dim / 2.0))


    print '''left x:%d right x:%d  scan dimension: %d''' % (left_x, right_x, scan_dim)

    left_rect = cvRect(left_x,both_y,scan_dim,scan_dim)
    
    left_worker_thread = threading.Thread(target=worker.detect,args=(left_rect,zone1))
    left_worker_thread.start()

    right_rect = cvRect(right_x,both_y,scan_dim,scan_dim)

    right_worker_thread = threading.Thread(target=worker.detect,args=(right_rect,zone2))
    right_worker_thread.start()

    connect_thread = threading.Thread(target=worker.connect)
    connect_thread.start()
    
    #worker_thread = threading.Thread(target=worker.detect)
    #worker_thread.start()
    
    # output_name = 'output_%s.mpg' % (name)
    
    # print str(cvGetCaptureProperty(capture,CV_CAP_PROP_FOURCC) )
    # result_vid = cvCreateVideoWriter( output_name, MSMPEG4V3, 30.0, cvSize(int(width),int(height)) )
    # result_vid = cvCreateVideoWriter(output_name, FAAD, 30.0, cvSize(640,360) )



    while 1:
        # temp_disp_buffer = cvQueryFrame( capture )
        worker.frame_buffer = cvQueryFrame( capture )
        # mirror the image
        # mirror_buff = cvCreateImage(cvSize(int(width),int(height)),8,3)
        # cvFlip(temp_disp_buffer,mirror_buff,1)
        # worker.frame_buffer = cvCloneImage(mirror_buff)

        # clip to the size specified
        disp_buffer = cvGetImage(cvGetSubRect(worker.frame_buffer, None, cvRect(0,0,int(width),int(height) )) )

        # resize to the size specified
        # disp_buffer = cvCreateImage(cvSize(int(width),int(height)),8,3)
        # cvResize(worker.frame_buffer,disp_buffer)
        
        # do nothing to the original frame but clone it
        # disp_buffer = cvCloneImage(worker.frame_buffer)

        # set a region of interest
        # cvSetImageROI(worker.frame_buffer, cvRect(0,diff,int(width),int(height)))


        
        if not disp_buffer:
            break
        # if worker.paint_buffer:
        #     cvShowImage(wndname, worker.paint_buffer)

        try:
            c = '%c' % (cvWaitKey (10) & 255)
            if c == '\x1b':
                # user has press the ESC key, so exit
                break

            if c in ['r', 'R']:
                worker.need_to_init = True
            elif c in ['c', 'C']:
                worker.of_points = [[], []]
            # k = cvWaitKey(5)
            # if k % 0x100 == 27:
                # break
        except TypeError:
            break

        disp_img = cvCloneImage(disp_buffer)
        try:
            cvRectangle(disp_img,cvPoint(disp_left_x,disp_both_y),cvPoint(disp_left_x+display_dim,disp_both_y+display_dim),worker.zonecolor[zone1], 3, CV_AA, 0)
            cvRectangle(disp_img,cvPoint(disp_right_x,disp_both_y),cvPoint(disp_right_x+display_dim,disp_both_y+display_dim),worker.zonecolor[zone2], 3, CV_AA, 0)
        except KeyError:
            # race condition, the threads haven't set colors yet
            pass

        # Using PIL for screen display
        # if worker.paint_buffer:
        #     # cvShowImage(paint_name,pil_to_ipl(worker.paint_buffer))
        #     image = ipl_to_pil(disp_img)
        #     image.convert('RGBA')
        #     disp_image = Image.composite(image,worker.paint_buffer,worker.paint_buffer)
        #     # ow,oh = disp_image.size
        #     # scale = 1280.0 / ow
        #     # big_w = int(ow * scale)
        #     # big_h = int(oh * scale)
        #     # disp_image = disp_image.resize((big_w,big_h))
        #     disp_img = pil_to_ipl(disp_image)

        scale = 1600.0 / disp_img.width
        big_w = int(disp_img.width * scale)
        big_h = int(disp_img.height * scale)
        big_image = cvCreateImage(cvSize(int(big_w),int(big_h)),8,3)
        cvResize(disp_img,big_image)
        # cvSmooth(big_image,big_image,)
        cvShowImage(wndname,big_image)


    print worker.stop()
    #cvReleaseVideoWriter(result_vid)
    #cvReleaseMemStorage( storage )
    cvDestroyWindow(wndname)
    #cvDestroyWindow(adapt)
    #cvDestroyWindow(canny)
    #cvDestroyWindow(test)

    print 'exiting'
    sys.exit(0)
