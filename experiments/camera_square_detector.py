#!/usr/bin/python
#
# The full "Square Detector" program.
# It loads several images subsequentally and tries to find squares in
# each image
#
import sys

#from opencv.cv import *
#from opencv.highgui import *

from ctypes_opencv import *

from math import sqrt

import random
import math

from optparse import OptionParser

import time

H263 = 0x33363255
H263I = 0x33363249
MSMPEG4V3 = 0x33564944
MPEG4 = 0x58564944
MSMPEG4V2 = 0x3234504D
MJPEG = 0x47504A4D
MPEG1VIDEO = 0x314D4950
AC3 = 0x2000
MP2 = 0x50
FLV1 = 0x31564C46

FAAD = 0x31637661

#img = None;
#img0 = None;
storage = None;

thresh = [45,15]

adapt_1 = 45
adapt_2 = 15

wndname = "SnapCode Detector"
adapt = "Threshold"
canny = "Canny"

def print_timing(func):
    def wrapper(*arg):
        t1 = time.time()
        res = func(*arg)
        t2 = time.time()
        print '%s took %0.3f ms' % (func.func_name, (t2-t1)*1000.0)
        return res
    return wrapper


def angle( pt0, pt1, pt2 ):
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

def right_angle_check(points):
    '''Checks the opposite angles in a closed shape created by four points and makes sure they're nearly 90 degrees.'''
    ang1 = angle(points[0],points[1],points[2])
    ang2 = angle(points[2],points[3],points[0])
    # print "opposite angles for this countour: %f and %f" % (ang1, ang2)
    # 1.5 radians ~ 86 deg
    if ang1 > 1.4 and ang2 > 1.4:
        return True
    else:
        return False


# check if the distance between all points is roughly the same
# in other words, a square
def equal_length_check( points ):
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
        print "Step. h:%d w:%d" % (oimg.height, oimg.width)
        working_img = cvCreateImage( cvSize(oimg.width*2, oimg.height*2), 8, 3 );
        cvPyrUp( oimg, working_img, CV_GAUSSIAN_5x5 );
        oimg = cvCloneImage( working_img )
    return working_img

#@print_timing
def find_contours( img, storage ):
    sz = cvSize( img.width, img.height )

    timg = cvCloneImage( img ) # make a copy of input image
    gray = cvCreateImage( sz, 8, 1 )

    #timg = down_sample(timg,1)
    #timg = up_sample(timg,1)

    cvCvtColor(timg ,gray, CV_RGB2GRAY)

    tgray = cvCreateImage( sz, 8, 1 );
    #cvAdaptiveThreshold(gray,tgray,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,15,15)
    #cvAdaptiveThreshold(gray,tgray,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,45,15)
    cvAdaptiveThreshold(gray,tgray,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,21,16)
    # cvSaveImage('adaptive_gray.jpg',tgray)
    cvShowImage(adapt, tgray)

    cgray = cvCreateImage( sz, 8, 1 );
    # print '%d and %d' % (thresh[0], thresh[1])
    cvCanny( gray, cgray, 1000, 2000, 5 );
    cvShowImage(canny, cgray)

    # cvSaveImage('canny_gray.jpg',cgray)


    # down-scale and upscale the image to filter out the noise
    # cvPyrDown( subimage, pyr, 7 );
    # cvSaveImage('down_test.jpg',subimage)
    # cvPyrUp( pyr, subimage, 7 );
    # cvSaveImage('up_test.jpg',subimage)
    # find squares in every color plane of the image
    contour_list = []
    # process the thresholded and canny edges
    test_images = [tgray,cgray]

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
        count, contours = cvFindContours( image, storage, mode=CV_RETR_LIST, method=CV_CHAIN_APPROX_SIMPLE) 

        #first_contour, sizeof(CvContour),
        #    CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0) );

        # test each contour
        for contour in contours.hrange():            
            # TODO: Minimum and maximum perimeter is a function
            # of image size + % of image the code will likely take up
            perim = cvContourPerimeter(contour)
            # contour must be at least 10 pixels on a side
            if perim > 20: #perim > 30 and perim < 300:
                # if cvMatchShapes(square,contour,2) < 0.001:
                    # print str(cvMatchShapes(square,contour,2))        
                    result = cvApproxPoly( contour, sizeof(CvContour), storage,
                        CV_POLY_APPROX_DP, perim*0.02, 0 )
                    if result.total == 4 and cvCheckContourConvexity(result):
                        contour_list.append(result)
            try:
                contour = contour.h_next
            except AttributeError:
                contour = None
    return contour_list

#@print_timing
def draw_contours(img, contour_list):
    cpy = cvCloneImage( img );
    max_x = 0
    max_y = 0
    min_x = img.width
    min_y = img.height

    for contour in contour_list:
        # cvDrawContours(cpy,contour,CV_RGB(255,0,0),CV_RGB(0,255,255),0,1)
        # points = []
        # perim = cvContourPerimeter(contour)
        # # if perim > 30 and perim < 300:
        # result = cvApproxPoly( contour, sizeof(CvContour), storage,
        #     CV_POLY_APPROX_DP, perim*0.02, 0 )
        points = contour.asarray(CvPoint)

        if equal_length_check(points):
            if right_angle_check(points):
                # find the outer extremes for a bounding box
                for pt in points:
                    max_x = max(max_x,pt.x + 15) 
                    max_y = max(max_y,pt.y + 15)
                    min_x = min(min_x,pt.x - 15)
                    min_y = min(min_y,pt.y - 15)
            
                cvPolyLine( cpy, [points], 1, CV_RGB(0,255,255), 1, CV_AA, 0 )
    bounds = [cvPoint(min_x,min_y),cvPoint(max_x,min_y),cvPoint(max_x,max_y),cvPoint(min_x,max_y)]
    # print '''x:%d y:%d height:%d width:%d''' % (min_x,min_y,max_x-min_x,max_y-min_y)
    cvPolyLine( cpy, [bounds], 1, CV_RGB(255,255,0), 3, CV_AA, 0 )
    cvSaveImage( 'test.jpg', cpy)
    return cpy # cvRect(min_x,min_y,max_x-min_x,max_y-min_y)


# def camShiftTest( img, box ):
#     window = box
#     cvCamShift( sample_image, window
#         const CvArr*     prob_image, 
#         CvRect           window, 
#         CvTermCriteria   criteria, 
#         CvConnectedComp* comp, 
#         CvBox2D*         box        = NULL 
#     ); 
    
def on_trackbar(a):
    thresh[0] = 3 + 2 * a

def on_trackbar2(a):
    thresh[1] = a


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
    cvNamedWindow(adapt, CV_WINDOW_AUTOSIZE)
    cvNamedWindow(canny, CV_WINDOW_AUTOSIZE)


    storage = cvCreateMemStorage(0);
    
    names =  [args[0]]
    name = names[0]
    #for name in names:
      
    capture = cvCreateCameraCapture( int(name) )
    cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH, 640 )
    cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT, 480 )

    img = cvQueryFrame( capture )

    # props = [CV_CAP_PROP_FOURCC]
    # for prop in props:    
    #     print str(cvGetCaptureProperty(capture, prop))

    (width,height) = [cvGetCaptureProperty(capture, prop) for prop in [CV_CAP_PROP_FRAME_WIDTH,CV_CAP_PROP_FRAME_HEIGHT]]
    print width
    print height
    output_name = 'output_%s.mpg' % (name)
    
    # print str(cvGetCaptureProperty(capture,CV_CAP_PROP_FOURCC) )
    # result_vid = cvCreateVideoWriter( output_name, MSMPEG4V3, 30.0, cvSize(int(width),int(height)) )
    # result_vid = cvCreateVideoWriter(output_name, FAAD, 30.0, cvSize(640,360) )
    
    while 1:
        img = cvQueryFrame( capture )

        if not img:
            break
        


        # template = cvLoadImage( "qr_small_corner.png", 1 );
        # img = cvLoadImage( name, 1 );
    
        # x:272 y:192 height:107 width:107
        # x:247 y:126 height:252 width:252
        # img0 = cvGetSubRect( img, cvRect(640,440,320,240))
        # img0 = cvGetSubRect( img, cvRect(247,126,252,252))
        # 
        # if not img0:
        #     print "Couldn't load %s" % name
        #     continue;
        img0 = cvCloneImage( img )

        #cvSetImageROI(img0, cvRect(640,440,320,240))

        cnt = find_contours(img0, storage)
        ximg = draw_contours(img0, cnt)        
        cvShowImage(wndname, ximg)

        try:
            k = cvWaitKey(5)
            if k % 0x100 == 27:
                #cvReleaseImage(img)
                #cvReleaseImage(img0)
                #cvReleaseImage(ximg)
                break
        except TypeError:
            break

        #cvWriteFrame(result_vid, ximg)

        # # clear memory storage - reset free space position
        cvClearMemStorage( storage )
    #cvReleaseVideoWriter(result_vid)
    #cvReleaseMemStorage( storage )
    cvDestroyWindow(wndname)
    cvDestroyWindow(adapt)
    cvDestroyWindow(canny)

    print 'exiting'
    sys.exit(0)
