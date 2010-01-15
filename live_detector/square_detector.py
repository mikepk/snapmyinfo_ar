#!/usr/bin/python
#
import sys

#from opencv.cv import *
#from opencv.highgui import *

from ctypes_opencv import *
import math
# from math import sqrt

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


class SquareDetector(object):
    def __init__(self):
        self.minimum_perimeter = 30

        # square_detector buffers
        self.sz = None
        self.grayscale_buffer = None
        self.threshold_buffer = None
        self.canny_buffer = None
        
        self.storage = cvCreateMemStorage(0)
        
        self.thresh_bot = 230
        self.thresh_top = 340
        

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

        return cvPoint((cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0)

        
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
        '''Checks the opposite angles in a 4 sided closed shape and makes sure they're nearly 90 degrees.'''
        ang1 = self.angle(points[0],points[1],points[2])
        ang2 = self.angle(points[2],points[3],points[0])
        # print "opposite angles for this countour: %f and %f" % (ang1, ang2)
        # 1.5 radians ~ 86 deg
        # 1.2 radians ~ 70 deg
        # 1 radian ~ 57 deg
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

    def init_image_buffers(self,size):
        '''Create image buffers for processing.'''
        self.sz = size
        self.grayscale_buffer = cvCreateImage(self.sz, 8, 1)
        self.threshold_buffer = cvCreateImage(self.sz, 8, 1)
        self.canny_buffer = cvCreateImage(self.sz, 8, 1)
        
    def get_max_square(self, img):
        '''Get the largest reasonable square to represent a code'''
        pass

    def find_squares( self, img):
        '''Extract squares using the contours from an image and testing for various square characteristics.'''

        # The images that will be scanned for contours
        test_images = []

        # create grayscale version of the image
        cvCvtColor(img ,self.grayscale_buffer, CV_RGB2GRAY)

        # compute the adaptive threshold
        cvAdaptiveThreshold(self.grayscale_buffer,self.threshold_buffer,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY_INV,13,16)
        # cvAdaptiveThreshold(self.grayscale_buffer,tgray,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,17,3)
        # cvThreshold(self.grayscale_buffer,tgray,190,255,CV_THRESH_BINARY_INV) 
        
        
        cvShowImage(adapt,self.threshold_buffer)
        test_images.append(self.threshold_buffer)
        

        
        # compute the canny edge detector version of the image
        # self.canny_buffer = cvCreateImage( sz, 8, 1 );
        cvCanny( self.grayscale_buffer, self.canny_buffer, self.thresh_bot, self.thresh_top, 3 );
        test_images.append(self.canny_buffer)
        # cvShowImage(adapt,self.canny_buffer)

        # list of all the extracted candidate squares
        square_list = []

        # iterate on all processed images
        for image in test_images:
            # extract contours
            count, contours = cvFindContours( image, self.storage, mode=CV_RETR_LIST, method=CV_CHAIN_APPROX_SIMPLE) 

            # test each contour
            if contours:
                for contour in contours.hrange():            
                    # the perimeter of the contour
                    perim = cvContourPerimeter(contour)

                    # contour must be at least minimum_perimeter size to be considered
                    # this filters out small squares that appear due to noise
                    if perim > self.minimum_perimeter: 
                        # approximate the contour with a polygon with fairly good
                        # precision        
                        result = cvApproxPoly( contour, sizeof(CvContour), self.storage,
                            CV_POLY_APPROX_DP, perim*0.03, 0 )
                        # Make sure the result is roughly a perfect square. Check side length
                        # and cross angles
                        # number of verticies == 4
                        if result.total == 4 and cvCheckContourConvexity(result):
                            # turn the contour sequence into a CvPoint array
                            points = result.asarray(CvPoint)

                            # check that the sides are roughly equal in length
                            # perim/4 * 0.25 means the sides must be within 25% of each other
                            # this value is rounded
                            if self.equal_length_check(points, int(perim/4.0*0.25 + 0.5)):
                                # check that opposite angles are roughly 90 degrees
                                if self.right_angle_check(points):
                                    # store the perimeter value, the verticies, and the center point
                                    square_list.append({"perim":perim,"points":points,"center":self.center_point(points)})
                    # if h_next throws an exception, 
                    # no contours in the image
                    try:
                        contour = contour.h_next
                    except AttributeError:
                        contour = None
        # will leak memory unless you clear the CV mem storage
        cvClearMemStorage( self.storage )
        return square_list
    
    
    def reorder_points(self, points):
        '''Reorder points of a square to create a path from top right, top left, bottom left, bottom right.'''

        # sort them all in ascending order by y
        pts = sorted(points,lambda x,y:x.y - y.y)

        # split the array into the top two and bottom two
        top = pts[:2]
        bot = pts[2:]

        # order the top two l->r and the bottom two r->l
        top = sorted(top, lambda x,y:x.x - y.x)
        bot = sorted(bot, lambda x,y:y.x - x.x)

        return top+bot

    def get_max_square(self,square_list):
        '''Run through all the squares in a list and determine the largest outer square.'''

        # this should also probably be smarter about measuring distances to the
        # center point and splitting objects
        
        # should be an object prop
        max_perim = 40
        max_square = None

        for square in square_list:
            if square['perim'] > max_perim:
                max_perim = square['perim']
                max_square = square

        return max_square


    def get_bounding_box(self,square_list,scale=1.0,padding=0):
        '''Return a bounding box for the squares.'''

        max_sq = self.get_max_square(square_list)
        if not max_sq:
            return ()
        x_array = sorted([int(pt.x * scale + 0.5) for pt in max_sq['points']])
        y_array = sorted([int(pt.y * scale + 0.5) for pt in max_sq['points']])
        
        x_size = x_array[3] - x_array[0]
        y_size = y_array[3] - y_array[0]
        
        x_pad = 0
        y_pad = 0
        if padding:
            x_pad = int(x_size * padding + 0.5)
            y_pad = int(y_size * padding + 0.5)

        return (x_array[0] - x_pad,y_array[0] - y_pad,x_size+int(x_pad*2.0),y_size+int(y_pad*2.0))
        
        
    def compute_perspective_warp(self,square):
        '''Find the distortion matrix for a series of points in a square, remapped to a perfect square.'''

        size = square['perim'] / 4.0
        
        # the projected square is a perfect square surrounding the center point. It's created of 
        # equal length sides that are the perimeter / 4
        new_square_top_left = cvPoint(int(square['center'].x - size / 2.0 + 0.5), int(square['center'].y - size / 2.0 + 0.5))
        dest_points = [new_square_top_left,cvPoint(new_square_top_left.x + size, new_square_top_left.y),cvPoint(new_square_top_left.x + size, new_square_top_left.y + size), cvPoint(new_square_top_left.x, new_square_top_left.y + size)]

        # compute distortion matrix
        pt_array = CvPoint2D32f * 4
        p_mat = cvCreateMat(3, 3, CV_32F)
        p_src = pt_array(*((p.x, p.y) for p in square['points'] ))
        p_dst = pt_array(*((p.x, p.y) for p in dest_points))

        cvGetPerspectiveTransform(p_src, p_dst, p_mat)
        return p_mat, dest_points
        

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

            # self.oflow_test = max_square['points'][:]

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