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


from square_detector import SquareDetector
from qrcode import qrcode

class CardUser(object):
    '''An object to track users in the frame.'''

    # have a small buffer of previous tracked bounding squares
    # needs a thread that operates on the object changing the state of the CardUser
    # states are "TooFar", "Scanning", "ID"

    def __init__(self):        
        self.frames = []



