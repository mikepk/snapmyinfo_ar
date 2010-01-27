#!/usr/bin/python
#
import sys

import threading
import time
# import pygame
# from pygame.locals import *
# 
# from square_detector import SquareDetector
# from qrcode import qrcode
from TrackedCard import TrackedCard


class TrackerPool(object):
    '''An object to manage several trackers, run the tracking functions, deal with idle and working states.'''

    # have a small buffer of previous tracked bounding squares
    # needs a thread that operates on the object changing the state of the CardUser
    # states are "TooFar", "Scanning", "ID"

    def __init__(self, my_buffer, number_of_trackers=1):
        # create a number of trackers and assign them to the pool
        #self.image_buffer = my_buffer

        self.active_trackers = []
        self.idle_trackers = []
        self.trackers = []
        for i in range(number_of_trackers):
            tc = TrackedCard(i,my_buffer)
            self.trackers.append(tc)
            self.idle_trackers.append(i)
            
            tracker_thread = threading.Thread(target=tc.analyze)
            #,args=(left_rect,zone1))
            tracker_thread.start()
            

        self.orphan_list = []
        self.orphan_frames = []


    def stop(self):
        print "Shutting down all trackers."
        for tracker in self.trackers:
            tracker.running = False
        time.sleep(0.25)

    def update(self):
        '''Update all trackers'''

        for tracker in self.trackers:
            tracker.update()

        # reset the active and idle tracker lists
        self.active_trackers = []
        self.idle_trackers = []
        for tracker in self.trackers:
            if tracker.idle:
                self.idle_trackers.append(tracker.id)
            else:
                self.active_trackers.append(tracker.id)
        
        # update oprhans
        self.orphan_frames.insert(0,self.orphan_list)
        self.orphan_list = []
        if len(self.orphan_frames) > 6:
            self.orphan_frames.pop()
        
    def check(self,squares):
        '''Check all candidate squares.'''
        num = 0
        for sq in squares:
            matched = False
            for track_id in self.active_trackers:
                matched = self.trackers[track_id].check(sq)
                # square matched one of the actives
                if matched:
                    break

            if matched:
                continue

            num += 1
            
            if self.check_orphans(sq):
                continue

            self.add_orphan(sq)

        # print "Number of squares: %d  No Match: %d" % (len(squares), num)


    def check_orphans(self,sq):
        # check that this square hits in the previous three
        # frames for tracking
        hits = 0
        for frame in self.orphan_frames:
            if sq["bound"].collidelist(frame) >= 0:
                hits += 1

        # if > 2 hits, then add this to the trackers
        # print str(self.idle_trackers)
        if hits > 5 and len(self.idle_trackers) > 0:
            new_id = self.idle_trackers.pop()
            self.trackers[new_id].add(sq)
            return True
        else:
            return False

    def add_orphan(self,sq):
        self.orphan_list.append(sq["bound"])
