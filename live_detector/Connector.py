#!/usr/bin/python
#
import sys
# import Image, ImageFont, ImageDraw
# import pygame

from SnapRemoteCard import SnapRemoteCard
from SnapCommand import SnapCommand

import threading
import time

class Connector():
    '''A Class to implement a thread that handles connecting users. Scans the Tracker pool for user ids and connects them. Keeps a cache of connections to avoid hitting the server unnesc.'''

    def __init__(self,pool,message_buffer):
        self.src = SnapRemoteCard()
        self.com = SnapCommand()
        self.pool = pool
        self.message_buffer = message_buffer
        self.connection_cache = []

    def start(self):
        '''Start the connector thread.'''
        connector_thread = threading.Thread(target=self.connection_check)
        connector_thread.start()
        
    def stop(self):
        '''Stop the connector thread'''
        self.running = False
    
    
    def add_to_cache(self,my_list):
        self.connection_cache.append(my_list)

    def check_cache(self,my_list):
        return my_list in self.connection_cache
    
    def connection_check(self):
        '''A thread to scan the pool of trackers for connections.'''
        self.running = True
        
        while self.running:
            # update 2 times a second
            time.sleep(0.5)

            visible_codes = []
            names = []
            for t_id in self.pool.active_trackers:
                #rect = pool.trackers[t_id].get_bound_rect()
                if self.pool.trackers[t_id].user_id:
                    visible_codes.append(self.pool.trackers[t_id].user_id["user_id"])
                    names.append(self.pool.trackers[t_id].user_id["full_name"])
                    # print '%d has user id of %s' % (t_id, str(self.pool.trackers[t_id].user_id))

            if len(visible_codes) > 1:
                visible_codes.sort()
                if not self.check_cache(visible_codes):
                    if self.src.connect(visible_codes):
                        # add a message to the message buffer saying who's connected now
                        who = " and ".join(names)
                        self.message_buffer.write('''%s are now connected.''' % who)
                        self.add_to_cache(visible_codes)
                        
                else:
                    time.sleep(2)