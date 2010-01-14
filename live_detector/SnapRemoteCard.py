#!/usr/bin/env python
# encoding: utf-8
"""
SnapRemoteCard.py

Object to create a connection to, and send commands to, the snapmyinfo card server.
Used for connecting to the snapmyinfo service over a socket, in the case of the
video station (for networking).

Created by mikepk on 2009-11-04.
Copyright (c) 2009 Michael Kowalchik. All rights reserved.
"""

import sys
import getopt
import socket
import pickle
import re
import time

class SnapRemoteCard():
    '''Interface to access the remote card server.'''

    def __init__(self,host='localhost',port=9555):
        self.host = host
        self.port = port
        self.start()


    def start(self):
        '''Connect to the socket in TCP mode.'''
        HOST, PORT = self.host,self.port
        # SOCK_STREAM == a TCP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #sock.setblocking(0)  # optional non-blocking
        self.sock.connect((HOST, PORT))
        self.sock.settimeout(2.0)


    def get_ready(self):
        '''Send newline codes until the server is in a known, ready, state.'''
        retries = 4
        data = ''
        while retries > 0:
            try:
                self.sock.send('\n')
                if re.search("READY",self.sock.recv(16384)):
                    return True
            except socket.timeout:
                retries -= 1
        return False



    def get_card(self,card_id):
        '''Method to call the card server requesting a card.'''
        if not card_id:
            return ''
        
        if not self.get_ready():
            return None

        # server is ready, send the commands
        self.sock.send("get:%s\n" % card_id)
        time.sleep(0.1)
        retries = 4
        while retries > 0:
            try:
                val = self.sock.recv(16384)
                return pickle.loads(val)
            except socket.timeout:
                #print "timeout... retrying"
                retries -= 1
        # We've run out of retries
        return None


    def connect(self,users):
        '''Method to call the card server requesting user connections.'''
        if not users:
            return ''

        if not self.get_ready():
            return False

        # server is ready, send the commands
        # create text array of user_ids
        users_data = [str(user_id) for user_id in users]
        user_string = ','.join(users_data)

        self.sock.send("connect:%s\n" % user_string)
        time.sleep(0.1)
        retries = 4
        while retries > 0:
            try:
                val = self.sock.recv(16384)
                return pickle.loads(val)
            except socket.timeout:
                #print "timeout... retrying"
                retries -= 1
        # we've run out of retries
        return False


def main(argv=None):
    pass

if __name__ == "__main__":
    src = SnapRemoteCard()
    codes = ['Q0B73s0p3ndlJK','9WDu17f29WQYoe','5N3CTckvHyT8A4']
    for c in codes:
        card = src.get_card(c)