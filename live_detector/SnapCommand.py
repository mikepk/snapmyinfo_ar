#!/usr/bin/env python
# encoding: utf-8
"""
SnapCommand.py

Object to create a connection to, and send commands to, the snapmyinfo async command server.

Created by mikepk on 2009-09-22.
Copyright (c) 2009 Michael Kowalchik. All rights reserved.
"""

import sys
import getopt
import socket
import re

class SnapCommand():
    '''Issue commands to the async snapmyinfo command processor.'''

    def __init__(self,host='localhost',port=9999):
        self.host = host
        self.port = port

    def start(self):
        '''Connect to the socket in TCP mode.'''
        HOST, PORT = self.host,self.port
        # SOCK_STREAM == a TCP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #sock.setblocking(0)  # optional non-blocking
        self.sock.connect((HOST, PORT))

    def send(self,command):
        '''Method to call the command server with a specific command.'''
        if not command:
            return ''
        
        self.start()
        retries = 0
        while not re.search("READY",self.sock.recv(16384)):
            retries+=1
            if retries > 10:
                self.sock.close()
                return ''
            self.sock.send("\n");

        retries=0
        # server is ready, send the commands
        self.sock.send("%s\n" % command)

        return self.sock.recv(16384)  



def main(argv=None):
    pass

if __name__ == "__main__":
    sys.exit(main())
