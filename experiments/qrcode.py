#!/usr/bin/env python
# encoding: utf-8
"""
qrcode.py

Created by mikepk on 2009-07-31.
Copyright (c) 2009 Michael Kowalchik. All rights reserved.
"""

import sys
import os
import unittest

import Image, ImageFile, ImageEnhance
import string

import socket
import re

import StringIO


class qrcode:
    def __init__(self,host='localhost',port=9090):
        self.host = host
        self.port = port


    def encode_image(self,value,size=480):
        '''Create a PIL image with a border and of the requested size with footer info appended.'''
        
        img = self.generate_qrimage(StringIO.StringIO(self.encode(value)),size)
        return img


    def encode_image_with_footer(self, value, size=480):
        '''Create an qrcode image with the footer.'''

        img = self.encode_image(value, size)

        footer = Image.open('/usr/share/enerd/snapmyinfo/content/images/qr_footer.png')        
        footer_size = (size, int(float(footer.size[1]) * size / footer.size[0])+1)
        footer = footer.resize(footer_size,Image.ANTIALIAS)
        
        canvas = Image.new("L",(img.size[0],img.size[1]+footer.size[1]),255)
        canvas.paste(img,(0,0,img.size[0],img.size[1]))
        canvas.paste(footer,(0,img.size[1],img.size[0],img.size[1]+footer.size[1]))
        return canvas


    
    def generate_qrimage(self,img,requested_size=480):
        '''Takes the custom qr image from the encoder and produce a standard image, scaled to [resolution] size.'''

        im = Image.open(img)

        multiplier = requested_size / im.size[0] + 1

        # add 5% border aroung image
        padding = int(requested_size * 0.05)+1
        im = im.resize((im.size[0]*multiplier,im.size[1]*multiplier))
        
        canvas = Image.new("L",(im.size[0]+padding*2,im.size[1]+padding*2),255)
        canvas.paste(im,(padding,padding,im.size[0]+padding,im.size[1]+padding))
        canvas = canvas.resize((requested_size,requested_size),Image.ANTIALIAS)
        return canvas
    
    def encode(self,value):
        '''Take a string and return the QR code in string representing the custom QR img format.'''
        self.start()
        data = self.encoder_client(value)
        self.done()
        return data
        
    def decode(self,img_file):
        '''Take any image file and return the value of the QR code in it.'''
        self.start()
        data = self.decoder_client(img_file)
        self.done()
        return data

    def decode_hard(self,img_file):
        '''Try various image processing techniques to enhance the image to try and extract the bar code.'''
        # modify brightness, then sharpness, then contrast, trying to decode at each
        grayscale_image = ImageEnhance.Color(img_file).enhance(0)
        for contrast_val in [1,0.75,1.25]:
            c = ImageEnhance.Contrast(grayscale_image).enhance(contrast_val)
            for sharp in [1,1.5,2]:
                s = ImageEnhance.Sharpness(c).enhance(sharp)
                for bright in [1,1.5,2,2.5,3]:
                    data = self.decode(ImageEnhance.Brightness(s).enhance(bright))
                    if data != 'NO BARCODE':
                        print '''contrast: %f, sharpness, %f, brightness: %f''' % (contrast_val,sharp,bright)
                        return data
        return data
    	
        # for contrast_val in [1,0.75,1.25]:
        #     for sharp in [1,1.5,2]:
        #         for bright in [1,1.5,2,2.5,3]:
        #             data = self.decode(ImageEnhance.Contrast(ImageEnhance.Brightness(ImageEnhance.Sharpness(img_file).enhance(sharp)).enhance(bright)).enhance(contrast_val))
        #             if data != 'NO BARCODE':
        #                 return data
        # return data


    def start(self):
        '''Connect to the socket in TCP mode.'''
        HOST, PORT = self.host,self.port
        # SOCK_STREAM == a TCP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #sock.setblocking(0)  # optional non-blocking
        self.sock.connect((HOST, PORT))
    
    def done(self):
        self.sock.send("done\n")
        reply = self.sock.recv(16384)  # wait for the server to disconnect
        # add a timeout here somehow?
        self.sock = None
    
    
    def decoder_client(self,img):
        '''Method to call the Encoder/Decoder service with the decode command.'''
        # turn the image into a string for transfer over the socket
        memfile = StringIO.StringIO()
        img.save(memfile, "JPEG")
        string = memfile.getvalue()
        memfile.close()
        
        #reply = sock.recv(16384)
        retries = 0
        while not re.search("QRService",self.sock.recv(16384)):
            #wait for the server to announce ready
            retries+=1
            if retries > 10:
                self.sock.close()
                return ''
            self.sock.send("\n");
        retries=0
        # server is ready, send the encode command
        self.sock.send("decode\n")

        while not re.search("data",self.sock.recv(16384)):
            # reply = self.sock.recv(16384)  # limit reply to 16K
            retries+=1
            if retries > 10:
                self.sock.close()
                return ''
            self.sock.send("decode\n")

        for line in string.split('\n'):
             self.sock.send(line+'\n')
        self.sock.send("\n")
        self.sock.send("EOF\n")

        # The next server response should be the decoded string
        return self.sock.recv(16384)  
        
    
    def encoder_client(self,value):
        '''Method to call the Encoder/Decoder service with the encode command.'''
        #sock = self.connect_socket()
        #self.sock.send("\n");
        #reply = sock.recv(16384)
        retries = 0
        while not re.search("QRService",self.sock.recv(16384)):
            retries+=1
            if retries > 10:
                self.sock.close()
                return ''
            self.sock.send("\n");
            
        retries=0
        # server is ready, send the encode command
        self.sock.send("encode\n")

        while not re.search("data",self.sock.recv(16384)):
            retries+=1
            if retries > 10:
                self.sock.close()
                return ''
            self.sock.send("encode\n");

        for line in value.split('\n'):
            self.sock.send(value+'\n')

        # The next server response should be the encoded string
        return self.sock.recv(16384)  


# A custom file reader for the compressed, minimized encoded QR code
# this is what the QR encoder produces (usually a 29px by 29px )
# byte stream of white and black (0, 255)
class QrCodeImage(ImageFile.ImageFile):
    '''Special custom QR format generated by the encoder'''
    format = "QR"
    format_description = "QR code raster image"

    def _open(self):
        header_size = 64
        # check header
        header = self.fp.read(header_size)
        if header[:2] != "QR":
            raise SyntaxError, "not a QR code"

        header = string.split(header)

        # size in pixels (width, height)
        self.size = int(header[1]), int(header[2])

        # set the mode to grayscale
        self.mode = "L"

        # data descriptor
        self.tile = [
            ("raw", (0, 0) + self.size, header_size, (self.mode, 0, 1))
        ]

# register the new format reader with the Image class
Image.register_open("QRCODE", QrCodeImage)
Image.register_extension("QRCODE", ".qr")


class qrcodeTests(unittest.TestCase):
    def setUp(self):
        pass


if __name__ == '__main__':
    unittest.main()