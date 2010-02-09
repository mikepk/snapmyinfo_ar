#!/usr/bin/python
#
import sys
import Image, ImageFont, ImageDraw
import pygame

from RoundedRect import RoundedRect

class Line(object):
    def __init__(self, text='', font=None, pos=None):
        self.text = text
        self.font = font
        if self.font:
            self.size = self.font.getsize(text)
            self.width = self.size[0]
            self.height = self.size[1]
        self.x = 0
        self.y = 0

        
class DebugDisplay():
    '''A Class to implement a tracking object. Accesses shared buffer and does some operations on it.'''
    def __init__(self):
        self.font_tahoma_blk_16 = ImageFont.truetype("tahomabd.ttf", 16)

        # self.font_tahoma_blk_13 = ImageFont.truetype("tahomabd.ttf", 14)
        # self.font_tahoma_12 = ImageFont.truetype("tahoma.ttf", 13)
        # self.font_tahoma_10 = ImageFont.truetype("tahoma.ttf", 12)
        # 
        # # self.font_arial_blk_13 = ImageFont.truetype("ariblk.ttf", 13)
        # # self.font_arial_12 = ImageFont.truetype("arial.ttf", 12)
        # # self.font_arial_10 = ImageFont.truetype("arial.ttf", 10)

        self.line_spacing = 2
        self.padding = 30

        self.min_width = 50
        self.min_height = 50


    def gen_sprite(self, fps):
        '''Create a graphical representation of the user info.'''

        # biz_card = RoundedRect(total_size, 10, (30,30,30,255), True)
        self.size = self.font_tahoma_blk_16.getsize(fps)
        closer_img = Image.new('RGBA',(90,self.size[1]), (0,0,0,255))

        draw = ImageDraw.Draw(closer_img)
        draw.text((0,0), fps, font=self.font_tahoma_blk_16, fill=(255,255,255,255))
            
        fps_sprite = pygame.image.frombuffer(closer_img.tostring(), closer_img.size, closer_img.mode).convert_alpha() 
        return fps_sprite
    
    
