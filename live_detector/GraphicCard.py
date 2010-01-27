#!/usr/bin/python
#
import sys
import Image, ImageFont, ImageDraw
import pygame

class GraphicCard():
    '''A Class to implement a tracking object. Accesses shared buffer and does some operations on it.'''
    def __init__(self):
        self.font_arial = ImageFont.truetype("arial.ttf", 15)
        self.line_spacing = 5
        self.padding = 20

        self.min_width = 50
        self.min_height = 50


    def gen_sprite(self, card_data):
        '''Create a graphical representation of the user info.'''
        
        name = card_data["full_name"]
        
        
        # font_arial = ImageFont.truetype("arial.ttf", 15)
        size = self.font_arial.getsize(name)


        closer_img = Image.new('RGBA',(size[0]+2*self.padding,size[1]+2*self.padding),(200,200,200,255))
        draw = ImageDraw.Draw(closer_img)
        draw.text((self.padding, self.padding), name, font=self.font_arial, fill=(7,7,7))

        py_closer = pygame.image.frombuffer(closer_img.tostring(), closer_img.size, closer_img.mode).convert_alpha() 

        # im = Image.new('RGBA',(200,100),(0,0,0,255))
        # draw = ImageDraw.Draw(im)
        # 
        # draw.text((2, 2), "Michael Kowalchik", font=self.font_arial)

        return py_closer
    