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

        
class GraphicCard():
    '''A Class to implement a tracking object. Accesses shared buffer and does some operations on it.'''
    def __init__(self):
        self.font_tahoma_blk_16 = ImageFont.truetype("tahomabd.ttf", 16)

        self.font_tahoma_blk_13 = ImageFont.truetype("tahomabd.ttf", 14)
        self.font_tahoma_12 = ImageFont.truetype("tahoma.ttf", 13)
        self.font_tahoma_10 = ImageFont.truetype("tahoma.ttf", 12)

        # self.font_arial_blk_13 = ImageFont.truetype("ariblk.ttf", 13)
        # self.font_arial_12 = ImageFont.truetype("arial.ttf", 12)
        # self.font_arial_10 = ImageFont.truetype("arial.ttf", 10)



        self.line_spacing = 2
        self.padding = 30

        self.min_width = 50
        self.min_height = 50

        self.font_map = [self.font_tahoma_blk_16,
                        self.font_tahoma_12,
                        self.font_tahoma_12,
                        self.font_tahoma_12]


    def gen_sprite(self, card_data):
        '''Create a graphical representation of the user info.'''
        text_lines = []
        lines = []
        if card_data["twitter"]:
            text_lines.insert(0,'@'+card_data["twitter"])
        if card_data["title"]:
            text_lines.insert(0,card_data["title"])
            if card_data['company']:
                if len(card_data['company'] + card_data['title']) < 30:
                    text_lines[0] = text_lines[0] + " " + card_data["company"]
                else:
                    text_lines.insert(0,card_data["company"])
        elif card_data["company"]:
            text_lines.insert(0,card_data["company"])
        if card_data["first_name"]:
            text_lines.insert(0,card_data["first_name"])
            if card_data["last_name"]:
                text_lines[0] = text_lines[0] + " " + card_data["last_name"]

        if not text_lines:
            text_lines.append(card_data["full_name"])

        total_size = (2*self.padding,2*self.padding)
        pos = (self.padding,self.padding)
        for i in range(len(text_lines)):
            text = text_lines[i]
            my_line = Line(text,self.font_map[i])
            my_line.x = pos[0]
            my_line.y = pos[1]
            pos = (pos[0], pos[1] + my_line.height + self.line_spacing)
            total_size = (max(total_size[0],my_line.width + 2*self.padding), total_size[1] + my_line.height + self.line_spacing)
            lines.append(my_line)

        biz_card = RoundedRect(total_size, 10, (30,30,30,255), True)
        closer_img = biz_card.compose()
        # closer_img = Image.new('RGBA',(total_size[0],total_size[1]),(200,200,200,255))
        draw = ImageDraw.Draw(closer_img)
        for line in lines:
            draw.text((line.x, line.y), line.text, font=line.font, fill=(255,255,255))
            
        # if card_data["first_name"]:
        #     first_line = card_data["first_name"]
        # if card_data["last_name"]:
        #     first_line += " " + card_data["last_name"]


        # else:
        #     first_line = card_data["full_name"]
        
        
        # font_arial = ImageFont.truetype("arial.ttf", 15)

        py_closer = pygame.image.frombuffer(closer_img.tostring(), closer_img.size, closer_img.mode).convert_alpha() 

        # im = Image.new('RGBA',(200,100),(0,0,0,255))
        # draw = ImageDraw.Draw(im)
        # 
        # draw.text((2, 2), "Michael Kowalchik", font=self.font_arial)

        return py_closer
    
    
    def gen_closer_sprite(self):
        '''Create a graphical representation of the user info.'''

        text_lines = ["Bring Closer!"]
        lines = []
        
        total_size = (2*self.padding,2*self.padding)
        pos = (self.padding,self.padding)
        # for i in range(len(text_lines)):
        text = "Bring Closer!"
        my_line = Line(text,self.font_tahoma_blk_16)
        my_line.x = pos[0]
        my_line.y = pos[1]
        pos = (pos[0], pos[1] + my_line.height + self.line_spacing)
        total_size = (max(total_size[0],my_line.width + 2*self.padding), total_size[1] + my_line.height + self.line_spacing)
        lines.append(my_line)

        closer_img = Image.new('RGBA',(total_size[0],total_size[1]),(128,128,128,255))
        draw = ImageDraw.Draw(closer_img)
        for line in lines:
            draw.text((line.x, line.y), line.text, font=line.font, fill=(255,255,255))

        # if card_data["first_name"]:
        #     first_line = card_data["first_name"]
        # if card_data["last_name"]:
        #     first_line += " " + card_data["last_name"]


        # else:
        #     first_line = card_data["full_name"]


        # font_arial = ImageFont.truetype("arial.ttf", 15)

        py_closer = pygame.image.frombuffer(closer_img.tostring(), closer_img.size, closer_img.mode).convert_alpha() 

        # im = Image.new('RGBA',(200,100),(0,0,0,255))
        # draw = ImageDraw.Draw(im)
        # 
        # draw.text((2, 2), "Michael Kowalchik", font=self.font_arial)

        return py_closer