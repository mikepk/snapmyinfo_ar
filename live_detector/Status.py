#!/usr/bin/python
#
import sys
import Image, ImageFont, ImageDraw
import pygame

from RoundedRect import RoundedRect

class StatusLine(object):
    '''A Class to implement individual status items.'''
    def __init__(self,text,font=None):
        self.text = text
        self.font = font
        if self.font:
            self.size = self.font.getsize(text)
            self.width = self.size[0]
            self.height = self.size[1]
        else:
            self.size = 0
            self.width = 0
            self.height = 0
        self.x = 0
        self.y = 0
        self.term = False

        
class Status():
    '''A Class to implement the status message buffer and graphical display.'''
    def __init__(self):
        self.font_tahoma_blk_18 = ImageFont.truetype("tahomabd.ttf", 18)
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

        self.font_map = [self.font_tahoma_blk_13,
                        self.font_tahoma_12,
                        self.font_tahoma_12,
                        self.font_tahoma_12]
                        
        self.text = None
        self.color = (50,50,100)
        self.transparency = 170
        
        self.status = []
        
        self.height = 50
        self.width = 200
        self.min_width = 200
        self.sprite = None
        self.max_width = 325


    def update(self):
        '''Update the status box.'''
        pass

    def wrap(self,message,myfont):
        '''Determine the word wrap and returns the proper number of lines.'''
        lines = []
        words = message.split(" ")
        temp_line = []
        size_accum = 0
        for word in words:
            size_accum += myfont.getsize(word+" ")[0]
            temp_line.append(word)
            if size_accum > self.max_width:
                lines.append(StatusLine(" ".join(temp_line), myfont))
                size_accum = 0
                temp_line = []

        if len(temp_line) > 0:
            lines.append(StatusLine(" ".join(temp_line), myfont))
        
        lines[-1].term = True
        #lines.reverse()

        tot_height = 0
        for line in lines:
            line.x = 20
            line.y = 20 + tot_height
            tot_height += line.height

        return lines      

    def write(self,message):
        '''Write a simple text message to the status display.'''
        new_lines = self.wrap(message,self.font_tahoma_blk_18)

        new_lines_height = sum([line.height for line in new_lines])
        # self.wrap(message,self.font_tahoma_blk_18)
        ht = 0
        if len(self.status) > 4:
            rem = self.status.pop()
            
        for line in self.status:
            line.y = line.y + new_lines_height
        
        new_lines[0].x = 20
        new_lines[0].y = 20

        self.status = new_lines + self.status
        self.height = 40 + sum([line.height for line in self.status])
        self.width = max(self.min_width, 40 + max([line.width for line in self.status]) )

        # self.width = max(self.width,)
        self.sprite = self.gen_sprite()

    
    def tweet(self,message,users):
        '''Show a twitter message on the display.'''
        pass
    
    


        
    
    def gen_sprite(self):
        '''Create a graphical representation of the status info.'''


        color = self.color + (self.transparency,)
        # status_img = Image.new('RGBA',(self.width,self.height),color)
        rect = RoundedRect((self.width,self.height),10,color,True)
        status_img = rect.compose() #self.round_rectangle((self.width,self.height),10,color)
        draw = ImageDraw.Draw(status_img)

        for line in self.status:
            draw.text((line.x, line.y), line.text, font=line.font, fill=(240,240,240) )
            if line.term and line is not self.status[-1]:
                draw.line(((10, line.y + line.height),(self.width - 10, line.y + line.height)), fill=self.color)


        return pygame.image.frombuffer(status_img.tostring(), status_img.size, status_img.mode).convert_alpha() 
