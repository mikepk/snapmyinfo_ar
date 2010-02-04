#!/usr/bin/python
#
import sys
import Image, ImageDraw

class RoundedRect(object):
    def __init__(self,size, radius, fill, hl=False):

        if hl:
            self.hl = True

        self.size = size
        self.radius = radius
        self.fill = fill

        self.hl_top = Image.open('images/light_hilite.png')
        self.hl_left = self.hl_top.rotate(90)

        self.hl_bot = Image.open('images/dark_hilite.png').rotate(180)
        self.hl_right = self.hl_bot.rotate(90)

        self.rect = Image.new('RGBA', self.size, self.fill)
        self.mask = self.gen_mask(self.size,self.radius)
        # base_rect.paste(rect,mask=self.mask)
        #return self.compose()

    def compose(self):
        base_rect = Image.new('RGBA', self.size, (0,0,0,0))
        if self.hl:
            self.apply_hilites()
        base_rect.paste(self.rect,mask=self.mask)
        return base_rect
        

    def round_corner(self,radius, fill):
        """Draw a round corner"""
        corner = Image.new('RGBA', (radius, radius), (0, 0, 0, 0))
        draw = ImageDraw.Draw(corner)
        draw.pieslice((0, 0, radius * 2, radius * 2), 180, 270, fill=fill)
        return corner

    def gen_mask(self,size,radius):
        width, height = size
        mask = Image.new('RGBA', size, (255,255,255,255))
        corner = self.round_corner(radius, (255,255,255,255))
        mask.paste(corner, (0, 0))
        mask.paste(corner.rotate(90), (0, height - radius)) # Rotate the corner and paste it
        mask.paste(corner.rotate(180), (width - radius, height - radius))
        mask.paste(corner.rotate(270), (width - radius, 0))
        return mask


    def apply_hilites(self):
        # rectangle = Image.new('RGBA', size, (0,0,0,0))
        width,height = self.size

        self.rect.paste(self.hl_top.convert('RGB'),(0,0),mask=self.hl_top)

        self.rect.paste(self.hl_left.convert('RGB'),(0,0),mask=self.hl_left)

        self.rect.paste(self.hl_bot.convert('RGB'),(0,height-self.hl_bot.size[1]),mask=self.hl_bot)

        self.rect.paste(self.hl_right.convert('RGB'),(width-self.hl_right.size[0],0),mask=self.hl_right)


# def round_rectangle(self,size, radius, fill):
#     """Draw a rounded rectangle"""
#     width, height = size
#     base = Image.new('RGBA', size, (0,0,0,0))
#     rectangle = Image.new('RGBA', size, fill)
# 
#     self.apply_hilites(size,rectangle)
# 
#     base.paste(rectangle,mask=mask)
#     
#     return base


