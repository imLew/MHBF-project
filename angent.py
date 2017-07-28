#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:31:30 2017

@author: dpenguin
"""
import numpy as np

class agent(object):
    
    bar_loc = 50
    staff_width = 20
    bar_height = 20
    bar_length = 40
    
    def __init__(self):
        self.position = np.array([0,0])
        
    def get_direction(self):
        direction = 2*np.pi*np.random.random()
        return direction
    
    def move(self):
        direction = self.get_direction()
        distance = np.random.normal(loc=3,scale=1.5)
        move = np.array([np.cos(direction)*distance,np.sin(direction)*distance])
        new_position = self.position+move
        if 0<new_position[0]<self.staff_width:
            print(new_position)

            return new_position
        elif self.position[1]>self.bar_height:
            print(new_position)
            return new_position
        elif self.position[1]<self.bar_height:
            print(new_position)
            print("You just ran into a wall!")

    