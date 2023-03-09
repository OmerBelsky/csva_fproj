# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 23:08:10 2023

@author: atver
"""
import datetime
def time_str():
    time_str = str(datetime.datetime.now()).split('.')[0].replace(':', ' ').replace('-', ' ').replace(' ', '_')
    return time_str