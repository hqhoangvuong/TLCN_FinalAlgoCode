# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 13:00:00 2020

@author: Hikari
"""

import cv2
import os
import os.path
from pathlib import Path

class VideoWriter():
    def __init__(self, filename = 'result', dimension = (1920, 1080)):
        self.fileDir = self.makeDirectory(filename)
        self.result = cv2.VideoWriter(self.fileDir, 
                                      cv2.VideoWriter_fourcc(*'MJPG'), 
                                      10, 
                                      dimension)
        
    def makeDirectory(self, filename):
        self.saveDir = r'video_out'
        if not Path(self.saveDir).is_dir():
            os.mkdir(Path(self.saveDir))
            
        fileDir = self.saveDir + '/{0}.{1}'.format(filename, '.avi')
        if Path(fileDir).is_file():
            os.remove(Path(fileDir))
            
        return fileDir
            
    def writeFrame(self, frame):
        self.result.write(frame)