# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 09:53:48 2020

@author: Hikari
"""

import insightface
import numpy as np
import cv2

class FaceDetector:
    def __init__(self, gpu_id):
        self.ctx_id = gpu_id
        self.model = insightface.app.FaceAnalysis()
        self.model.prepare(ctx_id = self.ctx_id, nms = 0.4)
        
    def detect(self, frame):
        self.faces = self.model.get(frame)
        return self.faces
    
    def draw_boundary_box(self, frame, user_face = None):
        faces_data = self.faces
        if user_face is not None:
            faces_data = user_face
        
        rects = []
            
        for idx, face in enumerate(faces_data):
            bbox = face.bbox.astype(np.int).flatten()
            rects.append(bbox)
            cv2.rectangle(frame, 
                          (bbox[0], bbox[1]), 
                          (bbox[2], bbox[3]), 
                          (0, 255, 0), 
                          2)
        return frame, rects
        
        
        