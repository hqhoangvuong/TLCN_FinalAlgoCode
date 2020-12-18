# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 09:53:48 2020

@author: Hikari
"""

import cv2
from video_reader import VideoReader
from frame_enhancer import LowLightEnhance
from face_detector import FaceDetector
from face_tagger import CentroidTracker
from video_writer import VideoWriter

video_reader = VideoReader()
low_light_enhancer = LowLightEnhance('snapshots/Epoch99.pth', 0)
face_detector = FaceDetector(gpu_id = 0)
face_tagger = CentroidTracker(maxDisappeared = 25)

video_reader.setVideoPath(r'videos/video2.mp4')
video_reader.setFrameSavePath(r'savedframes')

def main():
    ret = True
    frame_dim = video_reader.getVideoDimension()
    video_writer = VideoWriter('abcd', frame_dim)
    while ret:
        ret, frame, frame_no = video_reader.getFrame()
        if ret == False:
            break
        
        rects = []
        frame, rects = face_detector.draw_boundary_box(frame)
        objects, maxAppereds = face_tagger.update(rects)
            
        for (objectID, centroid) in objects.items():
            text = "ID {0}".format(objectID)
            text_color = (255, 0, 0)
            
            if maxAppereds.get(objectID) >= 50:
                text_color = (0, 255, 0)
                
            cv2.putText(frame, 
                        text, 
                        (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        text_color,
                        2)
            
            cv2.circle(frame, 
                       (centroid[0], centroid[1]), 
                       4, 
                       (0, 255, 0), 
                       -1)
            
        video_writer.writeFrame(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()

