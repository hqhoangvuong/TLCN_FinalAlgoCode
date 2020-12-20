# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 09:53:48 2020

@author: Hikari
"""

import cv2
from video_reader import VideoReader
from face_detector import FaceDetector
from face_tagger import CentroidTracker
from video_writer import VideoWriter
import time

video_reader = VideoReader()
face_detector = FaceDetector(gpu_id = 0)
face_tagger = CentroidTracker(maxDisappeared = 20)

video_reader.setVideoPath(r'videos/video1.mp4')
video_reader.setFrameSavePath(r'savedframes')

def main():
    ret = True
    frame_dim = video_reader.getVideoDimension()
    video_writer = VideoWriter('abcd', frame_dim)
    
    start_time = time.time()
    x = 1
    counter = 0

    while ret:
        ret, frame, frame_no = video_reader.getFrame()
        if ret == False:
            break
        face_detector.detect(frame)
        rects = []
        frame, rects = face_detector.draw_boundary_box(frame)
        objects, maxAppereds = face_tagger.update(rects)
            
        for (objectID, centroid) in objects.items():
            text_color = (0, 0, 255)
            
            if maxAppereds.get(objectID) >= 50:
                text_color = (0, 255, 0)
            text = "ID {0}, Nguoi {1}".format(objectID, 'quen' if maxAppereds.get(objectID) >= 50 else 'la')
            cv2.putText(frame, 
                        text, 
                        (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        text_color,
                        2)
            
        video_writer.writeFrame(frame)
        
        counter += 1
        if (time.time() - start_time) > x :
            print("FPS: ", counter / (time.time() - start_time))
            counter = 0
            start_time = time.time()
        
        if 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()

