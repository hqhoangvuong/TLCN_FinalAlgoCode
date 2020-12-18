# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 09:53:48 2020

@author: Hikari
"""

from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    def __init__(self, maxDisappeared = 50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxAppeared = OrderedDict()

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.maxAppeared[self.nextObjectID] = 1
        self.nextObjectID += 1
        
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.maxAppeared[objectID]
        
    def update(self, rects):
        # Khong co boundingbox nao duoc tra ve ==> k co khuon mat duoc
        # phat hien trong khung hinh
        # Cap nhat tat ca cac khuon mat da luu rang da bien mat them 1 lan
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects, self.maxAppeared
        
        # Khoi tao ma tran toan 0 len(rects) hang, 2 cot 
        # [[X, Y], [X1, Y1], ..., [Xn, Yn]]
        inputCentroids = np.zeros((len(rects), 2), dtype = "int")
        # Tinh trong tam (giao diem 2 duong cheo) cua cac 
        # boundingbox hinh chu nhat, sau do luu vao mang inputCentroids
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            
        # Neu self.objects = 0, tuc la chua co khuon mat nao duoc dang ki
        # de tracking, dang ki het tat ca cac khuon mat phat hien duoc
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            
            # Tinh khoang cach vector giua trong tam cua cac boundingbox
            # cua nhung khuon mat da dang ki voi cac trong tam tinh tu cac 
            # input bounding box.
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            
            # De thuc hien phep so sanh, can phait im gia tri nho nhat cua 
            # moi hang, sau do sap xep sao cho hang co gia tri nho nhat duoc 
            # day len dau cua danh sach
            rows = D.min(axis = 1).argsort()
            
            # Thuc hien tuong tu voi cac cot, tim gia tri nho nhat cua moi cot,
            # sau do sap xep cac cot su dung cac hang da tinh toan o tren
            cols = D.argmin(axis = 1)[rows]
            
            # De co the biet duoc can update, dang ki moi hoac huy dang ki khuon
            # mat nao, chung ta can biet cot nao hoac hang nao da duoc duyet
            usedRows = set()
            usedCols = set()
            
            for (row, col) in zip(rows, cols):
                # Neu mot hang hoac cot da duoc duyet, bo qua
                if row in usedRows or col in usedCols:
                    continue
                
                # Lay objectID cua hang hien tai, gan lai toa do trong tam,
                # reset bien disappeared (so lan bien mat) ve 0 va tang so lan
                # khuon mat do xuat hien len 1 don vi
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                self.maxAppeared[objectID] += 1
                
                # Danh dau rang da duyet qua hang va cot do
                usedRows.add(row)
                usedCols.add(col)
            
            # Lay danh sach cac hang va cot chua duoc duyet qua
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            
            # Xet truong hop so trong tam cua nhung khuon mat da duoc dang ki 
            # lon hon hoac bang so trong tam tinh tu bounding box cua 
            # cac khuon mat lay tu input, can kiem tra xem co khuon mat nao bi
            # bien mat hay khong
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
            
        return self.objects, self.maxAppeared