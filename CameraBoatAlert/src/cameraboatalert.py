# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import tensorflow as tf
import os
from camera.CameraAlertThread import CameraAlertThread
from utils import backbone

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

class CameraSource(object):
    def __init__(self, name="", source=0):
        self.name = name
        self.source = source
        
cameras = [CameraSource("Camera1", 0)]

if __name__ == "__main__":
    
    file_name="region.txt"
    capture_area = []
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            try:
                area = []
                file = open(file_name, "r") 
                for line in file: 
                    area=line.strip().split(',')
                    capture_area.append(tuple(map(int,area)))
            except : # whatever reader errors you care about
                print("File not exist")

    if len(capture_area)==0:
        capture_area = [(0,0),(0,0)]
    else:
        capture_area.append(capture_area[0])

    detection_graph, category_index = backbone.set_model('model')
    for c in cameras:
        CameraAlertThread(detection_graph, category_index, c.source, c.name, capture_area).start()

