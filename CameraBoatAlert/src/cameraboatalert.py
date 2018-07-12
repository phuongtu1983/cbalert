# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import cv2
import tensorflow as tf
from utils import backbone

from camera.CameraThread import CameraThread

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

class CameraSource(object):
    def __init__(self, name="", source=0):
        self.name = name
        self.source = source
        
cameras = [CameraSource("Camera1","3.mp4"),CameraSource("Camera2","4.mp4")]

if __name__ == "__main__":
    
    detection_graph, category_index = backbone.set_model('model')
    
    for c in cameras:
        CameraThread(detection_graph, category_index, c.source, c.name).start()

