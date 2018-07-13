# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
from threading import Thread, Lock
import cv2
import numpy as np
import tensorflow as tf

from utils import visualization_utils as vis_util


class CameraAlertThread :
    def __init__(self, detection_graph, category_index, src = 0, name = "", capture_area=[], width = 600, height = 600) :
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.started = False
        self.name = name
        self.detection_graph = detection_graph
        self.category_index = category_index
        self.capture_area = capture_area
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print("Already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.object_dectection, args=())
        self.thread.start()

    def read(self) :
        self.read_lock.acquire()
        (self.grabbed, self.frame) = self.stream.read()
        self.read_lock.release()
            
    def is_opened(self) :
        self.read_lock.acquire()
        if self.stream.isOpened():
            result = True
        else:
            result = False
        self.read_lock.release()
        return result

    def __exit__(self) :
        self.stream.release()
        
    def object_dectection(self):
        with self.detection_graph.as_default():
          with tf.Session(graph=self.detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

            targeted_objects = ["cell phone", "boat"]
            # for all the frames that are extracted from input video
            while(self.is_opened()):
                self.read()
                if not self.grabbed:
                    print("End of the video file...")
                    break
                    
                input_frame = self.frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # Visualization of the results of a detection.        
                is_vehicle_detected, predicted_direction  = vis_util.visualize_boxes_and_labels_on_image_array(input_frame, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), self.category_index, targeted_objects, self.capture_area, use_normalized_coordinates=True)
                                    
                lenght = len(self.capture_area)
                for i in range(0,lenght-1):
                    cv2.line(input_frame, self.capture_area[i], self.capture_area[i+1], (0, 0, 255), 2)
                
                if is_vehicle_detected:
                    cv2.putText(input_frame,predicted_direction, self.capture_area[0], cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255, 255), 2,cv2.FONT_HERSHEY_SIMPLEX)
    
                cv2.imshow(self.name,input_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            self.__exit__()
            cv2.destroyWindow(self.name)
