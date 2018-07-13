# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import numpy as np
import cv2
from camera.CameraThread import CameraThread

CANVAS_SIZE = (600,600)
FINAL_LINE_COLOR = (0, 0, 255)
WORKING_LINE_COLOR = (127, 127, 127)

class PolygonDrawer(object):
    def __init__(self, window_name, camera_thread):
        self.window_name = window_name # Name for our window
        self.camera_thread = camera_thread
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon


    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.points.pop(len(self.points)-1)


    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.window_name, np.zeros(CANVAS_SIZE, np.uint8))
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        frame = None
        while(not self.done):
            frame = self.camera_thread.read()
            if frame is None:
                print("End of file")
                break
            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv2.polylines(frame, np.array([self.points]), False, FINAL_LINE_COLOR, 1)
                # And  also show what the current segment would look like
                cv2.line(frame, self.points[-1], self.current, WORKING_LINE_COLOR)
            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) == 27 :
                break

        # User finised entering the polygon points, so let's make the final drawing
        if frame is not None:
            # of a filled polygon
            if (len(self.points) > 0):
                cv2.fillPoly(frame, np.array([self.points]), FINAL_LINE_COLOR)
            # And show it
            cv2.imshow(self.window_name, frame)
            
        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyWindow(self.window_name)


def save_region(points):
    file = open("region.txt","w")
    file.write('\n'.join('%s,%s' % x for x in points))
    file.close() 

if __name__ == "__main__":
    vs = CameraThread(0).start()
    pd = PolygonDrawer("Camera", vs)
    image = pd.run()
    print("Polygon = %s" % pd.points)
    save_region(pd.points)
    vs.__exit__()
    vs.stop()
    cv2.destroyAllWindows()
