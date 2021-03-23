from threading import Thread
import cv2
import time

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0, flag=cv2.CAP_ANY):
        self.stream = cv2.VideoCapture(src, flag) 
        self.frame = None
        self.grabbed = False
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        
    def start(self):
        self._ref = Thread(target=self.get, args=())
        self._ref.start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
                time.sleep(0.03)

    def stop(self):
        self.stopped = True
