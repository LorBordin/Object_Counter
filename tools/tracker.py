from norfair import Detection, Tracker
import numpy as np

def euclidean_distance(detection, tracked_object):
    """ Compute the Euclidean distance """
    return np.linalg.norm(detection.points - tracked_object.estimate)

def get_centroid(yolo_box):
    """ Get bbox centroids  """
    Xt, Yt, W, H = yolo_box
    return np.array([Xt-W/2, Yt-H/2])

class SortTracker:
    """ Simple Norfair tracker wrapper """
    def __init__(self, 
                 distance = euclidean_distance, 
                 distance_threshold = .25, 
                 initialization_delay = 2):    
        self.tracker = Tracker(euclidean_distance, distance_threshold, initialization_delay)

    def update(self, bboxes, frame_size):
        detections = [Detection(get_centroid(box)/frame_size, data={"box":box}) for box in bboxes]
        tracked_objects = self.tracker.update(detections, period=1)
        boxes, ids = [], []
        for track in tracked_objects:
            if not track.live_points: 
                continue
            boxes.append(track.box)
            ids.append(track.id)  
        if len(boxes)>0:          
            return np.array(boxes, dtype="int"), np.array(ids, dtype="int")
        else: 
            return np.empty((0,4), dtype="int"), np.empty(0, dtype="int") 
        