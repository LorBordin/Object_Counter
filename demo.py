# Import needed packages
from tools import draw_object_bbox
from tools import SortTracker
from tools import Options
from tools import YoloDarknet
from tools import Barrier
from tools import VideoGet
from tools import Memory
import config as cfg
import numpy as np
import time
import cv2
import sys

# Set up flags
opt = Options()
args  = opt.parse()

# Import the models
yolo_coco = YoloDarknet(cfg.coco_labels, cfg.coco_weights, cfg.coco_config, .2, .35, 3)
tracker = SortTracker()

# Set the variables
cam = VideoGet(src=args["video_path"])
cam.start()
n_frames = 0
obj_name = args["class"].capitalize()

if args["mode"]=="gate_crossing":
    barrier = Barrier(get_line=args["get_line"], coord1=args["X1"], coord2=args["X2"])
    prev_objs = Memory()
    frame = cam.frame.copy()
    barrier.get_line_coords(frame)
    if args["mode"]=="gate_crossing" and n_frames==0:    
        barrier.compute_rect_equation()

# set the video output 
if args.get("output", False):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_path = args["output"]
    fps = 24
    writer = None

print("[INFO] Processing video...")

# Start the video stream 
start = time.time()
while True:
    
    # grab frame from video
    if cam.grabbed:
        frame = cam.frame.copy()
        H, W = frame.shape[:2]
    else:
        break
    
    # get persons and ppes
    objects, class_ids, _ = yolo_coco.predict(frame)
    
    # select just cars
    objects = objects[class_ids==args["class"]]

    # update tracker
    objects, ids = tracker.update(objects, (W,H))

    # update counter
    if args["mode"]=="gate_crossing":
        p_pos = np.vstack([objects[:, 0]+objects[:, 2]/2, objects[:,1]+objects[:, 3]]).transpose() 
        object_dict = {idx: x for idx, x in zip(ids, p_pos)}
        prev_objs.update(object_dict)
        if n_frames > 1:
            barrier.count_objects(object_dict, prev_objs)
        barrier.draw_barrier(frame, obj_name)
    else:
        text = f"{obj_name} count: {len(objects)}"
        cv2.rectangle(frame, (0, 0), (len(text) * 20, 50), black, 2)
        cv2.rectangle(frame, (0, 0), (len(text) * 20, 50), white, -1)
        cv2.putText(frame, text ,(15, 31), 0, 1, black, 2)
    

    # draw results
    draw_object_bbox(frame, objects, ids)
    
    if args.get("show_output", False):
        cv2.imshow("canvas", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break

    # write the video
    if args.get("output", False):
        if writer is None:
            writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H), True)
        writer.write(frame)

    print(f"\r[INFO] Processed  {n_frames}th frame...", end="")
    n_frames += 1
    
end = time.time()
cv2.destroyAllWindows()
print("[INFO] Avg fps: {:.2f}.".format(n_frames/(end-start)))
cam.stop()
sys.exit()


