# Import needed packages
from tools import draw_object_bbox
from tools import draw_counter
from tools import SortTracker
from tools import YoloDarknet
from tools import Options
from tools import Barrier
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
yolo_coco = YoloDarknet(args["names"], args["weights"], args["config"], .4, .35, 1)
tracker = SortTracker()

# Set the variables
cam = cv2.VideoCapture(args["video_path"], cv2.CAP_ANY) 
f = open(args["names"], "r")
obj_name = f.read().strip().capitalize()
f.close()

if args["mode"]=="gate_crossing":
    barrier = Barrier(get_line=args["get_line"], coord1=args["X1"], coord2=args["X2"])
    prev_objs = Memory()
    _, frame = cam.read()
    barrier.get_line_coords(frame)
    if args["mode"]=="gate_crossing":    
        barrier.compute_rect_equation()

# set the video output 
if args.get("output", False):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_path = args["output"]
    fps = 24
    writer = None

print("[INFO] Processing video...")

# Start the video stream 
n_frames = 0
start = time.time()

while True:
    
    # grab frame from video
    grabbed, frame = cam.read()
    
    if not grabbed:
        break
    
    H, W = frame.shape[:2]
   
    # get persons and ppes
    objects, class_ids, _ = yolo_coco.predict(frame)

    # update tracker
    objects, ids = tracker.update(objects, (W,H))

    # draw results
    draw_object_bbox(frame, objects, ids)

    # update counter
    if args["mode"]=="gate_crossing":
        p_pos = np.vstack([objects[:, 0]+objects[:, 2]/2, objects[:,1]+objects[:, 3]]).transpose() 
        object_dict = {idx: x for idx, x in zip(ids, p_pos)}
        prev_objs.update(object_dict)
        if n_frames > 1:
            barrier.count_objects(object_dict, prev_objs)
        gate_coords = ((barrier.Xt, barrier.Yt),(barrier.Xb, barrier.Yb))
        draw_counter(frame, obj_name, barrier.counter, gate_coords=gate_coords)
    else:
        draw_counter(frame, obj_name, len(objects))
    
    if args["show_output"]:
        cv2.imshow("canvas", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break

    # write the video
    if args.get("output", False):
        if writer is None:
            writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H), True)
        writer.write(frame)

    print(f"\r[INFO] Processed {n_frames}th frame...", end="")
    n_frames += 1
    
end = time.time()
cv2.destroyAllWindows()
print("[INFO] Avg fps: {:.2f}.".format(n_frames/(end-start)))
sys.exit()


