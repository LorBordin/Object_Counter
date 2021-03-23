import numpy as np
import time
import cv2

# define some colors
green = (0, 255, 0)
red = (0, 0, 255)
blue = (255, 0, 0)
cyan = (255, 255, 0)
yellow = (0, 255, 255)
black = (0, 0, 0)
white = (255, 255, 255)

frame_color = {"person": white, "helmet": cyan, "vest": yellow}

def draw_object_bbox(img, coords, ids, print_info=True):
    '''
    inputs: - img: frame
            - coords: bbox coordinates list - format: (Xt, Yt, W, H)
            - ids: object ids list, 
    '''

    # change coords and fit inside frame
    H, W = img.shape[:2]
    coords[:, 2:] += coords[:, :2]
    coords[:, :2][coords[:, :2] < 0] = 0
    coords[:, 2][coords[:, 2] > W] = W
    coords[:, 3][coords[:, 3] > H] = H

    for box, Id in zip(coords, ids):
        Xmin, Ymin, Xmax, Ymax = box
        crop = img[Ymin:Ymax, Xmin:Xmax]
        # draw a bbox surrounding the object
        cv2.rectangle(img, (Xmin, Ymin), (Xmax, Ymax), white, 2)
        # color the inside of the bbox according to her status
        color = white 
        rect = (np.ones(crop.shape) * color).astype("uint8")
        img[Ymin:Ymax, Xmin:Xmax] = cv2.addWeighted(crop, 0.7, rect, 0.3, 1.0)
        # write info
        if print_info:
            text = f"Id {Id}"
            cv2.rectangle(img, (Xmin, Ymin-20), (Xmin + len(text) * 10, Ymin), white, -1)
            cv2.putText(img, text ,(Xmin, Ymin-5), 0, .5, blue, 1)