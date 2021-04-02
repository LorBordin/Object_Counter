import numpy as np
import cv2

green = (0, 255, 0)
red = (255, 0, 0)
white = (255, 255, 255)
black = (0, 0, 0)

class Barrier():
    """
    Stores information about the gate and count the # of objects passing thorugh.
    Params:
        - get_line: If True allows to draw the gate on the first frame
        - frame: must be passed if get_line is set to True
        - coord1(2): coordinates of the gate (optional)

    """
    def __init__(self, get_line=True, frame=None, coord1=None, coord2=None):
        self.get_line = get_line
        if coord1 is not None:
            assert (coord1 is not None), "First coords not provided."
            assert (coord2 is not None), "Second coords not provided."
            self.Xt, self.Yt = coord1
            self.Xb, self.Yb = coord2
        
    def compute_rect_equation(self):
        """ Get the slope and intercept of the rect corresponding to the gate """
        self.slope = (self.Yb - self.Yt) / (self.Xb - self.Xt)
        self.q = self.Yt - self.slope * self.Xt
        self.counter = 0
    
    def count_objects(self, skiers, prev_skiers):
        for idx, curr_pos in skiers.items():
            for frame in prev_skiers.storage[1:]:
                if idx in frame.keys():
                    prev_pos = frame[idx]
                    crossed_line = check_coords(prev_pos, (self.slope, self.q)) and not check_coords(curr_pos, (self.slope, self.q))
                    if crossed_line:
                        self.counter += 1
                    break 
    
    def get_line_coords(self, frame):
        """ To draw the gate on the frame """
        assert frame is not None, "You must provide a frame reference in order to get the coordinates"
        draw_line_widget = DrawLineWidget(frame, "Barrier Selection")
        canvas = draw_line_widget.window_name
        cv2.moveWindow(canvas, 0, 0)
        while True:
            cv2.imshow(canvas, draw_line_widget.padded_img)
            if len(draw_line_widget.img_coords) == 2:
                cv2.polylines(draw_line_widget.padded_img, [np.array(draw_line_widget.img_coords, dtype=np.int32)], True, green, 2)
                cv2.imshow(canvas, draw_line_widget.padded_img)
                cv2.waitKey(200)
                cv2.destroyWindow(canvas)
                break
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        self.Xt, self.Yt, self.Xb, self.Yb =  np.array(draw_line_widget.img_coords).reshape(4)
        print(f"[{self.Xt}, {self.Yt}], [{self.Xb}, {self.Yb}]")
        self.get_line = False
        
    
class DrawLineWidget(object):
    
    def __init__(self, img, window_name, pct_padding=0):
        self.img = img
        self.bottom_padding = int(pct_padding * self.img.shape[0])
        self.padded_img = cv2.copyMakeBorder(self.img.copy(), 0, self.bottom_padding, 0, 0, cv2.BORDER_CONSTANT, None, black)
        self.window_name = window_name
        self.img_coords = []
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.extract_coords)

    def extract_coords(self, event, x, y, flags, parameters):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.img_coords.append((x, y))
            cv2.polylines(self.padded_img, [np.array(self.img_coords, dtype=np.int32)], False, red, 2)
            cv2.imshow(self.window_name, self.padded_img)
       
    
def check_coords(coords, line):
    """ Return True if the coords lies above the line, False otherwise. 
        Input: - coords (X, Y),
               - line (slope, intercept)
    """
    X, Y = coords
    m, q = line
    result = True if (Y < m*X + q) else False  
    return True if (Y < m*X + q) else False  
    