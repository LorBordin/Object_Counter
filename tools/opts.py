import argparse
import pafy
import cv2

class Options():
    """ Initial settings """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-m", "--mode", required=True,
            help="Options: counter or gate_crossing")
        self.parser.add_argument("-v", "--video_path",
            help="Path to input video")
        self.parser.add_argument("-o", "--output", 
            help="[Optional] Path to output video")
        self.parser.add_argument("-s", "--show_output", type=int, default=1,
            help="If 0 it does not show the live output")
        self.parser.add_argument("-l", "--coords", 
            help="line coords - format: '[Xt,Yt] [Xb,Yb]'")
        self.parser.add_argument("-y", "--youtube",
            help="path to a YouTube video (url)")
        self.parser.add_argument("-c", "--class", default="person", 
            help="Object name. Options: person, bicycle, car")
        
    def parse(self):
        args = vars(self.parser.parse_args())
        if args.get("coords", False):
            x1, x2 = args["coords"].split()
            Xt, Yt = x1.split(",")
            Xb, Yb = x2.split(",")
            Xt, Yt = int(Xt[1:]), int(Yt[:-1])
            Xb, Yb = int(Xb[1:]), int(Yb[:-1])
            x1, x2 = (Xt, Yt), (Xb, Yb)
            get_line = False
        else:
            x1, x2 = None, None
            get_line = True  
        
        args["X1"], args["X2"] = x1, x2
        args["get_line"] = get_line

        if args.get("youtube", False):
            url = args["youtube"]
            video = pafy.new(url)
            best = video.getbest(preftype="mp4")
            args["video_path"] = best.url

        if args["class"]=="person":
            args["class"] = 0 
        elif args["class"]=="bicycle":
            args["class"] = 1
        else:
            args["class"] = 2

        return args
