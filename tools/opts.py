import argparse
import pafy
import cv2

class Options():
    """ Initial settings """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-v", "--video_path",
            help="Path to input video")
        self.parser.add_argument("-o", "--output", 
            help="[Optional] Path to output video")
        self.parser.add_argument("-s", "--show_output", type=int, default=1,
            help="If 0 it does not show the live outut")
        self.parser.add_argument("-l", "--coords", 
            help="line coords - format: '[Xt,Yt] [Xb,Yb]'")
        self.parser.add_argument("-u", "--youtube_url", type=int, default=0,
            help="if 1 allows to get video from YouTube")
        
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

        if args["youtube_url"]:
            url = args["video_path"]
            video = pafy.new(url)
            best = video.getbest(preftype="mp4")
            args["video_path"] = best.url

        return args
