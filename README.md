# Object Counter

Simple program object for detection and counting. There are two working modes, count objects (in the scene) or gate crossing count. 

To detect the objects the program uses a [Yolo network](https://github.com/AlexeyAB/darknet). While the demo version comes with a pretrained model that can detect persons, bicycles or cars, `object_count.py` is suited to detect custom objects. In that case you must provide a trained model.

<div align="center">
	<h4>Counting mode</h4>
	<img src="https://raw.githubusercontent.com/LorBordin/object_counter/master/examples/football.gif" width="500">
</div> 

<div align="center">
	<h4>Gate Crossing mode</h4>
	<img src="https://raw.githubusercontent.com/LorBordin/object_counter/master/examples/cars.gif" width="500">
</div> 


## Setup

- Install or build **OpenCV** (building is required for GPU support - recomended). 
- Clone this repo `git clone https://github.com/LorBordin/object_counter`.
- Change directory `cd object_counter`.
- Install  requirements `pip install -r requirements.txt`.

### To use the demo
- Download the Yolo-COCO [weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) and move them in `models/yolo-coco`.
- run `python demo.py -m counter -v PATH_TO_INPUT_VIDEO`.

**Optional arguments:**
	
  `-m MODE`: *Options: *counter* or *gate_crossing** 	
  `-v VIDEO_PATH`: Path to input video	
  `-y YOUTUBE_URL`: YouTube video URL	
  `-o VIDEO_PATH`: Path to output video		
  `-s 0_or_1`: If 0 doesn't show the live output - deafult 1	                   
  `-l GATE_COORDS`: Gate coords - format: `"[Xt,Yt] [Xb,Yb]"`	
  `-c CLASS`: Object class. Options: *person*, *bicycle*, *car*

### To use your custom weights

Coming soon...