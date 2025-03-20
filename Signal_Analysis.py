'''*** Import Section ***'''
from __future__ import division
from collections import Counter
import argparse
import os
import os.path as osp
import time
import torch
import cv2
import emoji
import warnings
from ultralytics import YOLO  # Updated YOLOv10 library
from util.parser import load_classes
from util.image_processor import preparing_image
from util.utils import non_max_suppression
from util.dynamic_signal_switching import switch_signal, avg_signal_oc_time

warnings.filterwarnings('ignore')

print('\033[1m' + '\033[91m' + "Kickstarting YOLO...\n")

# Argument Parsing
def arg_parse():
    parser = argparse.ArgumentParser(description="YOLO Vehicle Detection Model for Intelligent Traffic Management System")
    parser.add_argument("--images", dest='images', help="Image / Directory containing images to vehicle detection upon", 
                        default="vehicles-on-lanes", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1, type=int)
    parser.add_argument("--confidence_score", dest="confidence", help="Confidence Score to filter Vehicle Prediction", 
                        default=0.3, type=float)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshold", default=0.3, type=float)
    parser.add_argument("--weights", dest='weightsfile', help="weights file", 
                        default="weights/yolov10l.pt", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network", 
                        default="640", type=int)  # YOLOv10 works better with 640
    return parser.parse_args()

args = arg_parse()
images = args.images
batch_size = args.bs
confidence = args.confidence
nms_thesh = args.nms_thresh
CUDA = torch.cuda.is_available()

# Load class names
classes = load_classes("data/idd.names")

# Load YOLOv10 model
model = YOLO(args.weightsfile)
print('\033[0m' + "YOLO Neural Network Successfully Loaded..." + u'\N{check mark}')
inp_dim = args.reso

# Move model to GPU if available
if CUDA:
    model.to('cuda')

print('\033[1m' + '\033[92m' + "Performing Vehicle Detection with YOLO Neural Network..." + '\033[0m' + u'\N{check mark}')

# Load images
try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = [osp.join(osp.realpath('.'), images)]
except FileNotFoundError:
    print(f"No input found with the name {images}")
    exit()

loaded_ims = [cv2.imread(x) for x in imlist]
im_batches = list(map(preparing_image, loaded_ims, [inp_dim] * len(imlist)))
im_dim_list = torch.FloatTensor([(x.shape[1], x.shape[0]) for x in loaded_ims]).repeat(1, 2)

if CUDA:
    im_dim_list = im_dim_list.cuda()

lane_count_list = []
denser_lane = 0
lane_with_higher_count = 0
input_image_count = 0

print('\033[1m' + "SUMMARY")
print('\033[1m' + "-" * 120)

# Object Detection
for i, batch in enumerate(im_batches):
    if CUDA:
        batch = batch.to('cuda')

    start = time.time()
    results = model.predict(batch, conf=confidence)  # YOLOv10 prediction
    end = time.time()

    for im_num, image in enumerate(imlist[i * batch_size:min((i + 1) * batch_size, len(imlist))]):
        vehicle_count = 0
        input_image_count += 1
        objs = [classes[int(x.cls)] for x in results[0].boxes]  # Extract class names
        vc = Counter(objs)

        for obj in objs:
            if obj in ["car", "motorbike", "truck", "bicycle", "autorickshaw"]:
                vehicle_count += 1

        print(f'\033[1m Lane {input_image_count} - Number of Vehicles detected: {vehicle_count}')

        if vehicle_count > 0:
            lane_count_list.append(vehicle_count)

        if vehicle_count > lane_with_higher_count:
            lane_with_higher_count = vehicle_count
            denser_lane = input_image_count

        print('\033[0m' + "           Vehicle Type         Count")
        for key, value in sorted(vc.items()):
            if key in ["car", "motorbike", "truck", "bicycle"]:
                print(f"           {key:<15} {value}")

    if CUDA:
        torch.cuda.synchronize()

if not lane_count_list:
    print('\033[1m' + "No vehicles detected from the input.")

print('\033[1m' + "-" * 120)
print(emoji.emojize(':vertical_traffic_light:') + f'\033[1m Lane with denser traffic: Lane {denser_lane}\n')

switching_time = avg_signal_oc_time(lane_count_list)
switch_signal(denser_lane, switching_time)

print('\033[1m' + "-" * 120)

torch.cuda.empty_cache()
