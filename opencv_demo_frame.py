import argparse
parser = argparse.ArgumentParser(description="finetine ViCLIP for closed-set actdet")
parser.add_argument("--REC", action='store_true',
                    help="record stream")
parser.add_argument("--DISABLEEMPTY", action='store_true',
                    help="disable humans with undetected actions")
parser.add_argument("-rate", type=int, default=2,
                    help="framerate")
parser.add_argument("-act", type=lambda s: s.split(","),
                    help="Comma-separated list of actions")
parser.add_argument("-color", type=str, default='green',
                    help="color to plot predictions")
parser.add_argument("-font", type=float, default=1.0,
                    help="font size")
parser.add_argument("-line", type=int, default=1,
                    help="line thickness")
parser.add_argument("-weights", type=str,
                    default='weights/avak_aws_stats_flt_b16_txtaug_txtlora/avak_b16_10.pt',
                    #default='weights/k700_stats_flt_b16_txtaug_txtlora/avak_b16_11.pt',
                    help="path to trained weights")
parser.add_argument("-thresh", type=float, default=0.25,
                    help="cosine threshold")
args = parser.parse_args()

COLORS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}
color = COLORS[args.color]
font = args.font
thickness = args.line

import os
import numpy as np
import cv2
import time

'''
cv2.imshow('ffmpeg fix', np.array([1], dtype=np.uint8)) #temp pyav-cv2 fix
cv2.destroyAllWindows()
'''
outsize=(480,640) #(800,1280)
out_frame = np.full((outsize[0],outsize[1],3), 0.)
cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL) 
cv2.imshow('RealSense', out_frame)
print('Press any key to start buffering')
cv2.waitKey(0)

import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.ops import batched_nms
from sia import get_sia, PostProcessViz
from datasets import avatextaug

from util.box_ops import box_cxcywh_to_xyxy

device = "cuda:0" if torch.cuda.is_available() else "cpu"

############
# Load SIA #
############
postprocess = PostProcessViz()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_sia(size='b', pretrain=None, det_token_num=20, text_lora=True, num_frames=9)['sia']
model.load_state_dict(torch.load(args.weights, weights_only=True), strict=False)
model.to(device)
model.eval()
print('Model Loaded')

captions = args.act
if captions == None:
    captions = list(avatextaug.keys())
captions = [s.replace('_', ' ') for s in captions]
temp_num_classes = len(captions)

text_embeds = model.encode_text(captions)
text_embeds = F.normalize(text_embeds, dim=-1)

imgsize=(240,320) #(120,160) #(180,240) #(240,320)

#tfs = v2.Compose([v2.Resize(imgsize),
#                  v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

tfs = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

########################
# Initialize Realsense #
########################
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

if args.REC:
    writer = cv2.VideoWriter('realsense_res.mp4',  
                             cv2.VideoWriter_fourcc(*'MJPG'), 
                             20, (outsize[1],outsize[0])) 

###################
# Start streaming #
###################

buffer_max_len = 9 * args.rate # multiples of 9
mididx = buffer_max_len // 2
indices = np.arange(0, buffer_max_len, buffer_max_len // 9)
buffer = []
plotbuffer = []
init = 0
while True:
    #time.sleep(0.53)
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break
    raw_image = frame
    
    # show previous processed keyframe
    cv2.imshow('RealSense', out_frame)
    if init > buffer_max_len and args.REC:
        writer.write(out_frame)
    else:
        init += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

    # Convert images to numpy arrays
    plotbuffer.append(raw_image.transpose(2,0,1))
    raw_image = cv2.resize(raw_image, (imgsize[1], imgsize[0]), interpolation=cv2.INTER_NEAREST)
    color_image = raw_image.transpose(2,0,1)
    buffer.append(color_image)
    
    # start inference if buffer is full
    if len(buffer) > buffer_max_len:
        _ = buffer.pop(0)
        _ = plotbuffer.pop(0)
        clip_torch = torch.from_numpy(np.array(buffer)[indices]).to(device) / 255
        clip_torch = tfs(clip_torch)

        with torch.no_grad():
            outputs = model.encode_vision(clip_torch.unsqueeze(0))
            outputs['pred_logits'] = F.normalize(outputs['pred_logits'], dim=-1) @ text_embeds.T
            result = postprocess(outputs, outsize, human_conf=0.9, thresh=args.thresh)[0]
            result['text_labels'] = [[captions[e] for e in ele] for ele in result['labels']]
            boxes = result['boxes']
            labels = result['text_labels']
            scores = result['scores']

        out_frame = plotbuffer[mididx].transpose(1,2,0)
        for j in range(len(boxes)):
            box = boxes[j].cpu().detach().numpy()
            label = labels[j]
            score = scores[j]
            if 0 in score.shape and args.DISABLEEMPTY:
                continue
            start_point = (box[0], box[1])
            end_point = (box[2], box[3])
            out_frame = cv2.rectangle(out_frame, start_point, end_point, color, thickness)
            offset = 20
            for k in range(len(label)):
                act = label[k]
                sco = score[k]
                act = act + ' ' + str(round(sco.item(),2))
                out_frame = cv2.putText(out_frame, act, (box[0]-5, box[1]+offset), cv2.FONT_HERSHEY_SIMPLEX, font, color, thickness, cv2.LINE_AA)
                offset += 20
    end = time.time()
    print('fps =', 1/(end - start))
                
    # done inference if buffer is full
    
#cv2.destroyAllWindows()
if args.REC:
    writer.release()
# Stop streaming
