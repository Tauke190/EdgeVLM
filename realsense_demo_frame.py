import argparse
parser = argparse.ArgumentParser(description="finetine ViCLIP for closed-set actdet")
parser.add_argument("--REC", action='store_true',
                    help="record stream")
parser.add_argument("-rate", type=int, default=4,
                    help="framerate")
parser.add_argument("-act", type=lambda s: s.split(","),
                    help="Comma-separated list of actions")
parser.add_argument("-color", type=str, default='green',
                    help="color to plot predictions")
parser.add_argument("-weights", type=str,
                    default='weights/avak_aws_stats_flt_b16_txtaug_txtlora/avak_b16_6.pt',
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

import os
import pyrealsense2 as rs
import numpy as np
import cv2
import time

'''
cv2.imshow('ffmpeg fix', np.array([1], dtype=np.uint8)) #temp pyav-cv2 fix
cv2.destroyAllWindows()
'''
outsize=(800,1280)#(480,640)
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

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pretrain = os.path.join("weights", "viclip", "ViCLIP-B_InternVid-FLT-10M.pth")
model = get_sia(size='b', pretrain=pretrain, det_token_num=20, text_lora=True, num_frames=9)['sia']
model.load_state_dict(torch.load(args.weights, weights_only=True), strict=False)
model.to(device)
model.eval()
print('Model Loaded')

captions = args.act
if captions == None:
    captions = list(avatextaug.keys())
temp_num_classes = len(captions)

text_embeds = model.encode_text(captions)
text_embeds = F.normalize(text_embeds, dim=-1)

imgsize=(240,320) #(120,160) #(180,240) #(240,320)
'''
tfs = v2.Compose([v2.Resize(imgsize),
                  v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
'''
tfs = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

########################
# Initialize Realsense #
########################
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
rsdevice = pipeline_profile.get_device()
device_product_line = str(rsdevice.get_info(rs.camera_info.product_line))

found_rgb = False
for s in rsdevice.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

#config.enable_stream(rs.stream.color, outsize[1], outsize[0], rs.format.rgb8, 30)
config.enable_stream(rs.stream.color, 1280, 800, rs.format.rgb8, 30)

if args.REC:
    writer = cv2.VideoWriter('realsense_res.mp4',  
                             cv2.VideoWriter_fourcc(*'MJPG'), 
                             20, (outsize[1],outsize[0])) 

###################
# Start streaming #
###################
pipeline.start(config)

buffer_max_len = 9 * args.rate # multiples of 9
mididx = buffer_max_len // 2
indices = np.arange(0, buffer_max_len, buffer_max_len // 9)
buffer = []
plotbuffer = []
init = 0
while True:
    #time.sleep(0.53)
    start = time.time()
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    
    # show previous processed keyframe
    cv2.imshow('RealSense', out_frame)
    if init > buffer_max_len and args.REC:
        writer.write(out_frame)
    else:
        init += 1
    #print('collect')
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break
        
    if not color_frame:
        continue

    # Convert images to numpy arrays
    raw_image = np.asanyarray(color_frame.get_data()).copy()
    plotbuffer.append(raw_image.transpose(2,0,1))
    raw_image = cv2.resize(raw_image, (imgsize[1], imgsize[0]), interpolation=cv2.INTER_LANCZOS4)
    color_image = raw_image.transpose(2,0,1) #[:,120:120+outsize[0],320:320+outsize[1]]
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
            result = postprocess(outputs, outsize, human_conf=0.7, thresh=args.thresh)[0]
            result['text_labels'] = [[captions[e] for e in ele] for ele in result['labels']]
            boxes = result['boxes']
            labels = result['text_labels']
            scores = result['scores']

        out_frame = cv2.cvtColor(plotbuffer[mididx].transpose(1,2,0), cv2.COLOR_RGB2BGR)
        for j in range(len(boxes)):
            box = boxes[j].cpu().detach().numpy()
            label = labels[j]
            score = scores[j]
            start_point = (box[0], box[1])
            end_point = (box[2], box[3])
            thickness = 1
            font=1.0
            out_frame = cv2.rectangle(out_frame, start_point, end_point, color, thickness)
            offset = 20
            for k in range(len(label)):
                act = label[k]
                sco = score[k]
                act = act + ' ' + str(round(sco.item(),2))
                out_frame = cv2.putText(out_frame, act, (box[0]-5, box[1]+offset), cv2.FONT_HERSHEY_SIMPLEX, font, color, thickness, cv2.LINE_AA)
                offset += 20
    end = time.time()
    #print('Inference duration for 9x8 frames:', end - start)
    print('fps =', 1/(end - start))
                
    # done inference if buffer is full
    
#cv2.destroyAllWindows()
if args.REC:
    writer.release()
# Stop streaming
pipeline.stop()
