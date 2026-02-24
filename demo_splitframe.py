import os
import numpy as np
import cv2
import time
import argparse
parser = argparse.ArgumentParser(description="Offline Demo with SIA")
parser.add_argument("-thresh", type=float, default=0.25,
                    help="cosine threshold")
parser.add_argument("-act", type=lambda s: s.split(","), help="Comma-separated list of actions")
parser.add_argument("-color", type=str, default='green',
                    help="color to plot predictions")
parser.add_argument("-font", type=float, default=1.0,
                    help="font size")
parser.add_argument("-line", type=int, default=1,
                    help="line thickness")
parser.add_argument("-F", type=str,
                    help="file path")
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

cap = cv2.VideoCapture(args.F)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

outsize=(frame_height, frame_width)
out_frame = np.full((outsize[0],outsize[1],3), 0.)
cv2.namedWindow('Demo', cv2.WINDOW_NORMAL) 
cv2.imshow('Demo', out_frame)
print('Press any key to start buffering')
cv2.waitKey(0)

import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.ops import batched_nms
from sia import get_sia, PostProcessViz
from datasets import avatextaug

from util.box_ops import box_cxcywh_to_xyxy

##############
# Load SIA #
##############

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = get_sia(size='b', pretrain=None, det_token_num=20, text_lora=True, num_frames=9)['sia']
model.load_state_dict(torch.load('weights/avak_aws_stats_flt_b16_txtaug_txtlora/avak_b16_10.pt', weights_only=True), strict=False)
model.to(device)
model.eval()
print('Model Loaded')

tfs = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

####################
# Put Actions Here #
####################
#captions = list(avatextaug.keys())
captions = args.act
if captions == None:
    captions = list(avatextaug.keys())
else:
    captions = [act.replace('_',' ') for act in captions]
temp_num_classes = len(captions)

text_embeds = model.encode_text(captions)
text_embeds = F.normalize(text_embeds, dim=-1)

imgsize=(240,320) #(120,160) #(180,240) #(240,320)

writer = cv2.VideoWriter('pred_' + args.F.split('.')[0] +'.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         25, (frame_width,frame_height)) 

buffer_max_len = 72
mididx = buffer_max_len // 2
buffer = []
plotbuffer = []

postprocess = PostProcessViz()
init = 0
ret = True
while ret:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    raw_image = frame
    
    # show previous processed keyframe
    cv2.imshow('Demo', out_frame)
    if init > buffer_max_len:
        writer.write(out_frame)
    else:
        init += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

    # Convert images to numpy arrays
    plotbuffer.append(raw_image.transpose(2,0,1))
    #raw_image = cv2.resize(raw_image, (imgsize[1], imgsize[0]), interpolation=cv2.INTER_NEAREST)
    color_image = raw_image.transpose(2,0,1)
    buffer.append(color_image)
    
    # start inference if buffer is full
    if len(buffer) > buffer_max_len:
        _ = buffer.pop(0)
        _ = plotbuffer.pop(0)
        
        clip = np.array(buffer)[0:buffer_max_len:buffer_max_len//9]
        
        # split into 4 quadrants
        sub_H, sub_W = clip.shape[-2:]
        sub_H, sub_W = sub_H // 2, sub_W // 2
        clip1 = clip[:,:, :sub_H, :sub_W].transpose(2,3,0,1).reshape(sub_H, sub_W, -1) # topleft
        clip2 = clip[:,:, :sub_H, sub_W:].transpose(2,3,0,1).reshape(sub_H, sub_W, -1) # topright
        clip3 = clip[:,:, sub_H:, :sub_W].transpose(2,3,0,1).reshape(sub_H, sub_W, -1) # bottomleft
        clip4 = clip[:,:, sub_H:, sub_W:].transpose(2,3,0,1).reshape(sub_H, sub_W, -1) # bottomright
        print(clip1.shape)
        clip1 = cv2.resize(clip1, (imgsize[1], imgsize[0]), interpolation=cv2.INTER_NEAREST)
        clip2 = cv2.resize(clip2, (imgsize[1], imgsize[0]), interpolation=cv2.INTER_NEAREST)
        clip3 = cv2.resize(clip3, (imgsize[1], imgsize[0]), interpolation=cv2.INTER_NEAREST)
        clip4 = cv2.resize(clip4, (imgsize[1], imgsize[0]), interpolation=cv2.INTER_NEAREST)
        clip1 = clip1.reshape(imgsize[0], imgsize[1], 9, 3).transpose(2,3,0,1)
        clip2 = clip2.reshape(imgsize[0], imgsize[1], 9, 3).transpose(2,3,0,1)
        clip3 = clip3.reshape(imgsize[0], imgsize[1], 9, 3).transpose(2,3,0,1)
        clip4 = clip4.reshape(imgsize[0], imgsize[1], 9, 3).transpose(2,3,0,1)
        clip = np.stack([clip1, clip2, clip3, clip4])
        
        clip_torch = torch.tensor(clip) / 255
        clip_torch = tfs(clip_torch)

        with torch.no_grad():
            outputs = model.encode_vision(clip_torch.to(device))
            outputs['pred_logits'] = F.normalize(outputs['pred_logits'], dim=-1) @ text_embeds.T
            
            # combine detections from all 4 frames
            outputs['pred_boxes'][1] = outputs['pred_boxes'][1] + torch.tensor([1,0,0,0]).to(device)
            outputs['pred_boxes'][2] = outputs['pred_boxes'][2] + torch.tensor([0,1,0,0]).to(device)
            outputs['pred_boxes'][3] = outputs['pred_boxes'][3] + torch.tensor([1,1,0,0]).to(device)
            outputs['pred_boxes'] /= 2
            outputs['pred_logits'] = outputs['pred_logits'].view(1,-1,outputs['pred_logits'].shape[-1])
            outputs['pred_boxes'] = outputs['pred_boxes'].view(1,-1,outputs['pred_boxes'].shape[-1])
            outputs['human_logits'] = outputs['human_logits'].view(1,-1,outputs['human_logits'].shape[-1])
            
            result = postprocess(outputs, outsize, human_conf=0.0, thresh=args.thresh)[0]
            result['text_labels'] = [[captions[e] for e in ele] for ele in result['labels']]
            boxes = result['boxes']
            labels = result['text_labels']
            scores = result['scores']

        out_frame = plotbuffer[mididx].transpose(1,2,0)
        for j in range(len(boxes)):
            box = boxes[j].cpu().detach().numpy()
            label = labels[j]
            score = scores[j]
            start_point = (box[0], box[1])
            end_point = (box[2], box[3])
            out_frame = cv2.rectangle(out_frame, start_point, end_point, color, thickness)
            offset = 0
            for k in range(len(label)):
                act = label[k]
                sco = score[k]
                act = act + ' ' + str(round(sco.item(),2))
                out_frame = cv2.putText(out_frame, act, (box[0]-5, box[1]+offset), cv2.FONT_HERSHEY_SIMPLEX, font, color, thickness, cv2.LINE_AA)
                offset += 20
                if thickness > 1:
                    offset += 30
    end = time.time()
    print('Inference duration for 9x8 frames:', end - start)
                
    # done inference if buffer is full
writer.release() 
#cv2.destroyAllWindows()
