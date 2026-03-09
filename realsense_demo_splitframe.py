import os

import pyrealsense2 as rs
import numpy as np
import cv2
import time
'''
cv2.imshow('ffmpeg fix', np.array([1], dtype=np.uint8)) #temp pyav-cv2 fix
cv2.destroyAllWindows()
'''
outsize=(480,640)
out_frame = np.full((outsize[0],outsize[1],3), 0.)
cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL) 
cv2.imshow('RealSense', out_frame)
print('Press any key to start buffering')
cv2.waitKey(0)
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.ops import batched_nms
from ezact import get_ezact, PostProcessViz
from datasets import avatextaug

from util.box_ops import box_cxcywh_to_xyxy

##############
# Load EZACT #
##############

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pretrain = os.path.join("viclip", "ViCLIP-B_InternVid-FLT-10M.pth")
model = get_ezact(size='b', pretrain=pretrain, det_token_num=100, lora=False, text_lora=True, num_frames=9, merge=False)['ezact']
model.load_state_dict(
    torch.load(
        'weights/avak_stats_flt_b16_txtaug_txtlora/avak_b16_11.pt',
        map_location=device,
        weights_only=True,
    ),
    strict=False,
)
model.to(device)
model.eval()
print('Model Loaded')
captions = list(avatextaug.keys())
#captions = ['cheering', 'clapping', 'jumping']
temp_num_classes = len(captions)

text_embeds = model.encode_text(captions)
text_embeds = F.normalize(text_embeds, dim=-1)

imgsize=(120,160) #(120,160)
tfs = v2.Compose([v2.Resize(imgsize),
                  v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

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

config.enable_stream(rs.stream.color, outsize[1], outsize[0], rs.format.rgb8, 30)

writer = cv2.VideoWriter('realsense_res.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         3, (outsize[1],outsize[0])) 


# Start streaming
pipeline.start(config)

buffer_max_len = 9
mididx = buffer_max_len // 2
buffer = []

postprocess = PostProcessViz()
init = 0
while True:
    #time.sleep(0.53)
    start = time.time()
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    
    # show previous processed keyframe
    cv2.imshow('RealSense', out_frame)
    if init > 9:
        writer.write(out_frame)
    else:
        init += 1
    print('collect')
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break
        
    if not color_frame:
        continue

    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data()).transpose(2,0,1)
    
    buffer.append(color_image)
    
    # start inference if buffer is full
    if len(buffer) > 9:
        _ = buffer.pop(0)
        clip_torch = torch.tensor(np.array(buffer)) / 255
        
        # split into 4 quadrants
        sub_H, sub_W = clip_torch.shape[-2:]
        sub_H, sub_W = sub_H // 2, sub_W // 2
        clip1 = clip_torch[:,:, :sub_H, :sub_W] # topleft
        clip2 = clip_torch[:,:, :sub_H, sub_W:] # topright
        clip3 = clip_torch[:,:, sub_H:, :sub_W] # bottomleft
        clip4 = clip_torch[:,:, sub_H:, sub_W:] # bottomright
        clip_torch = torch.stack([clip1, clip2, clip3, clip4])
        
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
            
            result = postprocess(outputs, outsize, human_conf=0.7)[0]
            result['text_labels'] = [[captions[e] for e in ele] for ele in result['labels']]
            boxes = result['boxes']
            labels = result['text_labels']
            #scores = result['scores']

        out_frame = cv2.cvtColor(buffer[mididx].transpose(1,2,0), cv2.COLOR_RGB2BGR)
        for j in range(len(boxes)):
            box = boxes[j].cpu().detach().numpy()
            label = labels[j]
            start_point = (box[0], box[1])
            end_point = (box[2], box[3])
            color = (0, 255, 0)
            thickness = 1
            out_frame = cv2.rectangle(out_frame, start_point, end_point, color, thickness)
            offset = 10
            for act in label:
                out_frame = cv2.putText(out_frame, act, (box[0], box[1]+offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness, cv2.LINE_AA)
                offset += 10
    end = time.time()
    print('Inference duration for 9 frames:', end - start)
                
    # done inference if buffer is full
    
cv2.destroyAllWindows()
writer.release()
# Stop streaming
pipeline.stop()
