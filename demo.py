import os
import numpy as np
import cv2
import time
import argparse
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.ops import batched_nms
from sia import get_sia, PostProcessViz
from datasets import avatextaug

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
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {total_frames}")

outsize=(frame_height, frame_width)
out_frame = np.full((outsize[0],outsize[1],3), 0.)
cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Demo', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow('Demo', out_frame)
print('Press any key to start buffering')
cv2.waitKey(0)



from util.box_ops import box_cxcywh_to_xyxy

##############
# Load SIA #
##############

device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_sia(size='b', pretrain=None, det_token_num=20, text_lora=True, num_frames=9)['sia']
model.load_state_dict(
    torch.load('weights/avak_b16_11.pt', map_location=device, weights_only=True),
    strict=False,
)
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

# Measure FLOPS for a single forward pass
try:
    from thop import profile
    dummy_clip = torch.randn(1, 3, 9, 240, 320).to(device)  # [B, C, T, H, W]
    flops, params = profile(model, inputs=(dummy_clip,), verbose=False)
    print(f"FLOPs per forward pass: {flops / 1e9:.2f} GFLOPs")
except ImportError:
    print("thop not installed, skipping FLOPS measurement")

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
forward_passes = 0
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
    raw_image = cv2.resize(raw_image, (imgsize[1], imgsize[0]), interpolation=cv2.INTER_NEAREST)
    color_image = raw_image.transpose(2,0,1)
    buffer.append(color_image)
    
    # start inference if buffer is full
    if len(buffer) > buffer_max_len:
        forward_passes += 1
        _ = buffer.pop(0)
        _ = plotbuffer.pop(0)
        clip_torch = torch.tensor(np.array(buffer)[0:buffer_max_len:buffer_max_len//9]) / 255
        clip_torch = tfs(clip_torch)

        with torch.no_grad():
            outputs = model.encode_vision(clip_torch.unsqueeze(0).to(device))
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
                if font > 1 and thickness > 1:
                    offset += 50
    end = time.time()
    print('Inference duration for 9x8 frames:', end - start)
                
    # done inference if buffer is full
writer.release()
print(f"\nInference complete! Output saved to: pred_{args.F.split('.')[0]}.mp4")
print(f"Total frames processed: {total_frames}")
print(f"Total forward passes: {forward_passes}")
#cv2.destroyAllWindows()