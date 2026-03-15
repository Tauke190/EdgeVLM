import os
import numpy as np
import cv2
import time
import argparse
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from sia import get_sia, PostProcessViz
import json
# from datasets import avatextaug

# Load default actions from JSON file
with open('gpt/GPT_AVA.json', 'r') as f:
    gpt_data = json.load(f)
    DEFAULT_ACTIONS = list(gpt_data.keys())

parser = argparse.ArgumentParser(description="Offline Inference with SIA")
parser.add_argument("-F", type=str, required=True, help="file path")
parser.add_argument("-thresh", type=float, default=0.25, help="cosine threshold")
parser.add_argument("-act", type=lambda s: s.split(","), help="Comma-separated list of actions")
parser.add_argument("-color", type=str, default='green', help="color to plot predictions")
parser.add_argument("-font", type=float, default=0.3, help="font size")
parser.add_argument("-line", type=int, default=1, help="line thickness")
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

print(f"Loading video: {args.F}")
cap = cv2.VideoCapture(args.F)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
outsize = (frame_height, frame_width)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading model...")
model = get_sia(size='b', pretrain=None, det_token_num=20, text_lora=True, num_frames=9)['sia']
model.load_state_dict(
    torch.load('weights/avak_b16_11.pt', map_location=device, weights_only=True),
    strict=False,
)
model.to(device)
model.eval()
print("Model loaded")

tfs = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

captions = args.act
if captions == None:
    captions = DEFAULT_ACTIONS
else:
    captions = [act.replace('_', ' ') for act in captions]

print(f"Actions to detect: {len(captions)}")
text_embeds = model.encode_text(captions)
text_embeds = F.normalize(text_embeds, dim=-1)

imgsize = (240, 320)
output_path = 'pred_' + args.F.split('.')[0] + '.mp4'
writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (frame_width, frame_height))

buffer_max_len = 72
mididx = buffer_max_len // 2
buffer = []
plotbuffer = []

postprocess = PostProcessViz()
init = 0
ret = True
frame_count = 0

print("Starting inference...")
while ret:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    raw_image = frame
    out_frame = raw_image.copy()
    
    plotbuffer.append(raw_image.transpose(2, 0, 1))
    raw_image = cv2.resize(raw_image, (imgsize[1], imgsize[0]), interpolation=cv2.INTER_NEAREST)
    color_image = raw_image.transpose(2, 0, 1)
    buffer.append(color_image)
    
    if len(buffer) > buffer_max_len:
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

        out_frame = plotbuffer[mididx].transpose(1, 2, 0).astype(np.uint8)
        for j in range(len(boxes)):
            box = boxes[j].cpu().detach().numpy()
            label = labels[j]
            score = scores[j]
            start_point = (int(box[0]), int(box[1]))
            end_point = (int(box[2]), int(box[3]))
            out_frame = cv2.rectangle(out_frame, start_point, end_point, color, thickness)
            offset = 0
            for k in range(len(label)):
                act = label[k]
                sco = score[k]
                text = act + ' ' + str(round(sco.item(), 2))
                out_frame = cv2.putText(out_frame, text, (int(box[0])-5, int(box[1])+offset), cv2.FONT_HERSHEY_SIMPLEX, font, color, thickness, cv2.LINE_AA)
                offset += 20
    
    if init > buffer_max_len:
        writer.write(out_frame)
    else:
        init += 1
    
    end = time.time()
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames")

cap.release()
writer.release()
print(f"Inference complete! Output saved to: {output_path}")
