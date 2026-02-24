import os
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms
from tqdm import tqdm
from sia import get_sia, HungarianMatcher, SetCriterion, PostProcess
from datasets import AVA, K700, textaugavak, avatextaug, k700textaug

from util.box_ops import box_cxcywh_to_xyxy
from util.misc import reduce_dict
from mapcalc import calculate_map
import numpy as np

import json

import argparse

parser = argparse.ArgumentParser(description="finetine ViCLIP for closed-set actdet")
parser.add_argument("-SIZE", metavar="SIZE", type=str, default='b16',
                    help="B16 or L14")
parser.add_argument("-FRAMES", metavar="FRAMES", type=int, default=9,
                    help="number of input frames")
parser.add_argument("-WORKERS", metavar="WORKERS", type=int, default=8,
                    help="num of workers for dataloader")
parser.add_argument("-WIDTH", metavar="WIDTH", type=int, default=320,
                    help="width of video to resize to")
parser.add_argument("-HEIGHT", metavar="HEIGHT", type=int, default=240,
                    help="height of video to resize to")
parser.add_argument("-JSON", metavar="JSON", type=str, default='stats.json',
                    help="output json name")
                    
parser.add_argument("-DET", metavar="DET", type=int, default=10,
                    help="number of [det] tokens to use")
parser.add_argument("--TXTLORA", action='store_true',
                    help="use LoRA on text encoder")
    
parser.add_argument("-ANNOLISTAVA", metavar="ANNOLISTAVA", type=str, default='anno/ava_action_list_v2.2.pbtxt',
                    help="AVA list of actions anno path")
parser.add_argument("-ANNOLISTAVA2019", metavar="ANNOLISTAVA2019", type=str, default='anno/ava_action_list_v2.2_for_activitynet_2019.pbtxt',
                    help="AVA 60 list of actions anno path")
                    
parser.add_argument("-KINETICS", metavar="KINETICS", type=str,
                    help="KINETICS video directory")
parser.add_argument("-RATEKINETICS", metavar="RATEKINETICS", type=int, default=8,
                    help="sampling rate of input frames")
parser.add_argument("-ANNOTRAINKINETICS", metavar="ANNOTRAINKINETICS", type=str, default='anno/kinetics_train_v1.0.csv',
                    help="KINETICS train anno path")
parser.add_argument("-ANNOVALKINETICS", metavar="ANNOVALKINETICS", type=str, default='anno/kinetics_val_v1.0.csv',
                    help="KINETICS val anno path")
parser.add_argument("-ANNOLISTKINETICS", metavar="ANNOLISTKINETICS", type=str, default='anno/kinetics_700_labels.csv',
                    help="KINETICS list of actions anno path")
                    
parser.add_argument("--TXTAUG", action='store_true',
                    help="augment text labels")

parser.add_argument("-PRETRAINED", metavar="PRETRAINED", type=str, default='',
                    help="pretrained weights to use")

args = parser.parse_args()

assert args.SIZE in ('b16', 'l14'), 'choose either b16 or l14'
assert '.json' in args.JSON, 'filename must end in .json'

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('Using device:', device)

##############
# Load Model #
##############
if args.SIZE == 'b16':
    pretrain = "weights/viclip/ViCLIP-B_InternVid-FLT-10M.pth"

    model = get_sia(size='b', pretrain=pretrain, det_token_num=args.DET, text_lora=args.TXTLORA, num_frames=args.FRAMES)['sia']
else:
    pretrain = "weights/viclip/ViCLIP-L_InternVid-FLT-10M.pth"

    model = get_sia(size='l', pretrain=pretrain, det_token_num=args.DET, text_lora=args.TXTLORA, num_frames=args.FRAMES)['sia']

model.to(device)

print('Trainable parameters:')
for n, p in model.named_parameters():
    if p.requires_grad:
        print(n)
print()
parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters: %.3fM' % parameters)
parameters = model.parameters()
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Total Parameters: %.3fM' % parameters)
print()
del parameters

################
# Load Dataset #
################
num_neg_classes = 0
total_classes = num_neg_classes + 1

video_input_num_frames = args.FRAMES

num_workers = args.WORKERS
tfs = v2.Compose([v2.Resize((args.HEIGHT, args.WIDTH)),
                  v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = torch.stack(batch[0])
    return tuple(batch)
    
train_kinetics = K700(args.KINETICS, args.ANNOTRAINKINETICS, args.ANNOVALKINETICS, clsfile=args.ANNOLISTAVA, csvfile=args.ANNOLISTKINETICS, transforms=tfs, frames=args.FRAMES, rate=args.RATEKINETICS, split='train', enable_k700_labels=False)

def flatten(xss):
    return [x for xs in xss for x in xs]

if args.TXTAUG:
    # averaging matrix
    Ak700 = torch.zeros(16*total_classes, total_classes)
    for i in range(total_classes):
        start, end = 16*i, 16*(i+1)
        Ak700[start:end,i] = 1/16
    Ak700 = Ak700.to(device) # (16xtotal_classes, total_classes)
    
model.eval()

postprocess = PostProcess()

print('generating assignments for :', args.PRETRAINED)
model.load_state_dict(torch.load(args.PRETRAINED, map_location=device))
model.eval()

###############################################################################
# Load Hungarian Matcher for matching predicted human boxes to GT human boxes #
###############################################################################
cost_class = 2 #1
cost_bbox = 2 #5
cost_giou = 2 #2
cost_human = 2 #1

matcher = HungarianMatcher(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou, cost_human=cost_human)

def get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx
    
def get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx

aligned_anno = {}
thresh = 0.25
for i in tqdm(range(len(train_kinetics))):
    sample, gt = train_kinetics[i]
    gt['boxes'] = gt['boxes'].to(device)
    targets = (gt, )
    print(gt)
    aligned_anno[gt['frame_key']] = []
    
    with torch.no_grad():
        H, W = sample.shape[-2:]
        samples = sample.unsqueeze(0).to(device)
        
        # extract global class, negative classes and temp_label_mapping
        captions = [gt['global_cls'], ]
        captionstoidx = {v: k for k, v in enumerate(captions)}
        temp_num_classes = len(captions)
        
        if args.TXTAUG:
            captions_aug = flatten([k700textaug[ele] for ele in captions])
            outputs = model(samples.to(device), captions_aug)
            outputs['pred_logits']  = outputs['pred_logits'] @ Ak700
        else:
            outputs = model(samples.to(device), captions)
            
        indices = matcher(outputs, targets)
        srcidx = get_src_permutation_idx(indices)
        tgtidx = get_tgt_permutation_idx(indices)
        target_boxes_o = [t["boxes"][J] for t, (_, J) in zip(targets, indices)][0]
        
        src_logits = outputs['pred_logits'][srcidx] # B, HUMAN, CLS
        src_pred = src_logits[:,0]
        
        for i in range(len(src_logits)):
            pred = src_pred[i]
            if pred >= 0.25:
                aligned_anno[gt['frame_key']].append(tgtidx[1][i].cpu().detach().item())
                
        print(aligned_anno[gt['frame_key']])
        
json.dump(aligned_anno, open(args.JSON[:-5] + '_assigned_ws.json', 'w'))
