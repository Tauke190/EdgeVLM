from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .sia import SIA
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import batched_nms, sigmoid_focal_loss
import numpy as np
import cv2
import os

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from scipy.optimize import linear_sum_assignment

def get_sia(size='l', 
              pretrain=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViClip-InternVid-10M-FLT.pth"),
              text_lora=False,
              det_token_num=100,
              num_frames=9):
    
    tokenizer = _Tokenizer()
    sia_model = SIA(tokenizer=tokenizer,
                        size=size,
                        pretrain=pretrain,
                        det_token_num=det_token_num,
                        num_frames=num_frames,
                        text_lora=text_lora)
    m = {'sia':sia_model, 'tokenizer':tokenizer}
    
    return m

###################################
# For closed-set action detection #
###################################
# YOLOS: Added Hungarian Matcher
# REMOVED cost_class to avoid matching for actions
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_human: float = 1): #HUMAN
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class # remove in the future
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_human = cost_human #HUMAN
        assert cost_bbox != 0 or cost_giou != 0 or cost_human != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        out_human = outputs["human_logits"].flatten(0, 1).softmax(-1) #HUMAN

        # Also concat the target labels and boxes
        #tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_human = torch.zeros(len(tgt_bbox)).int().to(tgt_bbox.device) #HUMAN

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        
        cost_human = -out_human[:, tgt_human] #HUMAN

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou + self.cost_human * cost_human #HUMAN
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

# YOLOS: Added Loss
# Note: Modified for dynamic num of classes
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

    def loss_labels(self, outputs, targets, indices, num_boxes, num_classes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        #empty_weight = torch.ones(num_classes + 1).to(outputs['pred_logits'].device)
        #empty_weight[-1] = self.eos_coef # sigmoid focal loss does not use weights
        
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        
        # one-hot encoding here
        
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        '''
        target_classes = torch.full(src_logits.shape[:2], num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes = F.one_hot(target_classes)
        target_classes[idx] = target_classes_o
        '''
        target_classes = target_classes_o
        src_logits = src_logits[idx]

        loss_ce = F.cross_entropy(src_logits, target_classes.float())
        #loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes.transpose(1, 2).float(), empty_weight)
        #loss_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes.float(), empty_weight) # use one-hot encoded probs instead of int classes
        #loss_ce = sigmoid_focal_loss(src_logits, target_classes.float(), alpha=0.25, gamma=2.0, reduction='mean') # use one-hot encoded probs instead of int classes
        losses = {'loss_ce': loss_ce}
        return losses
        
    def loss_human(self, outputs, targets, indices, num_boxes, num_classes, log=True): #HUMAN
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        empty_weight = torch.ones(1 + 1).to(outputs['human_logits'].device)
        empty_weight[-1] = self.eos_coef
        
        num_classes = 1
        
        assert 'human_logits' in outputs
        src_logits = outputs['human_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([torch.zeros(len(t['labels'])).long() for t, (_, J) in zip(targets, indices)]).to(src_logits.device)

        target_classes = torch.full(src_logits.shape[:2], num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_human = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
        losses = {'loss_human': loss_human}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, num_classes): #quick fix for num_classes for loss_label
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, num_classes): #quick fix for num_classes for loss_label
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes, num_classes): #quick fix for num_classes for loss_label
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, num_classes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'human': self.loss_human #HUMAN
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        #return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
        return loss_map[loss](outputs, targets, indices, num_boxes, num_classes, **kwargs)

    def forward(self, outputs, targets, num_classes):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, num_classes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
        
class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, imgsize, human_conf=0.7, Aaug=None, thresh=0.25):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_human, out_logits, out_bbox = outputs['human_logits'], outputs['pred_logits'], outputs['pred_boxes']

        if Aaug is not None:
            out_logits = out_logits @ Aaug

        human_prob = F.softmax(out_human, -1)
        human_scores, human_labels = human_prob[...,].max(-1)
        
        prob = out_logits
        
        boxes = box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([imgsize[1], imgsize[0], imgsize[1], imgsize[0]]).to(boxes.device)
        boxes = boxes * scale_fct

        results = []
        bs = out_human.shape[0]
        for i in range(bs):
            human_idx = torch.where(human_labels[i] == 0) # obtain boxes where human is detected
            human_scores_kept = human_scores[i][human_idx] # filter boxes where human is detected
            prob_kept = prob[i][human_idx]
            boxes_kept = boxes[i][human_idx]
            
            human_idx = torch.where(human_scores_kept >= human_conf) # obtain boxes where human conf > thresh (default=0.7)
            human_scores_kept = human_scores_kept[human_idx] # filter boxes where human is detected
            prob_kept = prob_kept[human_idx]
            boxes_kept = boxes_kept[human_idx]
            
            human_idx = batched_nms(boxes_kept, human_scores_kept, torch.zeros(len(human_scores_kept)).to(human_scores_kept.device), 0.5) # extra NMS
            human_scores_kept = human_scores_kept[human_idx] # filter boxes where human is detected
            prob_kept = prob_kept[human_idx]
            boxes_kept = boxes_kept[human_idx].int()

            final_scores = []
            final_labels = []
            finalboxes = []
            for i in range(len(human_idx)):
                box = boxes_kept[i]
                gt = torch.where(prob_kept[i] >= thresh)[0]
                gt_conf = (prob_kept[i][gt] + 1) / 2
                final_scores.extend(gt_conf)
                final_labels.extend(gt)
                for _ in range(len(gt)):
                    finalboxes.append(box)
            final_scores = torch.stack(final_scores) if len(final_scores) != 0 else torch.empty(0)
            final_labels = torch.stack(final_labels) if len(final_labels) != 0 else torch.empty(0)
            finalboxes = torch.stack(finalboxes) if len(finalboxes) != 0 else torch.empty(0)
            
            results.append({'scores': final_scores,
                            'labels': final_labels,
                            'boxes': finalboxes})

        return results

class PostProcessViz(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, imgsize, human_conf=0.7, Aaug=None, thresh=0.25, return_stage_timings=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        import time

        out_human, out_logits, out_bbox = outputs['human_logits'], outputs['pred_logits'], outputs['pred_boxes']

        if Aaug is not None:
            out_logits = out_logits @ Aaug

        timings = {
            "human_filter_s": 0.0,
            "nms_s": 0.0,
            "threshold_s": 0.0,
        }

        human_prob = F.softmax(out_human, -1)
        human_scores, human_labels = human_prob[...,].max(-1)

        prob = out_logits

        boxes = box_cxcywh_to_xyxy(out_bbox)
        scale_fct = boxes.new_tensor([imgsize[1], imgsize[0], imgsize[1], imgsize[0]])
        boxes = boxes * scale_fct

        results = []
        bs = out_human.shape[0]
        for batch_idx in range(bs):
            human_filter_start = time.perf_counter()
            keep_mask = (human_labels[batch_idx] == 0) & (human_scores[batch_idx] >= human_conf)
            keep_indices = keep_mask.nonzero(as_tuple=False).flatten()
            timings["human_filter_s"] += time.perf_counter() - human_filter_start

            if keep_indices.numel() == 0:
                results.append({'scores': [], 'labels': [], 'boxes': []})
                continue

            human_scores_kept = human_scores[batch_idx].index_select(0, keep_indices)
            prob_kept = prob[batch_idx].index_select(0, keep_indices)
            boxes_kept = boxes[batch_idx].index_select(0, keep_indices)

            nms_start = time.perf_counter()
            nms_indices = batched_nms(
                boxes_kept,
                human_scores_kept,
                torch.zeros_like(human_scores_kept, dtype=torch.long),
                0.5,
            )
            timings["nms_s"] += time.perf_counter() - nms_start

            prob_kept = prob_kept.index_select(0, nms_indices)
            boxes_kept = boxes_kept.index_select(0, nms_indices).int()

            threshold_start = time.perf_counter()
            threshold_mask = prob_kept >= thresh
            shifted_scores = (prob_kept + 1) / 2
            final_scores = []
            final_labels = []
            finalboxes = []
            for box, box_mask, box_scores in zip(boxes_kept, threshold_mask, shifted_scores):
                label_ids = box_mask.nonzero(as_tuple=False).flatten()
                final_labels.append(label_ids)
                final_scores.append(box_scores.index_select(0, label_ids))
                finalboxes.append(box)
            timings["threshold_s"] += time.perf_counter() - threshold_start

            results.append({'scores': final_scores,
                            'labels': final_labels,
                            'boxes': finalboxes})

        if return_stage_timings:
            return results, timings
        return results
