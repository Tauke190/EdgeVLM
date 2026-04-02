import os
import sys
import cv2
import numpy as np
import argparse
import time
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from sia import get_sia, PostProcessViz
import json

parser = argparse.ArgumentParser(description="Multi-Tier Surveillance Inference: Motion → Person → Action")
parser.add_argument("-F", type=str, help="video file path")
parser.add_argument("-thresh", type=float, default=0.5, help="action cosine threshold")
parser.add_argument("-act", type=lambda s: s.split(","), help="comma-separated list of actions")
parser.add_argument("-color", type=str, default='green', help="color for visualization")
parser.add_argument("-font", type=float, default=0.5, help="font size")
parser.add_argument("-line", type=int, default=2, help="line thickness")
parser.add_argument("--camera-device", type=int, default=0, help="camera device index when reading live input")
parser.add_argument("--simulate-live", action='store_true',
                    help="pace video-file input against wall clock so it behaves like a live camera stream")
parser.add_argument("--target-fps", type=float, default=None,
                    help="override FPS used for live pacing when simulating a camera from file input")
parser.add_argument("--drop-frames", action='store_true',
                    help="drop file frames when processing falls behind live pacing")
parser.add_argument("--show-preview", action='store_true', help="show a live preview window")
parser.add_argument("--no-write-output", action='store_true', help="disable output video writing")

# Tier 1: Motion Detection
parser.add_argument("--motion-thresh", type=int, default= 1000, help="min contour area for motion")
parser.add_argument("--motion-frames", type=int, default=3, help="consecutive frames for motion trigger")
parser.add_argument("--cooldown", type=int, default=60, help="frames to stay active after motion stops")

# Tier 2: Person Detection
parser.add_argument("--person-model", type=str, default='yolov8n', choices=['yolov8n', 'mobilenet-ssd'],
                    help="person detector model")
parser.add_argument("--person-thresh", type=float, default=0.3, help="person confidence threshold")

# Tier 3: Action Detection
parser.add_argument("--skip-tier3", action='store_true', help="skip Tier 3 action detection (motion+person only)")
parser.add_argument("--debug", action='store_true', help="show motion mask and tier states")

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

if not args.F and args.simulate_live:
    parser.error("--simulate-live requires -F with a video file.")

source_label = args.F if args.F else f"camera:{args.camera_device}"
print(f"Loading source: {source_label}")
cap = cv2.VideoCapture(args.F if args.F else args.camera_device)
if not cap.isOpened():
    print(f"✗ Failed to open source: {source_label}")
    sys.exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
source_is_file = bool(args.F)
if fps <= 0:
    fps = args.target_fps if args.target_fps else 30.0
elif args.target_fps:
    fps = args.target_fps

if source_is_file:
    print(f"Video: {frame_width}x{frame_height}, {total_frames} frames @ {fps} fps")
else:
    print(f"Camera: {frame_width}x{frame_height} @ {fps} fps")

output_path = None
writer = None
if not args.no_write_output:
    output_name = 'multitier_' + (os.path.splitext(os.path.basename(args.F))[0] if args.F else f'camera_{args.camera_device}') + '.mp4'
    output_path = output_name
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

if args.show_preview:
    cv2.namedWindow("Multi-Tier Inference", cv2.WINDOW_NORMAL)

# ====================
# TIER 1: Motion Detection (CPU)
# ====================

print("\n=== Initializing Tier 1: MOG2 Motion Detection ===")
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=30,
    detectShadows=True
)
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
blur_kernel = (5, 5)

motion_frame_count = 0
cooldown_count = 0
motion_active = False
motion_roi = None

# ====================
# TIER 2: Person Detection
# ====================

print(f"=== Initializing Tier 2: {args.person_model} Person Detection ===")
person_detector = None
if args.person_model == 'yolov8n':
    try:
        from ultralytics import YOLO
        person_detector = YOLO('weights/yolov8n.pt')
        print("✓ YOLOv8n loaded")
    except ImportError:
        print("⚠ ultralytics not found. Install: pip install ultralytics")
        args.person_model = 'mobilenet-ssd'
        print("  Falling back to MobileNet-SSD...")

if args.person_model == 'mobilenet-ssd':
    mobilenet_proto = 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/MobileNetSSD_deploy.prototxt'
    mobilenet_model = 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/MobileNetSSD_deploy.caffemodel'
    try:
        print("Loading MobileNet-SSD from OpenCV DNN...")
        person_detector = cv2.dnn.readNetFromCaffe(mobilenet_proto, mobilenet_model)
        print("✓ MobileNet-SSD loaded")
    except:
        print("⚠ Failed to load MobileNet-SSD. Ensure OpenCV is built with DNN support.")
        person_detector = None

person_detected = False
person_boxes = []

# ====================
# TIER 3: SIA Action Detection
# ====================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"=== Initializing Tier 3: SIA Action Detection (device: {device}) ===")

# Initialize Tier 3 state (may be used even if skip_tier3 is True)
sia_active = False
sia_frame_count = 0
last_boxes = None
last_labels = None
last_scores = None
cached_action_text = None  # Cache last action text to display even without fresh predictions
model = None
text_embeds = None
tfs = None
postprocess = None
buffer = []
plotbuffer = []
buffer_max_len = 72

if not args.skip_tier3:
    try:
        # Load default actions
        with open('gpt/GPT_AVA.json', 'r') as f:
            gpt_data = json.load(f)
            DEFAULT_ACTIONS = list(gpt_data.keys())

        captions = args.act
        if captions is None:
            captions = DEFAULT_ACTIONS
        else:
            captions = [act.replace('_', ' ') for act in captions]

        print(f"Loading SIA model (actions: {len(captions)})...")
        model = get_sia(size='b', pretrain=None, det_token_num=20, text_lora=True, num_frames=9)['sia']
        model.load_state_dict(
            torch.load('weights/avak_b16_11.pt', map_location=device, weights_only=True),
            strict=False,
        )
        model.to(device)
        model.eval()
        print("✓ SIA model loaded")

        # Pre-encode text embeddings (one-time)
        text_embeds = model.encode_text(captions)
        text_embeds = F.normalize(text_embeds, dim=-1)

        tfs = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        postprocess = PostProcessViz()

        # Buffer management for Tier 3 (already initialized above)
        buffer = []
        plotbuffer = []
        
    except Exception as e:
        print(f"✗ Failed to load SIA: {e}")
        args.skip_tier3 = True

# ====================
# Statistics
# ====================

frame_count = 0
source_frame_index = 0
skipped_frames = 0
motion_triggers = 0
frames_with_motion = 0
person_detections = 0
frames_with_persons = 0
sia_forward_passes = 0

# FLOPs tracking
yolo_flops_per_pass = 8.7e9  # YOLOv8n: ~8.7 GFLOPs (from ultralytics)
sia_flops_per_pass = 0  # Will be measured
yolo_inference_count = 0
total_flops = 0.0

# Measure SIA FLOPs (one-time)
if not args.skip_tier3:
    try:
        from thop import profile
        dummy_clip = torch.randn(1, 3, 9, 240, 320).to(device)
        flops, _ = profile(model.vision_encoder, inputs=(dummy_clip,), verbose=False)
        sia_flops_per_pass = flops
        print(f"\nFLOPs per pass:")
        print(f"  YOLOv8n: {yolo_flops_per_pass/1e9:.2f} GFLOPs")
        print(f"  SIA Vision Encoder: {sia_flops_per_pass/1e9:.2f} GFLOPs")
    except ImportError:
        print("⚠ thop not installed, skipping FLOPs measurement")
        sia_flops_per_pass = 100e9  # Conservative estimate

# Warmup period: skip motion gating for first N frames to catch early persons
# (MOG2 needs time to learn background)
warmup_frames = int(fps * 3) if fps > 0 else 75  # 3 seconds or 75 frames
warmup_complete = False

print(f"\n{'='*60}")
print("Starting multi-tier inference...")
print(f"{'='*60}\n")

start_time = time.time()
stream_start_time = time.perf_counter()

while cap.isOpened():
    if args.simulate_live and source_is_file:
        elapsed_stream = time.perf_counter() - stream_start_time
        next_target_time = source_frame_index / fps
        if next_target_time > elapsed_stream:
            time.sleep(next_target_time - elapsed_stream)
        elif args.drop_frames:
            desired_frame_index = int(elapsed_stream * fps)
            frames_to_skip = max(0, desired_frame_index - source_frame_index)
            while frames_to_skip > 0:
                if not cap.grab():
                    break
                source_frame_index += 1
                skipped_frames += 1
                frames_to_skip -= 1

    ret, frame = cap.read()
    if not ret:
        break
    source_frame_index += 1

    frame_count += 1
    out_frame = frame.copy()

    # ==================== TIER 1: MOTION DETECTION ====================
    blurred = cv2.GaussianBlur(frame, blur_kernel, 0)
    fg_mask = bg_subtractor.apply(blurred, learningRate=0.001)
    fg_mask[fg_mask == 127] = 0  # Remove shadows
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > args.motion_thresh]
    motion_detected = len(valid_contours) > 0

    if motion_detected:
        motion_frame_count += 1
        if motion_frame_count >= args.motion_frames and not motion_active:
            motion_active = True
            motion_triggers += 1
        cooldown_count = 0
    else:
        motion_frame_count = 0

    if motion_active and not motion_detected:
        cooldown_count += 1
        if cooldown_count >= args.cooldown:
            motion_active = False
            cooldown_count = 0

    if motion_active:
        frames_with_motion += 1
        if valid_contours:
            all_points = np.vstack(valid_contours)
            x, y, w, h = cv2.boundingRect(all_points)
            motion_roi = (x, y, w, h)

    # Check if warmup complete
    if frame_count > warmup_frames:
        warmup_complete = True

    # ==================== TIER 2: PERSON DETECTION ====================
    person_detected = False
    person_boxes = []

    # Run person detection if: (1) motion detected OR (2) still in warmup period
    if (motion_active or not warmup_complete) and person_detector:
        yolo_inference_count += 1  # Track YOLO inference call
        try:
            if args.person_model == 'yolov8n':
                # YOLOv8n inference
                results = person_detector(frame, conf=args.person_thresh, verbose=False)
                for result in results:
                    for box in result.boxes:
                        if int(box.cls[0]) == 0:  # COCO class 0 = person
                            person_detected = True
                            person_boxes.append(box.xyxy[0].cpu().numpy())
                            person_detections += 1

            elif args.person_model == 'mobilenet-ssd':
                # MobileNet-SSD inference
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
                person_detector.setInput(blob)
                detections = person_detector.forward()

                h, w = frame.shape[:2]
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > args.person_thresh:
                        idx = int(detections[0, 0, i, 1])
                        if idx == 15:  # MobileNet-SSD person class
                            person_detected = True
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            person_boxes.append(box)
                            person_detections += 1

        except Exception as e:
            if frame_count == 1:  # Print error only once
                print(f"⚠ Person detection error: {e}")

    if person_detected:
        frames_with_persons += 1

    # ==================== TIER 3: SIA ACTION DETECTION ====================
    # Always fill the buffer (not gated by person detection)
    # This way inference is ready immediately when a person is detected
    if not args.skip_tier3:
        raw_image_resized = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_NEAREST)
        color_image = raw_image_resized.transpose(2, 0, 1)
        buffer.append(color_image)
        plotbuffer.append(frame.transpose(2, 0, 1))

        # Keep buffer at max size (sliding window)
        if len(buffer) > buffer_max_len:
            buffer.pop(0)
            plotbuffer.pop(0)

        # Run inference only when buffer is full AND person is detected
        if len(buffer) >= buffer_max_len and person_detected:
            sia_forward_passes += 1
            total_flops += sia_flops_per_pass  # Accumulate FLOPs
            sia_active = True
            sia_frame_count = 0

            clip_torch = torch.tensor(np.array(buffer)[0:buffer_max_len:buffer_max_len//9]) / 255
            clip_torch = tfs(clip_torch)

            with torch.no_grad():
                outputs = model.encode_vision(clip_torch.unsqueeze(0).to(device))
                outputs['pred_logits'] = F.normalize(outputs['pred_logits'], dim=-1) @ text_embeds.T
                result = postprocess(outputs, (frame_height, frame_width), human_conf=0.9, thresh=args.thresh)[0]
                result['text_labels'] = [[captions[e] for e in ele] for ele in result['labels']]
                last_boxes = result['boxes']
                last_labels = result['text_labels']
                last_scores = result['scores']
                
                # Cache the first detected action for reuse
                if len(last_boxes) > 0 and len(last_labels[0]) > 0:
                    max_idx = torch.argmax(last_scores[0]).item()
                    cached_action_text = f"{last_labels[0][max_idx]} {round(last_scores[0][max_idx].item(), 2)}"

    # Deactivate Tier 3 inference if no persons detected for cooldown period
    # BUT keep the cached action predictions (last_boxes/labels/scores) visible
    if sia_active and not person_detected:
        sia_frame_count += 1
        if sia_frame_count > args.cooldown:
            sia_active = False
            sia_frame_count = 0
            # Don't clear last_boxes - keep labels visible on YOLO boxes

    # ==================== VISUALIZATION ====================

    # Draw motion contours and ROI
    if valid_contours:
        cv2.drawContours(out_frame, valid_contours, -1, (200, 200, 0), 2)

    if motion_active and motion_roi:
        x, y, w, h = motion_roi
        cv2.rectangle(out_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Compute IoU between boxes for association
    def compute_iou(box1, box2):
        """Compute IoU between two boxes in xyxy format"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    # Draw person boxes with associated action labels from Tier 3
    if person_boxes:
        for yolo_box in person_boxes:
            x1, y1, x2, y2 = int(yolo_box[0]), int(yolo_box[1]), int(yolo_box[2]), int(yolo_box[3])
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue YOLO box
            
            # Display action label (from fresh predictions or cached last known action)
            if not args.skip_tier3:
                best_action = None
                best_score = None
                
                # Try to find associated action from fresh predictions
                if last_boxes is not None and len(last_boxes) > 0:
                    best_iou = 0.0
                    for j in range(len(last_boxes)):
                        sia_box = last_boxes[j]
                        if torch.is_tensor(sia_box):
                            sia_box = sia_box.cpu().detach().numpy()
                        
                        iou = compute_iou(yolo_box, sia_box)
                        if iou > best_iou:
                            best_iou = iou
                            label = last_labels[j]
                            score = last_scores[j]
                            
                            if len(label) > 0:
                                max_idx = torch.argmax(score).item()
                                best_action = label[max_idx]
                                best_score = score[max_idx].item()
                    
                    # Update cached action if meaningful overlap found
                    if best_action is not None and best_iou > 0.1:
                        cached_action_text = f"{best_action} {round(best_score, 2)}"
                
                # Always display cached action (never disappear)
                if cached_action_text is not None:
                    cv2.putText(out_frame, cached_action_text, (x1 - 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                               font, color, thickness, cv2.LINE_AA)

    # Tier status panel
    tier1_status = "MOTION" if motion_active else "IDLE"
    tier1_color = (0, 255, 0) if motion_active else (100, 100, 100)
    
    tier2_status = f"PERSON ({len(person_boxes)})" if person_detected else "IDLE"
    tier2_color = (255, 0, 0) if person_detected else (100, 100, 100)
    
    tier3_status = f"ACTION ({sia_forward_passes})" if sia_active and not args.skip_tier3 else "IDLE"
    tier3_color = (0, 255, 0) if sia_active and not args.skip_tier3 else (100, 100, 100)

    cv2.putText(out_frame, f"T1: {tier1_status}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, font * 0.8, tier1_color, 1)
    cv2.putText(out_frame, f"T2: {tier2_status}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, font * 0.8, tier2_color, 1)
    if not args.skip_tier3:
        cv2.putText(out_frame, f"T3: {tier3_status}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, font * 0.8, tier3_color, 1)

    frame_text = f"Frame: {frame_count}/{total_frames}" if source_is_file and total_frames > 0 else f"Frame: {frame_count}"
    cv2.putText(out_frame, frame_text, (10, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, font * 0.7, (255, 255, 255), 1)

    if args.simulate_live and source_is_file:
        cv2.putText(out_frame, f"Live Sim: {fps:.1f} fps | Dropped: {skipped_frames}", (10, frame_height - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, font * 0.6, (255, 255, 255), 1)

    # Debug: overlay motion mask
    if args.debug:
        fg_mask_vis = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        out_frame = cv2.addWeighted(out_frame, 0.7, fg_mask_vis, 0.3, 0)

    if writer is not None:
        writer.write(out_frame)

    if args.show_preview:
        cv2.imshow("Multi-Tier Inference", out_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        progress_text = f"{frame_count}/{total_frames}" if source_is_file and total_frames > 0 else f"{frame_count}"
        print(f"  Frame {progress_text} | "
              f"T1: {'ON' if motion_active else 'OFF'} | "
              f"T2: {'ON' if person_detected else 'OFF'} | "
              f"T3: {'ON' if sia_active and not args.skip_tier3 else 'OFF'} | "
              f"FPS: {frame_count/elapsed:.1f}")

cap.release()
if writer is not None:
    writer.release()
if args.show_preview:
    cv2.destroyAllWindows()

elapsed = time.time() - start_time

print(f"\n{'='*60}")
print(f"✓ Multi-tier inference complete!")
print(f"  Output: {output_path if output_path else 'disabled'}")
print(f"{'='*60}")

if frame_count == 0:
    print("Warning: No frames processed!")
    sys.exit(1)

print(f"\nTier 1 (Motion Detection):")
print(f"  Frames with motion detected: {frames_with_motion} ({100*frames_with_motion/frame_count:.1f}%)")
print(f"  Motion triggers: {motion_triggers}")
print(f"\nTier 2 (Person Detection):")
print(f"  Frames with persons detected: {frames_with_persons} ({100*frames_with_persons/frame_count:.1f}%)")
print(f"  Total person detections: {person_detections}")
if not args.skip_tier3:
    print(f"\nTier 3 (Action Detection):")
    print(f"  Forward passes: {sia_forward_passes}")
    print(f"  Effective speedup: {frame_count / max(sia_forward_passes, 1):.1f}x vs. always-on")

# Calculate total FLOPs (add YOLO FLOPs)
total_flops += yolo_inference_count * yolo_flops_per_pass

if not args.skip_tier3 and sia_flops_per_pass > 0:
    flops_per_frame_multitier = total_flops / frame_count
    flops_per_frame_alwayson = sia_flops_per_pass
    flops_speedup = flops_per_frame_alwayson / max(flops_per_frame_multitier, 1)
    
    print(f"\nComputational Efficiency (FLOPs):")
    print(f"  Multi-tier FLOPs/frame: {flops_per_frame_multitier/1e9:.2f} GFLOPs")
    print(f"  Always-on SIA FLOPs/frame: {flops_per_frame_alwayson/1e9:.2f} GFLOPs")
    print(f"  FLOPs reduction: {flops_speedup:.1f}x more efficient than always-on SIA")

print(f"\nProcessing:")
print(f"  Total frames: {frame_count}")
if args.simulate_live and source_is_file:
    print(f"  Source frames dropped to maintain live pacing: {skipped_frames}")
print(f"  Processing time: {elapsed:.1f}s ({frame_count/elapsed:.1f} fps)")
