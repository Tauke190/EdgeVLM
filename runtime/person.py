from pathlib import Path

import cv2


class PersonGate:
    def __init__(
        self,
        detector,
        weights,
        threshold,
        precision,
        device,
        stride,
        cooldown_frames,
        min_on_time,
        hit_threshold,
        scale,
        resize_width,
        min_box_area,
    ):
        if stride < 1:
            raise ValueError("person_stride must be >= 1.")
        if cooldown_frames < 0:
            raise ValueError("person_cooldown_frames must be >= 0.")
        if min_on_time < 0:
            raise ValueError("person_min_on_time must be >= 0.")
        if min_box_area < 0:
            raise ValueError("person_min_box_area must be >= 0.")

        self.detector_name = detector
        self.weights = weights
        self.threshold = float(threshold)
        self.precision = precision
        self.device = str(device)
        self.stride = int(stride)
        self.cooldown_frames = int(cooldown_frames)
        self.min_on_time = int(min_on_time)
        self.hit_threshold = float(hit_threshold)
        self.scale = float(scale)
        self.resize_width = int(resize_width)
        self.min_box_area = int(min_box_area)

        self.eligible_frame_count = 0
        self.frames_since_positive = self.cooldown_frames + 1
        self.cached_boxes = []
        self.cached_scores = []
        self.active_frame_age = 0

        self.use_fp16 = self.precision == "fp16" and self.device.startswith("cuda")
        if self.precision == "fp16" and not self.use_fp16:
            raise RuntimeError("Person detector FP16 precision is only supported on CUDA.")

        self.detector = None
        if self.detector_name == "yolov8n":
            self._init_yolov8n()
        elif self.detector_name == "hog":
            self._init_hog()
        else:
            raise ValueError(f"Unsupported person detector '{self.detector_name}'.")

    def _init_yolov8n(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for person_detector='yolov8n'."
            ) from exc

        weights_path = Path(self.weights)
        if not weights_path.is_file():
            raise FileNotFoundError(f"Person detector weights not found: {weights_path}")
        self.detector = YOLO(str(weights_path))

    def _init_hog(self):
        if self.scale <= 1.0:
            raise ValueError("person_scale must be > 1.0 for HOG detectMultiScale.")
        if self.resize_width < 64:
            raise ValueError("person_resize_width must be >= 64.")
        self.detector = cv2.HOGDescriptor()
        self.detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def reset(self):
        self.eligible_frame_count = 0
        self.frames_since_positive = self.cooldown_frames + 1
        self.cached_boxes = []
        self.cached_scores = []
        self.active_frame_age = 0

    def _prepare_frame(self, frame):
        frame_height, frame_width = frame.shape[:2]
        if frame_width <= self.resize_width:
            return frame, 1.0
        scale_ratio = self.resize_width / float(frame_width)
        resized_height = max(64, int(round(frame_height * scale_ratio)))
        resized = cv2.resize(frame, (self.resize_width, resized_height), interpolation=cv2.INTER_LINEAR)
        return resized, frame_width / float(self.resize_width)

    def _filter_boxes(self, boxes, scores):
        kept_boxes = []
        kept_scores = []
        for box, score in zip(boxes, scores):
            area = max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))
            if area < self.min_box_area:
                continue
            kept_boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
            kept_scores.append(float(score))
        return kept_boxes, kept_scores

    def _detect_people_yolo(self, frame):
        boxes = []
        scores = []
        results = self.detector(
            frame,
            conf=self.threshold,
            verbose=False,
            half=self.use_fp16,
            device=self.device,
        )
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) != 0:
                    continue
                boxes.append(box.xyxy[0].tolist())
                scores.append(float(box.conf[0]))
        return self._filter_boxes(boxes, scores)

    def _detect_people_hog(self, frame):
        resized, scale_back = self._prepare_frame(frame)
        rects, weights = self.detector.detectMultiScale(
            resized,
            winStride=(8, 8),
            padding=(8, 8),
            scale=self.scale,
        )

        boxes = []
        scores = []
        for (x, y, w, h), weight in zip(rects, weights):
            score = float(weight)
            if score < self.hit_threshold:
                continue
            boxes.append(
                [
                    int(round(x * scale_back)),
                    int(round(y * scale_back)),
                    int(round((x + w) * scale_back)),
                    int(round((y + h) * scale_back)),
                ]
            )
            scores.append(score)
        return self._filter_boxes(boxes, scores)

    def _detect_people(self, frame):
        if self.detector_name == "yolov8n":
            return self._detect_people_yolo(frame)
        return self._detect_people_hog(frame)

    def update(self, frame, enabled=True):
        if not enabled:
            self.reset()
            return {
                "person_detected": False,
                "person_active": False,
                "person_boxes": [],
                "person_scores": [],
                "person_detector": self.detector_name,
                "person_detector_ran": False,
            }

        self.eligible_frame_count += 1
        if self.cached_boxes and self.active_frame_age < self.min_on_time:
            self.active_frame_age += 1
            return {
                "person_detected": True,
                "person_active": True,
                "person_boxes": list(self.cached_boxes),
                "person_scores": list(self.cached_scores),
                "person_detector": self.detector_name,
                "person_detector_ran": False,
            }

        boxes = []
        scores = []
        should_run_detector = ((self.eligible_frame_count - 1) % self.stride) == 0
        if should_run_detector:
            boxes, scores = self._detect_people(frame)
            if boxes:
                self.cached_boxes = boxes
                self.cached_scores = scores
                self.frames_since_positive = 0
                self.active_frame_age = 0
            else:
                self.frames_since_positive += 1
        elif self.cached_boxes:
            self.frames_since_positive += 1

        if self.cached_boxes and self.frames_since_positive > self.cooldown_frames:
            self.cached_boxes = []
            self.cached_scores = []
            self.active_frame_age = 0

        person_active = bool(self.cached_boxes)
        return {
            "person_detected": person_active,
            "person_active": person_active,
            "person_boxes": list(self.cached_boxes),
            "person_scores": list(self.cached_scores),
            "person_detector": self.detector_name,
            "person_detector_ran": should_run_detector,
        }
