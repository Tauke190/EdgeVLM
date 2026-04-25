import cv2
import numpy as np


class MotionGate:
    def __init__(self, threshold_area, motion_frames, cooldown_frames, blur_kernel, learning_rate):
        if blur_kernel % 2 == 0:
            raise ValueError("motion_blur_kernel must be odd.")
        self.threshold_area = int(threshold_area)
        self.motion_frames = int(motion_frames)
        self.cooldown_frames = int(cooldown_frames)
        self.blur_kernel = int(blur_kernel)
        self.learning_rate = float(learning_rate)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((5, 5), np.uint8)
        self.motion_frame_count = 0
        self.cooldown_count = 0
        self.motion_active = False
        self.motion_roi = None

    def update(self, frame):
        blurred = cv2.GaussianBlur(frame, (self.blur_kernel, self.blur_kernel), 0)
        fg_mask = self.bg_subtractor.apply(blurred, learningRate=self.learning_rate)
        fg_mask[fg_mask == 127] = 0
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [contour for contour in contours if cv2.contourArea(contour) > self.threshold_area]
        motion_detected = len(valid_contours) > 0

        if motion_detected:
            self.motion_frame_count += 1
            if self.motion_frame_count >= self.motion_frames:
                self.motion_active = True
            self.cooldown_count = 0
        else:
            self.motion_frame_count = 0

        if self.motion_active and not motion_detected:
            self.cooldown_count += 1
            if self.cooldown_count >= self.cooldown_frames:
                self.motion_active = False
                self.cooldown_count = 0

        self.motion_roi = None
        if self.motion_active and valid_contours:
            all_points = np.vstack(valid_contours)
            self.motion_roi = cv2.boundingRect(all_points)

        return {
            "motion_detected": motion_detected,
            "motion_active": self.motion_active,
            "motion_roi": self.motion_roi,
        }
