import time

from runtime.buffer import SlidingWindowBuffer
from runtime.inference_core import SIARuntimeCore
from runtime.metrics import maybe_cuda_synchronize
from runtime.motion import MotionGate
from runtime.person import PersonGate
from runtime.preprocess import build_clip_tensor, build_normalizer, resize_frame
from runtime.visualize import draw_active_tier_overlay, draw_predictions, resolve_color


class AlwaysOnSIAPipeline:
    def __init__(self, config):
        self.config = config
        self.core = SIARuntimeCore(config)
        self.buffer = SlidingWindowBuffer(config.buffer_max_len, config.num_frames)
        self.normalizer = build_normalizer(config.normalize_mean, config.normalize_std)
        self.color = resolve_color(config.color)
        self.motion_gate = None
        self.person_gate = None
        uses_motion_gate = config.pipeline_mode in {"motion_only", "motion_person_sia"}
        uses_person_gate = config.pipeline_mode in {"person_only", "motion_person_sia"}
        if uses_motion_gate:
            self.motion_gate = MotionGate(
                threshold_area=config.motion_threshold_area,
                motion_frames=config.motion_frames,
                cooldown_frames=config.motion_cooldown_frames,
                blur_kernel=config.motion_blur_kernel,
                learning_rate=config.motion_learning_rate,
            )
        if uses_person_gate:
            self.person_gate = PersonGate(
                detector=config.person_detector,
                weights=config.person_weights,
                threshold=config.person_threshold,
                precision=config.person_precision,
                device=config.device,
                stride=config.person_stride,
                cooldown_frames=config.person_cooldown_frames,
                hit_threshold=config.person_hit_threshold,
                scale=config.person_scale,
                resize_width=config.person_resize_width,
                min_box_area=config.person_min_box_area,
            )
        self.last_sia_push_index = None
        self.last_sia_wall_time = None
        self.prev_motion_active = False
        self.prev_person_active = False
        self.prev_sia_active = False
        self.last_completed_predictions = None
        self.last_action_persist_deadline_s = None
        self.sia_inference_count = 0
        self.adaptive_cap_updates = 0
        self.adaptive_active_loop_ema_s = None
        self.current_sia_target_fps = 0.0 if config.adaptive_sia_target_fps else float(config.sia_target_fps)

    def _render_in_pipeline_enabled(self):
        return self.config.render_enabled and self.config.mode != "live"

    def _timing_stub(self, preprocess_time):
        return {
            "preprocess_s": preprocess_time,
            "inference_s": 0.0,
            "postprocess_s": 0.0,
            "postprocess_filter_s": 0.0,
            "postprocess_nms_s": 0.0,
            "postprocess_threshold_s": 0.0,
            "label_decode_s": 0.0,
            "render_s": 0.0,
        }

    def _freeze_predictions(self, inference_result):
        boxes = []
        for box in inference_result.get("boxes", []):
            if hasattr(box, "detach"):
                boxes.append([int(value) for value in box.detach().cpu().tolist()])
            else:
                boxes.append([int(value) for value in box])
        labels = [list(label_list) for label_list in inference_result.get("labels", [])]
        scores = []
        for score_list in inference_result.get("scores", []):
            scores.append([float(score) for score in score_list])
        return {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
        }

    def _render_with_persisted_predictions(self, frame, tier_status=None):
        if not self._render_in_pipeline_enabled():
            return None, 0.0
        maybe_cuda_synchronize(self.core.device, self.config.sync_cuda_timing)
        render_start = time.perf_counter()
        rendered_frame = frame.copy()
        if self.last_completed_predictions:
            rendered_frame = draw_predictions(
                rendered_frame,
                self.last_completed_predictions["boxes"],
                self.last_completed_predictions["labels"],
                self.last_completed_predictions["scores"],
                self.color,
                self.config.font_scale,
                self.config.line_thickness,
            )
        if self.config.show_active_tiers:
            rendered_frame = draw_active_tier_overlay(rendered_frame, tier_status or self._tier_status(active=False))
        maybe_cuda_synchronize(self.core.device, self.config.sync_cuda_timing)
        render_time = time.perf_counter() - render_start
        return rendered_frame, render_time

    def _person_count(self, gate_state):
        person_boxes = gate_state.get("person_boxes") or []
        return len(person_boxes)

    def _persisted_action_allowed(self, gate_state, active=False, now_s=None):
        if active:
            return True
        if not self.last_completed_predictions:
            return False
        if self.config.pipeline_mode == "always_on":
            if self.last_action_persist_deadline_s is None:
                return False
            if now_s is None:
                now_s = time.perf_counter()
            return now_s <= self.last_action_persist_deadline_s
        if self.config.pipeline_mode == "motion_only":
            return bool(gate_state.get("motion_active"))
        if self.config.pipeline_mode == "person_only":
            return bool(gate_state.get("person_active"))
        if self.config.pipeline_mode == "motion_person_sia":
            return bool(gate_state.get("person_active"))
        return False

    def _tier_status(self, gate_state=None, active=False, now_s=None):
        gate_state = gate_state or {}
        action_display_active = self._persisted_action_allowed(gate_state, active=active, now_s=now_s)
        return {
            "motion_active": bool(gate_state.get("motion_active")),
            "person_active": bool(gate_state.get("person_active")),
            "person_count": self._person_count(gate_state),
            "sia_active": bool(active),
            "action_display_active": action_display_active,
            "displaying_persisted_action": action_display_active and not bool(active),
            "scheduler_state": None,
            "action_inference_count": self.sia_inference_count,
        }

    def _scheduler_state(self, buffer_ready, gate_state, active, stride_wait=False, cooldown=False, rate_wait=False):
        if active:
            return "sia_active"
        if rate_wait:
            return "sia_rate_wait"
        if stride_wait:
            return "sia_stride_wait"
        if cooldown:
            return "cooldown"

        if not buffer_ready:
            if self.config.pipeline_mode == "always_on":
                return "buffering"
            if gate_state["person_active"]:
                return "person_confirmed_buffering"
            if gate_state["motion_active"]:
                return "motion_buffering"
            return "warming_up"

        if self.config.pipeline_mode == "always_on":
            return "idle"
        if self.config.pipeline_mode == "motion_only":
            return "motion_idle"
        if self.config.pipeline_mode == "person_only":
            return "person_check"
        if self.config.pipeline_mode == "motion_person_sia":
            if gate_state["motion_active"]:
                return "person_check"
            return "idle"
        return "idle"

    def _sia_trigger_reason(self, gate_state):
        if self.config.pipeline_mode != "motion_person_sia":
            return "always_allowed"

        motion_rising = bool(gate_state.get("motion_rising"))
        person_rising = bool(gate_state.get("person_rising"))
        if motion_rising and self.config.sia_retrigger_on_motion_edge:
            return "motion_edge"
        if person_rising and self.config.sia_retrigger_on_person_edge:
            return "person_edge"
        if self.last_sia_push_index is None:
            return "initial_activation"
        pushed_since_last_sia = self.buffer.total_pushed - self.last_sia_push_index
        if pushed_since_last_sia >= self.config.sia_min_new_frames:
            return "min_new_frames"
        return None

    def _cooldown_active(self, gate_state):
        motion_active = bool(gate_state.get("motion_active"))
        person_active = bool(gate_state.get("person_active"))
        if self.config.pipeline_mode == "motion_only":
            return not motion_active and self.prev_motion_active
        if self.config.pipeline_mode == "person_only":
            return not person_active and self.prev_person_active
        if self.config.pipeline_mode == "motion_person_sia":
            return not motion_active and (self.prev_motion_active or self.prev_person_active or self.prev_sia_active)
        return False

    def _sia_rate_wait_active(self, now_s):
        target_fps = float(self.current_sia_target_fps)
        if target_fps <= 0.0 or self.last_sia_wall_time is None:
            return False
        min_interval_s = 1.0 / target_fps
        return (now_s - self.last_sia_wall_time) < min_interval_s

    def _adaptive_cap_enabled(self):
        return bool(self.config.adaptive_sia_target_fps)

    def _adaptive_cap_ready(self):
        return self.sia_inference_count >= int(self.config.adaptive_sia_warmup_frames)

    def _adaptive_cap_ema_ms(self):
        if self.adaptive_active_loop_ema_s is None:
            return None
        return self.adaptive_active_loop_ema_s * 1000.0

    def _adaptive_cap_status(self):
        return {
            "sia_target_fps_effective": float(self.current_sia_target_fps),
            "adaptive_sia_target_fps_enabled": self._adaptive_cap_enabled(),
            "adaptive_sia_target_fps_ready": self._adaptive_cap_ready(),
            "adaptive_sia_target_fps_updates": int(self.adaptive_cap_updates),
            "adaptive_sia_active_loop_ema_ms": self._adaptive_cap_ema_ms(),
        }

    def _update_adaptive_sia_target_fps(self, active_loop_s):
        if not self._adaptive_cap_enabled():
            return
        smoothing = min(max(float(self.config.adaptive_sia_smoothing), 0.0), 1.0)
        if self.adaptive_active_loop_ema_s is None:
            self.adaptive_active_loop_ema_s = active_loop_s
        else:
            self.adaptive_active_loop_ema_s = (
                (1.0 - smoothing) * self.adaptive_active_loop_ema_s
                + smoothing * active_loop_s
            )
        if not self._adaptive_cap_ready():
            self.current_sia_target_fps = 0.0
            return
        ema_s = self.adaptive_active_loop_ema_s
        if ema_s <= 0.0:
            return
        measured_fps = 1.0 / ema_s
        utilization = max(float(self.config.adaptive_sia_utilization), 0.01)
        target_fps = measured_fps * utilization
        min_fps = max(float(self.config.adaptive_sia_min_fps), 0.0)
        max_fps = self.config.adaptive_sia_max_fps
        if max_fps is not None and max_fps > 0.0:
            target_fps = min(target_fps, float(max_fps))
        target_fps = max(target_fps, min_fps)
        self.current_sia_target_fps = target_fps
        self.adaptive_cap_updates += 1

    def _result(self, payload, gate_state, active, now_s=None):
        payload.update(self._adaptive_cap_status())
        return self._finalize_result(payload, gate_state, active=active, now_s=now_s)

    def _finalize_result(self, result, gate_state, active, now_s=None):
        tier_status = self._tier_status(gate_state=gate_state, active=active, now_s=now_s)
        tier_status["scheduler_state"] = result.get("scheduler_state")
        result["tier_status"] = tier_status
        self.prev_motion_active = bool(gate_state.get("motion_active"))
        self.prev_person_active = bool(gate_state.get("person_active"))
        self.prev_sia_active = bool(active)
        return result

    def process_frame(self, frame, frame_size):
        frame_start_s = time.perf_counter()
        preprocess_start = time.perf_counter()
        original_chw = frame.transpose(2, 0, 1)
        resized = resize_frame(frame, self.config.img_width, self.config.img_height)
        resized_chw = resized.transpose(2, 0, 1)
        self.buffer.push(resized_chw, original_chw)
        preprocess_time = time.perf_counter() - preprocess_start
        gate_state = {
            "motion_detected": None,
            "motion_active": None,
            "motion_roi": None,
            "person_detected": None,
            "person_active": None,
            "person_boxes": [],
            "person_scores": [],
            "person_detector": None,
            "person_detector_ran": False,
        }
        if self.motion_gate is not None:
            gate_state.update(self.motion_gate.update(frame))
        if self.person_gate is not None:
            person_gate_enabled = True
            if self.config.pipeline_mode == "motion_person_sia":
                person_gate_enabled = bool(gate_state["motion_active"])
            gate_state.update(self.person_gate.update(frame, enabled=person_gate_enabled))
        gate_state["motion_rising"] = bool(gate_state.get("motion_active")) and not self.prev_motion_active
        gate_state["person_rising"] = bool(gate_state.get("person_active")) and not self.prev_person_active

        if not self._persisted_action_allowed(gate_state, active=False, now_s=frame_start_s):
            self.last_completed_predictions = None
            self.last_action_persist_deadline_s = None

        buffer_ready = self.buffer.ready()
        cooldown_active = self._cooldown_active(gate_state)

        if not buffer_ready:
            scheduler_state = self._scheduler_state(buffer_ready, gate_state, active=False)
            rendered_frame, render_time = self._render_with_persisted_predictions(
                frame,
                tier_status=self._tier_status(gate_state=gate_state, active=False, now_s=frame_start_s),
            )
            return self._result({
                "active": False,
                "output_ready": False,
                "scheduler_state": scheduler_state,
                "sia_trigger_reason": None,
                "rendered_frame": rendered_frame,
                "detections": 0,
                "gate_state": gate_state,
                "timings": {
                    **self._timing_stub(preprocess_time),
                    "render_s": render_time,
                },
            }, gate_state, active=False, now_s=frame_start_s)

        if self.config.pipeline_mode == "motion_only" and not gate_state["motion_active"]:
            scheduler_state = self._scheduler_state(
                buffer_ready,
                gate_state,
                active=False,
                cooldown=cooldown_active,
            )
            rendered_frame, render_time = self._render_with_persisted_predictions(
                self.buffer.render_frame(),
                tier_status=self._tier_status(gate_state=gate_state, active=False, now_s=frame_start_s),
            )
            return self._result({
                "active": False,
                "output_ready": True,
                "scheduler_state": scheduler_state,
                "sia_trigger_reason": None,
                "rendered_frame": rendered_frame,
                "detections": 0,
                "gate_state": gate_state,
                "timings": {
                    **self._timing_stub(preprocess_time),
                    "render_s": render_time,
                },
            }, gate_state, active=False, now_s=frame_start_s)
        if self.config.pipeline_mode == "person_only" and not gate_state["person_active"]:
            scheduler_state = self._scheduler_state(
                buffer_ready,
                gate_state,
                active=False,
                cooldown=cooldown_active,
            )
            rendered_frame, render_time = self._render_with_persisted_predictions(
                self.buffer.render_frame(),
                tier_status=self._tier_status(gate_state=gate_state, active=False, now_s=frame_start_s),
            )
            return self._result({
                "active": False,
                "output_ready": True,
                "scheduler_state": scheduler_state,
                "sia_trigger_reason": None,
                "rendered_frame": rendered_frame,
                "detections": 0,
                "gate_state": gate_state,
                "timings": {
                    **self._timing_stub(preprocess_time),
                    "render_s": render_time,
                },
            }, gate_state, active=False, now_s=frame_start_s)
        if self.config.pipeline_mode == "motion_person_sia":
            if not gate_state["motion_active"] or not gate_state["person_active"]:
                scheduler_state = self._scheduler_state(
                    buffer_ready,
                    gate_state,
                    active=False,
                    cooldown=cooldown_active,
                )
                rendered_frame, render_time = self._render_with_persisted_predictions(
                    self.buffer.render_frame(),
                    tier_status=self._tier_status(gate_state=gate_state, active=False, now_s=frame_start_s),
                )
                return self._result({
                    "active": False,
                    "output_ready": True,
                    "scheduler_state": scheduler_state,
                    "sia_trigger_reason": None,
                    "rendered_frame": rendered_frame,
                    "detections": 0,
                    "gate_state": gate_state,
                    "timings": {
                        **self._timing_stub(preprocess_time),
                        "render_s": render_time,
                    },
                }, gate_state, active=False, now_s=frame_start_s)
            trigger_reason = self._sia_trigger_reason(gate_state)
            if trigger_reason is None:
                scheduler_state = self._scheduler_state(
                    buffer_ready,
                    gate_state,
                    active=False,
                    stride_wait=True,
                )
                rendered_frame, render_time = self._render_with_persisted_predictions(
                    self.buffer.render_frame(),
                    tier_status=self._tier_status(gate_state=gate_state, active=False, now_s=frame_start_s),
                )
                return self._result({
                    "active": False,
                    "output_ready": True,
                    "scheduler_state": scheduler_state,
                    "sia_trigger_reason": None,
                    "rendered_frame": rendered_frame,
                    "detections": 0,
                    "gate_state": gate_state,
                    "timings": {
                        **self._timing_stub(preprocess_time),
                        "render_s": render_time,
                    },
                }, gate_state, active=False, now_s=frame_start_s)
        else:
            trigger_reason = "always_allowed"

        if self._sia_rate_wait_active(frame_start_s):
            scheduler_state = self._scheduler_state(
                buffer_ready,
                gate_state,
                active=False,
                rate_wait=True,
            )
            rendered_frame, render_time = self._render_with_persisted_predictions(
                self.buffer.render_frame(),
                tier_status=self._tier_status(gate_state=gate_state, active=False, now_s=frame_start_s),
            )
            return self._result({
                "active": False,
                "output_ready": True,
                "scheduler_state": scheduler_state,
                "sia_trigger_reason": None,
                "rendered_frame": rendered_frame,
                "detections": 0,
                "gate_state": gate_state,
                "timings": {
                    **self._timing_stub(preprocess_time),
                    "render_s": render_time,
                },
            }, gate_state, active=False, now_s=frame_start_s)

        clip_tensor = build_clip_tensor(
            self.buffer.sampled_clip(),
            self.normalizer,
            self.core.device,
            self.core.input_use_fp16,
        )
        inference_result = self.core.infer_clip(clip_tensor, frame_size)
        self.last_completed_predictions = self._freeze_predictions(inference_result)
        self.last_action_persist_deadline_s = frame_start_s + (float(self.config.action_persist_ms) / 1000.0)
        self.sia_inference_count += 1
        self.last_sia_push_index = self.buffer.total_pushed
        self.last_sia_wall_time = frame_start_s
        rendered_frame = None
        render_time = 0.0
        if self._render_in_pipeline_enabled():
            render_base = self.buffer.render_frame()
            rendered_frame, render_time = self._render_with_persisted_predictions(
                render_base,
                tier_status=self._tier_status(gate_state=gate_state, active=True, now_s=frame_start_s),
            )
        active_loop_estimate_s = (
            preprocess_time
            + inference_result["timings"]["inference_s"]
            + inference_result["timings"]["postprocess_s"]
            + inference_result["timings"]["label_decode_s"]
            + render_time
        )
        self._update_adaptive_sia_target_fps(active_loop_estimate_s)
        return self._result({
            "active": True,
            "output_ready": True,
            "scheduler_state": self._scheduler_state(buffer_ready, gate_state, active=True),
            "sia_trigger_reason": trigger_reason,
            "rendered_frame": rendered_frame,
            "detections": inference_result["num_detections"],
            "boxes": inference_result["boxes"],
            "labels": inference_result["labels"],
            "scores": inference_result["scores"],
            "gate_state": gate_state,
            "timings": {
                "preprocess_s": preprocess_time,
                **inference_result["timings"],
                "render_s": render_time,
            },
        }, gate_state, active=True, now_s=frame_start_s)
