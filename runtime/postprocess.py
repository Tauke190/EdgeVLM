from sia import PostProcessViz


class RuntimePostProcessor:
    def __init__(self, threshold):
        self.threshold = float(threshold)
        self.postprocess = PostProcessViz()

    def __call__(self, outputs, frame_size, return_stage_timings=False):
        postprocess_result = self.postprocess(
            outputs,
            frame_size,
            human_conf=0.9,
            thresh=self.threshold,
            return_stage_timings=return_stage_timings,
        )
        if return_stage_timings:
            result, breakdown = postprocess_result
            result = result[0]
        else:
            result = postprocess_result[0]
            breakdown = None

        return {
            "boxes": result["boxes"],
            "label_ids": result["labels"],
            "scores": result["scores"],
            "stage_timings": breakdown,
        }
