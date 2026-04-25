from sia import PostProcessViz


class RuntimePostProcessor:
    def __init__(self, threshold):
        self.threshold = float(threshold)
        self.postprocess = PostProcessViz()

    def __call__(self, outputs, frame_size):
        result = self.postprocess(outputs, frame_size, human_conf=0.9, thresh=self.threshold)[0]
        return {
            "boxes": result["boxes"],
            "label_ids": result["labels"],
            "scores": result["scores"],
        }
