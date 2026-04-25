import cv2


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


def resolve_color(name):
    if name not in COLORS:
        raise ValueError(f"Unsupported color '{name}'. Expected one of: {sorted(COLORS)}")
    return COLORS[name]


def draw_predictions(frame, boxes, labels, scores, color, font_scale, thickness):
    rendered = frame.copy()
    for box, label_list, score_list in zip(boxes, labels, scores):
        box_np = box.detach().cpu().numpy()
        start_point = (int(box_np[0]), int(box_np[1]))
        end_point = (int(box_np[2]), int(box_np[3]))
        cv2.rectangle(rendered, start_point, end_point, color, thickness)
        offset = 0
        for label, score in zip(label_list, score_list):
            text = f"{label} {round(float(score), 2)}"
            cv2.putText(
                rendered,
                text,
                (int(box_np[0]) - 5, int(box_np[1]) + offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            offset += 20
    return rendered
