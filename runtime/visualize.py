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
        if hasattr(box, "detach"):
            x1, y1, x2, y2 = (int(value) for value in box.detach().cpu().tolist())
        else:
            x1, y1, x2, y2 = (int(value) for value in box)
        start_point = (x1, y1)
        end_point = (x2, y2)
        cv2.rectangle(rendered, start_point, end_point, color, thickness)
        offset = 0
        for label, score in zip(label_list, score_list):
            text = f"{label} {float(score):.2f}"
            cv2.putText(
                rendered,
                text,
                (x1 - 5, y1 + offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            offset += 20
    return rendered
