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


def draw_active_tier_overlay(frame, tier_status, font_scale=0.8, thickness=2):
    rendered = frame.copy()
    active_color = COLORS["green"]
    inactive_color = (120, 120, 120)
    person_color = COLORS["blue"]
    action_color = COLORS["yellow"]

    lines = [
        {
            "text": "T1: MOTION",
            "active": bool(tier_status.get("motion_active")),
            "active_color": active_color,
        },
        {
            "text": f"T2: PERSON ({int(tier_status.get('person_count', 0))})",
            "active": bool(tier_status.get("person_active")),
            "active_color": person_color,
        },
        {
            "text": f"T3: ACTION ({int(tier_status.get('action_inference_count', 0))})",
            "active": bool(tier_status.get("action_display_active")),
            "active_color": action_color,
        },
    ]

    y = 36
    for line in lines:
        color = line["active_color"] if line["active"] else inactive_color
        cv2.putText(
            rendered,
            line["text"],
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += int(34 * font_scale) + 10
    return rendered
