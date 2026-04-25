import cv2


def open_capture(config):
    if config.mode == "offline":
        if not config.video_path:
            raise ValueError("Offline runtime requires config.video_path.")
        capture = cv2.VideoCapture(config.video_path)
        source_name = config.video_path
    elif config.mode == "live":
        capture = cv2.VideoCapture(config.video_device)
        source_name = f"camera:{config.video_device}"
    else:
        raise ValueError(f"Unsupported runtime mode '{config.mode}'.")

    if not capture.isOpened():
        raise RuntimeError(f"Could not open source '{source_name}'.")
    return capture
