"""Phone / screen detection with YOLOv8n (pretrained on COCO: cell phone, tv, laptop).

If a phone is visible near the face in the clip, the attempt is rejected.
Limit: it only catches a phone the camera can actually see.
"""
import numpy as np
import cv2
import face_recognition
import config

_model = None


def _get_model():
    """Load YOLOv8n once (weights download automatically on first use)."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(config.PHONE_MODEL)
    return _model


def phone_near_face(phone_xyxy, face_box, margin=None):
    """True if the phone box overlaps the face box (grown by `margin`).

    phone_xyxy = (x1, y1, x2, y2); face_box = (top, right, bottom, left).
    """
    margin = config.PHONE_FACE_MARGIN if margin is None else margin
    px1, py1, px2, py2 = phone_xyxy
    top, right, bottom, left = face_box
    w, h = right - left, bottom - top
    fx1, fy1 = left - margin * w, top - margin * h
    fx2, fy2 = right + margin * w, bottom + margin * h
    return not (px2 < fx1 or px1 > fx2 or py2 < fy1 or py1 > fy2)


def check_phone(frames_bgr):
    """Scan a few frames of the clip. Returns dict: found, confidence, frame."""
    if not frames_bgr:
        return {"found": False, "confidence": 0.0, "frame": None}
    model = _get_model()
    n = min(config.PHONE_CHECK_FRAMES, len(frames_bgr))
    for i in np.linspace(0, len(frames_bgr) - 1, n).astype(int):
        frame = frames_bgr[i]
        if config.PHONE_DEBUG:                       # show everything YOLO sees (low confidence)
            seen = model.predict(frame, conf=0.1, verbose=False)[0]
            print("  YOLO saw:", ", ".join(f"{seen.names[int(c)]} {p:.2f}" for c, p in
                  zip(seen.boxes.cls.tolist(), seen.boxes.conf.tolist())) or "nothing")
        res = model.predict(frame, classes=config.PHONE_CLASSES,
                            conf=config.PHONE_CONF, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        faces = face_recognition.face_locations(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for xyxy, conf, cls in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
            if not faces or any(phone_near_face(xyxy, f) for f in faces):
                shown = frame.copy()
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(shown, f"{res.names[int(cls)].upper()} {conf:.2f}", (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return {"found": True, "confidence": float(conf), "frame": shown}
    return {"found": False, "confidence": 0.0, "frame": None}
