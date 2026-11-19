"""Blink-based liveness check (no extra sensors, regular webcam only).

Idea: a printed photo or a phone screen showing a still picture never blinks.
We measure the Eye Aspect Ratio (EAR) in every frame of a short clip. EAR drops
sharply when the eye closes, so a drop followed by recovery = one blink.
"""
import numpy as np
import cv2
import face_recognition
import config


def eye_aspect_ratio(eye):
    """eye = 6 (x, y) points. EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)."""
    p = np.array(eye, dtype=float)
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return vertical / (2.0 * horizontal) if horizontal else 0.0


def face_width_ok(chin_points, image_width):
    """True if the face (jaw width) is big enough. Tiny faces, such as a face
    shown on a phone screen, give noisy eye landmarks and are rejected."""
    xs = [p[0] for p in chin_points]
    return (max(xs) - min(xs)) >= config.MIN_FACE_WIDTH_RATIO * image_width


def frame_ear(rgb):
    """Average EAR of both eyes, or None if there is not exactly one big enough face."""
    landmarks = face_recognition.face_landmarks(rgb)
    if len(landmarks) != 1:
        return None
    lm = landmarks[0]
    if not face_width_ok(lm["chin"], rgb.shape[1]):
        return None
    return (eye_aspect_ratio(lm["left_eye"]) + eye_aspect_ratio(lm["right_eye"])) / 2


def count_blinks(ears):
    """Count open -> closed -> open transitions in a list of EAR values."""
    valid = [e for e in ears if e is not None]
    if len(valid) < 5:
        return 0
    baseline = np.percentile(valid, 80)          # typical "eyes open" value
    if baseline < 0.15:                          # eyes never open properly
        return 0
    threshold = baseline * config.EAR_DROP_RATIO
    blinks, closed = 0, False
    for e in ears:
        if e is None:
            continue
        if e < threshold:
            closed = True
        elif closed:
            blinks += 1
            closed = False
    return blinks


def check_liveness(frames_bgr):
    """Analyse a clip. Returns a dict with: live, blinks, reason, best_frame."""
    if not frames_bgr:
        return {"live": False, "blinks": 0, "reason": "no frames captured", "best_frame": None}

    ears = []
    for bgr in frames_bgr:
        h, w = bgr.shape[:2]
        if w > config.PROCESS_WIDTH:
            bgr = cv2.resize(bgr, (config.PROCESS_WIDTH, int(h * config.PROCESS_WIDTH / w)))
        ears.append(frame_ear(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))

    found = [e for e in ears if e is not None]
    face_ratio = len(found) / len(frames_bgr)
    # Best frame for recognition = eyes most open (clear, front-facing face)
    if found:
        best = frames_bgr[int(np.argmax([e if e is not None else -1 for e in ears]))]
    else:
        best = frames_bgr[len(frames_bgr) // 2]

    if face_ratio < config.MIN_FACE_FRAMES:
        return {"live": False, "blinks": 0, "best_frame": best,
                "reason": "need ONE clear, close face for most of the clip (move closer)"}

    blinks = count_blinks(ears)
    live = blinks >= config.MIN_BLINKS
    return {"live": live, "blinks": blinks, "best_frame": best,
            "reason": "blink detected" if live else "no blink detected (possible photo/screen)"}
