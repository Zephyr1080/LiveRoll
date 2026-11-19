"""Marking attendance: from a live webcam clip (with liveness) or from a photo."""
import os
from datetime import datetime
import cv2
import face_recognition
import config
from database import run_sql
from faces import match_face, draw_label
from camera import show_image, record_clip
from liveness import check_liveness


def now_parts():
    now = datetime.now(config.TZ)
    return now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")


def mark_attendance(reg_no, name, liveness="Not checked"):
    """Record attendance with date and time. Returns a status message."""
    date, time = now_parts()
    already = run_sql("SELECT 1 FROM attendance WHERE reg_no=? AND date=?",
                      (reg_no, date), fetch=True)
    if already:                                   # duplicate prevention
        return "Duplicate - already marked today"
    run_sql("INSERT INTO attendance(reg_no, name, date, time, status, liveness) "
            "VALUES (?,?,?,?,?,?)", (reg_no, name, date, time, "Present", liveness))
    return "Present - attendance marked"


def recognize_array(bgr, mark=True, liveness="Not checked"):
    """Detect faces in a BGR image, match them and (optionally) mark attendance."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)
    results = []
    for box, encoding in zip(locations, encodings):
        reg_no, name = match_face(encoding)
        status = "Not recognized"
        if reg_no is not None:
            status = mark_attendance(reg_no, name, liveness) if mark else "Recognized"
        draw_label(bgr, box, name, reg_no is not None)
        results.append({"name": name, "reg_no": reg_no, "status": status})
    return bgr, results


def recognize_faces(image_path, mark=True):
    """Photo mode (e.g. a classroom photo). No liveness check here."""
    return recognize_array(cv2.imread(image_path), mark, "Not checked")


def _rejected(image, reason):
    return image, [{"name": "Unknown", "reg_no": None, "status": "Rejected - " + reason}]


def live_attendance(frames=None):
    """Webcam mode: clip -> blink check -> phone check -> recognize -> mark attendance."""
    if frames is None:
        frames = record_clip(config.CLIP_SECONDS, config.CLIP_FPS)
    best = frames[len(frames) // 2]

    if config.LIVENESS_ENABLED:
        live = check_liveness(frames)
        print(f"Liveness: {'PASS' if live['live'] else 'FAIL'} "
              f"({live['blinks']} blink(s) - {live['reason']})")
        if not live["live"]:
            return _rejected(live["best_frame"] if live["best_frame"] is not None else frames[0],
                             live["reason"])
        best = live["best_frame"]

    if config.PHONE_CHECK_ENABLED:
        from phone_check import check_phone          # loaded only when enabled
        phone = check_phone(frames)
        print("Phone check:", f"FAIL (phone detected, {phone['confidence']:.2f})"
              if phone["found"] else "PASS")
        if phone["found"]:
            return _rejected(phone["frame"], "phone/screen detected near face")

    return recognize_array(best, mark=True,
                           liveness="Verified" if config.LIVENESS_ENABLED else "Not checked")


def run_attendance_session(students=3):
    """One student at a time in front of the camera, each with a liveness check."""
    for i in range(students):
        print(f"--- Student {i+1} of {students}: look at the camera and blink ---")
        image, results = live_attendance()
        show_image(image)
        show_results(results)


def recognize_photo(image_path):
    """Photo mode (e.g. a classroom photo). No liveness or phone check here."""
    if not os.path.isfile(image_path):
        print("File not found:", image_path); return
    image, results = recognize_faces(image_path)
    show_results(results)
    show_image(image)


def show_results(results):
    if not results:
        print("No face detected in this image.")
    for r in results:
        print(f"  {r['name']:<25} {str(r['reg_no']):<16} -> {r['status']}")
