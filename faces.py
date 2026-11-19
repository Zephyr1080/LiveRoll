"""Registering students and matching faces."""
import os
import tempfile
import cv2
import numpy as np
import face_recognition
import config
from database import run_sql, load_known_faces, known_encodings, known_names, known_regs
from camera import take_photo


def register_student(reg_no, name, image_path):
    """Detect the face in an image, extract its encoding and save the student."""
    rgb = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(rgb)
    if len(locations) != 1:
        return f"Not registered: found {len(locations)} faces in the image (need exactly 1)."
    encoding = face_recognition.face_encodings(rgb, locations)[0]
    run_sql("INSERT OR REPLACE INTO students(reg_no, name, encoding) VALUES (?,?,?)",
            (reg_no, name, encoding.tobytes()))
    safe_name = name.replace(" ", "_")
    cv2.imwrite(f"{config.DATASET_DIR}/{reg_no}_{safe_name}.jpg",
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    load_known_faces()
    return f"Registered: {name} ({reg_no})"


def register_from_folder(folder):
    """Register every photo in a folder named like  AP26122040014_Phani_Trinadh.jpg."""
    folder = folder.strip().strip('"')
    if not os.path.isdir(folder):
        print("Folder not found:", folder); return
    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(filename)[0]
        reg_no, _, name = stem.partition("_")
        print(register_student(reg_no, name.replace("_", " "), os.path.join(folder, filename)))


def register_from_webcam(reg_no, name):
    """Open the webcam preview (SPACE to capture) and register the student."""
    path = take_photo(os.path.join(tempfile.gettempdir(), f"register_{reg_no}.jpg"))
    if path is None:
        print("Cancelled.")
        return
    print(register_student(reg_no, name, path))


def match_face(encoding):
    """Return (reg_no, name) of the closest known face, or (None, 'Unknown')."""
    if known_encodings:
        distances = face_recognition.face_distance(known_encodings, encoding)
        best = int(np.argmin(distances))
        if distances[best] <= config.TOLERANCE:
            return known_regs[best], known_names[best]
    return None, "Unknown"


def draw_label(bgr, box, name, known):
    top, right, bottom, left = box
    colour = (0, 180, 0) if known else (0, 0, 255)      # green = known, red = unknown
    cv2.rectangle(bgr, (left, top), (right, bottom), colour, 3)
    cv2.rectangle(bgr, (left, bottom), (right, bottom + 34), colour, cv2.FILLED)
    cv2.putText(bgr, name, (left + 6, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
