"""Demo and test cases."""
import os
import numpy as np
import pandas as pd
import cv2
import config
from database import run_sql, known_encodings, known_names, known_regs
from faces import register_student
from attendance import recognize_faces, now_parts, show_results
from camera import show_image
from reports import daily_report, show_table
from liveness import count_blinks


def run_demo():
    """Register a sample face from scikit-image and recognise it."""
    from skimage import data
    sample = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    cv2.imwrite("sample_registered.jpg", sample)
    print(register_student("DEMO001", "Demo Student", "sample_registered.jpg"))
    image, results = recognize_faces("sample_registered.jpg")
    show_image(image)
    show_results(results)
    print("\nDaily attendance report:")
    show_table(daily_report())


def make_group_image(paths, height=400):
    """Join several face photos side by side to simulate a group photo."""
    tiles = []
    for p in paths:
        img = cv2.imread(p)
        w = int(img.shape[1] * height / img.shape[0])
        tiles.append(cv2.resize(img, (w, height)))
    cv2.imwrite("group_test.jpg", np.hstack(tiles))
    return "group_test.jpg"


def reset_today(reg_no):
    run_sql("DELETE FROM attendance WHERE reg_no=? AND date=?", (reg_no, now_parts()[0]))


def run_liveness_tests():
    """Camera-free tests of the blink logic using made-up EAR sequences."""
    rng = np.random.default_rng(0)
    open_eyes = list(0.30 + rng.normal(0, 0.005, 30))
    blink_seq = open_eyes[:12] + [0.12, 0.09] + open_eyes[12:]
    cases = [
        ("Normal blink", blink_seq, 1),
        ("Still photo (no blink)", open_eyes, 0),
        ("Two blinks", open_eyes[:8] + [0.1] + open_eyes[8:20] + [0.1, 0.1] + open_eyes[20:], 2),
        ("Eyes always closed", [0.08] * 30, 0),
        ("Face missing", [None] * 30, 0),
    ]
    rows = []
    for name, ears, expected in cases:
        got = count_blinks(ears)
        rows.append({"Liveness Test": name, "Expected Blinks": expected,
                     "Actual Blinks": got, "Result": "PASS" if got == expected else "FAIL"})
    show_table(pd.DataFrame(rows))


def run_test_cases():
    if not known_regs:
        print("Register at least one student first."); return
    reg, name = known_regs[0], known_names[0]
    img_path = [f for f in os.listdir(config.DATASET_DIR) if f.startswith(reg + "_")][0]
    img_path = f"{config.DATASET_DIR}/{img_path}"
    rows = []

    def record(test, expected, passed, actual):
        rows.append({"Test Case": test, "Expected Result": expected,
                     "Actual Result": actual, "Result": "PASS" if passed else "FAIL"})

    # TC1 registered student detected
    reset_today(reg)
    _, res = recognize_faces(img_path)
    record("Registered student detected", "Attendance marked",
           len(res) == 1 and res[0]["status"].startswith("Present"),
           res[0]["status"] if res else "No face")

    # TC2 unknown person detected (hide all known faces for this test)
    saved = (known_encodings[:], known_names[:], known_regs[:])
    known_encodings.clear(); known_names.clear(); known_regs.clear()
    _, res = recognize_faces(img_path, mark=False)
    known_encodings.extend(saved[0]); known_names.extend(saved[1]); known_regs.extend(saved[2])
    record("Unknown person detected", "Not recognized",
           len(res) == 1 and res[0]["name"] == "Unknown",
           res[0]["status"] if res else "No face")

    # TC3 same student detected again
    _, res = recognize_faces(img_path)
    record("Same student detected again", "Duplicate entry prevented",
           len(res) == 1 and res[0]["status"].startswith("Duplicate"),
           res[0]["status"] if res else "No face")

    # TC4 multiple students detected
    if len(known_regs) >= 2:
        paths = []
        for r in known_regs[:2]:
            f = [x for x in os.listdir(config.DATASET_DIR) if x.startswith(r + "_")][0]
            paths.append(f"{config.DATASET_DIR}/{f}"); reset_today(r)
        _, res = recognize_faces(make_group_image(paths))
        ok = sum(1 for r in res if r["status"].startswith("Present")) == 2
        record("Multiple students detected", "Each recognized student recorded", ok,
               f"{sum(1 for r in res if r['reg_no'])} students recognised")
    else:
        rows.append({"Test Case": "Multiple students detected",
                     "Expected Result": "Each recognized student recorded",
                     "Actual Result": "Register 2 students to run this test", "Result": "SKIPPED"})

    # TC5 attendance report requested
    report = daily_report()
    record("Attendance report requested", "Report generated",
           len(report) >= 1 and os.path.exists("daily_report.csv"),
           f"{len(report)} rows, daily_report.csv saved")

    show_table(pd.DataFrame(rows))


def run_phone_tests():
    """Camera-free tests of the phone-near-face overlap logic."""
    from phone_check import phone_near_face
    face = (100, 300, 300, 100)                       # top, right, bottom, left
    cases = [
        ("Phone in front of face", (150, 150, 250, 250), True),
        ("Phone held next to face", (310, 120, 380, 260), True),
        ("Phone far away on desk", (600, 400, 700, 460), False),
    ]
    rows = []
    for name, box, expected in cases:
        got = phone_near_face(box, face)
        rows.append({"Phone Test": name, "Expected": expected, "Actual": got,
                     "Result": "PASS" if got == expected else "FAIL"})
    show_table(pd.DataFrame(rows))
