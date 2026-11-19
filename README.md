# Smart Attendance System

A student attendance system that recognizes faces from a regular webcam and checks that a **real, live person** is in front of the camera. It uses face recognition for identity, blink detection against photos, and YOLOv8n phone detection against screens, with a Streamlit web interface and a terminal menu.

## Features

- **Face registration** from a webcam capture or uploaded photos (one face per photo)
- **Face recognition** using 128-number face encodings and a distance threshold
- **Blink detection** (Eye Aspect Ratio) to reject still photos
- **Phone / screen detection** (YOLOv8n) to reject a phone held near the face
- **Duplicate prevention**: one attendance record per student per day
- **Reports**: daily and monthly attendance, with CSV download
- **Two interfaces**: Streamlit web app and a terminal menu
- Runs locally with a **regular webcam**, no depth sensors or special hardware

## How it works

```
Register (once)         Live attendance (every session)
photo -> 128-number     5 s webcam clip
encoding -> SQLite        -> blink check   (rejects photos)
                          -> phone check   (rejects phones/screens)
                          -> face match    (identity)
                          -> save record   (Present, date, time)
```

1. **Registration:** exactly one face must be found in the photo. Its encoding is saved in SQLite and the photo goes to `dataset/`.
2. **Blink check:** the clip (about 50 frames) is analysed with the Eye Aspect Ratio from eye landmarks. A dip followed by recovery counts as one blink. Tiny faces (such as a face on a phone screen) are ignored because their landmarks are noisy.
3. **Phone check:** YOLOv8n looks for a phone, TV or laptop in 8 frames. It only counts if the object overlaps the area around the face.
4. **Recognition:** the live face encoding is compared with all saved encodings. The closest match within the tolerance wins, otherwise the face is `Unknown`.
5. **Marking:** attendance is stored with date and time. A second scan on the same day shows `Duplicate`.

## Project structure

```
attendance_system/
├── app.py             # Streamlit interface
├── main.py            # Terminal menu
├── config.py          # All settings and thresholds
├── database.py        # SQLite helpers, in-memory known faces
├── camera.py          # Webcam preview, photo and clip capture
├── faces.py           # Registration and face matching
├── liveness.py        # Blink detection (EAR) and face-size check
├── phone_check.py     # YOLOv8n phone / screen detection
├── attendance.py      # Marking logic, live and photo modes
├── reports.py         # Daily and monthly reports
├── tests.py           # Demo and test cases
├── requirements.txt
├── .streamlit/config.toml
└── dataset/           # Registered face photos (created automatically)
```

## Requirements

- Python **3.11 or 3.12** (Python 3.14 has no prebuilt `dlib`, so installation fails)
- A webcam
- Windows, macOS or Linux (instructions below are for Windows PowerShell)

## Installation

```powershell
# 1. Create and activate a virtual environment
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip

# 2. Install dlib (prebuilt) and face_recognition without building dlib
pip install dlib-bin
pip install "setuptools<81" Click Pillow numpy face_recognition_models
pip install face_recognition --no-deps

# 3. Install the rest
pip install opencv-python pandas scikit-image ultralytics tzdata streamlit
```

On macOS or Linux, `pip install -r requirements.txt` works if CMake and a C++ compiler are installed.

The first phone check downloads the `yolov8n.pt` weights automatically.

## Usage

**Web interface (recommended)**

```powershell
python -m streamlit run app.py
```

| Page | What it does |
|---|---|
| Dashboard | Registered students, present today, not yet marked |
| Take attendance | Live webcam check (blink + phone), or upload a classroom photo |
| Register students | Webcam capture, upload one or many photos, list and delete |
| Reports | Daily and monthly tables, attendance chart, CSV download |

**Terminal menu**

```powershell
python main.py
```

Options: register (webcam or folder), take attendance (webcam or photo), daily report, monthly report, run tests.

**Registering from files:** name each photo `RegNo_Name.jpg`, for example `AP26122040014_R_Manoj.jpg`. Underscores in the name become spaces.

> The Streamlit app reads the webcam on the computer that runs it, so run it on the same machine as the camera.

## Configuration

Edit `config.py` (some options are also in the app sidebar).

| Setting | Default | Meaning |
|---|---|---|
| `TOLERANCE` | 0.5 | Face match distance, lower is stricter |
| `CAMERA_INDEX` | 0 | Webcam to use (try 1 for an external one) |
| `CLIP_SECONDS` / `CLIP_FPS` | 5 / 10 | Length and frame rate of the liveness clip |
| `EAR_DROP_RATIO` | 0.75 | Eye counts as closed below 75% of its open value |
| `MIN_BLINKS` | 1 | Blinks needed to pass |
| `MIN_FACE_FRAMES` | 0.6 | Share of frames that must show one clear face |
| `MIN_FACE_WIDTH_RATIO` | 0.18 | Smallest allowed face width (fraction of frame) |
| `PHONE_CONF` | 0.2 | YOLO confidence, lower catches more phones |
| `PHONE_CHECK_FRAMES` | 8 | Frames checked for phones |
| `PHONE_DEBUG` | True | Print what YOLO sees (set False once tuned) |
| `LIVENESS_ENABLED` / `PHONE_CHECK_ENABLED` | True | Turn each check on or off |

## Database

SQLite file `attendance.db`:

- `students(reg_no PRIMARY KEY, name, encoding)`
- `attendance(id, reg_no, name, date, time, status, liveness, UNIQUE(reg_no, date))`

## Testing

Run menu option 7 in `main.py`. It covers:

- Five system cases: registered student, unknown person, duplicate, group photo, report
- Blink counting (5 synthetic cases), phone-near-face (3 cases) and face-size (2 cases), none of which need a camera

## Limitations

- The phone check only catches a phone the camera can see.
- Live mode expects one student per clip. Photo mode handles groups but has no liveness check.
- Poor lighting or a weak webcam reduces blink and matching accuracy.

## Future work

- Random challenge ("turn left", "smile") to stop video replays
- Several photos per student for better matching
- A dedicated anti-spoofing model such as Silent-Face-Anti-Spoofing
- Online deployment and absence alerts

## Troubleshooting

| Problem | Fix |
|---|---|
| `Failed building wheel for dlib` | Use Python 3.12 and install `dlib-bin` as shown above |
| `No time zone found with key Asia/Kolkata` | `pip install tzdata` |
| `streamlit` is not recognized | Activate the venv and use `python -m streamlit run app.py` |
| Cannot open webcam | Close other apps using the camera, or change `CAMERA_INDEX` |
| Real student rejected with "move closer" | Sit closer, or lower `MIN_FACE_WIDTH_RATIO` |
| Phone not detected | Lower `PHONE_CONF` and check the `YOLO saw:` lines in the terminal |

## Privacy note

`attendance.db` and `dataset/` contain students' face data. They are excluded by the included `.gitignore`, so do not commit them to a public repository.


## Acknowledgements

[face_recognition](https://github.com/ageitgey/face_recognition) and [dlib](http://dlib.net), [OpenCV](https://opencv.org), [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [Streamlit](https://streamlit.io).
