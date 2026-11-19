"""Central settings for the attendance system."""
import os
from zoneinfo import ZoneInfo

# ---- Recognition / storage ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMERA_INDEX = 0                # 0 = default webcam, try 1 for an external one
TOLERANCE = 0.5                 # lower = stricter face matching
DATASET_DIR = os.path.join(BASE_DIR, "dataset")        # registered face images are saved here
DB_PATH = os.path.join(BASE_DIR, "attendance.db")      # SQLite database file
TZ = ZoneInfo("Asia/Kolkata")   # Indian Standard Time

# ---- Liveness (blink detection) ----
LIVENESS_ENABLED = True
CLIP_SECONDS = 5                # how long the webcam clip lasts
CLIP_FPS = 10                   # frames captured per second
PROCESS_WIDTH = 480             # frames are shrunk to this width for speed
EAR_DROP_RATIO = 0.75           # eye counts as "closed" below 75% of its open value
MIN_BLINKS = 1                  # blinks needed to pass
MIN_FACE_FRAMES = 0.6           # at least 60% of frames must show exactly one face

os.makedirs(DATASET_DIR, exist_ok=True)

# ---- Phone / screen check (YOLOv8n) ----
PHONE_CHECK_ENABLED = True
PHONE_MODEL = "yolov8n.pt"      # downloaded automatically the first time
PHONE_CONF = 0.35               # lower = catches more phones, more false alarms
PHONE_CHECK_FRAMES = 5          # frames sampled from the clip
PHONE_FACE_MARGIN = 0.5         # face box is grown by 50% before testing overlap
